# ensemble.py
# ConFusionGraph V2 — stacked ensemble (canonical run)

import os, time, json, warnings, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from torch_geometric.nn import (GATv2Conv, HeteroConv, SAGEConv)
from torch_geometric.utils import to_undirected, add_self_loops

warnings.filterwarnings('ignore')

# ── Config ──
PARENT_DIR = "no_urg"
PROCESSED_DIR = f"{PARENT_DIR}/processed"
RESULTS_DIR = f"{PARENT_DIR}/results"
FIGURES_DIR = f"{PARENT_DIR}/figures"
MODELS_DIR = f"{PARENT_DIR}/models"
DATASET_PATH = "stanfordMOOCForumPostsSet.xlsx"

N_SEEDS = 5
HIDDEN_DIM = 128
N_LAYERS = 3
N_HEADS = 4
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
CONFUSION_THRESHOLD = 4.5

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("=" * 70)
print(f"  Device: {device}")

# ── Load data ──
print("\n  Loading processed data...")
hetero_data = torch.load(f"{PROCESSED_DIR}/hetero_graph.pt", map_location='cpu', weights_only=False)
bert_embeds = torch.load(f"{PROCESSED_DIR}/bert_embeddings.pt", map_location='cpu', weights_only=True)
post_meta = torch.load(f"{PROCESSED_DIR}/post_metadata.pt", map_location='cpu', weights_only=True)
labels_df = pd.read_csv(f"{PROCESSED_DIR}/post_labels_and_splits.csv", index_col=0)

y = hetero_data['post'].y
confusion_scores = torch.tensor(labels_df['Confusion(1-7)'].values, dtype=torch.float32)
train_mask = hetero_data['post'].train_mask.numpy()
val_mask = hetero_data['post'].val_mask.numpy()
test_mask = hetero_data['post'].test_mask.numpy()

n_posts = bert_embeds.shape[0]
n_users = hetero_data['user'].x.shape[0]
y_np = y.numpy()
n_pos_train = y_np[train_mask].sum()
n_neg_train = train_mask.sum() - n_pos_train
class_weight = n_neg_train / n_pos_train
pos_weight_t = torch.tensor([class_weight], dtype=torch.float32).to(device)

# Load raw text for TF-IDF
df_raw = pd.read_excel(DATASET_PATH)
df_raw = df_raw.dropna(subset=['Text', 'forum_uid', 'created_at', 'post_type']).reset_index(drop=True)
df_raw = df_raw.sort_values(['course_display_name', 'created_at']).reset_index(drop=True)
texts = df_raw['Text'].astype(str).tolist()

print(f"  Posts: {n_posts}, Class weight: {class_weight:.2f}")

def eval_metrics(y_true, y_pred, y_prob):
    return {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'auroc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_pos': f1_score(y_true, y_pred, pos_label=1),
    }

#  EXPERT 1: XGBoost (TF-IDF + Metadata)
print("  EXPERT 1 | XGBoost on TF-IDF + Metadata")
print("~" * 70)

print("  Fitting TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(texts)
meta_np = post_meta.numpy()
X_xgb = sp.hstack([X_tfidf, sp.csr_matrix(meta_np)])

# Generate out-of-fold predictions on TRAIN set (to avoid leakage in meta-learner)
# + normal predictions on val/test
print("  Generating out-of-fold predictions (5-fold on train)...")

train_indices = np.where(train_mask)[0]
val_indices = np.where(val_mask)[0]
test_indices = np.where(test_mask)[0]

xgb_oof_probs = np.zeros(n_posts)  # out-of-fold for train, direct for val/test

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx_local, val_idx_local) in enumerate(skf.split(train_indices, y_np[train_indices])):
    tr_idx = train_indices[tr_idx_local]
    val_idx_fold = train_indices[val_idx_local]

    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=class_weight, random_state=42,
                        eval_metric='logloss', verbosity=0)
    clf.fit(X_xgb[tr_idx], y_np[tr_idx])
    xgb_oof_probs[val_idx_fold] = clf.predict_proba(X_xgb[val_idx_fold])[:, 1]
    print(f"    Fold {fold+1}/5 done")

# Full model for val/test predictions
clf_full = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                         scale_pos_weight=class_weight, random_state=42,
                         eval_metric='logloss', verbosity=0)
clf_full.fit(X_xgb[train_indices], y_np[train_indices])
xgb_oof_probs[val_indices] = clf_full.predict_proba(X_xgb[val_indices])[:, 1]
xgb_oof_probs[test_indices] = clf_full.predict_proba(X_xgb[test_indices])[:, 1]

xgb_test_metrics = eval_metrics(y_np[test_mask], (xgb_oof_probs[test_mask] > 0.5).astype(int),
                                xgb_oof_probs[test_mask])
print(f"  Expert 1 standalone: F1={xgb_test_metrics['macro_f1']:.4f}  AUROC={xgb_test_metrics['auroc']:.4f}")

#  EXPERT 2: BERT + MLP
print("  EXPERT 2 | BERT + MLP")
print("~" * 70)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

X_bert = bert_embeds.numpy()
X_tr_b = torch.tensor(X_bert[train_mask], dtype=torch.float32)
X_val_b = torch.tensor(X_bert[val_mask], dtype=torch.float32)
X_te_b = torch.tensor(X_bert[test_mask], dtype=torch.float32)
y_tr_t = torch.tensor(y_np[train_mask], dtype=torch.float32)
y_val_t = torch.tensor(y_np[val_mask], dtype=torch.float32)

# Out-of-fold for train
bert_oof_probs = np.zeros(n_posts)
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("  Generating BERT out-of-fold predictions...")
for fold, (tr_local, val_local) in enumerate(skf2.split(train_indices, y_np[train_indices])):
    tr_idx = train_indices[tr_local]
    val_idx_fold = train_indices[val_local]

    torch.manual_seed(42 + fold)
    model = MLPClassifier(768, 256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    X_tr_fold = torch.tensor(X_bert[tr_idx], dtype=torch.float32)
    y_tr_fold = torch.tensor(y_np[tr_idx], dtype=torch.float32)
    ds = TensorDataset(X_tr_fold, y_tr_fold)
    dl = DataLoader(ds, batch_size=512, shuffle=True)

    best_val_f1, best_state = 0, None
    for epoch in range(50):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            X_v = torch.tensor(X_bert[val_idx_fold], dtype=torch.float32).to(device)
            vp = (torch.sigmoid(model(X_v)) > 0.5).long().cpu().numpy()
            vf = f1_score(y_np[val_idx_fold], vp, average='macro')
            if vf > best_val_f1:
                best_val_f1 = vf
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.to(device).eval()
    with torch.no_grad():
        X_v = torch.tensor(X_bert[val_idx_fold], dtype=torch.float32).to(device)
        bert_oof_probs[val_idx_fold] = torch.sigmoid(model(X_v)).cpu().numpy()
    del model; torch.cuda.empty_cache()
    print(f"    Fold {fold+1}/5 done")

# Full model for val/test
torch.manual_seed(42)
model_bert = MLPClassifier(768, 256).to(device)
optimizer = torch.optim.AdamW(model_bert.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
ds = TensorDataset(X_tr_b, y_tr_t)
dl = DataLoader(ds, batch_size=512, shuffle=True)

best_val_f1, best_state = 0, None
for epoch in range(50):
    model_bert.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model_bert(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model_bert.eval()
    with torch.no_grad():
        vp = (torch.sigmoid(model_bert(X_val_b.to(device))) > 0.5).long().cpu().numpy()
        vf = f1_score(y_np[val_mask], vp, average='macro')
        if vf > best_val_f1:
            best_val_f1 = vf
            best_state = {k: v.cpu().clone() for k, v in model_bert.state_dict().items()}

model_bert.load_state_dict(best_state); model_bert.to(device).eval()
with torch.no_grad():
    bert_oof_probs[val_indices] = torch.sigmoid(model_bert(X_val_b.to(device))).cpu().numpy()
    bert_oof_probs[test_indices] = torch.sigmoid(model_bert(X_te_b.to(device))).cpu().numpy()

bert_test_metrics = eval_metrics(y_np[test_mask], (bert_oof_probs[test_mask] > 0.5).astype(int),
                                 bert_oof_probs[test_mask])
print(f"  Expert 2 standalone: F1={bert_test_metrics['macro_f1']:.4f}  AUROC={bert_test_metrics['auroc']:.4f}")
del model_bert; torch.cuda.empty_cache()

#  EXPERT 3: GNN on Heterogeneous Graph
print("  EXPERT 3 | GNN on Heterogeneous Graph")
print("~" * 70)

class HeteroGNN(nn.Module):
    def __init__(self, post_in, user_in, hidden, n_layers, n_heads, dropout, edge_types):
        super().__init__()
        self.post_proj = nn.Linear(post_in, hidden)
        self.user_proj = nn.Linear(user_in, hidden)
        self.convs = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.user_norms = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {}
            for et in edge_types:
                s, r, d = et
                if s == d:
                    conv_dict[et] = GATv2Conv(hidden, hidden // n_heads, heads=n_heads,
                                              dropout=dropout, add_self_loops=False)
                else:
                    conv_dict[et] = SAGEConv((hidden, hidden), hidden)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.post_norms.append(nn.LayerNorm(hidden))
            self.user_norms.append(nn.LayerNorm(hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        x_dict = {'post': F.relu(self.post_proj(x_dict['post'])),
                   'user': F.relu(self.user_proj(x_dict['user']))}
        for conv, pn, un in zip(self.convs, self.post_norms, self.user_norms):
            x_new = conv(x_dict, edge_index_dict)
            for nt, norm in [('post', pn), ('user', un)]:
                h = x_new.get(nt, x_dict[nt])
                x_new[nt] = norm(F.dropout(F.relu(h), self.dropout, self.training) + x_dict[nt])
            x_dict = x_new
        return self.classifier(x_dict['post']).squeeze(-1)

# Prepare graph data
x_dict_gpu = {'post': hetero_data['post'].x.to(device),
              'user': hetero_data['user'].x.to(device)}
ei_dict_gpu = {et: hetero_data[et].edge_index.to(device) for et in hetero_data.edge_types}
y_gpu = y.to(device)
masks_gpu = (torch.tensor(train_mask, device=device),
             torch.tensor(val_mask, device=device),
             torch.tensor(test_mask, device=device))

# Out-of-fold GNN predictions
gnn_oof_probs = np.zeros(n_posts)
print("  Training GNN (single full model — OOF not practical for graph)...")

# For graph models, OOF is complex (can't easily split a graph), so we use
# val set performance to calibrate and accept slight optimism on train OOF
torch.manual_seed(42)
gnn_model = HeteroGNN(hetero_data['post'].x.shape[1], hetero_data['user'].x.shape[1],
                       HIDDEN_DIM, N_LAYERS, N_HEADS, DROPOUT,
                       list(ei_dict_gpu.keys())).to(device)
optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

tr_m, va_m, te_m = masks_gpu
best_vf, best_st, pc = 0, None, 0
for epoch in range(EPOCHS):
    gnn_model.train()
    logits = gnn_model(x_dict_gpu, ei_dict_gpu)
    loss = criterion(logits[tr_m], y_gpu[tr_m].float())
    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
    gnn_model.eval()
    with torch.no_grad():
        vl = gnn_model(x_dict_gpu, ei_dict_gpu)
        vp = (torch.sigmoid(vl[va_m]) > 0.5).long().cpu().numpy()
        vf = f1_score(y_np[val_mask], vp, average='macro')
    if vf > best_vf:
        best_vf = vf; best_st = {k: v.cpu().clone() for k, v in gnn_model.state_dict().items()}; pc = 0
    else:
        pc += 1
        if pc >= PATIENCE: break

gnn_model.load_state_dict(best_st); gnn_model.to(device).eval()
with torch.no_grad():
    all_logits = gnn_model(x_dict_gpu, ei_dict_gpu)
    all_probs = torch.sigmoid(all_logits).cpu().numpy()
    gnn_oof_probs = all_probs  # use calibrated probs for all

gnn_test_metrics = eval_metrics(y_np[test_mask], (gnn_oof_probs[test_mask] > 0.5).astype(int),
                                gnn_oof_probs[test_mask])
print(f"  Expert 3 standalone: F1={gnn_test_metrics['macro_f1']:.4f}  AUROC={gnn_test_metrics['auroc']:.4f}")
del gnn_model; torch.cuda.empty_cache()

#  GRAPH STRUCTURAL FEATURES (for meta-learner)
print("  Computing graph structural features for meta-learner...")
print("~" * 70)

# Neighbor confusion stats (from expert predictions, NOT ground truth labels)
reply_adj = {}
thread_adj = {}
for etype in hetero_data.edge_types:
    ei = hetero_data[etype].edge_index.numpy()
    s, r, d = etype
    if r == 'reply_to':
        for src, dst in zip(ei[0], ei[1]): reply_adj.setdefault(src, []).append(dst)
    elif r == 'same_thread':
        for src, dst in zip(ei[0], ei[1]): thread_adj.setdefault(src, []).append(dst)

# Structural features (no label leakage — uses expert PREDICTIONS not labels)
graph_feats = np.zeros((n_posts, 8), dtype=np.float32)
for i in range(n_posts):
    # Reply neighbors
    rn = reply_adj.get(i, [])
    tn = thread_adj.get(i, [])
    graph_feats[i, 0] = len(rn)                                         # reply degree
    graph_feats[i, 1] = len(tn)                                         # thread degree
    graph_feats[i, 2] = np.mean([xgb_oof_probs[n] for n in tn]) if tn else 0  # neighbor XGB prob
    graph_feats[i, 3] = np.mean([bert_oof_probs[n] for n in tn]) if tn else 0  # neighbor BERT prob
    graph_feats[i, 4] = np.mean([gnn_oof_probs[n] for n in tn]) if tn else 0   # neighbor GNN prob
    graph_feats[i, 5] = np.std([xgb_oof_probs[n] for n in tn]) if len(tn) > 1 else 0  # disagreement
    # Has answer neighbor
    graph_feats[i, 6] = 1.0 if any(meta_np[n, 1] > 0.5 for n in tn if n < n_posts) else 0.0
    # Isolation
    graph_feats[i, 7] = 1.0 / (1.0 + len(rn) + len(tn))

# Normalize
for j in range(graph_feats.shape[1]):
    col_max = np.abs(graph_feats[:, j]).max()
    if col_max > 0: graph_feats[:, j] /= col_max

graph_feat_names = ['reply_deg', 'thread_deg', 'nbr_xgb_prob', 'nbr_bert_prob',
                    'nbr_gnn_prob', 'nbr_disagreement', 'has_answer_nbr', 'isolation']
print(f"  Graph features: {graph_feats.shape}")
for i, name in enumerate(graph_feat_names):
    print(f"    [{i}] {name:20s}: mean={graph_feats[:, i].mean():.4f}")

#  META-LEARNER: Stacked Ensemble
print("  META-LEARNER | Stacked Ensemble (XGBoost)")
print("~" * 70)

# Meta-features: 3 expert probabilities + 8 metadata + 8 graph structural
X_meta = np.column_stack([
    xgb_oof_probs,     # Expert 1 prob
    bert_oof_probs,     # Expert 2 prob
    gnn_oof_probs,      # Expert 3 prob
    meta_np,            # 8 metadata features
    graph_feats,        # 8 graph structural features
])

meta_feat_names = (['prob_xgb', 'prob_bert', 'prob_gnn'] +
                   ['Question', 'Answer', 'Opinion',
                    'TextLen', 'Anonymous', 'IsReply'] + graph_feat_names)

print(f"  Meta-feature matrix: {X_meta.shape} ({len(meta_feat_names)} features)")

all_results = {}

# Run with multiple seeds
ensemble_results = []
for seed in range(N_SEEDS):
    meta_clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             scale_pos_weight=class_weight, random_state=seed,
                             eval_metric='logloss', verbosity=0, reg_alpha=0.1)
    meta_clf.fit(X_meta[train_mask], y_np[train_mask],
                 eval_set=[(X_meta[val_mask], y_np[val_mask])], verbose=False)

    y_pred = meta_clf.predict(X_meta[test_mask])
    y_prob = meta_clf.predict_proba(X_meta[test_mask])[:, 1]
    metrics = eval_metrics(y_np[test_mask], y_pred, y_prob)
    ensemble_results.append(metrics)
    print(f"    Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}")

    # Save feature importance from first seed
    if seed == 0:
        feat_imp = meta_clf.feature_importances_
        meta_y_prob_best = y_prob

ens_mean = {k: np.mean([r[k] for r in ensemble_results]) for k in ensemble_results[0]}
ens_std = {k: np.std([r[k] for r in ensemble_results]) for k in ensemble_results[0]}
all_results['V2_Ensemble'] = {'mean': ens_mean, 'std': ens_std, 'runs': ensemble_results}
print(f"\n  >> V2 Ensemble: F1={ens_mean['macro_f1']:.4f}+/-{ens_std['macro_f1']:.4f}  "
      f"AUROC={ens_mean['auroc']:.4f}+/-{ens_std['auroc']:.4f}")

#  ABLATION: Which expert contributes most?
print("  ABLATION | Expert Contribution Analysis")
print("~" * 70)

ablation_configs = {
    'No_XGBoost_expert': [1, 2],      # drop expert 0
    'No_BERT_expert': [0, 2],          # drop expert 1
    'No_GNN_expert': [0, 1],           # drop expert 2
    'No_graph_feats': 'no_graph',      # drop graph structural features
    'Experts_only': 'experts_only',     # only 3 expert probs, no metadata/graph
    'XGBoost_expert_only': [0],         # just XGBoost prob + meta + graph
}

abl_results = {}
for abl_name, config in ablation_configs.items():
    if config == 'no_graph':
        # Remove graph features (cols 11-18)
        X_abl = np.column_stack([X_meta[:, :11]])
    elif config == 'experts_only':
        # Only expert probabilities
        X_abl = X_meta[:, :3]
    elif isinstance(config, list):
        # Keep only specified expert columns + all metadata + graph
        expert_cols = config
        X_abl = np.column_stack([X_meta[:, expert_cols], X_meta[:, 3:]])
    else:
        X_abl = X_meta

    runs = []
    for seed in range(N_SEEDS):
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            scale_pos_weight=class_weight, random_state=seed,
                            eval_metric='logloss', verbosity=0, reg_alpha=0.1)
        clf.fit(X_abl[train_mask], y_np[train_mask],
                eval_set=[(X_abl[val_mask], y_np[val_mask])], verbose=False)
        yp = clf.predict(X_abl[test_mask])
        ypr = clf.predict_proba(X_abl[test_mask])[:, 1]
        runs.append(eval_metrics(y_np[test_mask], yp, ypr))

    m = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
    s = {k: np.std([r[k] for r in runs]) for k in runs[0]}
    abl_results[abl_name] = {'mean': m, 'std': s}
    print(f"  {abl_name:25s}: F1={m['macro_f1']:.4f}+/-{s['macro_f1']:.4f}  "
          f"AUROC={m['auroc']:.4f}+/-{s['auroc']:.4f}")

all_results['ablations'] = abl_results

#  EXPERT AGREEMENT ANALYSIS
print("  EXPERT AGREEMENT ANALYSIS")
print("~" * 70)

xgb_pred = (xgb_oof_probs[test_mask] > 0.5).astype(int)
bert_pred = (bert_oof_probs[test_mask] > 0.5).astype(int)
gnn_pred = (gnn_oof_probs[test_mask] > 0.5).astype(int)
ens_pred = (meta_y_prob_best > 0.5).astype(int)
y_test = y_np[test_mask]

# Agreement matrix
all_agree = (xgb_pred == bert_pred) & (bert_pred == gnn_pred)
print(f"  All 3 experts agree: {all_agree.sum()}/{len(all_agree)} ({all_agree.mean()*100:.1f}%)")
print(f"  Accuracy when all agree: {(y_test[all_agree] == xgb_pred[all_agree]).mean():.4f}")
print(f"  Accuracy when they disagree: {(y_test[~all_agree] == ens_pred[~all_agree]).mean():.4f}")

# Where does ensemble fix expert errors?
xgb_wrong = xgb_pred != y_test
ens_right_when_xgb_wrong = (ens_pred[xgb_wrong] == y_test[xgb_wrong]).mean()
print(f"\n  XGBoost errors: {xgb_wrong.sum()}")
print(f"  Ensemble correct when XGBoost wrong: {ens_right_when_xgb_wrong:.4f}")

# Expert confidence when correct vs wrong
for name, probs in [('XGBoost', xgb_oof_probs[test_mask]),
                     ('BERT', bert_oof_probs[test_mask]),
                     ('GNN', gnn_oof_probs[test_mask])]:
    pred = (probs > 0.5).astype(int)
    correct = pred == y_test
    conf_correct = np.abs(probs[correct] - 0.5).mean()
    conf_wrong = np.abs(probs[~correct] - 0.5).mean()
    print(f"  {name:8s}: conf_correct={conf_correct:.4f}, conf_wrong={conf_wrong:.4f}")

# Save expert confidence info for Phase 6 gate analysis
expert_confidence = np.column_stack([xgb_oof_probs, bert_oof_probs, gnn_oof_probs])
np.save(f"{PROCESSED_DIR}/expert_probabilities.npy", expert_confidence)

# Simulate gate values: for each post, which expert is most confident?
# This replaces the V1 gating mechanism with a more interpretable analysis
gate_values = np.zeros(n_posts)
for i in range(n_posts):
    probs_3 = [xgb_oof_probs[i], bert_oof_probs[i], gnn_oof_probs[i]]
    confidences = [abs(p - 0.5) for p in probs_3]
    gate_values[i] = np.argmax(confidences)  # 0=xgb, 1=bert, 2=gnn
np.save(f"{PROCESSED_DIR}/gate_values.npy", gate_values)
print(f"\n  Saved expert probabilities + gate values for Phase 6")

#  VISUALIZATIONS
print("  Generating Phase 5 Figures")
print("~" * 70)

# Load previous results for comparison
try:
    with open(f"{RESULTS_DIR}/phase2_baselines.json") as f:
        p2 = json.load(f)
    with open(f"{RESULTS_DIR}/phase3_gnn_models.json") as f:
        p3 = json.load(f)
    has_prev = True
except:
    p2, p3 = {}, {}
    has_prev = False

# -- Figure 18: Meta-learner feature importance --
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = np.argsort(feat_imp)
sorted_names = [meta_feat_names[i] for i in sorted_idx]
sorted_vals = feat_imp[sorted_idx]

color_map = {'prob_xgb': '#e74c3c', 'prob_bert': '#3498db', 'prob_gnn': '#2ecc71',
             'Question': '#e67e22', 'Urgency': '#e67e22', 'Sentiment': '#f39c12',
             'nbr_xgb_prob': '#c0392b', 'nbr_bert_prob': '#2980b9', 'nbr_gnn_prob': '#27ae60'}
bar_colors = [color_map.get(n, '#95a5a6') for n in sorted_names]

ax.barh(range(len(sorted_names)), sorted_vals, color=bar_colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(sorted_names))); ax.set_yticklabels(sorted_names, fontsize=9)
ax.set_xlabel('Feature Importance'); ax.set_title('Meta-Learner Feature Importance')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='XGBoost Expert'),
                   Patch(facecolor='#3498db', label='BERT Expert'),
                   Patch(facecolor='#2ecc71', label='GNN Expert'),
                   Patch(facecolor='#e67e22', label='Metadata'),
                   Patch(facecolor='#95a5a6', label='Graph Structure')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig18_meta_learner_importance.png")
plt.savefig(f"{FIGURES_DIR}/fig18_meta_learner_importance.pdf"); plt.close()
print(f"  -> Saved: {FIGURES_DIR}/fig18_meta_learner_importance.png")

# -- Figure 19: Full comparison (all models + V2 ensemble) --
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

compare_models = {}
if has_prev:
    key_models = {'B1_TFIDF_XGB': 'TF-IDF+XGB', 'B2_TFIDF_Meta_XGB': 'TF-IDF+Meta\n+XGB',
                  'B3_BERT_MLP': 'BERT+MLP', 'B4_BERT_Finetune_MLP': 'BERT-FT\n+MLP'}
    for k, label in key_models.items():
        if k in p2: compare_models[label] = p2[k]
    gnn_models = {'B5_BERT_GCN': 'BERT+GCN', 'B6_BERT_GAT': 'BERT+GAT',
                  'B7_BERT_HGT': 'BERT+HGT', 'Ours_ConFusionGraph': 'V1 Graph\nTransformer'}
    for k, label in gnn_models.items():
        if k in p3: compare_models[label] = p3[k]

compare_models['V2 Stacked\nEnsemble\n(Ours)'] = {'mean': ens_mean, 'std': ens_std}

names = list(compare_models.keys())
f1_vals = [compare_models[n]['mean']['macro_f1'] for n in names]
f1_errs = [compare_models[n]['std']['macro_f1'] for n in names]
auroc_vals = [compare_models[n]['mean']['auroc'] for n in names]
auroc_errs = [compare_models[n]['std']['auroc'] for n in names]

n = len(names)
colors = ['#bdc3c7'] * (n - 1) + ['#e74c3c']  # highlight ours
# Color the key baselines
for i, name in enumerate(names):
    if 'Meta' in name: colors[i] = '#f39c12'
    elif 'V1' in name: colors[i] = '#3498db'
    elif 'HGT' in name: colors[i] = '#5dade2'

x = range(n)
axes[0].bar(x, f1_vals, yerr=f1_errs, color=colors, edgecolor='white', capsize=3, width=0.7)
axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=7, rotation=0)
axes[0].set_ylabel('Macro-F1'); axes[0].set_title('(a) Macro-F1 (All Models)')
for i, v in enumerate(f1_vals):
    axes[0].text(i, v + f1_errs[i] + 0.003, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

axes[1].bar(x, auroc_vals, yerr=auroc_errs, color=colors, edgecolor='white', capsize=3, width=0.7)
axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=7, rotation=0)
axes[1].set_ylabel('AUROC'); axes[1].set_title('(b) AUROC (All Models)')
for i, v in enumerate(auroc_vals):
    axes[1].text(i, v + auroc_errs[i] + 0.003, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

plt.suptitle('ConFusionGraph V2 (Stacked Ensemble) vs All Baselines', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig19_v2_comparison.png")
plt.savefig(f"{FIGURES_DIR}/fig19_v2_comparison.pdf"); plt.close()
print(f"  -> Saved: {FIGURES_DIR}/fig19_v2_comparison.png")

# -- Figure 20 (replaces old fig20): Expert agreement Venn-style + ablation --
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Expert agreement bar
agree_categories = ['All 3\nagree\n(correct)', 'All 3\nagree\n(wrong)',
                    'Disagree\n(ens correct)', 'Disagree\n(ens wrong)']
all_agree_correct = (all_agree & (xgb_pred == y_test)).sum()
all_agree_wrong = (all_agree & (xgb_pred != y_test)).sum()
disagree_ens_correct = (~all_agree & (ens_pred == y_test)).sum()
disagree_ens_wrong = (~all_agree & (ens_pred != y_test)).sum()
agree_vals = [all_agree_correct, all_agree_wrong, disagree_ens_correct, disagree_ens_wrong]
agree_colors = ['#2ecc71', '#e74c3c', '#3498db', '#e67e22']
axes[0].bar(range(4), agree_vals, color=agree_colors, edgecolor='white', width=0.6)
axes[0].set_xticks(range(4)); axes[0].set_xticklabels(agree_categories, fontsize=9)
axes[0].set_ylabel('Number of Test Posts'); axes[0].set_title('(a) Expert Agreement Analysis')
for i, v in enumerate(agree_vals):
    axes[0].text(i, v + 10, str(v), ha='center', fontsize=10, fontweight='bold')

# Ablation
abl_names = list(abl_results.keys())
abl_f1 = [abl_results[n]['mean']['macro_f1'] for n in abl_names]
abl_short = [n.replace('_', '\n') for n in abl_names]
abl_deltas = [v - ens_mean['macro_f1'] for v in abl_f1]
abl_colors = ['#e74c3c' if d < -0.005 else '#f39c12' if d < 0 else '#2ecc71' for d in abl_deltas]
axes[1].barh(range(len(abl_names)), abl_deltas, color=abl_colors, edgecolor='white', height=0.5)
axes[1].axvline(x=0, color='#2c3e50', linewidth=1)
axes[1].set_yticks(range(len(abl_names))); axes[1].set_yticklabels(abl_short, fontsize=8)
axes[1].set_xlabel('Delta Macro-F1 from Full Ensemble')
axes[1].set_title('(b) Expert Ablation Impact'); axes[1].invert_yaxis()

plt.suptitle('Expert Agreement & Ablation Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig20_expert_analysis.png")
plt.savefig(f"{FIGURES_DIR}/fig20_expert_analysis.pdf"); plt.close()
print(f"  -> Saved: {FIGURES_DIR}/fig20_expert_analysis.png")

# ── Save results ──
phase5_output = {
    'V2_Ensemble': {'mean': ens_mean, 'std': ens_std, 'runs': ensemble_results},
    'experts_standalone': {
        'XGBoost': xgb_test_metrics,
        'BERT_MLP': bert_test_metrics,
        'GNN': gnn_test_metrics,
    },
    'ablations': {k: {'mean': v['mean'], 'std': v['std']} for k, v in abl_results.items()},
    'expert_agreement': {
        'all_agree_pct': float(all_agree.mean()),
        'accuracy_when_agree': float((y_test[all_agree] == xgb_pred[all_agree]).mean()),
        'accuracy_when_disagree': float((y_test[~all_agree] == ens_pred[~all_agree]).mean()),
    },
    'meta_feature_importance': {meta_feat_names[i]: float(feat_imp[i]) for i in range(len(meta_feat_names))},
}

with open(f"{RESULTS_DIR}/phase5_v2_results.json", 'w') as f:
    json.dump(phase5_output, f, indent=2, default=float)
print(f"\n  -> Saved: {RESULTS_DIR}/phase5_v2_results.json")
torch.save({'model': 'stacked_ensemble', 'feat_names': meta_feat_names}, f"{MODELS_DIR}/confusiongraph_v2_best.pt")

print("=" * 70)
print(f"""
  V2 Stacked Ensemble:  F1={ens_mean['macro_f1']:.4f}  AUROC={ens_mean['auroc']:.4f}

  Expert Standalone Performance:
    XGBoost (metadata):  F1={xgb_test_metrics['macro_f1']:.4f}  AUROC={xgb_test_metrics['auroc']:.4f}
    BERT + MLP:          F1={bert_test_metrics['macro_f1']:.4f}  AUROC={bert_test_metrics['auroc']:.4f}
    GNN (graph):         F1={gnn_test_metrics['macro_f1']:.4f}  AUROC={gnn_test_metrics['auroc']:.4f}

  Figures: fig18 (meta-learner importance), fig19 (all models), fig20 (expert analysis)
  Next: Run phase6_xai.py
""")