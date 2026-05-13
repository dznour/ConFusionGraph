# baselines.py
# Non-graph baselines: TF-IDF+XGBoost, TF-IDF+Meta, BERT+MLP, BERT fine-tuned

import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Config ──
PARENT_DIR = "no_urg"
PROCESSED_DIR = f"{PARENT_DIR}/processed"
RESULTS_DIR = f"{PARENT_DIR}/results"
FIGURES_DIR = f"{PARENT_DIR}/figures"
MODELS_DIR = f"{PARENT_DIR}/models"
DATASET_PATH = "stanfordMOOCForumPostsSet.xlsx"

N_SEEDS = 5
BERT_MODEL_NAME = "bert-base-uncased"
MLP_EPOCHS = 50
MLP_LR = 1e-3
MLP_HIDDEN = 256
MLP_BATCH = 512
FINETUNE_EPOCHS = 8
FINETUNE_LR = 2e-5
FINETUNE_BATCH = 32
FINETUNE_ACCUM = 2
FINETUNE_MAX_LEN = 256

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("=" * 70)
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ── Load data ──
print("\n  Loading processed data...")
bert_embeds = torch.load(f"{PROCESSED_DIR}/bert_embeddings.pt", map_location='cpu', weights_only=True)
post_meta = torch.load(f"{PROCESSED_DIR}/post_metadata.pt", map_location='cpu', weights_only=True)
labels_df = pd.read_csv(f"{PROCESSED_DIR}/post_labels_and_splits.csv", index_col=0)

labels = labels_df['confused'].values
splits = labels_df['split'].values
train_mask = splits == 'train'
val_mask = splits == 'val'
test_mask = splits == 'test'

# Load raw text for TF-IDF and fine-tuning
df_raw = pd.read_excel(DATASET_PATH)
df_raw = df_raw.dropna(subset=['Text', 'forum_uid', 'created_at', 'post_type']).reset_index(drop=True)
df_raw = df_raw.sort_values(['course_display_name', 'created_at']).reset_index(drop=True)
texts = df_raw['Text'].astype(str).tolist()

# Class weight
n_pos = labels[train_mask].sum()
n_neg = train_mask.sum() - n_pos
class_weight = n_neg / n_pos
print(f"  Posts: {len(labels):,} (train={train_mask.sum():,}, val={val_mask.sum():,}, test={test_mask.sum():,})")
print(f"  Class weight (pos): {class_weight:.2f}")

all_results = {}

def eval_metrics(y_true, y_pred, y_prob):
    return {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'auroc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_pos': f1_score(y_true, y_pred, pos_label=1),
    }

#  B1: TF-IDF + XGBoost
print("  B1 | TF-IDF + XGBoost")
print("~" * 70)

print("  Fitting TF-IDF (max_features=10000)...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform([texts[i] for i in range(len(texts))])

X_tr_tfidf = X_tfidf[train_mask]
X_val_tfidf = X_tfidf[val_mask]
X_te_tfidf = X_tfidf[test_mask]
y_tr, y_val, y_te = labels[train_mask], labels[val_mask], labels[test_mask]

b1_results = []
for seed in range(N_SEEDS):
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=class_weight, random_state=seed,
                        eval_metric='logloss', verbosity=0)
    clf.fit(X_tr_tfidf, y_tr, eval_set=[(X_val_tfidf, y_val)], verbose=False)
    y_pred = clf.predict(X_te_tfidf)
    y_prob = clf.predict_proba(X_te_tfidf)[:, 1]
    metrics = eval_metrics(y_te, y_pred, y_prob)
    b1_results.append(metrics)
    print(f"    Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}")

b1_mean = {k: np.mean([r[k] for r in b1_results]) for k in b1_results[0]}
b1_std = {k: np.std([r[k] for r in b1_results]) for k in b1_results[0]}
all_results['B1_TFIDF_XGB'] = {'mean': b1_mean, 'std': b1_std, 'runs': b1_results}
print(f"  >> B1 Mean: Macro-F1={b1_mean['macro_f1']:.4f}±{b1_std['macro_f1']:.4f}  AUROC={b1_mean['auroc']:.4f}±{b1_std['auroc']:.4f}")

#  B2: TF-IDF + Metadata + XGBoost
print("  B2 | TF-IDF + Metadata + XGBoost")
print("~" * 70)

import scipy.sparse as sp
X_tfidf_meta = sp.hstack([X_tfidf, sp.csr_matrix(post_meta.numpy())])
X_tr2 = X_tfidf_meta[train_mask]
X_val2 = X_tfidf_meta[val_mask]
X_te2 = X_tfidf_meta[test_mask]

b2_results = []
for seed in range(N_SEEDS):
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=class_weight, random_state=seed,
                        eval_metric='logloss', verbosity=0)
    clf.fit(X_tr2, y_tr, eval_set=[(X_val2, y_val)], verbose=False)
    y_pred = clf.predict(X_te2)
    y_prob = clf.predict_proba(X_te2)[:, 1]
    metrics = eval_metrics(y_te, y_pred, y_prob)
    b2_results.append(metrics)
    print(f"    Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}")

b2_mean = {k: np.mean([r[k] for r in b2_results]) for k in b2_results[0]}
b2_std = {k: np.std([r[k] for r in b2_results]) for k in b2_results[0]}
all_results['B2_TFIDF_Meta_XGB'] = {'mean': b2_mean, 'std': b2_std, 'runs': b2_results}
print(f"  >> B2 Mean: Macro-F1={b2_mean['macro_f1']:.4f}±{b2_std['macro_f1']:.4f}  AUROC={b2_mean['auroc']:.4f}±{b2_std['auroc']:.4f}")

#  B3: BERT-frozen + MLP
print("  B3 | BERT-frozen + MLP")
print("~" * 70)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

X_bert = bert_embeds.numpy()
X_tr_b = torch.tensor(X_bert[train_mask], dtype=torch.float32)
X_val_b = torch.tensor(X_bert[val_mask], dtype=torch.float32)
X_te_b = torch.tensor(X_bert[test_mask], dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
y_te_t = torch.tensor(y_te, dtype=torch.float32)

pos_weight = torch.tensor([class_weight], dtype=torch.float32).to(device)

def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, in_dim, seed, epochs=MLP_EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPClassifier(in_dim, MLP_HIDDEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=MLP_BATCH, shuffle=True)

    best_val_f1, best_state = 0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_pred = (torch.sigmoid(val_logits) > 0.5).long().cpu().numpy()
            val_f1 = f1_score(y_val.numpy(), val_pred, average='macro')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_logits = model(X_test.to(device))
        te_prob = torch.sigmoid(te_logits).cpu().numpy()
        te_pred = (te_prob > 0.5).astype(int)
    return eval_metrics(y_test.numpy(), te_pred, te_prob)

b3_results = []
for seed in range(N_SEEDS):
    metrics = train_mlp(X_tr_b, y_tr_t, X_val_b, y_val_t, X_te_b, y_te_t, 768, seed)
    b3_results.append(metrics)
    print(f"    Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}")

b3_mean = {k: np.mean([r[k] for r in b3_results]) for k in b3_results[0]}
b3_std = {k: np.std([r[k] for r in b3_results]) for k in b3_results[0]}
all_results['B3_BERT_MLP'] = {'mean': b3_mean, 'std': b3_std, 'runs': b3_results}
print(f"  >> B3 Mean: Macro-F1={b3_mean['macro_f1']:.4f}±{b3_std['macro_f1']:.4f}  AUROC={b3_mean['auroc']:.4f}±{b3_std['auroc']:.4f}")

#  B4: BERT-finetuned + MLP
print("  B4 | BERT-finetuned + MLP (last 2 layers unfrozen)")
print("~" * 70)

class BERTFinetune(nn.Module):
    def __init__(self, bert_model_name, hidden=128, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        # Freeze all but last 2 encoder layers + pooler
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).squeeze(-1)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Tokenize all texts
print("  Tokenizing texts...")
all_encodings = tokenizer(texts, padding='max_length', truncation=True,
                          max_length=FINETUNE_MAX_LEN, return_tensors='pt')
all_input_ids = all_encodings['input_ids']
all_attention_mask = all_encodings['attention_mask']
all_labels_t = torch.tensor(labels, dtype=torch.float32)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        return all_input_ids[i], all_attention_mask[i], all_labels_t[i]

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
test_idx = np.where(test_mask)[0]

b4_results = []
for seed in range(N_SEEDS):
    print(f"\n    Seed {seed}:")
    torch.manual_seed(seed); np.random.seed(seed)
    model = BERTFinetune(BERT_MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=FINETUNE_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dl = DataLoader(TextDataset(train_idx), batch_size=FINETUNE_BATCH, shuffle=True)
    val_dl = DataLoader(TextDataset(val_idx), batch_size=FINETUNE_BATCH * 2)
    test_dl = DataLoader(TextDataset(test_idx), batch_size=FINETUNE_BATCH * 2)

    best_val_f1, best_state = 0, None
    for epoch in range(FINETUNE_EPOCHS):
        model.train(); total_loss = 0; steps = 0
        optimizer.zero_grad()
        for batch_i, (ids, mask, labs) in enumerate(train_dl):
            ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
            logits = model(ids, mask)
            loss = criterion(logits, labs) / FINETUNE_ACCUM
            loss.backward(); total_loss += loss.item() * FINETUNE_ACCUM
            if (batch_i + 1) % FINETUNE_ACCUM == 0:
                optimizer.step(); optimizer.zero_grad()
            steps += 1
        # Leftover
        optimizer.step(); optimizer.zero_grad()

        # Validation
        model.eval(); val_preds, val_true = [], []
        with torch.no_grad():
            for ids, mask, labs in val_dl:
                ids, mask = ids.to(device), mask.to(device)
                logits = model(ids, mask)
                preds = (torch.sigmoid(logits) > 0.5).long().cpu()
                val_preds.extend(preds.numpy()); val_true.extend(labs.numpy())
        val_f1 = f1_score(val_true, val_preds, average='macro')
        print(f"      Epoch {epoch+1}/{FINETUNE_EPOCHS}: loss={total_loss/steps:.4f}, val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test
    model.load_state_dict(best_state); model.to(device).eval()
    te_preds, te_probs, te_true = [], [], []
    with torch.no_grad():
        for ids, mask, labs in test_dl:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            te_probs.extend(probs); te_preds.extend((probs > 0.5).astype(int)); te_true.extend(labs.numpy())

    metrics = eval_metrics(np.array(te_true), np.array(te_preds), np.array(te_probs))
    b4_results.append(metrics)
    print(f"    >> Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}")

    # Save best model for seed 0
    if seed == 0:
        torch.save(best_state, f"{MODELS_DIR}/b4_bert_finetune_best.pt")

    del model; torch.cuda.empty_cache()

b4_mean = {k: np.mean([r[k] for r in b4_results]) for k in b4_results[0]}
b4_std = {k: np.std([r[k] for r in b4_results]) for k in b4_results[0]}
all_results['B4_BERT_Finetune_MLP'] = {'mean': b4_mean, 'std': b4_std, 'runs': b4_results}
print(f"\n  >> B4 Mean: Macro-F1={b4_mean['macro_f1']:.4f}±{b4_std['macro_f1']:.4f}  AUROC={b4_mean['auroc']:.4f}±{b4_std['auroc']:.4f}")

#  RESULTS SUMMARY & VISUALIZATION
print("=" * 70)

print(f"\n  {'Model':<30} {'Macro-F1':>12} {'AUROC':>12} {'Precision':>12} {'Recall':>12}")
print(f"  {'─'*78}")
for name, res in all_results.items():
    m, s = res['mean'], res['std']
    print(f"  {name:<30} {m['macro_f1']:.4f}±{s['macro_f1']:.4f} "
          f"{m['auroc']:.4f}±{s['auroc']:.4f} "
          f"{m['precision']:.4f}±{s['precision']:.4f} "
          f"{m['recall']:.4f}±{s['recall']:.4f}")

# -- Figure: Baseline Comparison --
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

model_names = list(all_results.keys())
short_names = ['TF-IDF+XGB', 'TF-IDF+Meta\n+XGB', 'BERT-frozen\n+MLP', 'BERT-finetune\n+MLP']
colors = ['#95a5a6', '#7f8c8d', '#3498db', '#2980b9']

f1_means = [all_results[n]['mean']['macro_f1'] for n in model_names]
f1_stds = [all_results[n]['std']['macro_f1'] for n in model_names]
auroc_means = [all_results[n]['mean']['auroc'] for n in model_names]
auroc_stds = [all_results[n]['std']['auroc'] for n in model_names]

x = range(len(model_names))
axes[0].bar(x, f1_means, yerr=f1_stds, color=colors, edgecolor='white',
            capsize=4, width=0.6, error_kw={'linewidth': 1.5})
axes[0].set_xticks(x); axes[0].set_xticklabels(short_names, fontsize=9)
axes[0].set_ylabel('Macro-F1'); axes[0].set_title('(a) Macro-F1 Score')
axes[0].set_ylim(0, max(f1_means) * 1.15)
for i, v in enumerate(f1_means):
    axes[0].text(i, v + f1_stds[i] + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

axes[1].bar(x, auroc_means, yerr=auroc_stds, color=colors, edgecolor='white',
            capsize=4, width=0.6, error_kw={'linewidth': 1.5})
axes[1].set_xticks(x); axes[1].set_xticklabels(short_names, fontsize=9)
axes[1].set_ylabel('AUROC'); axes[1].set_title('(b) AUROC Score')
axes[1].set_ylim(0, max(auroc_means) * 1.15)
for i, v in enumerate(auroc_means):
    axes[1].text(i, v + auroc_stds[i] + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Phase 2: Non-Graph Baselines (5 seeds)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig10_phase2_baselines.png"); plt.savefig(f"{FIGURES_DIR}/fig10_phase2_baselines.pdf"); plt.close()
print(f"\n  -> Saved: {FIGURES_DIR}/fig10_phase2_baselines.png")

# Save results
with open(f"{RESULTS_DIR}/phase2_baselines.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"  -> Saved: {RESULTS_DIR}/phase2_baselines.json")

print("\n  Phase 2 complete. Next: Run phase3_gnn_models.py")