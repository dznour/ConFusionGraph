# gnn_models.py
# GNN baselines (BERT+GCN, BERT+GAT, BERT+HGT) and ConFusionGraph V1

import os, time, json, warnings, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm

import torch_geometric
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import (GCNConv, GATv2Conv, HGTConv, HeteroConv,
                                 SAGEConv, Linear, LayerNorm)
from torch_geometric.utils import to_undirected, add_self_loops

warnings.filterwarnings('ignore')

# ── Config ──
PARENT_DIR = "no_urg"
PROCESSED_DIR = f"{PARENT_DIR}/processed"
RESULTS_DIR = f"{PARENT_DIR}/results"
FIGURES_DIR = f"{PARENT_DIR}/figures"
MODELS_DIR = f"{PARENT_DIR}/models"

N_SEEDS = 5
HIDDEN_DIM = 128
N_LAYERS = 3
N_HEADS = 4
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("=" * 70)
print(f"  Device: {device}")

# ── Load data ──
print("\n  Loading processed data...")
hetero_data = torch.load(f"{PROCESSED_DIR}/hetero_graph.pt", map_location='cpu', weights_only=False)
labels_df = pd.read_csv(f"{PROCESSED_DIR}/post_labels_and_splits.csv", index_col=0)

y = hetero_data['post'].y
train_mask = hetero_data['post'].train_mask
val_mask = hetero_data['post'].val_mask
test_mask = hetero_data['post'].test_mask

n_pos = y[train_mask].sum().item()
n_neg = train_mask.sum().item() - n_pos
class_weight = n_neg / n_pos
pos_weight = torch.tensor([class_weight], dtype=torch.float32).to(device)

print(f"  Nodes: post={hetero_data['post'].x.shape[0]}, user={hetero_data['user'].x.shape[0]}")
print(f"  Edge types: {hetero_data.edge_types}")
print(f"  Labels: pos={n_pos}, neg={n_neg}, weight={class_weight:.2f}")

# ── Build homogeneous post-only graph for B5/B6 ──
print("\n  Building homogeneous post-only graph...")
post_x = hetero_data['post'].x
n_posts = post_x.shape[0]

# Collect all post-post edges
homo_edges = []
for etype in [('post', 'reply_to', 'post'), ('post', 'same_thread', 'post'), ('post', 'temporal_next', 'post')]:
    if etype in hetero_data.edge_types:
        homo_edges.append(hetero_data[etype].edge_index)
homo_edge_index = torch.cat(homo_edges, dim=1) if homo_edges else torch.zeros(2, 0, dtype=torch.long)
homo_edge_index = to_undirected(homo_edge_index, num_nodes=n_posts)
homo_edge_index, _ = add_self_loops(homo_edge_index, num_nodes=n_posts)
print(f"  Homogeneous edges: {homo_edge_index.shape[1]:,} (with self-loops)")

# ── Eval function ──
def eval_metrics(y_true, y_pred, y_prob):
    return {
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'auroc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_pos': f1_score(y_true, y_pred, pos_label=1),
    }

#  MODEL DEFINITIONS

# ── B5: GCN ──
class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        self.norms.append(nn.LayerNorm(hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if x.shape[-1] == x_new.shape[-1]:
                x_new = x_new + x
            x = x_new
        return self.classifier(x).squeeze(-1)

# ── B6: GAT ──
class GATModel(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, n_heads, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden // n_heads, heads=n_heads, dropout=dropout, add_self_loops=True))
        self.norms.append(nn.LayerNorm(hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GATv2Conv(hidden, hidden // n_heads, heads=n_heads, dropout=dropout, add_self_loops=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if x.shape[-1] == x_new.shape[-1]:
                x_new = x_new + x
            x = x_new
        return self.classifier(x).squeeze(-1)

# ── B7: HGT ──
class HGTModel(nn.Module):
    def __init__(self, hetero_metadata, post_in, user_in, hidden, n_layers, n_heads, dropout):
        super().__init__()
        self.post_proj = nn.Linear(post_in, hidden)
        self.user_proj = nn.Linear(user_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(HGTConv(hidden, hidden, hetero_metadata, heads=n_heads))
            self.norms.append(LayerNorm(hidden, mode='node'))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict, degree_feats=None):
        x_dict = {
            'post': F.relu(self.post_proj(x_dict['post'])),
            'user': F.relu(self.user_proj(x_dict['user'])),
        }
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x_dict, edge_index_dict)
            # Residual + norm for each node type
            for key in x_new:
                x_new[key] = norm(F.relu(x_new[key]))
                x_new[key] = F.dropout(x_new[key], p=self.dropout, training=self.training)
                x_new[key] = x_new[key] + x_dict[key]
            x_dict = x_new
        return self.classifier(x_dict['post']).squeeze(-1)

# ── Ours: ConFusionGraph ──
class ConFusionGraph(nn.Module):
    def __init__(self, post_in, user_in, hidden, n_layers, n_heads, dropout, edge_types_list):
        super().__init__()
        self.post_proj = nn.Linear(post_in, hidden)
        self.user_proj = nn.Linear(user_in, hidden)

        # Structural encodings
        self.degree_enc = nn.Linear(5, hidden)  # in/out degree per node type

        # Heterogeneous convolution layers (GPS-style local MPNN)
        self.convs = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.user_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(n_layers):
            conv_dict = {}
            for et in edge_types_list:
                src_type, rel, dst_type = et
                if src_type == dst_type:
                    conv_dict[et] = GATv2Conv(hidden, hidden // n_heads, heads=n_heads,
                                              dropout=dropout, add_self_loops=False)
                else:
                    conv_dict[et] = SAGEConv((hidden, hidden), hidden)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.post_norms.append(nn.LayerNorm(hidden))
            self.user_norms.append(nn.LayerNorm(hidden))
            # FFN (GPS-style)
            self.ffns.append(nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden), nn.Dropout(dropout)))
            self.ffn_norms.append(nn.LayerNorm(hidden))

        # Global context aggregation (efficient alternative to O(n^2) attention)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict, degree_feats=None):
        # Project node features
        x_dict = {
            'post': F.relu(self.post_proj(x_dict['post'])),
            'user': F.relu(self.user_proj(x_dict['user'])),
        }

        # Add degree encoding to post nodes
        if degree_feats is not None:
            x_dict['post'] = x_dict['post'] + self.degree_enc(degree_feats)

        # Message passing layers
        for i, (conv, post_norm, user_norm, ffn, ffn_norm) in enumerate(
                zip(self.convs, self.post_norms, self.user_norms, self.ffns, self.ffn_norms)):
            # Local MPNN
            x_new = conv(x_dict, edge_index_dict)

            # Residual + Norm per node type
            if 'post' in x_new:
                x_new['post'] = post_norm(x_new['post'] + x_dict['post'])
            else:
                x_new['post'] = x_dict['post']
            if 'user' in x_new:
                x_new['user'] = user_norm(x_new['user'] + x_dict['user'])
            else:
                x_new['user'] = x_dict['user']

            # FFN (applied to post nodes) with residual
            post_ffn = ffn(x_new['post'])
            x_new['post'] = ffn_norm(post_ffn + x_new['post'])

            x_dict = x_new

        # Global context: mean-pool all post embeddings, concat with local
        global_ctx = x_dict['post'].mean(dim=0, keepdim=True).expand_as(x_dict['post'])
        post_final = self.global_proj(torch.cat([x_dict['post'], global_ctx], dim=-1))

        return self.classifier(post_final).squeeze(-1)

# ── Training function ──
def train_gnn(model, data_dict, y, masks, seed, model_type='homo', epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_mask, val_mask, test_mask = masks
    best_val_f1, best_state, patience_ctr = 0, None, 0

    for epoch in range(epochs):
        model.train()
        if model_type == 'homo':
            logits = model(data_dict['x'], data_dict['edge_index'])
        else:
            logits = model(data_dict['x_dict'], data_dict['edge_index_dict'],
                           data_dict.get('degree_feats', None))
        loss = criterion(logits[train_mask], y[train_mask].float())
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if model_type == 'homo':
                val_logits = model(data_dict['x'], data_dict['edge_index'])
            else:
                val_logits = model(data_dict['x_dict'], data_dict['edge_index_dict'],
                                   data_dict.get('degree_feats', None))
            val_prob = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
            val_pred = (val_prob > 0.5).astype(int)
            val_f1 = f1_score(y[val_mask].cpu().numpy(), val_pred, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    # Test
    model.load_state_dict(best_state); model.to(device).eval()
    with torch.no_grad():
        if model_type == 'homo':
            te_logits = model(data_dict['x'], data_dict['edge_index'])
        else:
            te_logits = model(data_dict['x_dict'], data_dict['edge_index_dict'],
                              data_dict.get('degree_feats', None))
        te_prob = torch.sigmoid(te_logits[test_mask]).cpu().numpy()
        te_pred = (te_prob > 0.5).astype(int)
    metrics = eval_metrics(y[test_mask].cpu().numpy(), te_pred, te_prob)
    return metrics, best_state, epoch + 1

# ── Compute degree features for ConFusionGraph ──
print("\n  Computing structural degree encodings...")
deg_feats = torch.zeros(n_posts, 5, dtype=torch.float32)
for col_i, etype in enumerate([('post', 'reply_to', 'post'), ('post', 'same_thread', 'post'),
                                ('post', 'temporal_next', 'post')]):
    if etype in hetero_data.edge_types:
        ei = hetero_data[etype].edge_index
        for node_idx in ei[1].numpy():
            if node_idx < n_posts: deg_feats[node_idx, col_i] += 1
# authored-by out-degree (how many users this post is connected to = 1)
if ('post', 'authored_by', 'user') in hetero_data.edge_types:
    ei = hetero_data[('post', 'authored_by', 'user')].edge_index
    for node_idx in ei[0].numpy():
        deg_feats[node_idx, 3] += 1
# Total degree
deg_feats[:, 4] = deg_feats[:, :4].sum(dim=1)
# Log-normalize
deg_feats = torch.log1p(deg_feats)
deg_max = deg_feats.max(dim=0).values
deg_max[deg_max == 0] = 1
deg_feats = deg_feats / deg_max
print(f"  Degree features: {deg_feats.shape}")

all_results = {}

#  B5: BERT + GCN (homogeneous)
print("  B5 | BERT + GCN (homogeneous post-only graph)")
print("~" * 70)

homo_data = {
    'x': post_x.to(device),
    'edge_index': homo_edge_index.to(device),
}
masks = (train_mask.to(device), val_mask.to(device), test_mask.to(device))
y_dev = y.to(device)

b5_results = []
for seed in range(N_SEEDS):
    model = GCNModel(post_x.shape[1], HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
    metrics, _, ep = train_gnn(model, homo_data, y_dev, masks, seed, 'homo')
    b5_results.append(metrics)
    print(f"    Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}  (ep={ep})")
    del model; torch.cuda.empty_cache()

b5_mean = {k: np.mean([r[k] for r in b5_results]) for k in b5_results[0]}
b5_std = {k: np.std([r[k] for r in b5_results]) for k in b5_results[0]}
all_results['B5_BERT_GCN'] = {'mean': b5_mean, 'std': b5_std, 'runs': b5_results}
print(f"  >> B5 Mean: Macro-F1={b5_mean['macro_f1']:.4f}±{b5_std['macro_f1']:.4f}  AUROC={b5_mean['auroc']:.4f}±{b5_std['auroc']:.4f}")

#  B6: BERT + GAT (homogeneous)
print("  B6 | BERT + GAT (homogeneous post-only graph)")
print("~" * 70)

b6_results = []
for seed in range(N_SEEDS):
    model = GATModel(post_x.shape[1], HIDDEN_DIM, N_LAYERS, N_HEADS, DROPOUT).to(device)
    metrics, _, ep = train_gnn(model, homo_data, y_dev, masks, seed, 'homo')
    b6_results.append(metrics)
    print(f"    Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}  (ep={ep})")
    del model; torch.cuda.empty_cache()

b6_mean = {k: np.mean([r[k] for r in b6_results]) for k in b6_results[0]}
b6_std = {k: np.std([r[k] for r in b6_results]) for k in b6_results[0]}
all_results['B6_BERT_GAT'] = {'mean': b6_mean, 'std': b6_std, 'runs': b6_results}
print(f"  >> B6 Mean: Macro-F1={b6_mean['macro_f1']:.4f}±{b6_std['macro_f1']:.4f}  AUROC={b6_mean['auroc']:.4f}±{b6_std['auroc']:.4f}")

#  B7: BERT + HGT (heterogeneous)
print("  B7 | BERT + HGT (heterogeneous graph)")
print("~" * 70)

# Prepare hetero data for GPU
hetero_metadata = hetero_data.metadata()
x_dict_dev = {
    'post': hetero_data['post'].x.to(device),
    'user': hetero_data['user'].x.to(device),
}
edge_index_dict_dev = {}
for etype in hetero_data.edge_types:
    edge_index_dict_dev[etype] = hetero_data[etype].edge_index.to(device)

hetero_data_dict = {
    'x_dict': x_dict_dev,
    'edge_index_dict': edge_index_dict_dev,
}

b7_results = []
for seed in range(N_SEEDS):
    model = HGTModel(hetero_metadata, post_x.shape[1], hetero_data['user'].x.shape[1],
                     HIDDEN_DIM, N_LAYERS, N_HEADS, DROPOUT).to(device)
    metrics, _, ep = train_gnn(model, hetero_data_dict, y_dev, masks, seed, 'hetero')
    b7_results.append(metrics)
    print(f"    Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}  (ep={ep})")
    del model; torch.cuda.empty_cache()

b7_mean = {k: np.mean([r[k] for r in b7_results]) for k in b7_results[0]}
b7_std = {k: np.std([r[k] for r in b7_results]) for k in b7_results[0]}
all_results['B7_BERT_HGT'] = {'mean': b7_mean, 'std': b7_std, 'runs': b7_results}
print(f"  >> B7 Mean: Macro-F1={b7_mean['macro_f1']:.4f}±{b7_std['macro_f1']:.4f}  AUROC={b7_mean['auroc']:.4f}±{b7_std['auroc']:.4f}")

#  OURS: ConFusionGraph
print("  OURS | ConFusionGraph (Heterogeneous GPS-style Graph Transformer)")
print("~" * 70)

cfg_data = {
    'x_dict': x_dict_dev,
    'edge_index_dict': edge_index_dict_dev,
    'degree_feats': deg_feats.to(device),
}

ours_results = []
best_overall_f1 = 0
best_overall_state = None
for seed in range(N_SEEDS):
    model = ConFusionGraph(
        post_in=post_x.shape[1], user_in=hetero_data['user'].x.shape[1],
        hidden=HIDDEN_DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        dropout=DROPOUT, edge_types_list=list(edge_index_dict_dev.keys())
    ).to(device)
    metrics, state, ep = train_gnn(model, cfg_data, y_dev, masks, seed, 'hetero')
    ours_results.append(metrics)
    print(f"    Seed {seed}: Macro-F1={metrics['macro_f1']:.4f}  AUROC={metrics['auroc']:.4f}  (ep={ep})")
    if metrics['macro_f1'] > best_overall_f1:
        best_overall_f1 = metrics['macro_f1']
        best_overall_state = state
    del model; torch.cuda.empty_cache()

# Save best ConFusionGraph model
torch.save(best_overall_state, f"{MODELS_DIR}/confusiongraph_best.pt")
print(f"  -> Saved best model: {MODELS_DIR}/confusiongraph_best.pt")

ours_mean = {k: np.mean([r[k] for r in ours_results]) for k in ours_results[0]}
ours_std = {k: np.std([r[k] for r in ours_results]) for k in ours_results[0]}
all_results['Ours_ConFusionGraph'] = {'mean': ours_mean, 'std': ours_std, 'runs': ours_results}
print(f"  >> OURS Mean: Macro-F1={ours_mean['macro_f1']:.4f}±{ours_std['macro_f1']:.4f}  AUROC={ours_mean['auroc']:.4f}±{ours_std['auroc']:.4f}")

#  RESULTS SUMMARY & VISUALIZATION
print("=" * 70)

print(f"\n  {'Model':<30} {'Macro-F1':>14} {'AUROC':>14} {'Precision':>14} {'Recall':>14}")
print(f"  {'─'*86}")
for name, res in all_results.items():
    m, s = res['mean'], res['std']
    print(f"  {name:<30} {m['macro_f1']:.4f}±{s['macro_f1']:.4f}  "
          f"{m['auroc']:.4f}±{s['auroc']:.4f}  "
          f"{m['precision']:.4f}±{s['precision']:.4f}  "
          f"{m['recall']:.4f}±{s['recall']:.4f}")

# -- Figure: GNN Comparison --
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

model_names = list(all_results.keys())
short_names = ['GCN\n(homo)', 'GAT\n(homo)', 'HGT\n(hetero)', 'ConFusion-\nGraph (ours)']
colors = ['#95a5a6', '#7f8c8d', '#3498db', '#e74c3c']

f1_means = [all_results[n]['mean']['macro_f1'] for n in model_names]
f1_stds = [all_results[n]['std']['macro_f1'] for n in model_names]
auroc_means = [all_results[n]['mean']['auroc'] for n in model_names]
auroc_stds = [all_results[n]['std']['auroc'] for n in model_names]

x = range(len(model_names))
bars1 = axes[0].bar(x, f1_means, yerr=f1_stds, color=colors, edgecolor='white',
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

plt.suptitle('Phase 3: GNN Models (5 seeds)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig11_phase3_gnn_models.png")
plt.savefig(f"{FIGURES_DIR}/fig11_phase3_gnn_models.pdf")
plt.close()
print(f"\n  -> Saved: {FIGURES_DIR}/fig11_phase3_gnn_models.png")

# Load phase2 results and create combined figure
try:
    with open(f"{RESULTS_DIR}/phase2_baselines.json") as f:
        p2_results = json.load(f)
    combined = {**p2_results, **all_results}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    all_names = list(combined.keys())
    all_short = ['TF-IDF\n+XGB', 'TF-IDF+Meta\n+XGB', 'BERT\n+MLP', 'BERT-FT\n+MLP',
                 'BERT\n+GCN', 'BERT\n+GAT', 'BERT\n+HGT', 'ConFusion-\nGraph']
    all_colors = ['#bdc3c7', '#95a5a6', '#5dade2', '#2e86c1',
                  '#a3e4d7', '#48c9b0', '#3498db', '#e74c3c']

    f1_all = [combined[n]['mean']['macro_f1'] for n in all_names]
    f1_std_all = [combined[n]['std']['macro_f1'] for n in all_names]
    auroc_all = [combined[n]['mean']['auroc'] for n in all_names]
    auroc_std_all = [combined[n]['std']['auroc'] for n in all_names]

    x = range(len(all_names))
    axes[0].bar(x, f1_all, yerr=f1_std_all, color=all_colors, edgecolor='white',
                capsize=3, width=0.65, error_kw={'linewidth': 1.2})
    axes[0].set_xticks(x); axes[0].set_xticklabels(all_short, fontsize=8)
    axes[0].set_ylabel('Macro-F1'); axes[0].set_title('(a) Macro-F1 (All Models)')
    for i, v in enumerate(f1_all):
        axes[0].text(i, v + f1_std_all[i] + 0.003, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

    axes[1].bar(x, auroc_all, yerr=auroc_std_all, color=all_colors, edgecolor='white',
                capsize=3, width=0.65, error_kw={'linewidth': 1.2})
    axes[1].set_xticks(x); axes[1].set_xticklabels(all_short, fontsize=8)
    axes[1].set_ylabel('AUROC'); axes[1].set_title('(b) AUROC (All Models)')
    for i, v in enumerate(auroc_all):
        axes[1].text(i, v + auroc_std_all[i] + 0.003, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

    plt.suptitle('Complete Model Comparison (B1-B7 + ConFusionGraph)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig12_all_models_comparison.png")
    plt.savefig(f"{FIGURES_DIR}/fig12_all_models_comparison.pdf")
    plt.close()
    print(f"  -> Saved: {FIGURES_DIR}/fig12_all_models_comparison.png")
except FileNotFoundError:
    print("  (Phase 2 results not found - skipping combined figure)")

# Save results
with open(f"{RESULTS_DIR}/phase3_gnn_models.json", 'w') as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"  -> Saved: {RESULTS_DIR}/phase3_gnn_models.json")

# Save model config for Phase 4
config = {
    'hidden_dim': HIDDEN_DIM, 'n_layers': N_LAYERS, 'n_heads': N_HEADS,
    'dropout': DROPOUT, 'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
    'edge_types': [str(et) for et in hetero_data.edge_types],
}
with open(f"{RESULTS_DIR}/model_config.json", 'w') as f:
    json.dump(config, f, indent=2)

print("\n  Phase 3 complete. Next: Run phase4_ablations_analysis.py")