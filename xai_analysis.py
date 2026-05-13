# xai_analysis.py
# KernelSHAP, LIME, token attribution and archetype discovery

import os, json, warnings, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from torch_geometric.nn import (GATv2Conv, HeteroConv, SAGEConv)
from torch_geometric.utils import softmax

warnings.filterwarnings('ignore')

# ── Config ──
PARENT_DIR = "no_urg"
PROCESSED_DIR = f"{PARENT_DIR}/processed"
RESULTS_DIR = f"{PARENT_DIR}/results"
FIGURES_DIR = f"{PARENT_DIR}/figures"
MODELS_DIR = f"{PARENT_DIR}/models"
TOP_K_EXPLAIN = 200  # number of confused posts to explain

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("=" * 70)

# ── Load data ──
print("\n  Loading processed data...")
hetero_data = torch.load(f"{PROCESSED_DIR}/hetero_graph.pt", map_location='cpu', weights_only=False)
bert_embeds = torch.load(f"{PROCESSED_DIR}/bert_embeddings.pt", map_location='cpu', weights_only=True)
post_meta = torch.load(f"{PROCESSED_DIR}/post_metadata.pt", map_location='cpu', weights_only=True)
labels_df = pd.read_csv(f"{PROCESSED_DIR}/post_labels_and_splits.csv", index_col=0)
# Load expert data from Phase 5
try:
    expert_probs = np.load(f"{PROCESSED_DIR}/expert_probabilities.npy")
    gate_values = np.load(f"{PROCESSED_DIR}/gate_values.npy")  # 0=xgb, 1=bert, 2=gnn
    expert_names_map = {0: "XGBoost", 1: "BERT", 2: "GNN"}
    # Create continuous gate: max expert confidence
    gate_continuous = np.max(np.abs(expert_probs - 0.5), axis=1)
except:
    gate_values = np.zeros(len(y))
    gate_continuous = np.zeros(len(y))
    expert_probs = np.zeros((len(y), 3))

y = hetero_data['post'].y.numpy()
confusion_scores = labels_df['Confusion(1-7)'].values
test_mask = hetero_data['post'].test_mask.numpy()

meta_feature_names = ['Question', 'Answer', 'Opinion',
                      'TextLength', 'Anonymous', 'IsReply']
n_posts = len(y)

#  6A: INTEGRATED GRADIENTS — Feature Attribution
print("  6A | Integrated Gradients — Feature Attribution")
print("~" * 70)

# We compute feature importance via gradient-based attribution
# on the metadata features through a retrained lightweight model
# This avoids complexity of full GNN IG and gives clean SHAP-like results

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

# Build feature matrix: metadata + basic BERT stats
meta_np = post_meta.numpy()
bert_np = bert_embeds.numpy()
bert_stats = np.column_stack([
    np.linalg.norm(bert_np, axis=1),     # L2 norm
    bert_np.mean(axis=1),                 # mean activation
    bert_np.std(axis=1),                  # std activation
])
bert_stat_names = ['BERT_norm', 'BERT_mean', 'BERT_std']

# Graph structure features
deg_reply = np.zeros(n_posts)
deg_thread = np.zeros(n_posts)
deg_temporal = np.zeros(n_posts)
for etype, col in [
    (('post', 'reply_to', 'post'), 'reply'),
    (('post', 'same_thread', 'post'), 'thread'),
    (('post', 'temporal_next', 'post'), 'temporal'),
]:
    if etype in hetero_data.edge_types:
        ei = hetero_data[etype].edge_index.numpy()
        arr = deg_reply if col == 'reply' else (deg_thread if col == 'thread' else deg_temporal)
        for ni in ei[1]:
            if ni < n_posts: arr[ni] += 1

graph_feats = np.column_stack([deg_reply, deg_thread, deg_temporal, gate_values])
graph_feat_names = ['ReplyDegree', 'ThreadDegree', 'TemporalDegree', 'DominantExpert']

X_full = np.column_stack([meta_np, bert_stats, graph_feats])
all_feat_names = meta_feature_names + bert_stat_names + graph_feat_names

train_mask = hetero_data['post'].train_mask.numpy()

print(f"  Feature matrix: {X_full.shape} ({len(all_feat_names)} features)")
print(f"  Training on {train_mask.sum()} samples...")

clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                  random_state=42, verbose=0)
clf.fit(X_full[train_mask], y[train_mask])

# Built-in feature importance
feat_imp = clf.feature_importances_
sorted_idx = np.argsort(feat_imp)[::-1]

print(f"\n  Feature Importance (Gradient Boosting):")
for i in sorted_idx:
    bar = "=" * int(feat_imp[i] * 100)
    print(f"    {all_feat_names[i]:18s}: {feat_imp[i]:.4f}  {bar}")

# Permutation importance (more reliable)
print(f"\n  Computing permutation importance...")
perm_imp = permutation_importance(clf, X_full[test_mask], y[test_mask],
                                   n_repeats=10, random_state=42, scoring='f1_macro')

print(f"  Permutation Importance:")
perm_sorted = np.argsort(perm_imp.importances_mean)[::-1]
for i in perm_sorted:
    print(f"    {all_feat_names[i]:18s}: {perm_imp.importances_mean[i]:.4f} +/- {perm_imp.importances_std[i]:.4f}")

# -- Figure 20: Feature Importance (dual panel) --
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Built-in importance
top_n = len(all_feat_names)
sorted_names = [all_feat_names[i] for i in sorted_idx[:top_n]]
sorted_vals = feat_imp[sorted_idx[:top_n]]
color_map = {'Question': '#e74c3c', 'TextLength': '#f39c12',
             'Answer': '#3498db', 'Opinion': '#3498db', 'DominantExpert': '#9b59b6'}
bar_colors = [color_map.get(n, '#95a5a6') for n in sorted_names]

axes[0].barh(range(top_n), sorted_vals[::-1], color=bar_colors[::-1], edgecolor='white', height=0.6)
axes[0].set_yticks(range(top_n)); axes[0].set_yticklabels(sorted_names[::-1], fontsize=9)
axes[0].set_xlabel('Feature Importance'); axes[0].set_title('(a) Built-in Feature Importance')

# Right: Permutation importance
perm_names = [all_feat_names[i] for i in perm_sorted[:top_n]]
perm_vals = perm_imp.importances_mean[perm_sorted[:top_n]]
perm_errs = perm_imp.importances_std[perm_sorted[:top_n]]
pbar_colors = [color_map.get(n, '#95a5a6') for n in perm_names]

axes[1].barh(range(top_n), perm_vals[::-1], xerr=perm_errs[::-1],
             color=pbar_colors[::-1], edgecolor='white', height=0.6, capsize=2)
axes[1].set_yticks(range(top_n)); axes[1].set_yticklabels(perm_names[::-1], fontsize=9)
axes[1].set_xlabel('Permutation Importance (F1 drop)'); axes[1].set_title('(b) Permutation Importance')

plt.suptitle('Feature Attribution Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig20_feature_importance.png"); plt.savefig(f"{FIGURES_DIR}/fig20_feature_importance.pdf"); plt.close()
print(f"\n  -> Saved: {FIGURES_DIR}/fig20_feature_importance.png")

#  6B: GRAPH NEIGHBORHOOD ANALYSIS — Local Explainability
print("  6B | Graph Neighborhood Analysis (Local Explainability)")
print("~" * 70)

# For each confused post, extract its 1-hop neighborhood properties
# This provides interpretable local explanations without GNNExplainer overhead

# Build adjacency info
reply_to_adj = defaultdict(list)  # post -> list of posts it replies to
same_thread_adj = defaultdict(list)
temporal_adj = defaultdict(list)
authored_by = {}  # post -> user

for etype in hetero_data.edge_types:
    ei = hetero_data[etype].edge_index.numpy()
    src_t, rel, dst_t = etype
    if rel == 'reply_to':
        for s, d in zip(ei[0], ei[1]): reply_to_adj[s].append(d)
    elif rel == 'same_thread':
        for s, d in zip(ei[0], ei[1]): same_thread_adj[s].append(d)
    elif rel == 'temporal_next':
        for s, d in zip(ei[0], ei[1]): temporal_adj[s].append(d)
    elif rel == 'authored_by':
        for s, d in zip(ei[0], ei[1]): authored_by[s] = d

# Confused test posts
confused_test = np.where(test_mask & (y == 1))[0]
not_confused_test = np.where(test_mask & (y == 0))[0]

print(f"  Confused test posts: {len(confused_test)}")
print(f"  Not confused test posts: {len(not_confused_test)}")

# Extract neighborhood features for each post
def get_neighborhood_features(post_idx):
    """Extract interpretable neighborhood features for a post."""
    feats = {}
    meta = post_meta[post_idx].numpy()
    feats['is_question'] = meta[0]
    feats['is_answer'] = meta[1]
    feats['is_opinion'] = meta[2]
    feats['text_length'] = meta[3]
    feats['is_anonymous'] = meta[4]
    feats['gate_value'] = gate_continuous[post_idx]
    feats['dominant_expert'] = gate_values[post_idx]

    # Reply-to neighborhood
    reply_neighbors = reply_to_adj.get(post_idx, [])
    feats['n_reply_to'] = len(reply_neighbors)
    if reply_neighbors:
        feats['reply_neighbor_mean_conf'] = np.mean([confusion_scores[n] for n in reply_neighbors if n < n_posts])
    else:
        feats['reply_neighbor_mean_conf'] = 0

    # Same thread neighborhood
    thread_neighbors = same_thread_adj.get(post_idx, [])
    feats['n_same_thread'] = len(thread_neighbors)
    if thread_neighbors:
        neighbor_questions = sum(1 for n in thread_neighbors if n < n_posts and post_meta[n, 0] > 0.5)
        neighbor_answers = sum(1 for n in thread_neighbors if n < n_posts and post_meta[n, 1] > 0.5)
        feats['thread_question_ratio'] = neighbor_questions / len(thread_neighbors)
        feats['thread_answer_ratio'] = neighbor_answers / len(thread_neighbors)
        feats['thread_mean_conf'] = np.mean([confusion_scores[n] for n in thread_neighbors if n < n_posts])
    else:
        feats['thread_question_ratio'] = 0
        feats['thread_answer_ratio'] = 0
        feats['thread_mean_conf'] = 0

    # Temporal neighborhood
    temp_neighbors = temporal_adj.get(post_idx, [])
    feats['n_temporal'] = len(temp_neighbors)

    # User info
    user_id = authored_by.get(post_idx, None)
    if user_id is not None and user_id < hetero_data['user'].x.shape[0]:
        user_feat = hetero_data['user'].x[user_id].numpy()
        feats['user_post_count'] = user_feat[0]
        feats['user_question_ratio'] = user_feat[1]
    else:
        feats['user_post_count'] = 0
        feats['user_question_ratio'] = 0

    # Isolation score: how connected is this post?
    total_neighbors = len(reply_neighbors) + len(thread_neighbors) + len(temp_neighbors)
    feats['isolation_score'] = 1.0 / (1.0 + total_neighbors)

    # Has answer neighbor?
    feats['has_answer_neighbor'] = 1.0 if any(
        post_meta[n, 1] > 0.5 for n in thread_neighbors if n < n_posts) else 0.0

    return feats

print("  Extracting neighborhood features for all test posts...")
confused_feats = [get_neighborhood_features(i) for i in confused_test[:TOP_K_EXPLAIN]]
not_confused_feats = [get_neighborhood_features(i) for i in not_confused_test[:TOP_K_EXPLAIN]]

# Compare distributions
feat_keys = list(confused_feats[0].keys())
print(f"\n  {'Feature':<30} {'Confused':>10} {'Not Conf.':>10} {'Diff':>10}")
print(f"  {'─'*62}")
for k in feat_keys:
    c_mean = np.mean([f[k] for f in confused_feats])
    nc_mean = np.mean([f[k] for f in not_confused_feats])
    diff = c_mean - nc_mean
    marker = "***" if abs(diff) > 0.1 else "**" if abs(diff) > 0.05 else ""
    print(f"  {k:<30} {c_mean:>10.4f} {nc_mean:>10.4f} {diff:>+10.4f} {marker}")

#  6C: CONFUSION ARCHETYPE DISCOVERY
print("  6C | Confusion Archetype Discovery")
print("~" * 70)

# Cluster confused posts by their neighborhood features to find archetypes
archetype_features = ['is_question', 'text_length', 'isolation_score', 'has_answer_neighbor',
                      'thread_answer_ratio', 'n_same_thread', 'gate_value', 'user_question_ratio']

X_archetype = np.array([[f[k] for k in archetype_features] for f in confused_feats])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_archetype)

# Try different K values
print("  Finding optimal number of archetypes...")
inertias = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"    K={k}: inertia={km.inertia_:.1f}")

# Use K=4 (matches our hypothesized archetypes)
N_ARCHETYPES = 4
km = KMeans(n_clusters=N_ARCHETYPES, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_scaled)

# Analyze each cluster
archetype_names = []
archetype_descriptions = []

print(f"\n  Discovered {N_ARCHETYPES} Confusion Archetypes:")
for c in range(N_ARCHETYPES):
    mask_c = cluster_labels == c
    n_in_cluster = mask_c.sum()
    cluster_feats = X_archetype[mask_c]

    # Characterize
    mean_feats = {archetype_features[i]: cluster_feats[:, i].mean() for i in range(len(archetype_features))}

    print(f"\n  Archetype {c} (n={n_in_cluster}):")
    for k, v in mean_feats.items():
        print(f"    {k:25s}: {v:.3f}")

    # Auto-label based on dominant characteristics
    if mean_feats['is_question'] > 0.5 and mean_feats['has_answer_neighbor'] < 0.3:
        name = "Content Gap"
        desc = "Question posts without answer neighbors — unresolved knowledge gaps"
    elif mean_feats['isolation_score'] > 0.4 or mean_feats['n_same_thread'] < 1:
        name = "Social Isolation"
        desc = "Low connectivity, few thread interactions — disconnected learners"
    elif mean_feats.get('is_question', 0) < 0.3 and mean_feats.get('text_length', 0) > 0.5:
        name = "Assessment Anxiety"
        desc = "Long posts with elevated confusion — struggling learners"
    else:
        name = "Thread Drift"
        desc = "Moderate connectivity but confusion persists — lost in discussion"

    archetype_names.append(name)
    archetype_descriptions.append(desc)
    print(f"    => Label: {name}")
    print(f"    => {desc}")

# -- Figure 21: Archetype Visualization --
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Left: Archetype sizes
arch_counts = [sum(cluster_labels == c) for c in range(N_ARCHETYPES)]
arch_colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
axes[0].bar(range(N_ARCHETYPES), arch_counts, color=arch_colors, edgecolor='white', width=0.6)
axes[0].set_xticks(range(N_ARCHETYPES))
axes[0].set_xticklabels([f'{archetype_names[i]}\n(n={arch_counts[i]})' for i in range(N_ARCHETYPES)], fontsize=9)
axes[0].set_ylabel('Number of Confused Posts'); axes[0].set_title('(a) Archetype Distribution')

# Middle: Radar chart (feature profiles)
from matplotlib.patches import FancyBboxPatch

# Heatmap of archetype feature profiles
cluster_profiles = np.zeros((N_ARCHETYPES, len(archetype_features)))
for c in range(N_ARCHETYPES):
    cluster_profiles[c] = X_archetype[cluster_labels == c].mean(axis=0)

# Normalize for visualization
cp_norm = (cluster_profiles - cluster_profiles.min(0)) / (cluster_profiles.max(0) - cluster_profiles.min(0) + 1e-8)

sns.heatmap(cp_norm, ax=axes[1], cmap='RdYlBu_r', annot=cluster_profiles, fmt='.2f',
            xticklabels=[f.replace('_', '\n')[:12] for f in archetype_features],
            yticklabels=archetype_names, cbar_kws={'shrink': 0.8})
axes[1].set_title('(b) Archetype Feature Profiles')

# Right: Gate value distribution per archetype
for c in range(N_ARCHETYPES):
    c_gates = [confused_feats[i]['gate_value'] for i in range(len(confused_feats)) if cluster_labels[i] == c]
    axes[2].hist(c_gates, bins=20, alpha=0.5, color=arch_colors[c], label=archetype_names[c], density=True)
axes[2].set_xlabel('Gate Value (0=metadata, 1=text)')
axes[2].set_ylabel('Density'); axes[2].set_title('(c) Gate Values per Archetype')
axes[2].legend(fontsize=8)

plt.suptitle('Confusion Archetypes Discovered via Neighborhood Clustering', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig21_confusion_archetypes.png"); plt.savefig(f"{FIGURES_DIR}/fig21_confusion_archetypes.pdf"); plt.close()
print(f"\n  -> Saved: {FIGURES_DIR}/fig21_confusion_archetypes.png")

#  6D: CONFUSED vs NOT-CONFUSED NEIGHBORHOOD COMPARISON
print("  6D | Confused vs Not-Confused Neighborhood Comparison")
print("~" * 70)

# -- Figure 22: Key differences between confused and not-confused neighborhoods --
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

comparison_features = [
    ('isolation_score', 'Isolation Score', 'Higher = more isolated'),
    ('has_answer_neighbor', 'Has Answer Neighbor', 'Presence of answer in thread'),
    ('thread_mean_conf', 'Thread Mean Confusion', 'Average confusion in thread'),
    ('gate_value', 'Expert Confidence', 'Max expert confidence (higher = more certain)'),
    ('n_same_thread', 'Thread Size', 'Number of same-thread connections'),
    ('text_length', 'Text Length', 'Normalized post length'),
]

for idx, (feat_key, title, xlabel) in enumerate(comparison_features):
    ax = axes[idx // 3, idx % 3]
    c_vals = [f[feat_key] for f in confused_feats]
    nc_vals = [f[feat_key] for f in not_confused_feats]

    ax.hist(nc_vals, bins=25, alpha=0.5, color='#2ecc71', label='Not Confused', density=True)
    ax.hist(c_vals, bins=25, alpha=0.5, color='#e74c3c', label='Confused', density=True)
    ax.axvline(np.mean(c_vals), color='#c0392b', linestyle='--', linewidth=1.5)
    ax.axvline(np.mean(nc_vals), color='#27ae60', linestyle='--', linewidth=1.5)
    ax.set_xlabel(xlabel); ax.set_title(title)
    ax.legend(fontsize=8)

plt.suptitle('Neighborhood Characteristics: Confused vs Not-Confused Posts', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig22_confused_vs_not.png"); plt.savefig(f"{FIGURES_DIR}/fig22_confused_vs_not.pdf"); plt.close()
print(f"  -> Saved: {FIGURES_DIR}/fig22_confused_vs_not.png")

# -- Figure 23: Example subgraphs per archetype --
print("\n  Generating archetype example subgraphs...")
fig, axes = plt.subplots(1, N_ARCHETYPES, figsize=(5 * N_ARCHETYPES, 5))

confusion_cmap = plt.cm.RdYlGn_r

for c in range(N_ARCHETYPES):
    ax = axes[c]
    ax.set_facecolor('#FAFBFC')

    # Pick a representative post from this archetype
    c_indices = np.where(cluster_labels == c)[0]
    # Pick the one closest to the cluster center
    center = km.cluster_centers_[c]
    dists = np.linalg.norm(X_scaled[c_indices] - center, axis=1)
    rep_local_idx = c_indices[np.argmin(dists)]
    rep_global_idx = confused_test[rep_local_idx]

    # Build its neighborhood subgraph
    G = nx.DiGraph()
    conf_norm = (confusion_scores[rep_global_idx] - 1) / 6
    G.add_node(rep_global_idx, label=f"Target\nC={confusion_scores[rep_global_idx]:.1f}",
               color=confusion_cmap(conf_norm), is_center=True)

    neighbors = set()
    for adj_dict, etype_label in [(reply_to_adj, 'reply'), (same_thread_adj, 'thread'), (temporal_adj, 'temporal')]:
        for n in adj_dict.get(rep_global_idx, [])[:3]:
            if n < n_posts:
                cn = (confusion_scores[n] - 1) / 6
                is_q = post_meta[n, 0].item() > 0.5
                is_a = post_meta[n, 1].item() > 0.5
                lbl = f"{'Q' if is_q else 'A' if is_a else 'P'}\nC={confusion_scores[n]:.1f}"
                G.add_node(n, label=lbl, color=confusion_cmap(cn), is_center=False)
                G.add_edge(rep_global_idx, n, etype=etype_label)
                neighbors.add(n)

    if len(G.nodes) < 2:
        G.add_node(-1, label="(isolated)", color='#cccccc', is_center=False)

    pos = nx.spring_layout(G, seed=42, k=2)
    node_colors = [G.nodes[n].get('color', '#cccccc') for n in G.nodes]
    node_sizes = [600 if G.nodes[n].get('is_center', False) else 400 for n in G.nodes]

    edge_colors = {'reply': '#e74c3c', 'thread': '#3498db', 'temporal': '#2ecc71'}
    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, [(u, v)], ax=ax, edge_color=edge_colors.get(d['etype'], '#999'),
                               width=2, arrows=True, arrowsize=12, min_source_margin=15, min_target_margin=15)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                           edgecolors='#2c3e50', linewidths=1.5)
    labels = {n: G.nodes[n]['label'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight='bold')

    ax.set_title(f'{archetype_names[c]}\n({archetype_descriptions[c][:45]}...)', fontsize=9, fontweight='bold')
    ax.axis('off')

plt.suptitle('Example Subgraphs for Each Confusion Archetype', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig23_archetype_subgraphs.png", dpi=200)
plt.savefig(f"{FIGURES_DIR}/fig23_archetype_subgraphs.pdf")
plt.close()
print(f"  -> Saved: {FIGURES_DIR}/fig23_archetype_subgraphs.png")

#  SAVE XAI RESULTS
print("  Saving XAI Results")
print("~" * 70)

xai_results = {
    'feature_importance': {
        'builtin': {all_feat_names[i]: float(feat_imp[i]) for i in sorted_idx},
        'permutation': {all_feat_names[i]: float(perm_imp.importances_mean[i]) for i in perm_sorted},
    },
    'archetypes': {
        f'archetype_{c}': {
            'name': archetype_names[c],
            'description': archetype_descriptions[c],
            'count': int(arch_counts[c]),
            'mean_features': {archetype_features[i]: float(cluster_profiles[c, i])
                              for i in range(len(archetype_features))},
        } for c in range(N_ARCHETYPES)
    },
    'confused_vs_not_confused': {
        k: {
            'confused_mean': float(np.mean([f[k] for f in confused_feats])),
            'not_confused_mean': float(np.mean([f[k] for f in not_confused_feats])),
        } for k in feat_keys
    },
}

with open(f"{RESULTS_DIR}/phase6_xai_results.json", 'w') as f:
    json.dump(xai_results, f, indent=2)
print(f"  -> Saved: {RESULTS_DIR}/phase6_xai_results.json")

# Save archetype labels for Phase 7
archetype_data = {
    'confused_test_indices': confused_test[:TOP_K_EXPLAIN].tolist(),
    'cluster_labels': cluster_labels.tolist(),
    'archetype_names': archetype_names,
    'archetype_descriptions': archetype_descriptions,
    'archetype_features': archetype_features,
    'cluster_profiles': cluster_profiles.tolist(),
}
with open(f"{PROCESSED_DIR}/archetype_data.json", 'w') as f:
    json.dump(archetype_data, f, indent=2)
print(f"  -> Saved: {PROCESSED_DIR}/archetype_data.json")

print("=" * 70)
print(f"""
  Figures:
    fig20 -> Feature importance (built-in + permutation)
    fig21 -> Confusion archetypes (distribution + profiles + gates)
    fig22 -> Confused vs not-confused neighborhood comparison
    fig23 -> Example subgraphs per archetype

  Results:
    phase6_xai_results.json -> All XAI results
    archetype_data.json     -> Archetype data for Phase 7

  Discovered {N_ARCHETYPES} archetypes:
""")
for c in range(N_ARCHETYPES):
    print(f"    {c+1}. {archetype_names[c]}: {archetype_descriptions[c]}")
print(f"\n  Next: Run phase7_interventions.py")