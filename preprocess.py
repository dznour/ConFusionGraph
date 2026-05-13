# preprocess.py
# Data preprocessing and heterogeneous graph construction

import os, time, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter
import torch
from transformers import BertTokenizer, BertModel
from torch_geometric.data import HeteroData
from tqdm import tqdm
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# ── Configuration ──
DATASET_PATH = "stanfordMOOCForumPostsSet.xlsx"
CONFUSION_THRESHOLD = 4.5
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 256
BERT_BATCH_SIZE = 64
RANDOM_SEED = 42
SAME_THREAD_CAP = 5

OUTPUT_DIR = "no_urg/figures"
PROCESSED_DIR = "no_urg/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.facecolor': 'white', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

print("=" * 70)
print("=" * 70)

#  1.1  LOAD & EXPLORE DATASET
print("  1.1 | Loading Dataset")
print("~" * 70)

df = pd.read_excel(DATASET_PATH)
print(f"  Total posts:        {len(df):,}")
print(f"  Columns:            {len(df.columns)}")
print(f"  Unique courses:     {df['course_display_name'].nunique()}")
print(f"  Unique users:       {df['forum_uid'].nunique()}")
print(f"  Date range:         {df['created_at'].min().date()} -> {df['created_at'].max().date()}")
print(f"  Post types:         {dict(df['post_type'].value_counts())}")
print(f"  Null text posts:    {df['Text'].isna().sum()}")

df = df.dropna(subset=['Text', 'forum_uid', 'created_at', 'post_type']).reset_index(drop=True)
df['Text'] = df['Text'].astype(str)
df['text_len'] = df['Text'].str.len()
print(f"\n  After cleaning:     {len(df):,} posts remaining")

#  1.2  CONFUSION DISTRIBUTION ANALYSIS
print("  1.2 | Confusion Distribution Analysis")
print("~" * 70)

conf = df['Confusion(1-7)']
print(f"  Mean: {conf.mean():.3f}  Std: {conf.std():.3f}  Median: {conf.median():.1f}  Mode: {conf.mode().values[0]:.1f}")
print(f"\n  Value counts:")
for val, cnt in conf.value_counts().sort_index().items():
    bar = "=" * int(cnt / 500)
    print(f"    {val:4.1f}: {cnt:6,}  {bar}")

# -- Figure 1: Confusion Distribution --
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ['#2ecc71' if v < CONFUSION_THRESHOLD else '#e74c3c' for v in sorted(conf.unique())]
counts = conf.value_counts().sort_index()
axes[0].bar(counts.index, counts.values, width=0.4, color=colors, edgecolor='white', linewidth=0.5)
axes[0].axvline(x=CONFUSION_THRESHOLD, color='#2c3e50', linestyle='--', linewidth=1.5, label=f'Threshold = {CONFUSION_THRESHOLD}')
axes[0].set_xlabel('Confusion Score'); axes[0].set_ylabel('Number of Posts')
axes[0].set_title('(a) Confusion Score Distribution'); axes[0].legend(fontsize=9)

course_order = ['Education', 'Humanities', 'Medicine']
df_ct = df.dropna(subset=['CourseType'])
palette = {'Education': '#3498db', 'Humanities': '#9b59b6', 'Medicine': '#e67e22'}
sns.boxplot(data=df_ct, x='CourseType', y='Confusion(1-7)', order=course_order, palette=palette, ax=axes[1], fliersize=2, width=0.5)
axes[1].set_xlabel('Course Domain'); axes[1].set_ylabel('Confusion Score')
axes[1].set_title('(b) Confusion by Course Domain')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_confusion_distribution.png"); plt.savefig(f"{OUTPUT_DIR}/fig1_confusion_distribution.pdf"); plt.close()
print(f"\n  -> Saved: {OUTPUT_DIR}/fig1_confusion_distribution.png")

#  1.3  FEATURE CORRELATIONS WITH CONFUSION
print("  1.3 | Feature Correlations with Confusion")
print("~" * 70)

corr_cols = ['Question(1/0)', 'Answer(1/0)', 'Opinion(1/0)', 'Sentiment(1-7)', 'Urgency(1-7)', 'up_count', 'reads', 'text_len']
corr_values = {}
for c in corr_cols:
    r = df[c].corr(df['Confusion(1-7)'])
    corr_values[c] = r
    direction = "+" if r > 0 else "-"
    bar = "=" * int(abs(r) * 40)
    print(f"  {c:20s}  r = {r:+.4f}  {direction} {bar}")

# -- Figure 2: Correlations + Question/PostType analysis --
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sorted_corr = sorted(corr_values.items(), key=lambda x: x[1])
names = [k.replace('(1/0)', '').replace('(1-7)', '').strip() for k, _ in sorted_corr]
vals = [v for _, v in sorted_corr]
bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in vals]
axes[0].barh(names, vals, color=bar_colors, edgecolor='white', height=0.6)
axes[0].axvline(x=0, color='#2c3e50', linewidth=0.8)
axes[0].set_xlabel('Pearson Correlation with Confusion'); axes[0].set_title('(a) Feature Correlations')

groups = [df[df['Question(1/0)'] == 0]['Confusion(1-7)'], df[df['Question(1/0)'] == 1]['Confusion(1-7)']]
bp = axes[1].boxplot(groups, labels=['Non-Question', 'Question'], patch_artist=True, widths=0.5, flierprops=dict(markersize=2))
bp['boxes'][0].set_facecolor('#3498db'); bp['boxes'][1].set_facecolor('#e74c3c')
axes[1].set_ylabel('Confusion Score'); axes[1].set_title('(b) Questions vs Non-Questions')

groups2 = [df[df['post_type'] == 'CommentThread']['Confusion(1-7)'], df[df['post_type'] == 'Comment']['Confusion(1-7)']]
bp2 = axes[2].boxplot(groups2, labels=['Thread Root', 'Reply'], patch_artist=True, widths=0.5, flierprops=dict(markersize=2))
bp2['boxes'][0].set_facecolor('#9b59b6'); bp2['boxes'][1].set_facecolor('#f39c12')
axes[2].set_ylabel('Confusion Score'); axes[2].set_title('(c) Roots vs Replies')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_correlations_and_flags.png"); plt.savefig(f"{OUTPUT_DIR}/fig2_correlations_and_flags.pdf"); plt.close()
print(f"\n  -> Saved: {OUTPUT_DIR}/fig2_correlations_and_flags.png")

#  1.4  THREAD STRUCTURE ANALYSIS
print("  1.4 | Thread Structure Analysis")
print("~" * 70)

roots = df[df['post_type'] == 'CommentThread'].copy()
replies = df[df['post_type'] == 'Comment'].copy()
root_id_set = set(roots['forum_post_id'].dropna())
linkable_replies = replies[replies['comment_thread_id'].isin(root_id_set)]
orphan_replies = replies[~replies['comment_thread_id'].isin(root_id_set)]

print(f"  Thread roots:        {len(roots):,}")
print(f"  Linkable replies:    {len(linkable_replies):,}")
print(f"  Orphan replies:      {len(orphan_replies):,}")

thread_sizes = linkable_replies.groupby('comment_thread_id').size()
print(f"  Thread size: mean={thread_sizes.mean():.2f}, median={thread_sizes.median():.1f}, max={thread_sizes.max()}")

# Confusion trajectory
multi_reply_threads = linkable_replies.groupby('comment_thread_id').filter(lambda x: len(x) >= 3)
trajectories = []
for tid, group in multi_reply_threads.groupby('comment_thread_id'):
    group = group.sort_values('created_at')
    vals = group['Confusion(1-7)'].values
    first_half = vals[:len(vals) // 2].mean()
    second_half = vals[len(vals) // 2:].mean()
    trajectories.append(second_half - first_half)
trajectories = np.array(trajectories)
print(f"  Confusion trajectories (threads >= 3 replies, n={len(trajectories)}):")
print(f"    Increasing (d > 0.5):  {(trajectories > 0.5).sum()}")
print(f"    Stable (|d| <= 0.5):   {((trajectories >= -0.5) & (trajectories <= 0.5)).sum()}")
print(f"    Decreasing (d < -0.5): {(trajectories < -0.5).sum()}")

# -- Figure 3: Thread Structure --
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

thread_sz_counts = thread_sizes.value_counts().sort_index()
axes[0].bar(thread_sz_counts.index[:20], thread_sz_counts.values[:20], color='#3498db', edgecolor='white')
axes[0].set_xlabel('Number of Replies'); axes[0].set_ylabel('Number of Threads')
axes[0].set_title('(a) Thread Size Distribution'); axes[0].set_yscale('log')

axes[1].hist(trajectories, bins=30, color='#9b59b6', edgecolor='white', alpha=0.8)
axes[1].axvline(x=0, color='#2c3e50', linestyle='-', linewidth=1)
axes[1].axvline(x=0.5, color='#e74c3c', linestyle='--', linewidth=1, label='+/-0.5 boundary')
axes[1].axvline(x=-0.5, color='#e74c3c', linestyle='--', linewidth=1)
axes[1].set_xlabel('Confusion Change (2nd half - 1st half)'); axes[1].set_ylabel('Number of Threads')
axes[1].set_title('(b) Confusion Trajectory in Threads'); axes[1].legend(fontsize=9)

thread_root_conf = {}
for tid in thread_sizes[thread_sizes >= 2].index:
    root_row = roots[roots['forum_post_id'] == tid]
    if len(root_row) == 0: continue
    root_conf = root_row['Confusion(1-7)'].values[0]
    reply_conf = linkable_replies[linkable_replies['comment_thread_id'] == tid]['Confusion(1-7)'].mean()
    thread_root_conf[tid] = (root_conf, reply_conf)

if thread_root_conf:
    rc = [v[0] for v in thread_root_conf.values()]
    rpc = [v[1] for v in thread_root_conf.values()]
    axes[2].scatter(rc, rpc, alpha=0.3, s=15, color='#e67e22', edgecolors='none')
    z = np.polyfit(rc, rpc, 1); p = np.poly1d(z)
    x_line = np.linspace(min(rc), max(rc), 100)
    axes[2].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'slope={z[0]:.3f}')
    axes[2].set_xlabel('Root Post Confusion'); axes[2].set_ylabel('Mean Reply Confusion')
    axes[2].set_title('(c) Root Confusion -> Reply Confusion'); axes[2].legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_thread_structure.png"); plt.savefig(f"{OUTPUT_DIR}/fig3_thread_structure.pdf"); plt.close()
print(f"\n  -> Saved: {OUTPUT_DIR}/fig3_thread_structure.png")

#  1.5  BINARY LABEL CREATION
print(f"  1.5 | Binary Label Creation (threshold = {CONFUSION_THRESHOLD})")
print("~" * 70)

df['confused'] = (df['Confusion(1-7)'] >= CONFUSION_THRESHOLD).astype(int)
n_pos = df['confused'].sum(); n_neg = len(df) - n_pos; ratio = n_neg / n_pos
print(f"  Confused (>= {CONFUSION_THRESHOLD}):     {n_pos:,} ({n_pos/len(df)*100:.1f}%)")
print(f"  Not confused (< {CONFUSION_THRESHOLD}): {n_neg:,} ({n_neg/len(df)*100:.1f}%)")
print(f"  Imbalance ratio:         1:{ratio:.1f}")
for t in [4.0, 4.5, 5.0, 5.5]:
    p = (df['Confusion(1-7)'] >= t).sum(); n = len(df) - p
    print(f"    >= {t}: {p:5,} pos ({p/len(df)*100:5.1f}%), ratio 1:{n/p:.1f}")

# -- Figure 4: Class Balance --
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].pie([n_neg, n_pos], labels=[f'Not Confused\n({n_neg:,})', f'Confused\n({n_pos:,})'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 10}, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[0].set_title(f'(a) Class Balance (threshold >= {CONFUSION_THRESHOLD})')

thresholds = np.arange(3.5, 6.5, 0.5)
pos_rates = [(df['Confusion(1-7)'] >= t).mean() * 100 for t in thresholds]
ratios_t = [(df['Confusion(1-7)'] < t).sum() / max((df['Confusion(1-7)'] >= t).sum(), 1) for t in thresholds]
ax2 = axes[1].twinx()
axes[1].bar(thresholds, pos_rates, width=0.35, color='#e74c3c', alpha=0.7, label='% Positive')
ax2.plot(thresholds, ratios_t, 'o-', color='#2c3e50', linewidth=2, markersize=6, label='Imbalance Ratio')
axes[1].axvline(x=CONFUSION_THRESHOLD, color='#3498db', linestyle='--', linewidth=1.5, label=f'Chosen: {CONFUSION_THRESHOLD}')
axes[1].set_xlabel('Confusion Threshold'); axes[1].set_ylabel('% Positive Class', color='#e74c3c')
ax2.set_ylabel('Imbalance Ratio (neg:pos)', color='#2c3e50')
axes[1].set_title('(b) Threshold Sensitivity'); axes[1].legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_class_balance.png"); plt.savefig(f"{OUTPUT_DIR}/fig4_class_balance.pdf"); plt.close()
print(f"\n  -> Saved: {OUTPUT_DIR}/fig4_class_balance.png")

#  1.6  TEMPORAL TRAIN / VAL / TEST SPLIT
print("  1.6 | Temporal Train/Val/Test Split (70/15/15 per course)")
print("~" * 70)

df = df.sort_values(['course_display_name', 'created_at']).reset_index(drop=True)
df['split'] = ''
for course in df['course_display_name'].dropna().unique():
    mask = df['course_display_name'] == course; idx = df[mask].index; n = len(idx)
    train_end = int(n * 0.70); val_end = int(n * 0.85)
    df.loc[idx[:train_end], 'split'] = 'train'
    df.loc[idx[train_end:val_end], 'split'] = 'val'
    df.loc[idx[val_end:], 'split'] = 'test'
no_course = df['split'] == ''
if no_course.any(): df.loc[no_course, 'split'] = 'train'

print(f"\n  {'Split':<8} {'Posts':>8} {'Confused':>10} {'% Pos':>8}")
print(f"  {'---'*10}")
for split in ['train', 'val', 'test']:
    sub = df[df['split'] == split]; pos = sub['confused'].sum()
    print(f"  {split:<8} {len(sub):>8,} {pos:>10,} {pos/len(sub)*100:>7.1f}%")

# -- Figure 5: Temporal Split --
fig, ax = plt.subplots(figsize=(14, 5))
split_colors = {'train': '#3498db', 'val': '#f39c12', 'test': '#e74c3c'}
courses_sorted = df.groupby('course_display_name').size().sort_values(ascending=False).index
for i, course in enumerate(courses_sorted):
    sub = df[df['course_display_name'] == course]
    for split in ['train', 'val', 'test']:
        ss = sub[sub['split'] == split]
        if len(ss) > 0:
            ax.barh(i, (ss['created_at'].max() - ss['created_at'].min()).days,
                    left=(ss['created_at'].min() - df['created_at'].min()).days,
                    color=split_colors[split], height=0.6, alpha=0.8)
short_names = [c.split('/')[-1][:25] for c in courses_sorted]
ax.set_yticks(range(len(courses_sorted))); ax.set_yticklabels(short_names, fontsize=8)
ax.set_xlabel('Days from Dataset Start'); ax.set_title('Temporal Train/Val/Test Split by Course')
legend_patches = [mpatches.Patch(color=c, label=s.title()) for s, c in split_colors.items()]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9); ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_temporal_split.png"); plt.savefig(f"{OUTPUT_DIR}/fig5_temporal_split.pdf"); plt.close()
print(f"\n  -> Saved: {OUTPUT_DIR}/fig5_temporal_split.png")

#  1.7  NODE FEATURE ENGINEERING
print("  1.7 | Node Feature Engineering")
print("~" * 70)

# Post node metadata features (8-dim)
df['is_comment'] = (df['post_type'] == 'Comment').astype(float)
df['is_anonymous'] = df['anonymous'].fillna(0).astype(float)
df['log_text_len'] = np.log1p(df['text_len'])
df['log_text_len'] = df['log_text_len'] / df['log_text_len'].max()

post_meta_cols = ['Question(1/0)', 'Answer(1/0)', 'Opinion(1/0)',
                  'log_text_len', 'is_anonymous', 'is_comment']
post_meta = df[post_meta_cols].values.astype(np.float32)
print(f"  Post metadata features: {post_meta.shape}")
for i, col in enumerate(post_meta_cols):
    print(f"    [{i}] {col:20s}  mean={post_meta[:, i].mean():.3f}  std={post_meta[:, i].std():.3f}")

# User node features (8-dim)
user_ids = df['forum_uid'].unique()
user_id_map = {uid: i for i, uid in enumerate(user_ids)}
n_users = len(user_ids)
user_feats = np.zeros((n_users, 6), dtype=np.float32)
for uid, group in df.groupby('forum_uid'):
    idx = user_id_map[uid]
    user_feats[idx, 0] = np.log1p(len(group))
    user_feats[idx, 1] = group['Question(1/0)'].mean()
    user_feats[idx, 2] = group['Answer(1/0)'].mean()
    user_feats[idx, 3] = group['Opinion(1/0)'].mean()
    user_feats[idx, 4] = group['course_display_name'].nunique()
    user_feats[idx, 5] = np.log1p(group['text_len'].mean())
for j in range(6):
    col_max = user_feats[:, j].max()
    if col_max > 0: user_feats[:, j] /= col_max

user_feat_names = ['log_posts', 'question_ratio', 'answer_ratio', 'opinion_ratio',
                   'course_count', 'mean_log_textlen']
print(f"\n  User features: {user_feats.shape}")
for i, name in enumerate(user_feat_names):
    print(f"    [{i}] {name:20s}  mean={user_feats[:, i].mean():.3f}  std={user_feats[:, i].std():.3f}")

#  1.8  EDGE CONSTRUCTION (5 types)
print("  1.8 | Edge Construction (5 heterogeneous edge types)")
print("~" * 70)

post_id_map = {pid: i for i, pid in enumerate(df['forum_post_id'].values)}
n_posts = len(df)

# Edge 1: reply-to
print("\n  [1/5] reply-to edges (reply -> thread root)...")
reply_src, reply_dst = [], []
for _, row in replies.iterrows():
    if row['comment_thread_id'] in post_id_map and row['forum_post_id'] in post_id_map:
        reply_src.append(post_id_map[row['forum_post_id']])
        reply_dst.append(post_id_map[row['comment_thread_id']])
reply_to_edges = torch.tensor([reply_src, reply_dst], dtype=torch.long)
print(f"    Count: {reply_to_edges.shape[1]:,}")

# Edge 2: same-thread
print(f"  [2/5] same-thread edges (capped at {SAME_THREAD_CAP} per post)...")
same_src, same_dst = [], []
for tid, group in linkable_replies.groupby('comment_thread_id'):
    if tid not in post_id_map: continue
    post_indices = [post_id_map[tid]]
    for _, row in group.iterrows():
        if row['forum_post_id'] in post_id_map:
            post_indices.append(post_id_map[row['forum_post_id']])
    post_indices = list(set(post_indices))
    if len(post_indices) <= SAME_THREAD_CAP + 1:
        for i in range(len(post_indices)):
            for j in range(i + 1, len(post_indices)):
                same_src.extend([post_indices[i], post_indices[j]])
                same_dst.extend([post_indices[j], post_indices[i]])
    else:
        for i, idx_i in enumerate(post_indices):
            neighbors = post_indices[max(0, i - SAME_THREAD_CAP // 2):i] + \
                        post_indices[i + 1:i + 1 + SAME_THREAD_CAP // 2]
            for idx_j in neighbors[:SAME_THREAD_CAP]:
                if idx_i != idx_j:
                    same_src.extend([idx_i, idx_j]); same_dst.extend([idx_j, idx_i])
same_thread_edges = torch.tensor([same_src, same_dst], dtype=torch.long)
print(f"    Count: {same_thread_edges.shape[1]:,} (bidirectional)")

# Edge 3: authored-by
print("  [3/5] authored-by edges (post -> user)...")
auth_src, auth_dst = [], []
for i, row in df.iterrows():
    if row['forum_uid'] in user_id_map:
        auth_src.append(i); auth_dst.append(user_id_map[row['forum_uid']])
authored_by_edges = torch.tensor([auth_src, auth_dst], dtype=torch.long)
print(f"    Count: {authored_by_edges.shape[1]:,}")

# Edge 4: temporal-next
print("  [4/5] temporal-next edges (consecutive posts per course)...")
temp_src, temp_dst, temp_weights = [], [], []
TAU = 24 * 3600
for course in df['course_display_name'].dropna().unique():
    course_df = df[df['course_display_name'] == course].sort_values('created_at')
    indices = course_df.index.tolist(); times = course_df['created_at'].values
    for k in range(len(indices) - 1):
        dt = (pd.Timestamp(times[k + 1]) - pd.Timestamp(times[k])).total_seconds()
        weight = np.exp(-dt / TAU) if dt > 0 else 1.0
        temp_src.append(indices[k]); temp_dst.append(indices[k + 1]); temp_weights.append(weight)
temporal_edges = torch.tensor([temp_src, temp_dst], dtype=torch.long)
temporal_weights = torch.tensor(temp_weights, dtype=torch.float32)
print(f"    Count: {temporal_edges.shape[1]:,}")
print(f"    Weights: mean={temporal_weights.mean():.3f}, min={temporal_weights.min():.3f}, max={temporal_weights.max():.3f}")

# Edge 5: user-co-thread
print("  [5/5] user-co-thread edges...")
thread_users = defaultdict(set)
for _, row in linkable_replies.iterrows():
    if row['forum_uid'] in user_id_map:
        thread_users[row['comment_thread_id']].add(user_id_map[row['forum_uid']])
for _, row in roots.iterrows():
    if row['forum_post_id'] in set(linkable_replies['comment_thread_id']) and row['forum_uid'] in user_id_map:
        thread_users[row['forum_post_id']].add(user_id_map[row['forum_uid']])
cothread_pairs = set()
for tid, users in thread_users.items():
    users = list(users)
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            cothread_pairs.add((min(users[i], users[j]), max(users[i], users[j])))
cothread_src, cothread_dst = [], []
for u1, u2 in cothread_pairs:
    cothread_src.extend([u1, u2]); cothread_dst.extend([u2, u1])
user_cothread_edges = torch.tensor([cothread_src, cothread_dst], dtype=torch.long)
print(f"    Count: {user_cothread_edges.shape[1]:,} (bidirectional)")

total_edges = sum(e.shape[1] for e in [reply_to_edges, same_thread_edges, authored_by_edges, temporal_edges, user_cothread_edges])
print(f"\n  EDGE SUMMARY:")
print(f"    reply-to (post->post):       {reply_to_edges.shape[1]:>8,}")
print(f"    same-thread (post<->post):   {same_thread_edges.shape[1]:>8,}")
print(f"    authored-by (post->user):    {authored_by_edges.shape[1]:>8,}")
print(f"    temporal-next (post->post):  {temporal_edges.shape[1]:>8,}")
print(f"    user-co-thread (user<->user):{user_cothread_edges.shape[1]:>8,}")
print(f"    TOTAL EDGES:                 {total_edges:>8,}")
print(f"    TOTAL NODES:                 {n_posts + n_users:>8,} ({n_posts} posts + {n_users} users)")

#  1.9  GRAPH STATISTICS & DEGREE DISTRIBUTIONS
print("  1.9 | Graph Statistics & Degree Distributions")
print("~" * 70)

post_degree = np.zeros(n_posts, dtype=int)
for edges in [reply_to_edges, same_thread_edges, temporal_edges]:
    if edges.shape[1] > 0:
        for node_idx in edges[0].numpy():
            if node_idx < n_posts: post_degree[node_idx] += 1
        for node_idx in edges[1].numpy():
            if node_idx < n_posts: post_degree[node_idx] += 1

user_degree = np.zeros(n_users, dtype=int)
if authored_by_edges.shape[1] > 0:
    for node_idx in authored_by_edges[1].numpy(): user_degree[node_idx] += 1
if user_cothread_edges.shape[1] > 0:
    for node_idx in user_cothread_edges[0].numpy(): user_degree[node_idx] += 1

print(f"  Post degree: mean={post_degree.mean():.1f}, median={np.median(post_degree):.0f}, max={post_degree.max()}")
print(f"  User degree: mean={user_degree.mean():.1f}, median={np.median(user_degree):.0f}, max={user_degree.max()}")
isolated = (post_degree == 0).sum()
print(f"  Isolated posts (no post-post edges): {isolated:,} ({isolated/n_posts*100:.1f}%)")

# -- Figure 6: Degree Distributions --
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
deg_counts = Counter(post_degree); degs = sorted(deg_counts.keys())[:30]
axes[0].bar(degs, [deg_counts[d] for d in degs], color='#3498db', edgecolor='white')
axes[0].set_xlabel('Degree'); axes[0].set_ylabel('Number of Post Nodes')
axes[0].set_title('(a) Post Node Degree Distribution'); axes[0].set_yscale('log')

udeg_counts = Counter(user_degree); udegs = sorted(udeg_counts.keys())[:30]
axes[1].bar(udegs, [udeg_counts[d] for d in udegs], color='#e67e22', edgecolor='white')
axes[1].set_xlabel('Degree'); axes[1].set_ylabel('Number of User Nodes')
axes[1].set_title('(b) User Node Degree Distribution'); axes[1].set_yscale('log')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_degree_distributions.png"); plt.savefig(f"{OUTPUT_DIR}/fig6_degree_distributions.pdf"); plt.close()
print(f"\n  -> Saved: {OUTPUT_DIR}/fig6_degree_distributions.png")

#  1.10  PUBLICATION-QUALITY SUBGRAPH VISUALIZATION
print("  1.10 | Publication-Quality Subgraph Visualization (ALL node/edge types)")
print("~" * 70)

# Find best thread: 4-10 replies, multiple users, varied confusion
best_thread, best_score = None, 0
for tid in thread_sizes[thread_sizes >= 4].index[:500]:
    root_row = roots[roots['forum_post_id'] == tid]
    if len(root_row) == 0: continue
    thread_reps = linkable_replies[linkable_replies['comment_thread_id'] == tid]
    all_thread = pd.concat([root_row, thread_reps])
    n_unique_users = all_thread['forum_uid'].nunique()
    conf_std = all_thread['Confusion(1-7)'].std()
    n_reps = len(thread_reps)
    score = n_unique_users * 2 + conf_std * 5 + min(n_reps, 8)
    if 4 <= n_reps <= 10 and n_unique_users >= 3 and score > best_score:
        best_score = score; best_thread = tid

if best_thread is None: best_thread = thread_sizes.idxmax()
print(f"  Selected thread: {best_thread}")

root_row = roots[roots['forum_post_id'] == best_thread]
thread_reps = linkable_replies[linkable_replies['comment_thread_id'] == best_thread].sort_values('created_at')
thread_posts = pd.concat([root_row, thread_reps]).reset_index(drop=True)
print(f"  Posts: {len(thread_posts)}, Users: {thread_posts['forum_uid'].nunique()}")
print(f"  Confusion scores: {thread_posts['Confusion(1-7)'].values.tolist()}")

# Build networkx visualization graph
G = nx.DiGraph()
confusion_cmap = plt.cm.RdYlGn_r
post_nodes = []
for i, (_, row) in enumerate(thread_posts.iterrows()):
    node_id = f"P{i}"
    conf_score = row['Confusion(1-7)']
    is_root = (row['post_type'] == 'CommentThread')
    is_q = (row['Question(1/0)'] == 1)
    parts = []
    if is_root: parts.append("ROOT")
    if is_q: parts.append("Q")
    parts.append(f"C={conf_score:.1f}")
    G.add_node(node_id, ntype='post', confusion=conf_score, conf_norm=(conf_score - 1) / 6,
               is_root=is_root, is_question=is_q, label="\n".join(parts), user=row['forum_uid'])
    post_nodes.append(node_id)

user_set = thread_posts['forum_uid'].unique()
user_node_map = {}
for j, uid in enumerate(user_set):
    node_id = f"U{j}"; n_up = df[df['forum_uid'] == uid].shape[0]
    G.add_node(node_id, ntype='user', label=f"User {j+1}\n({n_up} posts)")
    user_node_map[uid] = node_id

# Edges
for i in range(1, len(post_nodes)):
    G.add_edge(post_nodes[i], post_nodes[0], etype='reply-to')
for i in range(len(post_nodes) - 1):
    G.add_edge(post_nodes[i], post_nodes[i + 1], etype='same-thread')
    G.add_edge(post_nodes[i + 1], post_nodes[i], etype='same-thread')
for i, (_, row) in enumerate(thread_posts.iterrows()):
    if row['forum_uid'] in user_node_map:
        G.add_edge(post_nodes[i], user_node_map[row['forum_uid']], etype='authored-by')
for i in range(len(post_nodes) - 1):
    G.add_edge(post_nodes[i], post_nodes[i + 1], etype='temporal-next')
user_list = list(user_node_map.values())
for i in range(len(user_list)):
    for j in range(i + 1, len(user_list)):
        G.add_edge(user_list[i], user_list[j], etype='user-co-thread')
        G.add_edge(user_list[j], user_list[i], etype='user-co-thread')

print(f"  Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# -- Draw --
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_facecolor('#FAFBFC')

# Layout: posts in a line, users below
pos = {}
for i, pn in enumerate(post_nodes):
    pos[pn] = (i * 2.2, 0)
for j, (uid, un) in enumerate(user_node_map.items()):
    user_post_xs = [pos[post_nodes[i]][0] for i, (_, row) in enumerate(thread_posts.iterrows()) if row['forum_uid'] == uid]
    cx = np.mean(user_post_xs) if user_post_xs else j * 2.2
    pos[un] = (cx, -2.8 - 0.9 * (j % 2))

edge_styles = {
    'reply-to':       {'color': '#e74c3c', 'style': 'solid', 'width': 2.5, 'alpha': 0.8},
    'same-thread':    {'color': '#3498db', 'style': 'dashed', 'width': 1.5, 'alpha': 0.5},
    'authored-by':    {'color': '#95a5a6', 'style': 'dotted', 'width': 1.2, 'alpha': 0.6},
    'temporal-next':  {'color': '#2ecc71', 'style': 'solid', 'width': 1.5, 'alpha': 0.5},
    'user-co-thread': {'color': '#f39c12', 'style': 'dashed', 'width': 1.5, 'alpha': 0.5},
}

for etype, style in edge_styles.items():
    edge_list = [(u, v) for u, v, d in G.edges(data=True) if d['etype'] == etype]
    if not edge_list: continue
    seen = set(); unique_edges = []
    for u, v in edge_list:
        pair = tuple(sorted([u, v]))
        if pair not in seen: seen.add(pair); unique_edges.append((u, v))
    is_directed = etype in ['reply-to', 'temporal-next', 'authored-by']
    rad = 0.15 if etype == 'same-thread' else (0.1 if etype == 'temporal-next' else 0.0)
    nx.draw_networkx_edges(G, pos, edgelist=unique_edges, ax=ax,
        edge_color=style['color'], style=style['style'], width=style['width'],
        alpha=style['alpha'], arrows=is_directed, arrowsize=15,
        connectionstyle=f'arc3,rad={rad}', min_source_margin=22, min_target_margin=22)

post_colors = [confusion_cmap(G.nodes[pn]['conf_norm']) for pn in post_nodes]
post_sizes = [1000 if G.nodes[pn]['is_root'] else 700 for pn in post_nodes]
nx.draw_networkx_nodes(G, pos, nodelist=post_nodes, ax=ax, node_color=post_colors,
                       node_size=post_sizes, node_shape='s', edgecolors='#2c3e50', linewidths=2)
nx.draw_networkx_nodes(G, pos, nodelist=user_list, ax=ax, node_color='#f0f0f0',
                       node_size=750, node_shape='o', edgecolors='#e67e22', linewidths=2.5)

post_labels = {pn: G.nodes[pn]['label'] for pn in post_nodes}
user_labels = {un: G.nodes[un]['label'] for un in user_list}
nx.draw_networkx_labels(G, pos, post_labels, ax=ax, font_size=7, font_weight='bold', font_color='white')
nx.draw_networkx_labels(G, pos, user_labels, ax=ax, font_size=7, font_color='#2c3e50')

legend_elements = [
    mpatches.Patch(facecolor='#a0a0a0', edgecolor='#2c3e50', label='Post Node (square)'),
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor='#f0f0f0', markeredgecolor='#e67e22', markersize=10, label='User Node (circle)'),
    mlines.Line2D([], [], color='#e74c3c', linewidth=2.5, label='reply-to'),
    mlines.Line2D([], [], color='#3498db', linewidth=1.5, linestyle='--', label='same-thread'),
    mlines.Line2D([], [], color='#95a5a6', linewidth=1.2, linestyle=':', label='authored-by'),
    mlines.Line2D([], [], color='#2ecc71', linewidth=1.5, label='temporal-next'),
    mlines.Line2D([], [], color='#f39c12', linewidth=1.5, linestyle='--', label='user-co-thread'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95,
          edgecolor='#bdc3c7', title='Graph Components', title_fontsize=11)

sm = plt.cm.ScalarMappable(cmap=confusion_cmap, norm=plt.Normalize(1, 7)); sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
cbar.set_label('Confusion Score', fontsize=10); cbar.set_ticks([1, 2, 3, 4, 5, 6, 7])

ax.set_title('ConFusionGraph: Heterogeneous Subgraph with All Node and Edge Types',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_subgraph_visualization.png", dpi=300)
plt.savefig(f"{OUTPUT_DIR}/fig7_subgraph_visualization.pdf")
plt.close()
print(f"\n  -> Saved: {OUTPUT_DIR}/fig7_subgraph_visualization.png (300 DPI)")
print(f"  -> Saved: {OUTPUT_DIR}/fig7_subgraph_visualization.pdf")

#  1.11  BERT EMBEDDING EXTRACTION
print("  1.11 | BERT Embedding Extraction")
print("~" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\n  Loading {BERT_MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device)
bert_model.eval()

texts = df['Text'].tolist()
n_batches = (len(texts) + BERT_BATCH_SIZE - 1) // BERT_BATCH_SIZE
print(f"  Extracting: {len(texts):,} posts, batch_size={BERT_BATCH_SIZE}, max_len={BERT_MAX_LEN}")

embeddings = []
start_time = time.time()
with torch.no_grad():
    for i in tqdm(range(0, len(texts), BERT_BATCH_SIZE), desc="  BERT", ncols=80):
        batch_texts = texts[i:i + BERT_BATCH_SIZE]
        encoded = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=BERT_MAX_LEN, return_tensors='pt').to(device)
        outputs = bert_model(**encoded)
        cls_embeds = outputs.last_hidden_state[:, 0, :].cpu()
        embeddings.append(cls_embeds)

bert_embeddings = torch.cat(embeddings, dim=0)
elapsed = time.time() - start_time
print(f"\n  Done: {bert_embeddings.shape} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"  Size: {bert_embeddings.nelement() * 4 / 1e6:.1f} MB")

#  1.12  ASSEMBLE PyG HeteroData
print("  1.12 | Assembling PyG HeteroData Object")
print("~" * 70)

post_features = torch.cat([bert_embeddings, torch.tensor(post_meta, dtype=torch.float32)], dim=1)
print(f"  Post features: {post_features.shape} (768 BERT + 6 metadata, no urgency/sentiment)")

data = HeteroData()
data['post'].x = post_features
data['post'].y = torch.tensor(df['confused'].values, dtype=torch.long)
data['user'].x = torch.tensor(user_feats, dtype=torch.float32)

data['post'].train_mask = torch.tensor((df['split'] == 'train').values, dtype=torch.bool)
data['post'].val_mask = torch.tensor((df['split'] == 'val').values, dtype=torch.bool)
data['post'].test_mask = torch.tensor((df['split'] == 'test').values, dtype=torch.bool)

data['post', 'reply_to', 'post'].edge_index = reply_to_edges
data['post', 'same_thread', 'post'].edge_index = same_thread_edges
data['post', 'authored_by', 'user'].edge_index = authored_by_edges
data['post', 'temporal_next', 'post'].edge_index = temporal_edges
data['post', 'temporal_next', 'post'].edge_attr = temporal_weights.unsqueeze(-1)
data['user', 'co_thread', 'user'].edge_index = user_cothread_edges
data['user', 'authored', 'post'].edge_index = torch.stack([authored_by_edges[1], authored_by_edges[0]])

print(f"\n  HeteroData object:")
print(f"  {data}")
print(f"\n  Node types: {data.node_types}")
print(f"  Edge types: {data.edge_types}")
print(f"  Labels: pos={data['post'].y.sum().item()}, neg={(1 - data['post'].y).sum().item()}")
print(f"  Train/Val/Test: {data['post'].train_mask.sum()}/{data['post'].val_mask.sum()}/{data['post'].test_mask.sum()}")
