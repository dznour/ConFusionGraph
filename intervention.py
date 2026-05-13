# intervention.py
# Multi-strategy XAI-driven intervention system

import os, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict, Counter

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

PARENT_DIR = "no_urg"
PROCESSED_DIR = f"{PARENT_DIR}/processed"
RESULTS_DIR = f"{PARENT_DIR}/results"
FIGURES_DIR = f"{PARENT_DIR}/figures"
DATASET_PATH = "stanfordMOOCForumPostsSet.xlsx"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("  MULTI-STRATEGY XAI-DRIVEN INTERVENTION SYSTEM")
print("=" * 70)

# ── Load data ──
print("\n  Loading data...")
hetero_data = torch.load(f"{PROCESSED_DIR}/hetero_graph.pt", map_location='cpu', weights_only=False)
post_meta = torch.load(f"{PROCESSED_DIR}/post_metadata.pt", map_location='cpu', weights_only=True).numpy()
labels_df = pd.read_csv(f"{PROCESSED_DIR}/post_labels_and_splits.csv", index_col=0)

with open(f"{PROCESSED_DIR}/archetype_data.json") as f:
    arch_data = json.load(f)

y = hetero_data['post'].y.numpy()
confusion_scores = labels_df['Confusion(1-7)'].values
n_posts = len(y)

df_raw = pd.read_excel(DATASET_PATH)
df_raw = df_raw.dropna(subset=['Text', 'forum_uid', 'created_at', 'post_type']).reset_index(drop=True)
df_raw = df_raw.sort_values(['course_display_name', 'created_at']).reset_index(drop=True)
texts = df_raw['Text'].astype(str).tolist()
courses = df_raw['course_display_name'].values
post_types = df_raw['post_type'].values
forum_uids = df_raw['forum_uid'].values

# meta: [0]=Question, [1]=Answer, [2]=Opinion, [3]=TextLength, [4]=Anonymous, [5]=IsReply

confused_indices = np.array(arch_data['confused_test_indices'])
cluster_labels = np.array(arch_data['cluster_labels'])
archetype_names = arch_data['archetype_names']
archetype_descriptions = arch_data['archetype_descriptions']

print(f"  Confused posts: {len(confused_indices)}")
print(f"  Archetypes: {dict(Counter(archetype_names))}")

# Build adjacency
print("  Building adjacency structures...")
reply_adj = defaultdict(list)
thread_adj = defaultdict(list)
thread_root = {}  # post_idx -> root_idx (via reply_to)

for et in hetero_data.edge_types:
    ei = hetero_data[et].edge_index.numpy()
    s, r, d = et
    if r == 'reply_to':
        for a, b in zip(ei[0], ei[1]):
            reply_adj[a].append(b)
            thread_root[a] = b
    elif r == 'same_thread':
        for a, b in zip(ei[0], ei[1]):
            thread_adj[a].append(b)

# ── Load sentence encoder ──
print("  Loading sentence encoder...")
enc_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
enc_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
enc_model.eval()

def encode_texts(text_list, batch_size=64):
    all_emb = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        enc = enc_tokenizer(batch, padding=True, truncation=True,
                            max_length=256, return_tensors='pt').to(device)
        with torch.no_grad():
            out = enc_model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1)
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb)

# Pre-encode answer posts
answer_mask = post_meta[:, 1] > 0.5
answer_indices = np.where(answer_mask)[0]
print(f"  Answer posts for retrieval: {len(answer_indices)}")
print("  Encoding answer posts...")
answer_texts = [texts[i][:512] for i in answer_indices]
answer_embeds = encode_texts(answer_texts)
answer_courses = [courses[i] if i < len(courses) else '' for i in answer_indices]

# Build user profiles
print("  Building user profiles...")
user_activity = {}
for i in range(n_posts):
    uid = forum_uids[i] if i < len(forum_uids) else None
    if uid is None:
        continue
    if uid not in user_activity:
        user_activity[uid] = {'posts': [], 'courses': set(), 'n_answers': 0, 'conf_scores': []}
    user_activity[uid]['posts'].append(i)
    if i < len(courses):
        user_activity[uid]['courses'].add(courses[i])
    if post_meta[i, 1] > 0.5:
        user_activity[uid]['n_answers'] += 1
    user_activity[uid]['conf_scores'].append(confusion_scores[i] if i < len(confusion_scores) else 4.0)

for uid in user_activity:
    user_activity[uid]['mean_conf'] = np.mean(user_activity[uid]['conf_scores'])

#  STRATEGY 1: Peer-Answer Retrieval

def strategy_peer_answer(post_idx, top_k=3):
    """Find existing answer posts most semantically similar to the confused post."""
    query = texts[post_idx][:512]
    query_emb = encode_texts([query])
    query_course = courses[post_idx] if post_idx < len(courses) else None

    # Prefer same course, fall back to all
    if query_course:
        course_mask = np.array([c == query_course for c in answer_courses])
        if course_mask.sum() < 5:
            course_mask = np.ones(len(answer_courses), dtype=bool)
    else:
        course_mask = np.ones(len(answer_courses), dtype=bool)

    valid_emb = answer_embeds[course_mask]
    valid_idx = answer_indices[course_mask]

    if len(valid_emb) == 0:
        return {'found': False, 'answers': []}

    sims = cosine_similarity(query_emb, valid_emb)[0]
    top = np.argsort(sims)[-top_k:][::-1]

    answers = []
    for t in top:
        gi = valid_idx[t]
        answers.append({
            'post_idx': int(gi),
            'similarity': float(sims[t]),
            'text': texts[gi][:400],
            'course': courses[gi] if gi < len(courses) else '',
        })
    return {'found': True, 'answers': answers, 'best_sim': float(sims[top[0]])}

#  STRATEGY 2: Peer Connector

def strategy_peer_connector(post_idx, top_k=3):
    """Find helpful, low-confusion users in the same course."""
    query_course = courses[post_idx] if post_idx < len(courses) else None
    query_uid = forum_uids[post_idx] if post_idx < len(forum_uids) else None

    candidates = []
    for uid, info in user_activity.items():
        if uid == query_uid:
            continue
        if query_course and query_course not in info['courses']:
            continue
        if info['mean_conf'] >= 4.5:
            continue
        if len(info['posts']) < 3:
            continue

        helpfulness = (info['n_answers'] * 3 +
                       len(info['posts']) * 0.5 +
                       (4.5 - info['mean_conf']) * 2)
        candidates.append({
            'user_id': str(uid)[:12],
            'helpfulness': float(helpfulness),
            'n_posts': len(info['posts']),
            'n_answers': info['n_answers'],
            'mean_conf': float(info['mean_conf']),
        })

    candidates.sort(key=lambda x: -x['helpfulness'])
    found = len(candidates) > 0
    return {'found': found, 'peers': candidates[:top_k],
            'n_available': len(candidates)}

#  STRATEGY 3: Thread Summarizer

def strategy_thread_summary(post_idx, max_posts=15):
    """Extract and summarize thread structure around a confused post."""
    # Collect thread neighbors
    neighbors = set()
    neighbors.add(post_idx)

    # Direct connections
    for n in reply_adj.get(post_idx, []):
        neighbors.add(n)
    for n in thread_adj.get(post_idx, []):
        neighbors.add(n)

    # Root of thread
    if post_idx in thread_root:
        root = thread_root[post_idx]
        neighbors.add(root)
        # Other replies to same root
        for n in thread_adj.get(root, []):
            neighbors.add(n)

    neighbors = sorted(n for n in neighbors if n < n_posts)[:max_posts]

    if len(neighbors) <= 1:
        return {'found': False, 'summary': 'No thread context available'}

    # Analyze thread structure
    posts_info = []
    n_questions = 0
    n_answers = 0
    n_confused = 0
    conf_values = []

    for n in neighbors:
        is_root = (post_types[n] == 'CommentThread') if n < len(post_types) else False
        is_q = post_meta[n, 0] > 0.5
        is_a = post_meta[n, 1] > 0.5
        conf = confusion_scores[n] if n < len(confusion_scores) else 4.0

        if is_q: n_questions += 1
        if is_a: n_answers += 1
        if conf >= 4.5: n_confused += 1
        conf_values.append(conf)

        posts_info.append({
            'idx': int(n), 'is_root': bool(is_root),
            'is_question': bool(is_q), 'is_answer': bool(is_a),
            'confusion': float(conf),
            'text_preview': texts[n][:150] if n < len(texts) else '',
        })

    # Find root text
    root_post = next((p for p in posts_info if p['is_root']), posts_info[0])

    summary = {
        'found': True,
        'n_posts': len(posts_info),
        'n_questions': n_questions,
        'n_answers': n_answers,
        'n_confused': n_confused,
        'has_resolution': n_answers > 0,
        'is_unresolved': n_questions > n_answers,
        'mean_confusion': float(np.mean(conf_values)),
        'max_confusion': float(np.max(conf_values)),
        'root_text': root_post['text_preview'],
        'thread_narrative': _build_narrative(posts_info),
    }
    return summary

def _build_narrative(posts_info):
    """Build a human-readable thread narrative."""
    parts = []
    root = next((p for p in posts_info if p['is_root']), None)
    if root:
        parts.append(f"Thread started with: \"{root['text_preview'][:100]}...\"")

    questions = [p for p in posts_info if p['is_question'] and not p.get('is_root')]
    answers = [p for p in posts_info if p['is_answer']]
    confused = [p for p in posts_info if p['confusion'] >= 4.5]

    if questions:
        parts.append(f"{len(questions)} follow-up questions were asked.")
    if answers:
        parts.append(f"{len(answers)} answer-type responses were provided.")
    if confused:
        parts.append(f"{len(confused)} posts show elevated confusion (score >= 4.5).")

    if len(questions) > len(answers):
        parts.append("KEY ISSUE: More questions than answers — unresolved confusion.")
    elif len(answers) > 0 and len(confused) > len(answers):
        parts.append("KEY ISSUE: Answers exist but confusion persists — explanations may be unclear.")

    return " ".join(parts)

#  STRATEGY 4: Resource Recommender

COURSE_RESOURCES = {
    'How_to_Learn_Math': {
        'modules': ['Growth Mindset Foundations', 'Mathematical Reasoning Strategies', 'Collaborative Problem Solving'],
        'actions': ['Review the number sense activities from Week 2', 'Try the visual math exercises', 'Join the peer study group'],
    },
    'Statistics_in_Medicine': {
        'modules': ['Statistical Foundations Review', 'Hypothesis Testing Walkthrough', 'Clinical Trial Design Basics'],
        'actions': ['Work through the practice dataset', 'Review the formula reference sheet', 'Watch the supplementary stats video'],
    },
    'StatLearning': {
        'modules': ['Regression Fundamentals', 'Cross-Validation Concepts', 'Regularization Methods'],
        'actions': ['Run the R lab exercises', 'Review the mathematical derivations', 'Check the textbook chapters 3-5'],
    },
    'SciWrite': {
        'modules': ['Scientific Writing Structure', 'Citation and Evidence', 'Revision Strategies'],
        'actions': ['Use the paper outline template', 'Review the before/after writing examples', 'Submit a draft for peer review'],
    },
}

ARCHETYPE_ACTIONS = {
    'Active Question': [
        'Check if similar questions were answered in earlier forum threads',
        'Break your question into smaller sub-questions for clearer responses',
        'Tag your post with the specific lecture/module number for better visibility',
    ],
    'Unanswered Question': [
        'Repost your question in the general discussion with more context',
        'Try phrasing your question differently — include what you DO understand',
        'Flag the thread for instructor attention',
    ],
    'Silent Struggler': [
        'You are not alone — many students find this challenging',
        'Try posting a question in the forum — the community is here to help',
        'Schedule time for the optional review materials before moving forward',
    ],
    'Implicit Confusion': [
        'Re-read the relevant lecture notes focusing on the core concept',
        'Try explaining the concept in your own words to identify the gap',
        'Discuss with a peer to clarify your understanding',
    ],
    'Confusion Contagion': [
        'Focus on the original question before reading all replies',
        'Look for instructor-endorsed answers in the thread',
        'If the thread is too long, start a new focused thread with your specific question',
    ],
    'Contextual Confusion': [
        'Review prerequisite materials for this module',
        'Watch the lecture video again at 0.75x speed',
        'Write down your specific confusion point and post it as a new question',
    ],
    'Isolated Learner': [
        'Join the weekly study group session',
        'Introduce yourself in the general discussion thread',
        'Find a study partner through the course peer-matching tool',
    ],
}

def strategy_resource_recommend(post_idx, archetype):
    """Recommend course-specific + archetype-specific resources."""
    course = courses[post_idx] if post_idx < len(courses) else ''
    course_short = course.split('/')[-1] if isinstance(course, str) else ''

    course_recs = COURSE_RESOURCES.get(course_short, {
        'modules': ['Review the current module materials'],
        'actions': ['Re-watch the relevant lecture video'],
    })

    arch_actions = ARCHETYPE_ACTIONS.get(archetype, ARCHETYPE_ACTIONS.get('Contextual Confusion'))

    return {
        'course': course_short,
        'course_modules': course_recs['modules'],
        'course_actions': course_recs['actions'],
        'archetype_actions': arch_actions,
    }

#  STRATEGY 5: XAI-Conditioned Clarification

CLARIFICATION_TEMPLATES = {
    'Active Question': (
        "Your question about \"{topic}\" is a great one and shows active engagement with the material. "
        "{peer_line}"
        "Try breaking the concept into smaller pieces — start with what you DO understand, "
        "then identify the specific gap. The teaching team is here to help."
    ),
    'Unanswered Question': (
        "I see your question about \"{topic}\" hasn't received a response yet. "
        "This is a common confusion point in this module. "
        "{peer_line}"
        "Consider rephrasing your question with more context about what you've tried so far — "
        "this helps peers and instructors give more targeted help."
    ),
    'Silent Struggler': (
        "Your post about \"{topic}\" suggests you might be working through some challenging concepts. "
        "That's completely normal! Many students find this section difficult. "
        "I'd encourage you to post specific questions — the forum community is very supportive. "
        "{peer_line}"
    ),
    'Confusion Contagion': (
        "This thread has generated a lot of discussion with {n_posts} posts, and confusion seems "
        "to have built up through the conversation. {narrative} "
        "My suggestion: focus on the original question and look for instructor-endorsed answers. "
        "If still unclear, start a fresh thread with your specific point of confusion."
    ),
    'Isolated Learner': (
        "Welcome! Your thoughts on \"{topic}\" are valuable. "
        "I notice you haven't interacted much with other learners yet — "
        "connecting with peers can really help clarify concepts. "
        "{peer_line}"
        "Consider joining the course study group or responding to other students' questions."
    ),
}

def strategy_clarification(post_idx, archetype, peer_answer_result=None, thread_result=None, peer_connector_result=None):
    """Generate XAI-conditioned clarification response."""
    topic = texts[post_idx][:60].strip().rstrip('.')
    template = CLARIFICATION_TEMPLATES.get(archetype, CLARIFICATION_TEMPLATES.get('Active Question'))

    # Build peer line
    peer_line = ""
    if peer_answer_result and peer_answer_result.get('found') and peer_answer_result['answers']:
        best = peer_answer_result['answers'][0]
        if best['similarity'] > 0.25:
            peer_line = f"A peer previously wrote: \"{best['text'][:120]}...\" which may help. "

    # Build narrative from thread summary
    narrative = ""
    n_thread = 0
    if thread_result and thread_result.get('found'):
        narrative = thread_result.get('thread_narrative', '')
        n_thread = thread_result.get('n_posts', 0)

    response = template.format(
        topic=topic,
        peer_line=peer_line,
        narrative=narrative,
        n_posts=n_thread,
    )
    return response

#  INTERVENTION ROUTER

def route_intervention(post_idx, archetype):
    """Route confused post to the best strategy based on archetype."""
    # Always run all strategies to have them available
    s1 = strategy_peer_answer(post_idx)
    s2 = strategy_peer_connector(post_idx)
    s3 = strategy_thread_summary(post_idx)
    s4 = strategy_resource_recommend(post_idx, archetype)
    s5 = strategy_clarification(post_idx, archetype, s1, s3, s2)

    # Routing decision
    if 'Question' in archetype or 'Unanswered' in archetype:
        if s1['found'] and s1['best_sim'] > 0.3:
            primary = 'Peer-Answer Retrieval'
            response = f"A relevant answer exists in your course forum:\n\n\"{s1['answers'][0]['text'][:250]}...\"\n\n(Similarity: {s1['best_sim']:.0%})"
        else:
            primary = 'XAI Clarification'
            response = s5
    elif 'Silent' in archetype or 'Isolated' in archetype:
        if s2['found'] and s2['peers']:
            primary = 'Peer Connector'
            p = s2['peers'][0]
            response = (f"You're not alone in finding this challenging. I've identified "
                       f"{s2['n_available']} active, helpful learners in your course. "
                       f"The most active peer has contributed {p['n_answers']} answers "
                       f"and has a low confusion score ({p['mean_conf']:.1f}/7). "
                       f"Consider reaching out or checking their recent posts for insights.")
        else:
            primary = 'XAI Clarification'
            response = s5
    elif 'Contagion' in archetype:
        if s3['found'] and s3['n_posts'] >= 3:
            primary = 'Thread Summarizer'
            response = (f"Thread Summary ({s3['n_posts']} posts):\n"
                       f"Root: \"{s3['root_text'][:150]}...\"\n"
                       f"Status: {s3['n_questions']} questions, {s3['n_answers']} answers, "
                       f"{'UNRESOLVED' if s3['is_unresolved'] else 'partially resolved'}.\n"
                       f"Mean confusion: {s3['mean_confusion']:.1f}/7.\n\n"
                       f"{s3['thread_narrative']}")
        else:
            primary = 'XAI Clarification'
            response = s5
    else:
        primary = 'XAI Clarification'
        response = s5

    return {
        'primary_strategy': primary,
        'primary_response': response,
        'resources': s4,
        'all_strategies': {
            'peer_answer': {'available': s1['found'],
                           'best_sim': s1.get('best_sim', 0),
                           'n_retrieved': len(s1.get('answers', []))},
            'peer_connector': {'available': s2['found'],
                              'n_peers': s2.get('n_available', 0)},
            'thread_summary': {'available': s3['found'],
                              'n_posts': s3.get('n_posts', 0),
                              'unresolved': s3.get('is_unresolved', False)},
            'resources': s4,
            'clarification': s5[:200],
        },
    }

#  GENERATE ALL INTERVENTIONS
print("  Generating Multi-Strategy Interventions")
print("~" * 70)

interventions = []
n_process = min(len(confused_indices), len(cluster_labels))
print(f"  Processing {n_process} confused posts...")

t0 = time.time()
for i in range(n_process):
    post_idx = confused_indices[i]
    archetype = archetype_names[cluster_labels[i]]

    result = route_intervention(post_idx, archetype)

    interventions.append({
        'post_idx': int(post_idx),
        'archetype': archetype,
        'confusion_score': float(confusion_scores[post_idx]) if post_idx < len(confusion_scores) else 0,
        'post_text': texts[post_idx][:300] if post_idx < len(texts) else '',
        'primary_strategy': result['primary_strategy'],
        'primary_response': result['primary_response'],
        'resources': result['resources'],
        'all_strategies': result['all_strategies'],
    })

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / (i+1)) * (n_process - i - 1) / 60
        print(f"    {i+1}/{n_process} | Elapsed: {elapsed:.0f}s | ETA: {eta:.1f}min")

print(f"  Done: {len(interventions)} interventions in {(time.time()-t0)/60:.1f} min")

#  EVALUATION
print("  Evaluating Interventions")
print("~" * 70)

print("  Encoding responses for evaluation...")
responses = [iv['primary_response'][:512] for iv in interventions]
post_texts_eval = [iv['post_text'][:512] for iv in interventions]

resp_embeds = encode_texts(responses)
post_embeds = encode_texts(post_texts_eval)

# Relevance
relevance = np.array([cosine_similarity(post_embeds[i:i+1], resp_embeds[i:i+1])[0, 0]
                       for i in range(len(interventions))])

# Specificity (vs generic response)
generic = "Thank you for your question. Please review the course materials and feel free to ask for help."
generic_emb = encode_texts([generic])
specificity = np.array([1 - cosine_similarity(resp_embeds[i:i+1], generic_emb)[0, 0]
                         for i in range(len(interventions))])

# Word count
word_counts = np.array([len(iv['primary_response'].split()) for iv in interventions])

for iv, rel, spec in zip(interventions, relevance, specificity):
    iv['relevance'] = float(rel)
    iv['specificity'] = float(spec)
    iv['word_count'] = len(iv['primary_response'].split())

# Strategy distribution
strat_counts = Counter(iv['primary_strategy'] for iv in interventions)
arch_counts = Counter(iv['archetype'] for iv in interventions)

print(f"\n  Overall Metrics:")
print(f"    Relevance:   {relevance.mean():.4f} +/- {relevance.std():.4f}")
print(f"    Specificity: {specificity.mean():.4f} +/- {specificity.std():.4f}")
print(f"    Word Count:  {word_counts.mean():.0f} +/- {word_counts.std():.0f}")

print(f"\n  Strategy Distribution:")
for s, c in strat_counts.most_common():
    mask = [iv['primary_strategy'] == s for iv in interventions]
    rel_m = relevance[mask].mean()
    spec_m = specificity[mask].mean()
    print(f"    {s:<25} n={c:4d}  rel={rel_m:.4f}  spec={spec_m:.4f}")

print(f"\n  Archetype Distribution:")
for a, c in arch_counts.most_common():
    mask = [iv['archetype'] == a for iv in interventions]
    rel_m = relevance[mask].mean()
    spec_m = specificity[mask].mean()
    print(f"    {a:<25} n={c:4d}  rel={rel_m:.4f}  spec={spec_m:.4f}")

del enc_model; torch.cuda.empty_cache()

#  VISUALIZATIONS
print("  Generating Figures")
print("~" * 70)

strat_colors = {
    'Peer-Answer Retrieval': '#e74c3c',
    'Peer Connector': '#3498db',
    'Thread Summarizer': '#f39c12',
    'XAI Clarification': '#2ecc71',
    'Resource Recommender': '#9b59b6',
}
arch_colors = {
    'Active Question': '#e74c3c', 'Unanswered Question': '#c0392b',
    'Silent Struggler': '#3498db', 'Implicit Confusion': '#2980b9',
    'Confusion Contagion': '#f39c12', 'Isolated Learner': '#5dade2',
    'Contextual Confusion': '#2ecc71', 'Latent Confusion': '#27ae60',
    'Discussion Overload': '#e67e22',
}

# ── Figure 1: Strategy Routing Analysis ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# (a) Strategy distribution
strats = list(strat_counts.keys())
scnts = [strat_counts[s] for s in strats]
scols = [strat_colors.get(s, '#95a5a6') for s in strats]
axes[0].bar(range(len(strats)), scnts, color=scols, edgecolor='white', width=0.6)
axes[0].set_xticks(range(len(strats)))
axes[0].set_xticklabels([s.replace(' ', '\n') for s in strats], fontsize=8)
axes[0].set_ylabel('Count')
axes[0].set_title('(a) Primary Strategy Distribution', fontweight='bold')
for i, v in enumerate(scnts):
    axes[0].text(i, v + 2, str(v), ha='center', fontsize=10, fontweight='bold')

# (b) Archetype -> Strategy routing heatmap
arch_unique = sorted(set(iv['archetype'] for iv in interventions))
strat_unique = sorted(set(iv['primary_strategy'] for iv in interventions))
routing = np.zeros((len(arch_unique), len(strat_unique)))
for iv in interventions:
    ai = arch_unique.index(iv['archetype'])
    si = strat_unique.index(iv['primary_strategy'])
    routing[ai, si] += 1
# Normalize rows
row_sums = routing.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
routing_pct = routing / row_sums * 100

sns.heatmap(routing_pct, ax=axes[1], cmap='YlOrRd', annot=True, fmt='.0f',
            xticklabels=[s[:15] for s in strat_unique],
            yticklabels=[a[:18] for a in arch_unique],
            cbar_kws={'label': '% of archetype'})
axes[1].set_title('(b) Archetype -> Strategy Routing (%)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

# (c) Relevance by strategy
bp_data = []
bp_labels = []
bp_colors_list = []
for s in strat_unique:
    vals = [iv['relevance'] for iv in interventions if iv['primary_strategy'] == s]
    bp_data.append(vals)
    bp_labels.append(s[:15])
    bp_colors_list.append(strat_colors.get(s, '#95a5a6'))

bp = axes[2].boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.5)
for patch, c in zip(bp['boxes'], bp_colors_list):
    patch.set_facecolor(c)
axes[2].set_ylabel('Relevance Score')
axes[2].set_title('(c) Relevance by Strategy', fontweight='bold')
axes[2].tick_params(axis='x', rotation=15)

plt.suptitle('Multi-Strategy Intervention Routing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/interv_fig1_routing.png", dpi=200)
plt.savefig(f"{FIGURES_DIR}/interv_fig1_routing.pdf"); plt.close()
print(f"  -> Saved: interv_fig1_routing.png")

# ── Figure 2: Example Interventions ──
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
shown = {}
for iv in interventions:
    s = iv['primary_strategy']
    if s not in shown or iv['relevance'] > shown[s]['relevance']:
        shown[s] = iv

display_strats = list(shown.keys())[:4]
for idx, strat in enumerate(display_strats):
    ax = axes[idx // 2, idx % 2]
    ax.axis('off')
    iv = shown[strat]

    display = (
        f"STRATEGY: {strat}\n"
        f"ARCHETYPE: {iv['archetype']}\n"
        f"{'=' * 55}\n\n"
        f"CONFUSED POST (score={iv['confusion_score']:.1f}/7):\n"
        f"\"{iv['post_text'][:180]}...\"\n\n"
        f"INTERVENTION:\n"
        f"\"{iv['primary_response'][:280]}...\"\n\n"
        f"Relevance: {iv['relevance']:.3f}  |  "
        f"Specificity: {iv['specificity']:.3f}  |  "
        f"Words: {iv['word_count']}\n\n"
        f"RESOURCES:\n"
    )
    for action in iv['resources'].get('archetype_actions', [])[:2]:
        display += f"  - {action}\n"

    ax.text(0.03, 0.97, display, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=strat_colors.get(strat, '#f0f0f0'), alpha=0.1))
    ax.set_title(strat, fontsize=12, fontweight='bold',
                 color=strat_colors.get(strat, '#2c3e50'))

plt.suptitle('Best Example Intervention per Strategy', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/interv_fig2_examples.png", dpi=200)
plt.savefig(f"{FIGURES_DIR}/interv_fig2_examples.pdf"); plt.close()
print(f"  -> Saved: interv_fig2_examples.png")

# ── Figure 3: System Overview ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) Pipeline
ax = axes[0]; ax.axis('off')
pipeline = (
    "ConFusionGraph v2: Full Pipeline\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "1. DETECTION\n"
    "   Stacked Ensemble\n"
    "   (XGBoost + BERT + GNN)\n"
    "   → Confusion Prediction\n\n"
    "2. EXPLANATION (SHAP + LIME)\n"
    "   Global: Feature ranking\n"
    "   Local: Per-post explanation\n"
    "   Token: Word-level attribution\n"
    "   → Archetype Classification\n\n"
    "3. INTERVENTION (5 Strategies)\n"
    "   Archetype → Strategy Router:\n"
    "   ├─ Active Q     → Peer Answer\n"
    "   ├─ Silent        → Peer Connector\n"
    "   ├─ Contagion   → Thread Summary\n"
    "   ├─ All             → Resources\n"
    "   └─ Fallback     → XAI Clarification"
)
ax.text(0.03, 0.97, pipeline, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
ax.set_title('(a) System Architecture', fontweight='bold')

# (b) Overall quality
met_names = ['Relevance', 'Specificity']
met_vals = [relevance.mean(), specificity.mean()]
met_stds = [relevance.std(), specificity.std()]
bars = axes[1].bar(range(2), met_vals, yerr=met_stds,
                   color=['#3498db', '#e67e22'], edgecolor='white', capsize=5, width=0.4)
axes[1].set_xticks(range(2)); axes[1].set_xticklabels(met_names, fontsize=11)
axes[1].set_ylabel('Score'); axes[1].set_ylim(0, 1)
axes[1].set_title('(b) Intervention Quality', fontweight='bold')
for i, (m, s) in enumerate(zip(met_vals, met_stds)):
    axes[1].text(i, m + s + 0.02, f'{m:.3f}', ha='center', fontsize=11, fontweight='bold')

# (c) Archetype pie
a_names = list(arch_counts.keys())
a_vals = [arch_counts[n] for n in a_names]
a_cols = [arch_colors.get(n, '#95a5a6') for n in a_names]
axes[2].pie(a_vals, labels=[f'{n}\n({v})' for n, v in zip(a_names, a_vals)],
            colors=a_cols, autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 8})
axes[2].set_title('(c) Confusion Archetypes', fontweight='bold')

plt.suptitle('XAI-Driven Multi-Strategy Intervention System', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/interv_fig3_overview.png", dpi=200)
plt.savefig(f"{FIGURES_DIR}/interv_fig3_overview.pdf"); plt.close()
print(f"  -> Saved: interv_fig3_overview.png")

# ── Figure 4: Strategy availability analysis ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# How often each strategy was AVAILABLE vs CHOSEN
avail_counts = {
    'Peer-Answer\n(sim>0.3)': sum(1 for iv in interventions
                                   if iv['all_strategies']['peer_answer']['best_sim'] > 0.3),
    'Peer Connector\n(peers found)': sum(1 for iv in interventions
                                          if iv['all_strategies']['peer_connector']['available']),
    'Thread Summary\n(3+ posts)': sum(1 for iv in interventions
                                       if iv['all_strategies']['thread_summary']['n_posts'] >= 3),
    'Resources': len(interventions),
    'Clarification': len(interventions),
}

chosen_counts = {
    'Peer-Answer\n(sim>0.3)': strat_counts.get('Peer-Answer Retrieval', 0),
    'Peer Connector\n(peers found)': strat_counts.get('Peer Connector', 0),
    'Thread Summary\n(3+ posts)': strat_counts.get('Thread Summarizer', 0),
    'Resources': 0,
    'Clarification': strat_counts.get('XAI Clarification', 0),
}

x = range(len(avail_counts))
w = 0.35
axes[0].bar([i - w/2 for i in x], list(avail_counts.values()), w,
            label='Available', color='#3498db', edgecolor='white', alpha=0.7)
axes[0].bar([i + w/2 for i in x], list(chosen_counts.values()), w,
            label='Chosen as Primary', color='#e74c3c', edgecolor='white', alpha=0.7)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(list(avail_counts.keys()), fontsize=8)
axes[0].set_ylabel('Count')
axes[0].set_title('(a) Strategy Availability vs Selection', fontweight='bold')
axes[0].legend(fontsize=9)

# Relevance vs confusion score
conf_arr = np.array([iv['confusion_score'] for iv in interventions])
for strat in strat_unique:
    mask = np.array([iv['primary_strategy'] == strat for iv in interventions])
    if mask.any():
        axes[1].scatter(conf_arr[mask], relevance[mask], alpha=0.4, s=20,
                       color=strat_colors.get(strat, '#999'), label=strat[:15])
axes[1].set_xlabel('Original Confusion Score')
axes[1].set_ylabel('Intervention Relevance')
axes[1].set_title('(b) Confusion Score vs Relevance', fontweight='bold')
axes[1].legend(fontsize=7, loc='upper left')

plt.suptitle('Strategy Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/interv_fig4_analysis.png", dpi=200)
plt.savefig(f"{FIGURES_DIR}/interv_fig4_analysis.pdf"); plt.close()
print(f"  -> Saved: interv_fig4_analysis.png")

#  SAVE RESULTS
print("  Saving Results")
print("~" * 70)

output = {
    'n_interventions': len(interventions),
    'strategy_distribution': dict(strat_counts),
    'archetype_distribution': dict(arch_counts),
    'overall_metrics': {
        'relevance_mean': float(relevance.mean()),
        'relevance_std': float(relevance.std()),
        'specificity_mean': float(specificity.mean()),
        'specificity_std': float(specificity.std()),
        'word_count_mean': float(word_counts.mean()),
    },
    'per_strategy': {
        s: {
            'count': int(strat_counts.get(s, 0)),
            'relevance': float(np.mean([iv['relevance'] for iv in interventions if iv['primary_strategy'] == s])) if strat_counts.get(s, 0) > 0 else 0,
            'specificity': float(np.mean([iv['specificity'] for iv in interventions if iv['primary_strategy'] == s])) if strat_counts.get(s, 0) > 0 else 0,
        } for s in strat_unique
    },
    'sample_interventions': [{
        'post_idx': iv['post_idx'], 'archetype': iv['archetype'],
        'strategy': iv['primary_strategy'],
        'post_text': iv['post_text'][:200],
        'response': iv['primary_response'][:300],
        'relevance': iv['relevance'], 'specificity': iv['specificity'],
    } for iv in interventions[:30]],
}

with open(f"{RESULTS_DIR}/interventions_v2_results.json", 'w') as f:
    json.dump(output, f, indent=2, default=float)
print(f"  -> Saved: {RESULTS_DIR}/interventions_v2_results.json")

# Save LLM-ready prompts for upgrade
llm_prompts = []
for iv in interventions[:50]:
    llm_prompts.append({
        'archetype': iv['archetype'],
        'confusion_score': iv['confusion_score'],
        'post_text': iv['post_text'],
        'thread_context': iv['all_strategies'].get('thread_summary', {}),
        'available_peer_answer': iv['all_strategies']['peer_answer'].get('best_sim', 0) > 0.3,
        'prompt': f"You are a MOOC teaching assistant. A student (archetype: {iv['archetype']}) "
                  f"posted: \"{iv['post_text'][:300]}\"\n\n"
                  f"Generate a helpful, personalized response addressing their specific confusion. "
                  f"Keep it under 150 words, warm and encouraging.",
    })

with open(f"{RESULTS_DIR}/interventions_llm_prompts.json", 'w') as f:
    json.dump(llm_prompts, f, indent=2, default=float)
print(f"  -> Saved: {RESULTS_DIR}/interventions_llm_prompts.json (50 prompts for LLM upgrade)")

print("  INTERVENTION SYSTEM COMPLETE")
print("=" * 70)
print(f"""
  Generated: {len(interventions)} interventions

  Strategy Distribution:
""")
for s, c in strat_counts.most_common():
    print(f"    {s:<25} {c:4d} ({c/len(interventions)*100:.1f}%)")
print(f"""
  Quality:
    Relevance:   {relevance.mean():.3f} +/- {relevance.std():.3f}
    Specificity: {specificity.mean():.3f} +/- {specificity.std():.3f}

  Figures:
    interv_fig1 -> Strategy routing (distribution + heatmap + relevance)
    interv_fig2 -> Best example per strategy
    interv_fig3 -> System architecture overview
    interv_fig4 -> Availability vs selection + confusion vs relevance

  For LLM-quality upgrade:
    Use {RESULTS_DIR}/interventions_llm_prompts.json with Claude/GPT-4

  All experiments complete! Ready for paper writing.
""")