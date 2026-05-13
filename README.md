# ConFusionGraph

**An Explainable Graph Enhanced Ensemble for Confusion Detection and Multiple Strategy Intervention in Massive Open Online Course Discussion Forums**

> Abdennour Redjaibia, Samia Drissi, Karima Boussaha, Sevinç Gülseçen, Yacine Lafifi  
> *Journal of Communications Software and Systems (JCOMSS), 2026*

---

## Overview

ConFusionGraph is an end-to-end six-stage framework for:
1. Detecting learner confusion in MOOC discussion forums
2. Explaining predictions via dual XAI analysis (KernelSHAP + LIME)
3. Routing confused posts to differentiated intervention strategies

**Key results on Stanford MOOCPost dataset (29,584 posts, 11 courses):**
- Accuracy: **89.2%**
- AUROC: **0.923**
- Macro-F1: **0.795**
- Intervention specificity: **0.742**

---

## Dataset

This work uses the publicly available **Stanford MOOCPost dataset**:
- 29,604 forum posts from 11 courses across Education, Medicine, and Humanities
- Annotated with confusion scores (1–7 Likert scale), Question/Answer/Opinion flags, Sentiment and Urgency scores
- Available at: https://snap.stanford.edu/mooc/

Place the downloaded file in the root directory as `stanfordMOOCForumPostsSet.xlsx` before running any script.

---

## Repository Structure

```
ConFusionGraph/
├── README.md
├── requirements.txt
│
├── preprocess.py            # Stage 1 — data cleaning, BERT embeddings, graph construction
├── baselines.py             # Stage 2 — non-graph baselines (TF-IDF+XGB, BERT+MLP, BERT fine-tuned)
├── gnn_models.py            # Stage 3 — GNN baselines (BERT+GCN/GAT/HGT) + ConFusionGraph V1
├── ensemble.py              # Stage 4 — ConFusionGraph V2 stacked ensemble (canonical run)
├── xai_analysis.py          # Stage 5 — KernelSHAP, LIME, token attribution, archetype discovery
├── intervention.py          # Stage 6 — multi-strategy XAI-driven intervention system
├── confusion_chains.py      # Stage 7 — confusion chain extraction and contagion analysis
│
├── sensitivity_thresh_4_0.ipynb   # Threshold sensitivity at τ = 4.0
├── sensitivity_thresh_5_0.ipynb   # Threshold sensitivity at τ = 5.0
└── silhouette_k2_k6.ipynb         # Silhouette analysis K=2 to K=6
```

---

## Running the Pipeline

Run scripts in order. Each script reads from the outputs of the previous stage.

```bash
python preprocess.py         # ~30 min on GPU — builds graph and BERT embeddings
python baselines.py          # non-graph baselines
python gnn_models.py         # GNN baselines + ConFusionGraph V1
python ensemble.py           # ConFusionGraph V2 — reproduces paper results
python xai_analysis.py       # SHAP + LIME explainability
python intervention.py       # intervention system evaluation
python confusion_chains.py   # chain and contagion analysis
```

All outputs are written to `no_urg/results/` and `no_urg/figures/`.

> **Reproducing paper numbers:** `ensemble.py` is the canonical script. Running it should produce Accuracy = 89.2%, AUROC = 0.923, Macro-F1 = 0.795, averaged over 5 random seeds. Minor variation (< 0.001) is expected due to GNN training stochasticity.

---

## Additional Experiments

These notebooks address specific reviewer requests and can be run independently after `preprocess.py`:

- **`sensitivity_thresh_4_0.ipynb`** — full V2 pipeline at confusion threshold τ = 4.0
- **`sensitivity_thresh_5_0.ipynb`** — full V2 pipeline at confusion threshold τ = 5.0
- **`silhouette_k2_k6.ipynb`** — silhouette scores for K=2 through K=6 on SHAP archetype profiles

Each reads from `no_urg/processed/` and writes to its own output directory without overwriting existing results.

---

## Environment

Python 3.10, CUDA 12.1 (experiments run on NVIDIA RTX 3060, 12 GB VRAM).

```bash
pip install -r requirements.txt
```

CPU execution is supported but significantly slower for `preprocess.py` (BERT encoding) and `gnn_models.py` (GNN training).

---

## Citation

```bibtex
@article{redjaibia2026confusiongraph,
  title   = {An Explainable Graph Enhanced Ensemble for Confusion Detection
             and Multiple Strategy Intervention in Massive Open Online Course
             Discussion Forums},
  author  = {Redjaibia, Abdennour and Drissi, Samia and Boussaha, Karima and
             G{\"u}lse{\c{c}}en, Sevin{\c{c}} and Lafifi, Yacine},
  journal = {Journal of Communications Software and Systems},
  year    = {2026}
}
```

---

## License

This code is released under the MIT License for research reproducibility.  
The Stanford MOOCPost dataset is subject to its own license — please refer to the original dataset source.
