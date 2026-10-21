# Unsupervised Learning in Python

A KGP Talkie course on finding structure in data with no labels: compressing it, spotting what does not fit, grouping it, and learning representations that other models build on.

| Section | Topic | Data |
|---|---|---|
| 1 | Unsupervised learning in the ML ecosystem (slides only) | |
| 2 | Dimensionality reduction: PCA and its variants, t-SNE, UMAP | MNIST digits |
| 3 | Anomaly detection: reconstruction error, Isolation Forest, LOF, ECOD | Credit card fraud |
| 4 | Clustering: k-means, hierarchical, DBSCAN, HDBSCAN, Gaussian mixtures, customer segments | MNIST, UCI Wholesale Customers |
| 5 | Autoencoders with Keras 3, and semi-supervised learning with a few labels | Credit card fraud |
| 6 | Time series clustering: shape-based distance, DTW, k-Shape | ECGFiveDays, ECG5000 |
| 7 | Recommender systems: item similarity, implicit ALS, EASE, SASRec | MovieLens 1M |
| 8 | Clustering text with embeddings, UMAP, HDBSCAN, BERTopic and a local LLM | Banking77 |

Each section has two files with the same name:

- `slides/NN_*.pdf` is the deck. All the explanation lives here.
- `notebooks/NN_*.ipynb` is the code. Every dataset downloads inside the notebook, with no account or login.
- Run the notebooks from the `notebooks/` folder.

## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

On Windows and Linux, PyTorch is installed with CUDA 13.0 support and uses an NVIDIA GPU when there is one. Everything also runs on a CPU; the SASRec model in Section 7 is the one that takes noticeably longer.

Keras runs on the PyTorch backend. The Section 5 notebook sets `KERAS_BACKEND=torch` in its first cells, so nothing else is needed.

## Ollama, for Section 8

Install from [ollama.com](https://ollama.com), then:

```bash
ollama pull qwen3.5:4b
```

## Downloads

| Section | Download |
|---|---|
| 2, 4 | MNIST from OpenML, cached after the first run |
| 3, 5 | Credit card transactions, about 144 MB, on every run |
| 4 | Wholesale customers, one small file from the UCI repository |
| 6 | ECG datasets through aeon, cached after the first run |
| 7 | MovieLens 1M, about 6 MB |
| 8 | Banking77 and a small embedding model from Hugging Face |
