# Vector Search with FAISS and OpenRouter

This repository demonstrates how to build and query semantic vector search engines using **FAISS** (Facebook AI Similarity Search) and embedding models via **OpenRouter**.

---

## 📌 What is FAISS?

**FAISS (Facebook AI Similarity Search)** is an open-source library developed by Meta (Facebook) for fast, dense vector similarity search and clustering. 

When text, images, or audio are converted into mathematical representations called **embeddings** (vectors of high-dimensional numbers), standard databases struggle to search through them efficiently. FAISS allows you to search through millions or billions of vector embeddings in milliseconds to find items that are semantically similar.

## ⚡ Quickstart & Setup Guide

### Prerequisites
* Python 3.9+
* An API key from [OpenRouter](https://openrouter.ai/)

### 1. Installation

Clone the repository and install the required dependencies:

```bash
pip install openrouter faiss-cpu numpy python-dotenv

how to run?
`uv run main-flatl2.py` to run FLATL2 implementation method
`uv run IVF.py` to run IVF method
`uv run HNSW.py` to run HNSW method

```
---

## 🔬 Comparison of the 3 Search Methods

All three files generate embeddings for text documents using OpenAI's `text-embedding-3-small` model (via OpenRouter) and search for the most relevant document using a query. However, each file uses a different **FAISS Indexing Strategy**.

| Feature | `FlatL2` (Exact Search) | `IVFFlat` (Clustered Search) | `HNSWFlat` (Graph Search) |
| :--- | :--- | :--- | :--- |
| **Search Mechanism** | Compares query against **every** single vector. | Groups vectors into **clusters**; searches only the nearest clusters. | Builds a multi-layer **graph network** to navigate directly to neighbors. |
| **Accuracy** | 100% (Exact match / Brute-force) | Approximate (High accuracy depending on `nprobe`) | Approximate (Very high accuracy) |
| **Search Speed** | Slow for large datasets | Fast | Extremely Fast |
| **Training Required?** | ❌ No | ✅ Yes (`index.train()`) | ❌ No |
| **Best Used For** | Small datasets (<10,000 vectors) | Very large datasets (Millions of vectors) | Large-scale production systems needing low latency |

---

### 1. `IndexFlatL2` (Brute-Force / Exact Match)
* **How it works:** Calculates the exact Euclidean distance ($L_2$ distance) between the query vector and every vector stored in the index.
* **Pros:** Guaranteed to find the absolute closest vectors.
* **Cons:** Becomes very slow as dataset size grows.

---

### 2. `IndexIVFFlat` (Inverted File Index / Clustering)
* **How it works:** Uses $K$-Means clustering to divide vector space into regions (`nlist`). At search time, it only searches within the `nprobe` closest clusters rather than the entire dataset.
* **Pros:** Drastically reduces search time for huge datasets.
* **Cons:** Requires a preliminary training step (`index.train()`) and might miss the absolute best match if it falls outside the searched clusters.

---

### 3. `IndexHNSWFlat` (Hierarchical Navigable Small World / Graphs)
* **How it works:** Constructs a multi-layer graph where vectors are nodes connected by edges (`M` edges per node). Searching behaves like traversing a highway system: high layers leap across large distances, while lower layers fine-tune local proximity.
* **Pros:** Incredible speed and scaling; ideal for real-time web services.
* **Cons:** Consumes more RAM to maintain graph connections.

---

## 🚀 Real-World Applications: Where & How FAISS is Used

FAISS is widely used in modern AI and software engineering, particularly for:

1. **Retrieval-Augmented Generation (RAG):** Fetching relevant documentation or knowledge base entries to provide context to Large Language Models (LLMs).
2. **Semantic Search Engines:** Searching for concepts rather than exact keyword matches (e.g., searching "pet near fire" brings up "A dog was resting near the fireplace").
3. **Recommendation Systems:** Suggesting products, articles, or videos based on user embedding similarity (e.g., Spotify, Netflix, E-commerce).
4. **Duplicate Detection:** Detecting near-identical images or plagiarized text in massive databases.

## Top companies using FAISS in production:
⚬	Meta (Facebook & Instagram): Created FAISS to run large-scale content recommendation feeds, image recognition, and duplicate content/spam detection across billions of user posts.
⚬	eBay: Uses FAISS to power visual image search, allowing shoppers to upload photos and instantly discover visually similar products across billions of active listings.
⚬	Shopify: Leverages FAISS to index product catalog embeddings, delivering instant personalized product recommendations for thousands of online merchants.
⚬	Pinterest: Uses FAISS in its visual search pipeline to match uploaded image pins against billions of catalog images in real time.
⚬	Instacart: Uses FAISS-backed vector search to understand user search intent and retrieve relevant grocery items even when queries don't match exact keywords.

---


