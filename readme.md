# FAISS-based Citation Retrieval System (FLCR)

🚀 A production-style semantic retrieval system for citation recommendation.

- 📄 Large-scale retrieval over scientific documents  
- ⚡ FAISS-based nearest neighbor search for low-latency serving  
- 🧠 Dual-encoder embedding model with hard-negative training  
- 🌐 FastAPI demo for interactive querying  

---

## 🔥 Key Highlights

- Replaces brute-force cosine reranking with ANN search (FAISS)
- End-to-end pipeline: training → indexing → serving
- Supports local-context citation recommendation (more precise than document-level)
- Group-aware retrieval for diversified recommendation results
- Modular system design with clear separation between training, indexing, and serving

---

## 📊 Scale & Performance

- Designed for million-scale passage retrieval
- Efficient ANN search using FAISS indexing
- Low-latency query serving via prebuilt index
- Significantly faster than naive pairwise similarity search

---

## 🧠 System Overview

FLCR maps citation context and candidate papers into a shared embedding space, and performs nearest-neighbor retrieval using FAISS.

### Pipeline

1. Build citation-context dataset from parsed papers
2. Train dual-encoder retrieval model
3. Encode candidate papers into dense vectors
4. Build FAISS index for fast ANN search
5. Serve results via FastAPI or run offline evaluation

---

## 🏗️ Architecture

- Model: Dual-encoder (sentence-transformers / PyTorch)
- Retrieval backend: FAISS (ANN search)
- Serving layer: FastAPI
- Data storage: SQLite + local artifacts

---

## 🚀 Demo

Start the API server:

```bash
uvicorn apps.demos.demo1.app:app --reload
```

Then open:

http://127.0.0.1:8000

Available routes:

- /demo — input form  
- /demo/outcome?text=... — standard retrieval  
- /demo/outcomeCluster?text=... — group-aware retrieval  

---

## ⚙️ Quick Start

Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run core pipeline:

```bash
python3 -m flcr.train
python3 -m flcr.index
python3 -m flcr.evaluate
python3 -m flcr.qualitative
```

---

## 📦 Repository Structure

```
flcr/
  ├── model.py
  ├── train.py
  ├── index.py
  ├── evaluate.py
  ├── qualitative.py
  ├── search.py
  ├── data_processing/

apps/
  └── demos/

experiments/
  └── legacy/

artifacts/
```

---

## 🧪 Typical Workflow

1. Prepare dataset and embedding assets
2. Train retrieval model
3. Build FAISS index
4. Run evaluation
5. Launch demo

---

## 🧩 Dependencies

- PyTorch
- sentence-transformers
- FAISS
- FastAPI
- Uvicorn
- pandas / scikit-learn

---

## 💡 Engineering Focus

- Retrieval-oriented ML system design
- Large-scale embedding + ANN search
- Efficient inference pipeline construction
- Practical ML system deployment with FastAPI

---

## 📌 Notes

- Legacy versions under experiments/legacy/
- Main pipeline under flcr/
- Retrieval backend migrated from KDTree → FAISS
- Demo stack upgraded from Flask → FastAPI
