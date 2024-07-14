# FLCR

Fast Local Citation Recommendation for multi-topic scientific writing.

FLCR is an embedding-based retrieval system for citation recommendation. It maps local citation context and candidate cited papers into a shared vector space, then serves recommendations with nearest-neighbor search. The later system versions also support group-aware retrieval so the same input context can produce different recommendation sets.

## Highlights

- Local-context citation recommendation instead of full-document recommendation
- Dual-encoder style retrieval with hard-negative training
- Nearest-neighbor serving with `FAISS`
- Group-aware recommendation for diversified citation suggestions
- Lightweight FastAPI demos for interactive inspection

## System Pipeline

1. Build citation-context samples from parsed papers and citation links.
2. Encode text into dense sentence embeddings.
3. Train a retrieval model that aligns context embeddings with cited-paper embeddings.
4. Build a FAISS nearest-neighbor index over encoded cited papers.
5. Run offline evaluation or serve recommendations through the demo apps.

## Current Status

The actively maintained code lives under `flcr/`.

- Retrieval backend: `FAISS`
- Training/evaluation stack: `PyTorch`
- Demo stack: `FastAPI` + Jinja templates
- Data dependencies: local SQLite, pickle artifacts, local sentence-transformer weights

The repository is code-complete, but it is not fully self-contained. You need local model/data assets to actually train, index, evaluate, or run the demos.

## Quick Start

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Typical main entry points from the repository root:

```bash
python3 -m flcr.train
python3 -m flcr.index
python3 -m flcr.evaluate
python3 -m flcr.qualitative
```

Typical demo startup commands:

```bash
uvicorn apps.demos.demo1.app:app --reload
uvicorn apps.demos.demo2.app:app --reload
```

Open the app at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Required Local Assets

Several paths in the code point to local artifacts that are not committed to the repository. The most important ones are configured in [flcr/config.py](/Users/maxwellyin/Documents/GitHub/FLCR/flcr/config.py).

Expected assets include:

- grouped training/evaluation samples in SQLite form, for example `data/samplesGroups.db`
- MAG-style cited-paper resources in pickle form, for example `experiments/legacy/v3/check_point/smallMag2.pkl`
- trained checkpoints under `artifacts/checkpoints/`
- local sentence-transformer weights under `data/all-mpnet-base-v2`
- demo-specific checkpoints and FAISS index files under each demo's `resource/` directory

If these assets are missing, imports may still succeed but training, indexing, evaluation, and demo inference will fail at runtime.

## Repository Structure

- `flcr/`: current main implementation
- `flcr/model.py`: model definition, dataset logic, and training loop
- `flcr/train.py`: training entry point
- `flcr/index.py`: cited-paper encoding and FAISS index build
- `flcr/evaluate.py`: offline evaluation for Recall, MRR, MAP, and nDCG
- `flcr/qualitative.py`: manual qualitative retrieval workflow
- `flcr/search.py`: FAISS index helpers
- `flcr/data_processing/`: data-processing utilities used by the latest pipeline
- `apps/demos/`: interactive FastAPI demos
- `scripts/legacy/data_pipeline/`: older internal preprocessing utilities
- `scripts/public/data_export/`: public-data export helpers
- `artifacts/`: checkpoint notes and generated runtime artifacts
- `experiments/legacy/`: archived earlier model iterations and baselines

## Current vs. Legacy Code

The repository is now organized around a single primary implementation in `flcr/`.

Earlier iterations are preserved under `experiments/legacy/`:

- `v3`: baseline local citation retrieval
- `v3_1`: adds citing title and abstract features
- `v4`: adds group-aware conditioning
- `sql`: earlier SQL-oriented baseline code

This keeps the current system easy to navigate while still preserving the project history.

## Dependencies

Python dependencies are listed in `requirements.txt`.

Core runtime dependencies include:

- `torch`
- `sentence-transformers`
- `faiss-cpu`
- `fastapi`
- `uvicorn`
- `pandas`
- `scikit-learn`

The pinned `requirements.txt` reflects the project's original environment and includes a large number of notebook and experiment packages. If you later want to slim this down, do it as a separate cleanup task.

## Typical Workflow

1. Prepare or copy the local dataset and embedding assets into the expected paths.
2. Train the retrieval model with `python3 -m flcr.train`.
3. Build the cited-paper FAISS index with `python3 -m flcr.index`.
4. Run evaluation with `python3 -m flcr.evaluate`.
5. Run a qualitative check with `python3 -m flcr.qualitative`.
6. Start one of the demo apps with `uvicorn`.

## Demo Notes

- `apps/demos/demo1`: earlier recommendation demo
- `apps/demos/demo2`: grouped recommendation demo

Both demos now run on FastAPI and load local checkpoints plus FAISS index files from their own `resource/` directories.

The demo routes are:

- `/`: home page
- `/demo`: input form
- `/demo/outcome?text=...`: standard recommendation output
- `/demo/outcomeCluster?text=...`: clustered or group-aware output

## Migration Notes

Relative to older versions of this repository:

- `KDTree`-based retrieval has been replaced by `FAISS`
- Flask demo apps have been replaced by FastAPI apps
- the main implementation has been consolidated under `flcr/`
- older versions remain available under `experiments/legacy/`

## Engineering Focus

This project highlights:

- retrieval-oriented ML system design
- PyTorch training and evaluation workflows
- efficient candidate search with nearest-neighbor indexing
- practical handling of structured local datasets with SQLite
- iterative system improvement with a clean separation between current and legacy code

## Limitations

- The repository does not include the full training/evaluation datasets.
- Some paths still assume a local workstation-style directory layout and prebuilt artifacts.
- Demo resources are versioned separately from the main `flcr/` pipeline and may need manual refresh if model formats change.

## Related Docs

- App-specific notes: [apps/README.md](/Users/maxwellyin/Documents/GitHub/FLCR/apps/README.md)
- Legacy snapshots: [experiments/legacy/README.md](/Users/maxwellyin/Documents/GitHub/FLCR/experiments/legacy/README.md)
- Current configuration paths: [flcr/config.py](/Users/maxwellyin/Documents/GitHub/FLCR/flcr/config.py)
