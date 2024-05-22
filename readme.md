# FLCR

Fast Local Citation Recommendation for multi-topic scientific writing.

FLCR is an embedding-based retrieval system for citation recommendation. It maps local citation context and candidate cited papers into a shared vector space, then serves recommendations with nearest-neighbor search. The later system versions also support group-aware retrieval so the same input context can produce different recommendation sets.

## Highlights

- Local-context citation recommendation instead of full-document recommendation
- Dual-encoder style retrieval with hard-negative training
- Nearest-neighbor serving with `KDTree`
- Group-aware recommendation for diversified citation suggestions
- Lightweight Flask demos for interactive inspection

## System Pipeline

1. Build citation-context samples from parsed papers and citation links.
2. Encode text into dense sentence embeddings.
3. Train a retrieval model that aligns context embeddings with cited-paper embeddings.
4. Build a nearest-neighbor index over encoded cited papers.
5. Run offline evaluation or serve recommendations through the demo apps.

## Repository Structure

- `flcr/`: current main implementation
- `flcr/model.py`: model definition, dataset logic, and training loop
- `flcr/train.py`: latest training entrypoint
- `flcr/index.py`: cited-paper encoding and retrieval index build
- `flcr/evaluate.py`: offline evaluation for Recall, MRR, MAP, and nDCG
- `flcr/qualitative.py`: manual qualitative retrieval workflow
- `flcr/data_processing/`: data-processing utilities used by the latest pipeline
- `apps/demos/`: Flask demos and interactive app entry points
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

## Top-Level Layout

The top level is organized by responsibility:

- `flcr/`: core training, indexing, evaluation, and retrieval code
- `apps/`: interactive demos
- `scripts/`: auxiliary data-preparation and export utilities
- `artifacts/`: checkpoints and generated assets
- `experiments/`: archived earlier iterations

## Dependencies

Python dependencies are listed in `requirements.txt`.

The full pipeline expects several local assets that are not stored in this repository, including:

- processed SQLite datasets
- pickled MAG-style resources
- trained checkpoints
- local sentence-transformer weights

## Typical Entry Points

From the repository root:

```bash
python -m flcr.train
python -m flcr.index
python -m flcr.evaluate
```

## Engineering Focus

This project highlights:

- retrieval-oriented ML system design
- PyTorch training and evaluation workflows
- efficient candidate search with nearest-neighbor indexing
- practical handling of structured local datasets with SQLite
- iterative system improvement with a clean separation between current and legacy code
