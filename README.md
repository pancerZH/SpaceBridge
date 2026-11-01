# SpaceBridge

Zero-reindexing cross-model query alignment: query a frozen vector index built with one
embedding model using a different (newer, smaller, or multilingual) query encoder, by fitting
a query-side affine map on a few thousand calibration anchors.

See [spacebridge_design_and_benchmark.md](spacebridge_design_and_benchmark.md) for the design
and benchmark plan.

## Setup

```bash
uv sync            # Python 3.12, PyTorch (CUDA 12.8 wheels), benchmark + dev dependencies
uv run pytest      # synthetic-data tests, no downloads or GPU required
```

## Layout

- `src/spacebridge/`: alignment engine (anchor selection, solvers, refinement, `SpaceBridge`).
- `src/spacebridge_bench/`: benchmark harness (model registry, cached encoding, datasets, metrics, runner).
- `tests/`: unit tests on synthetic data.
- `scripts/`: studies, summary tables, and `steps.sh` for restartable experiment drivers.

Datasets and embeddings are downloaded at runtime and cached under `data/` (git-ignored);
no dataset text is committed to this repository.

## Long runs

Encoding, reranker scores and benchmark results are saved as they are produced, so a crash or power loss costs at most a few minutes of work: rerun the same command and it continues from what is on disk. `scripts/steps.sh` gives drivers restartable steps and a synced progress log.
