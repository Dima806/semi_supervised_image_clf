Run a fast 2-epoch end-to-end smoke test — safe for Codespaces (no GPU required).

```bash
make smoke-data   # download train split only (~120 MB on disk), skip if already done
make smoke        # SimCLR 2 epochs + FixMatch 2 epochs
```

`make smoke` does **not** require the unlabeled or test splits. Missing splits are replaced
with `_SyntheticDataset` (random PIL images), so the full training pipeline is exercised
without a 2.5 GB download.

## Disk budget on Codespaces (32 GB total, ~4 GB free after setup)

| Item | Size |
|---|---|
| `.venv` (CPU-only torch via uv) | ~1.5 GB |
| STL-10 train split (`make smoke-data`) | ~120 MB on disk |
| STL-10 test split (optional, `make data`) | ~190 MB on disk |
| STL-10 unlabeled (`make data-unlabeled`) | ~2.5 GB — **Kaggle only** |

## Why the tar.gz doesn't appear on disk

`dataset.py` streams `stl10_binary.tar.gz` over HTTP and extracts only the requested
binary files directly from the stream. The 2.6 GB archive is never written to disk.

## Why torchvision.datasets.STL10 is not used

`torchvision.datasets.STL10._check_integrity` requires **all five** binary files
(train_X, train_y, unlabeled_X, test_X, test_y) to be present even when only one
split is opened. The tiered download omits unused splits, so the integrity check
always fails. `_STL10Split` (in `dataset.py`) reads only the files it needs.

## Why the venv is small

`pyproject.toml` configures uv to install torch and torchvision from PyTorch's CPU-only
wheel index (`[[tool.uv.index]]`). This saves ~3.5 GB by excluding CUDA and triton
packages. Kaggle notebooks install via pip and already have CUDA torch pre-installed.
