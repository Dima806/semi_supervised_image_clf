Run a fast 2-epoch smoke test on a small synthetic subset — safe for Codespaces (no GPU, no large data download needed).

```bash
make smoke-data   # download train split only (~120 MB on disk), skip if already done
make smoke        # SimCLR 2 epochs + FixMatch 2 epochs on 1k images
```

## Disk budget on Codespaces (32 GB total, ~4 GB free after setup)

| Item | Size |
|---|---|
| `.venv` (CPU-only torch via uv) | ~1.5 GB |
| STL-10 train split (`make smoke-data`) | ~120 MB on disk |
| STL-10 test split (optional, `make data`) | ~190 MB on disk |
| STL-10 unlabeled (`make data-unlabeled`) | ~2.5 GB — **Kaggle only** |

`make smoke-data && make smoke` fits comfortably within the available space.
Do **not** run `make data-unlabeled` in Codespaces.

## Why the tar.gz doesn't appear on disk

`dataset.py` streams `stl10_binary.tar.gz` over HTTP and extracts only the requested
binary files directly from the stream. The 2.6 GB archive is never written to disk.

## Why the venv is small

`pyproject.toml` configures uv to install torch and torchvision from PyTorch's CPU-only
wheel index (`[[tool.uv.index]]`). This saves ~3.5 GB by excluding the CUDA and triton
packages. Kaggle notebooks install via pip and already have CUDA torch pre-installed,
so they are unaffected.
