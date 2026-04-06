Download STL-10 data. Choose the tier that fits your disk and use case.

The archive is streamed from the network — the 2.6 GB tar.gz is **never written to disk**.
Only the requested binary files are extracted, so disk usage matches the sizes below.

| Target | Disk | Use |
|---|---|---|
| `make smoke-data` | ~120 MB | Train split only — Codespaces-safe |
| `make data` | ~310 MB | Train + test — supervised baseline and FixMatch eval |
| `make data-unlabeled` | ~2.5 GB | Unlabeled split — required for SimCLR pretraining (Kaggle only) |

```bash
make smoke-data        # Codespaces-safe
make data              # adds test split
make data-unlabeled    # Kaggle / large-disk only
```

To remove downloaded data:
```bash
make clean
```

Data lands in `data/stl10_binary/` (gitignored). The `data/.gitkeep` file keeps the directory tracked.
