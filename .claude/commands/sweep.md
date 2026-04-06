Run the full FixMatch label-fraction sweep across 4 / 10 / 25 / 100 / 500 labels per class.

Requires SimCLR checkpoint and full data (train + test + unlabelled).

```bash
# Prerequisites
make data              # train + test splits (~310 MB)
make data-unlabeled    # unlabeled split (~2.5 GB on disk)
make pretrain          # SimCLR pretraining (~70 min on T4)

# Sweep
make sweep             # FixMatch at all label fractions (~90 min on T4)

# Results
make plot              # label efficiency curve → plots/label_efficiency.png
make export            # best model → checkpoints/model.onnx
```

For GPU training, use the Kaggle notebooks instead:
- `kaggle/simclr_pretrain.ipynb` — template
- `kaggle/fixmatch_sweep.ipynb` — template
- `kaggle/simclr-pretrain.ipynb` — trained version with outputs
- `kaggle/fixmatch-sweep.ipynb` — trained version with outputs

## Measured results (Kaggle T4, April 2026)

> **Important**: SimCLR checkpoint was not available during the sweep run below
> (separate Kaggle sessions). SimCLR + FixMatch end-to-end is pending.

| Method | Labels / class | Test accuracy |
|---|---|---|
| Supervised baseline | 500 | 55.9% |
| FixMatch (random init) | 4 | 10.3% ⚠ |
| FixMatch (random init) | 10 | 10.0% ⚠ |
| FixMatch (random init) | 25 | 10.0% ⚠ |
| FixMatch (random init) | 100 | 41.6% |
| FixMatch (random init) | 500 | 61.8% |

⚠ Near-random: pseudo-label threshold (0.95) never met with random init + ≤25 labels/class.
SimCLR encoder init is expected to resolve this significantly.
