Run the full FixMatch label-fraction sweep across 4 / 10 / 25 / 100 / 500 labels per class.

Requires SimCLR checkpoint and full data (train + test + unlabelled).

```bash
# Prerequisites
make data              # train + test splits (~310 MB)
make data-unlabeled    # unlabeled split (~2.5 GB on disk)
make pretrain          # SimCLR pretraining (~90 min on T4)

# Sweep
make sweep             # FixMatch at all label fractions

# Results
make plot              # label efficiency curve → plots/label_efficiency.png
make export            # best model → checkpoints/model.onnx
```

For GPU training, use the Kaggle notebooks instead:
- `kaggle/simclr_pretrain.ipynb`
- `kaggle/fixmatch_sweep.ipynb`
