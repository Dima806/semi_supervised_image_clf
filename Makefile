PYTHON := python
SRC    := src/semi_supervised_image_clf

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help:
	@echo "Setup"
	@echo "  sync          Install/sync dependencies with uv"
	@echo ""
	@echo "Data  (streamed — tar.gz never written to disk)"
	@echo "  smoke-data        Download train split only (~120 MB on disk)"
	@echo "  data              Download train + test splits (~310 MB on disk)"
	@echo "  data-unlabeled    Download unlabeled split (~2.5 GB on disk, needed for SimCLR)"
	@echo ""
	@echo "Training"
	@echo "  pretrain      Run SimCLR pretraining on unlabelled data"
	@echo "  supervised    Train supervised baseline (all 5000 labels)"
	@echo "  fixmatch      Run FixMatch (labels_per_class from config)"
	@echo "  ssl_fixmatch  SimCLR pretrain + FixMatch fine-tune"
	@echo "  sweep         Run all label fractions (4, 10, 25, 100, 500 per class)"
	@echo "  smoke         Fast smoke test on 10%% data (Codespaces-safe)"
	@echo ""
	@echo "Evaluation & export"
	@echo "  evaluate      Evaluate all checkpoints on test set"
	@echo "  plot          Generate label efficiency curve"
	@echo "  export        Export best model to ONNX"
	@echo ""
	@echo "Quality"
	@echo "  test          Run pytest with coverage"
	@echo "  lint          ruff format + ruff check --fix + ty"
	@echo ""
	@echo "Kaggle"
	@echo "  kaggle_push   Push notebooks to Kaggle for GPU training"
	@echo ""
	@echo "Cleanup"
	@echo "  clean         Remove downloaded data, checkpoints, plots, mlruns"
	@echo ""
	@echo "Pipelines"
	@echo "  all           data -> pretrain -> sweep -> plot"

# ============================================================================
# Setup
# ============================================================================

.PHONY: sync
sync:
	uv sync --dev

# ============================================================================
# Data
# ============================================================================

.PHONY: data smoke-data data-unlabeled

# The archive is streamed — the 2.6 GB tar.gz is never written to disk.
# Only the requested binary files are extracted, so disk usage matches the
# sizes shown below.

# train split only (~120 MB on disk) — safe for Codespaces
smoke-data:
	$(PYTHON) -m semi_supervised_image_clf.dataset --download --data-dir data/ \
		--splits train

# train + test (~310 MB on disk) — adds test split for evaluation
data:
	$(PYTHON) -m semi_supervised_image_clf.dataset --download --data-dir data/ \
		--splits train test

# unlabeled split (~2.5 GB on disk) — required for SimCLR pretraining (Kaggle only)
data-unlabeled:
	$(PYTHON) -m semi_supervised_image_clf.dataset --download --data-dir data/ \
		--splits unlabeled

# ============================================================================
# Training
# ============================================================================

.PHONY: pretrain supervised fixmatch ssl_fixmatch sweep smoke

pretrain:
	$(PYTHON) -m semi_supervised_image_clf.simclr \
		--config config/simclr.yaml \
		--checkpoint-dir checkpoints/

supervised:
	$(PYTHON) -m semi_supervised_image_clf.supervised \
		--config config/supervised.yaml \
		--checkpoint-dir checkpoints/

fixmatch:
	$(PYTHON) -m semi_supervised_image_clf.fixmatch \
		--config config/fixmatch.yaml \
		--checkpoint-dir checkpoints/

ssl_fixmatch:
	$(PYTHON) -m semi_supervised_image_clf.fixmatch \
		--config config/fixmatch.yaml \
		--pretrained-simclr checkpoints/simclr_encoder.pt \
		--checkpoint-dir checkpoints/

sweep:
	@for n in 4 10 25 100 500; do \
		echo "=== labels_per_class=$$n ==="; \
		$(PYTHON) -m semi_supervised_image_clf.fixmatch \
			--config config/fixmatch.yaml \
			--labels-per-class $$n \
			--checkpoint-dir checkpoints/; \
	done

smoke:
	$(PYTHON) -m semi_supervised_image_clf.simclr \
		--config config/simclr.yaml \
		--smoke-test \
		--checkpoint-dir checkpoints/
	$(PYTHON) -m semi_supervised_image_clf.fixmatch \
		--config config/fixmatch.yaml \
		--smoke-test \
		--checkpoint-dir checkpoints/

# ============================================================================
# Evaluation & export
# ============================================================================

.PHONY: evaluate plot export

evaluate:
	$(PYTHON) -m semi_supervised_image_clf.evaluate \
		--checkpoint-dir checkpoints/ \
		--config config/fixmatch.yaml

plot:
	$(PYTHON) -m semi_supervised_image_clf.plot \
		--results-dir checkpoints/ \
		--output plots/label_efficiency.png

export:
	$(PYTHON) -m semi_supervised_image_clf.export \
		--checkpoint checkpoints/best_model.pt \
		--output checkpoints/model.onnx

# ============================================================================
# Quality
# ============================================================================

.PHONY: test lint

test:
	pytest tests/ -v

lint:
	ruff format src/ tests/
	ruff check --fix src/ tests/
	ty check src/ --python $(shell which python)

# ============================================================================
# Cleanup
# ============================================================================

.PHONY: clean
clean:
	rm -rf data/stl10_binary/ checkpoints/*.pt plots/*.png mlruns/ results.json
	@echo "Cleaned data, checkpoints, plots, mlruns."

# ============================================================================
# Kaggle
# ============================================================================

.PHONY: kaggle_push
kaggle_push:
	$(PYTHON) -m semi_supervised_image_clf.kaggle_push \
		--notebooks kaggle/simclr_pretrain.ipynb kaggle/fixmatch_sweep.ipynb

# ============================================================================
# Pipelines
# ============================================================================

.PHONY: all
all: data data-unlabeled pretrain sweep plot
