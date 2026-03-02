# Five-Fold Split Artifacts

This directory stores fold-wise train/test JSON files generated from the curated AbAgKer dataset.

## Split Policy
- **K-fold setting**: 5 folds.
- **Output format**: `fold_{k}_train.json` and `fold_{k}_test.json`.
- **Usage**: each fold is used as test once while the remaining folds are used for training.

These files are consumed directly by the training configuration files under `any_tests/`.
