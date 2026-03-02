# AbAgKer: A Unified Semi-Supervised Framework for Antigen-Antibody Binding Affinity and Kinetics Prediction

## Abstract
To the best of our knowledge, we are the first to achieve full K-value prediction for antigen-antibody interactions (covering Kd, koff and kon), providing a comprehensive tool for antibody screening and analysis.


## Installation

### 1. Environment
We recommend Python **3.9-3.10**.

### 2. Core Training Dependencies
Install framework dependencies used by `main_wandb.py` and `taming/` modules:

```bash
pip install torch==1.12.0 torchvision torchaudio
pip install pytorch-lightning==1.5.10 torchmetrics transformers omegaconf
pip install numpy scipy scikit-learn pandas tqdm wandb pudb
pip install antiberty
pip install biopython jupyter ipykernel
```

### 3. Data Pipeline
To run notebooks under `data_pipline/` and `notebook/`, install:
```bash
  pip install b2bTools
  
  # ANARCI
  conda install -c conda-forge biopython -y
  conda install -c bioconda hmmer=3.3.2 -y
  cd ANARCI
  python setup.py install
```

## Architecture Description
- **Training entrypoint**: `main_wandb.py` provides config-driven model/data instantiation, deterministic setup, logger/callback registration, and Trainer execution.
- **Lightning wrapper**: `taming/models/AbAgKer_newLLM.py` defines `AbAgKerTrainer`, including `training_step`, `validation_step`, metric tracking, optimizer/scheduler setup, and feature assembly.
- **Antibody encoder**: `taming/modules/autoencoder/AbModule.py` contains Cattention module and token poolers.
- **Antigen poolers**: `taming/modules/autoencoder/AgModule.py` MEI module.
- **Cross modules**: `taming/modules/autoencoder/AbAgCross.py` implements bidirectional co-attention blocks.
- **Losses/metrics**: `taming/modules/losses/loss_Ab.py` and `taming/modules/metrics/metrics.py` define optimization targets and evaluation criteria.

## Data Format Explanation
Training/validation data are JSON lists where each sample includes:
- `pdb`: complex identifier.
- `H`, `L`: antibody heavy/light chain sequences.
- `X`: antigen sequence.
- `AbAgA`: supervised affinity target (used as primary regression signal).
- `AbAgI`, `AbAgAoff`: additional kinetic/auxiliary fields used by specific settings.

The dataloader utilities are defined in `taming/data/PM_Data.py` and `taming/data/utils.py`, with custom collation in `custom_collate`.


## Data Processing Pipeline
The preprocessing workflow under `data_pipline/` converts raw public resources into train-ready JSON splits and auxiliary tensors. The process is notebook-driven and reproducible at the directory level.

### Stage A: Source Extraction
- **SabDab parsing**: `data_pipline/sabdab/SabDab.ipynb`
  - Reads SabDab metadata (`sabdab_summary_all.tsv`) and structure archive (`SabDab.zip`).
  - Extracts antibody/antigen chain sequences from PDB entries.
  - Exports normalized records to `data_pipline/sabdab/result/`.
- **SKEMPI cleaning**: `data_pipline/skempi/skempi_clean_2509.ipynb`
  - Reads mutation/affinity tables and PDB mapping files.
  - Applies chain-role normalization and mutation consistency checks.
  - Produces curated records under `data_pipline/skempi/result/`.

### Stage B: Unified Sample Construction
- **Merge and split orchestration**: `data_pipline/AbAgKer_all/collectall_andsplit.ipynb`
  - Merges SabDab and SKEMPI-derived records.
  - Generates random and PDB-aware 5-fold train/test splits.
  - Writes fold artifacts to `data_pipline/AbAgKer_all/split_fivefold/`.

### Stage C: Auxiliary Feature Generation
- **SSF features**: `data_pipline/AbAgKer_all/ssf/ssf_get.ipynb`
  - Computes per-residue sequence-side features used by antigen pooling modules.
- **CDR masks/features**: `data_pipline/AbAgKer_all/cdrs_ssf/cdrs_compute.ipynb`
  - Uses ANARCI-based numbering to derive CDR region indicators.
  - Exports tensors consumed by the antibody Cattention pipeline.

### Runtime Linkage
Generated JSON and feature tensors are mirrored into runtime directories (`data/`, `data/data_aug/`) and consumed by `taming/data/PM_Data.py` during training and evaluation.

## Training Instructions
1. Prepare a YAML config (examples are provided in `any_tests/config_1120*` and related folders).
2. Launch training with:
```bash
python main_wandb.py -t True -b path/to/config.yaml --save-dir logs
```
3. For deterministic behavior, set `--seed` explicitly (default is already fixed in the script).
4. Resume from an existing run by passing `--resume path/to/logdir_or_ckpt`.

## Inference Instructions
For checkpoint-based inference/prediction:
1. Load the trained checkpoint through the corresponding config.
2. Reuse the same dataset schema for test JSON input.
3. Execute Lightning test/predict flow (or notebook workflows in `notebook/mutation_prediction/`).

Recommended practice:
- Keep preprocessing consistent with training (tokenization, padding lengths, SSF/CDR feature extraction).
- Validate output scale against the training target normalization policy.

## Configuration Explanation
Configuration is OmegaConf/YAML-based and typically includes:
- `model`: target class path and model hyperparameters.
- `data`: dataset class paths and file locations.
- `lightning.trainer`: accelerator/devices/precision/training schedule.
- `opt_config`: optimizer, gradient clipping, warmup/cosine scheduling.
- `loss_config`: objective weights and optional auxiliary-loss controls.

## File Structure Overview
```text
.
├── main_wandb.py                     # Training entrypoint (Lightning + configs)
├── taming/
│   ├── models/                       # LightningModule wrappers
│   ├── modules/
│   │   ├── autoencoder/              # Encoders, poolers, cross-attention blocks
│   │   ├── losses/                   # Task losses
│   │   └── metrics/                  # Evaluation metrics
│   └── data/                         # Dataset classes and collation utilities
├── any_tests/                        # Example experiment configs
├── data/                             # Fold splits and processed JSON data
├── data_pipline/                     # Data preparation artifacts/scripts
└── notebook/                         # Analysis and prediction notebooks
```

