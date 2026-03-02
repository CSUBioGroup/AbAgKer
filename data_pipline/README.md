# Data Pipeline Documentation

This directory contains the full data-engineering workflow for AbAgKer. It transforms raw SabDab/SKEMPI resources into standardized JSON samples, auxiliary SSF/CDR tensors, and fold-based evaluation splits.

## Workflow Map

### 1) Source-specific extraction
- `sabdab/SabDab.ipynb`
  - Loads `sabdab_summary_all.tsv` and `SabDab.zip`.
  - Parses structure files and exports normalized antibody/antigen sequence records.
- `skempi/skempi_clean_2509.ipynb`
  - Loads SKEMPI tables and mapping archives.
  - Applies mutation-aware sequence reconstruction and record-level cleaning.

### 2) Unified dataset assembly
- `AbAgKer_all/collectall_andsplit.ipynb`
  - Merges curated SabDab and SKEMPI entries.
  - Builds 5-fold random and PDB-prefix-aware splits.
  - Writes split artifacts under `AbAgKer_all/split_fivefold/`.

### 3) Auxiliary feature construction
- `AbAgKer_all/ssf/ssf_get.ipynb`
  - Generates sequence-side feature tensors (SSF) for antigen tokens.
- `AbAgKer_all/cdrs_ssf/cdrs_compute.ipynb`
  - Computes CDR indicators and position-aligned tensors via ANARCI numbering.

## Primary Outputs
- Curated JSON files in `sabdab/result/` and `skempi/result/`.
- Five-fold split JSON files in `AbAgKer_all/split_fivefold/`.
- SSF and CDR tensor artifacts referenced by training-time data loading.

## Reproducibility Guidelines
- Keep raw source snapshots fixed (TSV/CSV/ZIP/TGZ files tracked in this directory).
- Regenerate splits using fixed random seeds to preserve fold identity.
- Use the same SSF/CDR generation notebooks and parameters as training-time preprocessing.
