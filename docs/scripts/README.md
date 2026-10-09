# Scripts Index

This folder contains operational scripts grouped by purpose.

## Current layout

- `analysis/`
  - Cross-validated modeling scripts by dataset (`mimic/`, `hirid/`, `eicu/`).
- `trajectory/`
  - Canonical trajectory probability extraction scripts by dataset.
- `preprocess/`
  - Canonical preprocessing entry points by dataset.
- `launchers/`
  - `nohup` / `tmux` wrappers for long-running analysis jobs.
- `slurm/`
  - Cluster submission scripts and cohort-merge helpers.
- `traj_scripts/`
  - Legacy compatibility wrappers during the migration.
- `*_preprocessing/`
  - Legacy preprocessing implementations kept for compatibility.
- `notebooks/`
  - Notebook helpers and notebook-adjacent utilities.

## Recommended entry points

- Analysis (quick test):
  - `python scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 1 --cv-splits 2`
- Trajectory extraction:
  - `python scripts/trajectory/hirid/aki_trajs.py`
- HiRiD vitals/labs extraction:
  - `python scripts/trajectory/hirid/extract_vitals_labs.py`
- Launchers:
  - `./scripts/launchers/hirid/run_hirid_sepsis_nohup.sh --train-n-jobs 8`
- Subset smoke test:
  - `python scripts/trajectory/run_circulatory_failure_bayes_subset.py --dry-run`

## Portability notes

- Launchers should avoid hardcoded absolute workspace paths.
- SLURM scripts now use environment-variable defaults for portability:
  - `REPO_ROOT` (auto-detected from `pyproject.toml`)
  - `DATA_ROOT` (default: `/home/gaga/data/physionet`)
  - `RESULTS_ROOT` (default: `$REPO_ROOT/results`)
  - `LOG_ROOT` (default: `$REPO_ROOT/logs`)

Useful checks:
- `make slurm-check-hardcoded`
- `make slurm-check-dataroot`

## Proposed reorganization (non-breaking, phased)

### Phase 1 (safe)
- Keep existing paths, add compatibility wrappers when introducing new paths.
- Add per-folder README files (`analysis/`, `trajectory/`, `preprocess/`, `slurm/`).

### Phase 2 (structure)
- Move preprocessing scripts to `scripts/preprocess/{dataset}/`.
- Split SLURM scripts by purpose:
  - `scripts/slurm/run/{dataset}/`
  - `scripts/slurm/merge/{dataset}/`
  - `scripts/slurm/retry/{dataset}/`

### Phase 3 (naming consistency)
- Normalize trajectory file names to remove duplicated dataset prefixes inside dataset folders:
  - e.g. `scripts/traj_scripts/hirid/hirid_aki_trajs.py` → `scripts/trajectory/hirid/aki_trajs.py`
- Keep old paths as thin wrappers for one release cycle.

### Phase 4 (cleanup)
- Remove wrappers after references are updated in docs and SLURM/launcher scripts.
