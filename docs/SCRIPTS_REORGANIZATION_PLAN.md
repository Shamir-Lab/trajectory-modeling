# Scripts Reorganization Plan

Status: proposal

This plan minimizes breakage by introducing wrappers and staged migration.

## Goals

1. Reduce path drift between docs and runnable scripts.
2. Remove machine-specific absolute paths from launchers/SLURM.
3. Make script discovery easier for new users.

## Proposed target structure

```text
scripts/
  analysis/
    mimic/
    hirid/
    eicu/
  trajectory/                 # rename from traj_scripts
    mimic/
    hirid/
    eicu/
  preprocess/
    mimic/
    hirid/
    eicu/
  launchers/
    mimic/
    hirid/
    eicu/
  slurm/
    run/
      mimic/
      hirid/
      eicu/
    merge/
      mimic/
      hirid/
      eicu/
    retry/
      mimic/
      hirid/
      eicu/
  dev/
    notebooks/
    checks/
```

## Migration steps

### Step 1 — portability first
- Replace hardcoded paths with variables:
  - `REPO_ROOT`, `DATA_ROOT`, `RESULTS_ROOT`, `LOG_ROOT`.
- Update launcher generator and regenerate launcher scripts.
- Keep behavior unchanged.

### Step 2 — folder moves with wrappers
- Move:
  - `scripts/traj_scripts/*` → `scripts/trajectory/*`
  - `scripts/*_preprocessing/*` → `scripts/preprocess/{dataset}/*`
- Leave old files as wrappers that exec/import the new script path.

### Step 2b — name cleanup
- Prefer short canonical names inside dataset folders.
- Keep dataset-prefixed filenames only as compatibility wrappers.

### Step 3 — SLURM normalization
- Move SLURM files to `run/`, `merge/`, `retry/` subtrees.
- Standardize job logs under `${LOG_ROOT:-$REPO_ROOT/logs}`.

### Step 4 — docs freeze
- Canonical docs:
  - root README
  - `docs/scripts/README.md`
- Mark migration/conversion summaries as historical.

## Acceptance checks

- `make help` lists only existing commands.
- `README` commands run without path edits.
- Launchers work from any clone location.
- SLURM scripts run after setting only environment variables.
