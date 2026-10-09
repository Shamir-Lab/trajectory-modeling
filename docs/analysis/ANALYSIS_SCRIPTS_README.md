# Trajectory-Based Prediction Analysis Scripts

Unified Python scripts for running cross-validated model comparisons across all prediction tasks (Circulatory Failure, Sepsis, Liver, AKI, Ventilator) and datasets (MIMIC, HiRiD, eICU).

## Architecture

### Normalized Preprocessing (src/analysis/analysis_utils.py)

Shared utilities for all tasks:
- **Column auto-detection**: `pick_id_col()`, `pick_time_cols()`
- **Safe data loading**: `load_named_table()` (parquet-first, CSV fallback)
- **Robust biomarker statistics**: `biomarker_summary_stats()` (mean, max, min, trend, std)
- **Safe merging**: `safe_left_merge()` (duplicate key handling + one-to-one validation)
- **Feature organization**: `organize_feature_sets()` (trajectory, summary, baseline classification)
- **Dataset config**: `get_dataset_config()` (dataset-specific paths & ID columns)
- **Model utilities**: `get_default_models()` (LogisticRegression, RandomForest, HistGradientBoosting, XGBoost)

### Base Analysis Template (src/analysis/analysis_template.py)

Provides:
- `AnalysisConfig`: Configuration class (override DATASET, TASK, BIOMARKERS, TARGET_COL, etc.)
- `TrajectoryAnalysis`: Main analysis class with:
  - `load_data()`: Merge prediction dataset + outcomes
  - `prepare_features()`: Compute biomarker summary stats + merge
  - `run_cv()`: Cross-validated model comparison (GroupKFold, joblib parallelization)
  - `save_results()`: Export to CSV
- `main_cli()`: Generic CLI entry point with argument parsing

### Task-Specific Scripts (`scripts/analysis/{dataset}/`)

#### MIMIC (5 tasks)
- `mimic_circulatory_failure_analysis.py` (existing, already converted)
- `mimic_sepsis_analysis.py` (NEW)
- `mimic_liver_analysis.py` (NEW)
- `mimic_aki_analysis.py` (NEW)
- `mimic_ventilator_analysis.py` (NEW)

#### HiRiD (5 tasks)
- `hirid_circulatory_failure_analysis.py` (existing, already converted)
- `hirid_sepsis_analysis.py` (NEW, with parallelization)
- `hirid_liver_analysis.py` (NEW, with parallelization)
- `hirid_aki_analysis.py` (NEW, with parallelization)
- `hirid_ventilator_analysis.py` (NEW, with parallelization)

#### eICU (5 tasks)
- `eicu_circulatory_failure_analysis.py` (existing, already converted)
- `eicu_sepsis_analysis.py` (NEW)
- `eicu_liver_analysis.py` (NEW)
- `eicu_aki_analysis.py` (NEW)
- `eicu_ventilator_analysis.py` (NEW)

## Usage

### Quick Start (Single Script)

```bash
cd /home/gaga/tamarw1/trajectory-modeling

# Run with defaults
python scripts/analysis/mimic/mimic_sepsis_analysis.py

# Run with custom parameters
python scripts/analysis/hirid/hirid_liver_analysis.py \
    --cv-repeats 2 \
    --cv-splits 3 \
    --train-n-jobs 4 \
    --train-backend threads

# Run with custom data directory
python scripts/analysis/eicu/eicu_aki_analysis.py \
    --base-dir /path/to/eicu/aki \
    --output-dir /path/to/results
```

### Persistent Execution (nohup or tmux)

All tasks have auto-generated launcher scripts:

#### nohup (fire-and-forget)
```bash
# MIMIC Sepsis
./scripts/launchers/mimic/run_mimic_sepsis_nohup.sh --train-n-jobs 2

# HiRiD Liver (with parallelization)
./scripts/launchers/hirid/run_hirid_liver_nohup.sh --train-n-jobs 4 --train-backend threads

# eICU AKI
./scripts/launchers/eicu/run_eicu_aki_nohup.sh

# Monitor output
tail -f logs/outs/mimic/mimic_sepsis_analysis.out

# Check PID
cat .mimic_sepsis_analysis.pid

# Kill process
kill $(cat .mimic_sepsis_analysis.pid)
```

#### tmux (interactive session)
```bash
# HiRiD Ventilator (parallelized)
./scripts/launchers/hirid/run_hirid_ventilator_tmux.sh --train-n-jobs 8 --train-backend processes

# Attach to session
tmux attach -t hirid_ventilator_analysis

# List sessions
tmux list-sessions

# Kill session
tmux kill-session -t hirid_ventilator_analysis
```

## Parameters

### Data Directories (auto-detected)
- **--base-dir**: Input data directory (default: `/home/gaga/data/physionet/{dataset}/{task}/`)
- **--output-dir**: Results directory (default: `results/{dataset}/{task}/`)

### Cross-Validation
- **--cv-repeats**: Number of CV repeats (default: 10)
- **--cv-splits**: Number of folds per repeat (default: 5)
- **--lookback-window**: Lookback window for summary stats in native time units
- **--lookback-unit**: Unit label (`days` or `hours`) for clarity/logging
- **--lookahead-hours**: Deprecated alias for `--lookback-window`

### Parallelization (joblib)
- **--train-n-jobs**: Number of parallel jobs (default: 1)
  - `>0`: use N workers
  - `-1`: use all CPU cores
- **--train-backend**: Backend for parallelization (default: `threading`)
  - `threading`: Fast for I/O-bound work, shared memory
  - `processes`: Slower startup, true parallelism for CPU-bound work

## Dataset-Specific Configurations

### MIMIC
- **ID Column**: `hadm_id` (hospital admission ID)
- **Time Units**: Hours (`time_hours`, `time_hour`)
- **Data Path**: `/home/gaga/data/physionet/mimic/{task}/`
- **Biomarkers**:
  - Circulatory Failure: lactate, heartrate, systolic
  - Sepsis: lactate, wbc, platelet
  - Liver: bilirubin
  - AKI: creatinine
  - Ventilator: pao2, peep

### HiRiD
- **ID Column**: `patientid`
- **Time Units**: Hours (`time_hours`, `time_hour`)
- **Data Path**: `/home/gaga/data/physionet/hirid/{task}/`
- **Note**: CV repeats are parallelized via joblib for faster execution

### eICU
- **ID Column**: `stay_id` (ICU stay ID)
- **Time Units**: Days (`time_days`, `time_day`)
- **Data Path**: `/home/gaga/data/physionet/eicu/{task}/`

## Output

Results are saved to CSV with columns:
- `model`: Model name (LogisticRegression, RandomForest, XGBoost, HistGradientBoosting)
- `feature_set`: Feature configuration (Trajectory Only, Summary Stats Only, Trajectory + Summary)
- `auroc_mean`: Mean AUROC across folds
- `auroc_std`: Std AUROC
- `aupr_mean`: Mean AUPR (Average Precision)
- `aupr_std`: Std AUPR
- `n_folds`: Total number of folds

Example:
```
model,feature_set,auroc_mean,auroc_std,aupr_mean,aupr_std,n_folds
XGBoost,Trajectory Only,0.7823,0.0125,0.3456,0.0234,50
XGBoost,Summary Stats Only,0.8234,0.0098,0.4123,0.0198,50
XGBoost,Trajectory + Summary,0.8567,0.0087,0.4789,0.0167,50
```

## Feature Sets

Each analysis automatically creates three feature configurations:

1. **Trajectory Only**: Single/multi-biomarker trajectory probabilities (stable, gradual, rapid)
2. **Summary Stats Only**: Windowed statistics (mean, max, min, trend, change, std)
3. **Trajectory + Summary**: Combined features

Missing biomarkers are automatically skipped with logging.

## Advanced: Custom Analysis

To create a custom analysis for a new task/dataset:

1. Create a config class in your script:
```python
from analysis_template import AnalysisConfig

class CustomConfig(AnalysisConfig):
    DATASET = "mimic"  # or 'hirid', 'eicu'
    TASK = "custom_task"
    
    BIOMARKERS = {
        "biomarker1": {
            "file": "biomarker1_timeseries.csv",
            "value_col": "biomarker1",
        },
    }
    
    TARGET_COL = "target_outcome"
    OUTCOME_FILE = "outcomes.csv"
    OUTCOME_MERGE_COLS = ["hadm_id", "time_day", "target_outcome"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKBACK_WINDOW = 12
    LOOKBACK_UNIT = "hours"
```

2. Run via CLI:
```python
from analysis_template import main_cli

if __name__ == "__main__":
    main_cli(CustomConfig, description="Custom Task Analysis")
```

## Troubleshooting

### Import Errors
Ensure `src/` is in PYTHONPATH:
```bash
export PYTHONPATH="/home/gaga/tamarw1/trajectory-modeling/src:$PYTHONPATH"
python scripts/analysis/mimic/mimic_sepsis_analysis.py
```

### Missing Data Files
Check data exists:
```bash
ls -lh /home/gaga/data/physionet/mimic/sepsis/*.csv
```

### Memory Issues
Reduce `--cv-repeats` or `--cv-splits`:
```bash
python scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 2 --cv-splits 3
```

### Slow Execution
Enable parallelization:
```bash
python scripts/analysis/hirid/hirid_sepsis_analysis.py --train-n-jobs 8 --train-backend threads
```

## Validation Checklist

- [x] Preprocessing normalized across all tasks (analysis_utils.py)
- [x] Base template handles single/multi-biomarker variants
- [x] Dataset-specific ID/time column auto-detection
- [x] Safe merging with duplicate key handling
- [x] joblib parallelization for CV repeats
- [x] All 15 analysis scripts created (5 MIMIC + 5 HiRiD + 5 eICU)
- [x] Launcher scripts for persistent execution (30 total: 15 nohup + 15 tmux)
- [x] Backward compatibility with existing circulatory_failure scripts

## File Structure

```
trajectory-modeling/
├── src/
│   ├── analysis/
│   │   ├── analysis_utils.py      # Shared preprocessing utilities
│   │   └── analysis_template.py   # Base analysis class + CLI
│   └── traj_features/             # (existing trajectory modeling code)
├── scripts/
│   ├── analysis/                  # dataset-specific analysis scripts
│   ├── launchers/                 # nohup/tmux wrappers by dataset
│   └── traj_scripts/              # legacy compatibility wrappers during the migration
└── results/
    ├── mimic/
    │   ├── circulatory_failure/
    │   ├── sepsis/
    │   ├── liver/
    │   ├── aki/
    │   └── ventilator/
    ├── hirid/
    └── eicu/
```
