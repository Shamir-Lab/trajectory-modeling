# Quick Start Guide - Trajectory Analysis Scripts

> **Important (April 2026):** canonical script locations are:
> - Analysis scripts: `scripts/analysis/{dataset}/...`
> - Trajectory scripts: `scripts/traj_scripts/{dataset}/...`
> - Launcher scripts: `scripts/launchers/{dataset}/...`

## Fast Start

### Option 1: run one analysis script
```bash
cd /home/gaga/tamarw1/trajectory-modeling
python scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 1 --cv-splits 2
```

### Option 2: run with nohup launcher
```bash
./scripts/launchers/hirid/run_hirid_circulatory_failure_nohup.sh --train-n-jobs 4
tail -f logs/outs/hirid/hirid_circulatory_failure_analysis.out
```

### Option 3: run with tmux launcher
```bash
./scripts/launchers/hirid/run_hirid_cf_tmux.sh --train-n-jobs 8 --train-backend threads
tmux attach -t hirid_cf_analysis
```

### Option 4: trajectory-only smoke test
```bash
python scripts/traj_scripts/run_circulatory_failure_bayes_subset.py --dry-run
```

---

## Available Analysis Scripts

### MIMIC
```
python scripts/analysis/mimic/mimic_circulatory_failure_analysis.py
python scripts/analysis/mimic/mimic_sepsis_analysis.py
python scripts/analysis/mimic/mimic_liver_analysis.py
python scripts/analysis/mimic/mimic_aki_analysis.py
python scripts/analysis/mimic/mimic_ventilator_analysis.py
```

### HiRiD
```
python scripts/analysis/hirid/hirid_circulatory_failure_analysis.py
```

### eICU
```
python scripts/analysis/eicu/eicu_circulatory_failure_analysis.py
python scripts/analysis/eicu/eicu_sepsis_analysis.py
python scripts/analysis/eicu/eicu_liver_analysis.py
python scripts/analysis/eicu/eicu_aki_analysis.py
python scripts/analysis/eicu/eicu_ventilator_analysis.py
```

---

## Parallelization Options

### Recommended Configurations

| Task | Cores | Backend | Command |
|------|-------|---------|---------|
| MIMIC (quick test) | 1 | threading | `python scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 2 --cv-splits 2` |
| MIMIC (production) | 4 | threads | `./scripts/launchers/mimic/run_mimic_sepsis_nohup.sh --train-n-jobs 4` |
| HiRiD (production) | 8 | threads | `./scripts/launchers/hirid/run_hirid_circulatory_failure_nohup.sh --train-n-jobs 8 --train-backend threads` |
| eICU (all cores) | -1 | processes | `./scripts/launchers/eicu/run_eicu_circulatory_failure_nohup.sh --train-n-jobs -1 --train-backend processes` |

### Parallelization Flags

```
--train-n-jobs N           # Number of parallel workers
  1       = sequential (default)
  2-8     = threading (fast, low memory)
  -1      = use all CPU cores

--train-backend BACKEND    # Joblib backend
  threading  = fast startup, good for I/O-bound (default)
  processes  = true parallelism, slower startup, high memory
```

---

## Monitoring Execution

### nohup (fire-and-forget)
```bash
# Start
./scripts/launchers/mimic/run_mimic_sepsis_nohup.sh --train-n-jobs 2

# Monitor
tail -f logs/outs/mimic/mimic_sepsis_analysis.out

# Check process
ps aux | grep mimic_sepsis_analysis.py
ps -p $(cat .mimic_sepsis_analysis.pid)

# Stop
kill $(cat .mimic_sepsis_analysis.pid)
```

### tmux (interactive)
```bash
# Start
./scripts/launchers/hirid/run_hirid_cf_tmux.sh --train-n-jobs 4

# Attach
tmux attach -t hirid_cf_analysis

# Detach (Ctrl+B, then D)
# Kill session
tmux kill-session -t hirid_liver_analysis

# List all sessions
tmux list-sessions
```

---

## Output

Results saved to: `results/{dataset}/{task}/{task}_cv_results.csv`

Example results:
```
model,feature_set,auroc_mean,auroc_std,aupr_mean,aupr_std,n_folds
XGBoost,Trajectory Only,0.8234,0.0087,0.4123,0.0198,50
XGBoost,Summary Stats Only,0.8456,0.0076,0.4567,0.0176,50
XGBoost,Trajectory + Summary,0.8789,0.0065,0.5012,0.0154,50
```

---

## Common Use Cases

### Quick Test (1 minute)
```bash
python scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 1 --cv-splits 2
```

### Full Production Run (4 hours, parallelized)
```bash
./scripts/launchers/hirid/run_hirid_circulatory_failure_nohup.sh --train-n-jobs 8 --train-backend threads
```

### Batch Run All MIMIC Tasks
```bash
for task in sepsis liver aki ventilator; do
   ./scripts/launchers/mimic/run_mimic_${task}_nohup.sh --train-n-jobs 2
    sleep 10
done

# Monitor all
for f in logs/outs/mimic/mimic_*_analysis.out; do
    echo "=== $(basename $f) ===" && tail -1 "$f"
done
```

### Compare All Datasets (same task)
```bash
# Run all sepsis tasks in parallel tmux sessions
./scripts/launchers/mimic/run_mimic_sepsis_tmux.sh --train-n-jobs 4
./scripts/launchers/hirid/run_hirid_cf_tmux.sh --train-n-jobs 4
./scripts/launchers/eicu/run_eicu_circulatory_failure_nohup.sh --train-n-jobs 4

# Compare results
echo "MIMIC:" && tail -1 results/mimic/circulatory_failure/*_cv_results.csv
echo "HiRiD:" && tail -1 results/hirid/circulatory_failure/*_cv_results.csv
echo "eICU:" && tail -1 results/eicu/circulatory_failure/*_cv_results.csv
```

---

## Key Features

### ✅ Normalized Preprocessing
- Auto-detection of ID, time columns across datasets
- Robust biomarker summary statistics (handles NaN, degenerate windows)
- Safe merging with duplicate key handling

### ✅ Parallelization
- CV repeats parallelized via joblib
- ~10x speedup on 8-core machine
- Configurable threading vs processes backend

### ✅ Persistent Execution
- nohup: Runs in background, survives SSH disconnect
- tmux: Interactive monitoring without detaching from shell

### ✅ Task Normalization
- Single/multi-biomarker variants (sepsis, circulatory failure)
- Consistent feature organization (trajectory, summary stats)
- Dataset-specific config (MIMIC: hadm_id, HiRiD: patientid, eICU: stay_id)

---

## Troubleshooting

### Error: "No module named 'analysis_utils'"
```bash
export PYTHONPATH="/home/gaga/tamarw1/trajectory-modeling/src:$PYTHONPATH"
python scripts/analysis/mimic/mimic_sepsis_analysis.py
```

### Error: "Missing table for circulatory_failure_prediction_dataset"
Check data exists:
```bash
ls -lh /home/gaga/data/physionet/mimic/sepsis/*.csv
```

### Slow execution
Reduce CV repeats or increase parallelism:
```bash
python scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 2 --cv-splits 3 --train-n-jobs 4
```

### Memory issues
Reduce parallel workers or switch to threading:
```bash
./scripts/launchers/mimic/run_mimic_sepsis_nohup.sh --train-n-jobs 2 --train-backend threading
```

---

## Next Steps

1. **Run a quick test**
   ```bash
   python scripts/analysis/mimic/mimic_sepsis_analysis.py --cv-repeats 1 --cv-splits 2
   ```

2. **Run production on HiRiD (parallelized)**
   ```bash
   ./scripts/launchers/hirid/run_hirid_sepsis_nohup.sh --train-n-jobs 8
   ```

3. **Monitor results**
   ```bash
   tail -f logs/outs/hirid/hirid_sepsis_analysis.out
   ```

4. **Compare results across datasets**
   ```bash
   for ds in mimic hirid eicu; do
       echo "=== ${ds^^} ===" && tail -1 results/$ds/sepsis/*_cv_results.csv
   done
   ```

---

For detailed documentation, see `docs/analysis/ANALYSIS_SCRIPTS_README.md`.
