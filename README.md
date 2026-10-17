# Trajectory Modeling for Disease Progression Prediction

Code for the M.Sc. thesis *Clinical parameter trajectory modeling for disease progression
prediction* (Tammy Wertheimer, Tel Aviv University; advisor Prof. Ron Shamir).

The recent course of a clinical parameter (for example lactate over the last three days) is
represented by scores for three trajectory types. The scores are estimated from many sampled
spline trajectories, drawn either by Bayesian MCMC or by bootstrap resampling. They are then
tested as features in five ICU prediction tasks on MIMIC-III, eICU and HiRID.

## Pipeline

```text
preprocessing -> trajectory features (MCMC / bootstrap) -> prediction analyses (10x5 grouped CV)
              -> boxplots, tables and significance tests -> thesis figures
```

| Task key | Thesis name | Datasets | Target clinical parameters |
| --- | --- | --- | --- |
| `sepsis` | Septic Shock | MIMIC-III, eICU | lactate, WBC, platelets |
| `liver` | Acute-on-Chronic Liver Failure (ACLF) | MIMIC-III, eICU | bilirubin |
| `aki` | AKI Stage 3 | MIMIC-III, eICU | creatinine |
| `ventilator` | Ventilator Discontinuation Success | MIMIC-III, eICU | P/F ratio |
| `circulatory_failure` | Circulatory Failure | MIMIC-III, eICU, HiRID | lactate, heart rate, systolic BP |

The lookback window, gap, prediction window and thresholds of each task are listed in thesis
Table 2.2. All commands below are run from the repository root.

### 1. Preprocessing

`scripts/{mimic,eicu,hirid}_preprocessing/<task>_preprocess.py` (no arguments) builds the cohort,
the outcome labels with their gap and prediction window, the static and dynamic features, and the
time series of the target parameters. Inputs are DuckDB copies of the PhysioNet databases:

- MIMIC-III: `data/mimic/mimiciii.duckdb` (relative to the repository root), with
  `vital_metadata.csv` and `labs_metadata.csv` next to it
- eICU: `$TRAJ_DATA_ROOT/eicu.duckdb`
- HiRID: `$TRAJ_DATA_ROOT/HiRiD/hirid.duckdb`

`TRAJ_DATA_ROOT` defaults to `/home/gaga/data/physionet`. The outputs go to
`$TRAJ_DATA_ROOT/<dataset>/<task>/`, which is where the next steps read them. The MIMIC liver and
sepsis scripts are the exception: they write to `results/mimic/<task>/`, and their outputs have to
be moved to `$TRAJ_DATA_ROOT/mimic/<task>/`.

### 2. Trajectory features

**MCMC.** `scripts/trajectory/<dataset>/<task>_trajs.py` (eICU septic shock:
`sepsis_all_biomarkers.py`) fits the Bayesian spline model with 4 chains of 300 warm-up and
200 sampling iterations (NUTS, target acceptance 0.99). The cohort can be split with
`--cohort-splits N --cohort-index k`; on SLURM the array jobs in `scripts/slurm/<dataset>/run_*.slurm`
do this, and `scripts/slurm/<dataset>/merge_*.sh` combines the parts. The circulatory-failure
features of the thesis were computed in batches inside the g++ container
(`docker/run_cf_batches.sh`, see [docs/CONTAINER_GUIDE.md](docs/CONTAINER_GUIDE.md)).

**Bootstrap.** For circulatory failure, `scripts/trajectory/<dataset>/circulatory_failure_trajs_bootstrap.py`.
For the other tasks the analysis computes the bootstrap scores on its first run (200 resamples)
and caches them as `<parameter>_trajectory_probs_bootstrap.csv` in the task's data directory.

The library code is in `src/traj_features/`: `backends/bayes/` (model, sampling, classification)
and `backends/bootstrap/`.

### 3. Prediction analyses

`scripts/analysis/<dataset>/<dataset>_<task>_analysis.py`. The circulatory-failure scripts are
standalone; the others are configurations of `src/analysis/analysis_template.py`. Each runs
logistic regression, random forest, histogram gradient boosting and XGBoost on the feature sets
SD, SD+S, SD+T, SD+S+T, SD+T (B) and SD+S+T (B) (SD: static and dynamic features, S: summary
statistics of the lookback window, T: trajectory type scores, (B): bootstrap instead of MCMC).

Evaluation is 10 repeats of 5-fold cross-validation grouped by patient (`GroupKFold`), with new
folds in every repeat (`shuffle=True, random_state=920 + repeat`). Results are written to
`--output-dir` as `<task>_cv_results.csv` and `<task>_cv_results_fold_metrics.csv` (the
circulatory-failure scripts write `<dataset>_circulatory_failure_*` files).

The thesis runs used these settings:

```bash
# AKI, liver, sepsis, ventilator (MIMIC-III and eICU)
python scripts/analysis/mimic/mimic_sepsis_analysis.py \
    --base-dir $TRAJ_DATA_ROOT/mimic/sepsis --output-dir results/mimic/sepsis \
    --cv-repeats 10 --cv-splits 5 --train-n-jobs 10 --train-backend threading \
    --parallel-axis repeat --oof-scope representative-k --oof-k 10 --plot-k 10

# Circulatory failure
python scripts/analysis/eicu/eicu_circulatory_failure_analysis.py \
    --base-dir $TRAJ_DATA_ROOT/eicu/circulatory_failure --output-dir results/eicu/circulatory_failure --train-n-jobs 10
python scripts/analysis/hirid/hirid_circulatory_failure_analysis.py \
    --base-dir $TRAJ_DATA_ROOT/hirid/circulatory_failure --output-dir results/hirid/circulatory_failure \
    --eval-profile full --n-jobs 8 --train-n-jobs 10
python scripts/analysis/mimic/mimic_circulatory_failure_analysis.py \
    --base-dir $TRAJ_DATA_ROOT/mimic/circulatory_failure --output-dir results/mimic/circulatory_failure \
    --n-repeats 10 --n-folds 5 --seed 920
```

The MIMIC-III circulatory-failure analysis was split across three servers with
`scripts/rerun/cf_shard.py`, which gives the same results as one run (`scripts/rerun/cf_shard_test.py`).

### 4. Boxplots, tables and significance tests

```bash
# AUROC/AUPR boxplots per dataset and task -> results/poster/<dataset>/<task>/mimic_eicu_selected_sets_v1/
python scripts/analysis/poster/run_poster_pipeline.py --config configs/poster/mimic_eicu_default.yaml

# Cohort overview and CV metrics tables -> results/tables/
python scripts/analysis/tables/build_summary_tables.py --data-root $TRAJ_DATA_ROOT --results-root results

# XGBoost feature-set comparisons: paired Wilcoxon tests over (repeat, fold), BH q-values
python scripts/rerun/significance.py results results/significance.csv
python scripts/rerun/gen_tables.py results/significance.csv tables_3_2-3_4.json   # rows of Tables 3.2-3.4
python scripts/rerun/gen_s1.py results/poster table_s1.json                       # blocks of Table S1
```

See [docs/poster_visualization_pipeline.md](docs/poster_visualization_pipeline.md) for the
boxplot configuration. The scripts in `scripts/rerun/` are the helpers of the thesis runs, kept
unchanged; some use the server paths of those runs (see [docs/thesis_results.md](docs/thesis_results.md)).

### 5. Thesis figures

- **Qualitative validation** (Fig. 3.1, Supplementary Figs. S1-S4):
  `scripts/figures/qualitative_validation.py`. The thesis figures show the examples selected on
  2026-09-21. Regenerate them with `scripts/rerun/qv_replay.py <selected_examples_manifest.csv> <output_dir>`,
  which plots the windows listed in that manifest. Running `qualitative_validation.py` on its own
  selects different examples. The manifest contains patient-level values and is not in this
  repository.
- **Threshold sensitivity** (Figs. 3.4-3.5, eICU septic shock): build the slope cache with
  `scripts/slurm/eicu/run_eicu_sepsis_bayes_slope_cache_container.slurm` and
  `merge_eicu_sepsis_bayes_slope_cache.sh`, run the grid with
  `bash scripts/analysis/eicu/run_sensitivity_grid.sh lactate wbc platelet`
  (`eicu_sepsis_sensitivity_analysis.py`), then draw the figures with
  `scripts/analysis/eicu/plot_sensitivity_combined.py`.
- **Runtime and memory of MCMC vs bootstrap** (Section 3.3.3): `scripts/benchmarks/bench_backends.py`.

## Timeline convention

A prediction is made at an anchor time *a*: the integer `time_day` of a row (`time_hour` for
circulatory failure). Its features include all data recorded during the index day [*a*, *a*+1);
the trajectory features use the lookback window of width *W* that ends at the anchor. The gap δ and
the prediction window Δ are counted from *a*+1, so the outcome is evaluated in
[*a*+δ+1, *a*+δ+Δ+1] (thesis Section 2.4.2, Figure 2.1). Time is in days, and in hours for
circulatory failure.

## Trajectory types and thresholds

Each sampled trajectory is evaluated on a uniform grid and classified by its interval slopes
(`src/traj_features/backends/bayes/classify.py`; the bootstrap backend uses the same rules):

| Type (thesis) | Rule | Code |
| --- | --- | --- |
| Stable | at least a fraction α of the slopes ≤ τ_inc and \|mean slope\| ≤ τ_flat | `nonprogression` |
| Linear progression | at least a fraction α of the slopes > τ_inc | `linear` |
| Non-linear progression | mean of the upper half of the sorted slopes minus mean of the lower half > γ_gap | `nonlinear` |

α = 0.8 for all tasks. The thresholds are arguments of the classification functions and of the
trajectory scripts: τ_flat = `flat_thr`, τ_inc = `decline_thr`, γ_gap = `nonlinear_gap`.
`pos_flags_from_traj` is used for parameters whose increase indicates progression,
`flags_from_traj` for parameters whose decrease does (platelets, systolic BP).

The type score T̂(c) is the fraction of the sampled trajectories classified as type c. The three
indicators are evaluated independently, so the scores need not sum to one. The output columns keep
older names: `prob_stable` / `<parameter>_stable` for Stable, `prob_gradual_*` / `<parameter>_gradual`
for Linear progression, and `prob_rapid_*` / `<parameter>_rapid` for Non-linear progression
(bootstrap columns end in `_boot`). Despite the `prob_` prefix, these are type scores.

## Installation

```bash
pip install -e .[all]        # or .[bayes], .[bootstrap], .[evaluation]
```

The analyses also need `xgboost`, the preprocessing needs `duckdb`, and parquet files need
`pyarrow`. The thesis runs used the `pymc_env` conda environment on the TAU rack-shamir servers
(PyMC 5.23, nutpie 0.16, scikit-learn 1.7.2, xgboost 3.1.2).

The PhysioNet data (MIMIC-III, eICU, HiRID) requires credentialed access and is not part of
this repository. Nothing derived from it at patient level may be committed.

## Python API

```python
from traj_features.backends.bootstrap import BootstrapTraj, BootstrapConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj

# df: one row per measurement, columns patientid, time (days), time_window (anchor day), lab_value
types = ("stable", "linear_progression", "nonlinear_progression")
model = BootstrapTraj(BootstrapConfig(
    window_years=3.0,          # lookback window W, in the time unit of the data
    n_bootstrap=200,
    flat_thr=10.0, decline_thr=30.0, nonlinear_gap=20.0,   # tau_flat, tau_inc, gamma_gap
    class_func=pos_flags_from_traj,
    traj_types=types,
    label_map={"nonprogression": types[0], "linear": types[1], "nonlinear": types[2]},
    pids="patientid", values="lab_value", time_col="time", windowing_col="time_window",
))
scores = model.embed(df)       # one row per (patient, anchor): trajtype_<type>_prob columns
```

`BayesianTraj` / `BayesConfig` in `traj_features.backends.bayes` work the same way.

## Tests

```bash
pytest tests/test_smoke_backends.py -m "not slow"   # add the slow tests with -m "not slow or slow"
```

On the cluster: `sbatch scripts/slurm/smoke_test.slurm`.

## Repository layout

```text
src/traj_features/        trajectory backends (Bayesian MCMC, bootstrap)
src/analysis/             analysis template, shared utilities, boxplot pipeline (poster/)
scripts/*_preprocessing/  cohorts, outcomes and features per dataset
scripts/trajectory/       trajectory feature scripts per dataset
scripts/analysis/         prediction analyses, sensitivity analysis, tables, boxplots
scripts/figures/          qualitative-validation figures
scripts/rerun/            helpers of the thesis runs (significance, table rows, figure replay, sharding)
scripts/benchmarks/       MCMC vs bootstrap runtime benchmark
scripts/slurm/, docker/   cluster and container drivers for the trajectory computations
configs/poster/           boxplot configuration used for the thesis
docs/                     container guide, boxplot pipeline, provenance of the thesis results
tests/                    smoke tests
```

`scripts/ckd_*.py` belong to a separate All of Us CKD analysis and are not part of the thesis.
`docs/thesis/` holds early drafts of the methods text; the thesis itself is the reference.
