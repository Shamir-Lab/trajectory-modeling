# How the thesis results were produced

The results in the thesis come from runs on the TAU rack-shamir servers (rack-shamir4, 5 and 8;
the `pymc_env` environment does not run on rack-shamir3, which lacks AVX). The result files are
in the server checkout's `results/` directory, which is not tracked. Its
`results/rerun_2026-10/README.md` is the detailed record; this page summarizes it.

## Prediction results (Tables 3.1-3.4, S1, the boxplots and the results text)

1. **2026-10-10/11, commit 4c57801.** All five tasks were preprocessed again with the corrected
   cohort definitions, and the 11 dataset-task analyses were run. The inputs were staged under
   `/tmp/tw_rerun/` on each server; the shared data root was not modified.
2. **2026-10-14, commit 1f56f72.** The 10 CV repeats had reused the same five patient folds,
   because `GroupKFold` ignores row order. The fix draws new folds in every repeat
   (`shuffle=True, random_state=920 + repeat`), and all 11 analyses were repeated with unchanged
   trajectory features. Tables 3.2-3.4, S1, the results text, both abstracts and the boxplots use
   this run.

The analysis commands are listed in the [README](../README.md#3-prediction-analyses). The
MIMIC-III circulatory-failure analysis ran in six shards across the three servers
(`scripts/rerun/cf_shard.py run`) that were merged afterwards (`cf_shard.py merge`). Then
`run_poster_pipeline.py` produced the boxplots, `build_summary_tables.py` the cohort and metrics
tables, `scripts/rerun/significance.py` the paired Wilcoxon tests, and `gen_tables.py` /
`gen_s1.py` the LaTeX rows of Tables 3.2-3.4 and S1. The boxplot configuration was
`configs/poster/mimic_eicu_default.yaml` with `results_root` and `outputs.root` pointing to the
gathered results.

## Trajectory features

The October runs reused the cached trajectory features in the data root, with these exceptions:

- **Circulatory failure:** the merged datasets were rebuilt from the per-marker caches for the new
  cohorts (`scripts/rerun/merge_cf_from_caches.py`), with the same merge as the pipeline.
- **HiRID circulatory failure, bootstrap:** recomputed for the whole new cohort in 24 splits with
  `scripts/trajectory/hirid/circulatory_failure_trajs_bootstrap.py`, then combined with
  `scripts/rerun/concat_hirid_boot.py`. The MCMC features were kept.
- **eICU ventilation:** MCMC and bootstrap features computed only for the windows that the
  corrected cohort added, each from the stay's full history (`scripts/rerun/vent_newwin_run.py`,
  then `vent_newwin_append.py`).
- **eICU ACLF:** features computed for the stays that the corrected cohort added, and appended to
  the caches with `scripts/rerun/append_newstays.py`.
- **Other tasks:** the merged datasets were rebuilt from the new preprocessed rows and the cached
  features (`scripts/rerun/rebuild_traj.py`); nothing was recomputed.

The cached circulatory-failure features were computed before October with the container drivers
in `docker/` (MCMC: `run_cf_batches.sh`; MIMIC-III bootstrap: `run_mimic_cf_bootstrap_batches.sh`;
see [CONTAINER_GUIDE.md](CONTAINER_GUIDE.md)).

## Figures

- **Qualitative validation (Fig. 3.1, Supplementary Figs. S1-S4).** The figures show the windows
  selected on 2026-09-21. They were redrawn for the thesis notation (commits 9b364a5, 2c42071,
  66c4e58, 5f53f05 and 8c15b8d) by replaying that selection with `scripts/rerun/qv_replay.py`; all 126
  windows, scores and values are identical. The manifest is on the server at
  `results/_archive_pre_rerun_2026-10/figures/qualitative_validation/selected_examples_manifest.csv`.
  It contains patient-level values, so it must not be committed.
- **Threshold sensitivity (Figs. 3.4-3.5).** Produced with `run_sensitivity_grid.sh`,
  `eicu_sepsis_sensitivity_analysis.py` and `plot_sensitivity_combined.py` before the CV-fold fix,
  and not rerun afterwards.

## Runtime benchmark (Section 3.3.3)

`scripts/benchmarks/bench_backends.py` was run on rack-shamir4 in September 2026 on eICU windows
(septic-shock lactate; circulatory-failure lactate and systolic BP), with g++ available to
PyTensor. Each backend processed the same windows on one CPU core with the production settings.
The rows used in the thesis are the matched single-core pairs.
