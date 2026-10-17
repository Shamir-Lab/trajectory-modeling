# Poster Visualization Pipeline

This pipeline generates poster-ready figure panels from existing cross-validation outputs:

- main panel: AUROC + AUPR boxplots (side-by-side), stratified by model type
- best-model panel: AUROC + AUPR boxplots for the best model (selected by AUPR), with significance stars

## Entry point

- `scripts/analysis/poster/run_poster_pipeline.py`

## Config

Use YAML configs in `configs/poster/`.

Default:
- `configs/poster/mimic_eicu_default.yaml` (MIMIC-III, eICU and HiRID; all tasks found under
  `results/{dataset}/`). The thesis boxplots were made with this config, with `inputs.results_root`
  and `outputs.root` pointing to the gathered results of the thesis run.

```bash
python scripts/analysis/poster/run_poster_pipeline.py --config configs/poster/mimic_eicu_default.yaml
```

## Input contract

For each dataset/task folder under `results/{dataset}/{task}`:

- `{task}_cv_results.csv`
- `{task}_cv_results_fold_metrics.csv`
- `{task}_cv_results_metadata.json` (optional)

The circulatory-failure scripts write `{dataset}_{task}_summary.csv` and
`{dataset}_{task}_fold_results.parquet` instead; both naming schemes are read.

Expected fold metrics columns:

- `model`, `feature_set`, `fold`, `repeat`, `auroc`, `aupr`

## Selected canonical feature sets

- `sd` = Static + Dynamic
- `sd_sum` = Static + Dynamic + Summary
- `sd_traj` = Static + Dynamic + Trajectory
- `sd_sum_traj` = Static + Dynamic + Summary + Trajectory
- `sd_traj_boot` = Static + Dynamic + Trajectory (Bootstrap)
- `sd_sum_traj_boot` = Static + Dynamic + Summary + Trajectory (Bootstrap)

For multi-marker tasks, mapping prefers multi-marker labels where available.

## Output

Artifacts are written to:

- `results/poster/{dataset}/{task}/{run_name}/`

Files:

- `panel_main_auroc_aupr.png/.pdf`
- `panel_best_model_stars.png/.pdf`
- `metrics_selected_models.csv`
- `fold_metrics_selected_models.csv`
- `best_model_summary.csv`
- `best_model_significance.csv`
- `manifest.json`
