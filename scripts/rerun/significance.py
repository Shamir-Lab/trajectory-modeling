# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/significance.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""XGBoost feature-set comparisons for the thesis tables.

Paired two-sided Wilcoxon signed-rank tests on per-fold AUROC/AUPR (folds paired
explicitly on repeat and fold), Benjamini-Hochberg q-values across the six poster
comparisons within each dataset/task/metric, and mean +/- SD per feature set.

usage: significance.py <results_root> <out_csv>"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, '/home/gaga/tamarw1/trajectory-modeling/src')
from analysis.poster.io_loader import load_task_tables  # noqa: E402
from analysis.poster.feature_set_mapper import resolve_feature_mapping  # noqa: E402

CANON = ['sd', 'sd_traj', 'sd_traj_boot', 'sd_sum', 'sd_sum_traj', 'sd_sum_traj_boot']
PAIRS = [('sd', 'sd_traj'), ('sd', 'sd_traj_boot'), ('sd_traj', 'sd_traj_boot'),
         ('sd_sum', 'sd_sum_traj'), ('sd_sum', 'sd_sum_traj_boot'), ('sd_sum_traj', 'sd_sum_traj_boot')]


def bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1); return out


root, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for ds_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in ('mimic', 'eicu', 'hirid')):
    for task_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
        try:
            tt = load_task_tables(root, ds_dir.name, task_dir.name)
        except FileNotFoundError:
            continue
        f = tt.fold_df.copy()
        cols = {c.lower(): c for c in f.columns}
        f = f.rename(columns={cols[k]: k for k in ['model', 'feature_set', 'fold', 'repeat', 'auroc', 'aupr'] if k in cols})
        mapping = resolve_feature_mapping(sorted(f.feature_set.unique()), CANON, None, True)
        x = f[f.model.astype(str).str.lower().str.replace(' ', '') == 'xgboost']
        for metric in ['auroc', 'aupr']:
            block = []
            for a, b in PAIRS:
                if a not in mapping or b not in mapping:
                    continue
                A = x[x.feature_set == mapping[a]].set_index(['repeat', 'fold'])[metric]
                B = x[x.feature_set == mapping[b]].set_index(['repeat', 'fold'])[metric]
                m = pd.concat([A, B], axis=1, keys=['a', 'b']).dropna()
                p = wilcoxon(m.a, m.b).pvalue if len(m) and (m.a != m.b).any() else 1.0
                block.append(dict(dataset=ds_dir.name, task=task_dir.name, metric=metric, left=a, right=b,
                                  left_feature_set=mapping[a], right_feature_set=mapping[b], n_pairs=len(m),
                                  left_mean=m.a.mean(), left_sd=m.a.std(), right_mean=m.b.mean(), right_sd=m.b.std(),
                                  left_median=m.a.median(), right_median=m.b.median(),
                                  median_delta=(m.b - m.a).median(), p_value=p))
            if block:
                for r, q in zip(block, bh([r['p_value'] for r in block])):
                    r['q_value_bh'] = q
                rows += block
res = pd.DataFrame(rows)
res.to_csv(out_csv, index=False)
print(f'wrote {out_csv}: {len(res)} rows, {res.groupby(["dataset", "task"]).ngroups} dataset-task combinations')
