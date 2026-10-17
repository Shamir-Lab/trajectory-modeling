# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/cf_shard_test.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/gaga/tamarw1/rerun_scripts')
import cf_shard as cs
cfa = cs.cfa
rng = np.random.RandomState(0)
n = 4000
df = pd.DataFrame({'id': rng.randint(0, 400, n), 'y': rng.binomial(1, 0.1, n)})
for c in ['a', 'b', 'lactate_stable', 'lactate_rapid', 'lactate_mean_12h']:
    df[c] = rng.randn(n) + df['y'] * 0.5
fs = {'S1': ['a'], 'S2': ['a', 'b'], 'T': ['lactate_stable', 'lactate_rapid'],
      'All': ['a', 'b', 'lactate_stable', 'lactate_rapid', 'lactate_mean_12h']}
ref, ref_sum, _ = cfa.run_cv(df, 'y', 'id', fs, n_repeats=3, n_folds=5, seed=920, train_n_jobs=4)
from sklearn.ensemble import RandomForestClassifier
orig = RandomForestClassifier.fit
def fit(self, X, y, sample_weight=None):
    self.n_jobs = 4
    return orig(self, X, y, sample_weight)
RandomForestClassifier.fit = fit
tmp = Path(tempfile.mkdtemp(dir='/tmp'))
for k in range(3):
    d = tmp / f's{k}'
    cfa.Parallel = cs.shard_parallel(k, 3, d, 4)
    r, s, dd = cfa.run_cv(df, 'y', 'id', fs, n_repeats=3, n_folds=5, seed=920, train_n_jobs=4)
    cfa.save_results(r, s, dd, d)
cfa.Parallel = cs.RealParallel
cs.merge(tmp / 'm', [tmp / f's{k}' for k in range(3)])
m = pd.read_parquet(tmp / 'm' / cs.FOLD_FILE)
ms = pd.read_csv(tmp / 'm' / 'mimic_circulatory_failure_summary.csv')
print('rows ref/merged', len(ref), len(m), 'same order of keys:',
      (ref[['Model', 'Feature Set', 'repeat', 'fold']].values == m[['Model', 'Feature Set', 'repeat', 'fold']].values).all())
print('max |diff| AUROC', np.abs(ref['ROC-AUC'].values - m['ROC-AUC'].values).max(),
      'AUPR', np.abs(ref['AUPR'].values - m['AUPR'].values).max())
a = ref_sum.reset_index(drop=True)[['Model', 'Feature Set', 'ROC_AUC_mean', 'AUPR_mean']]
b = ms[['Model', 'Feature Set', 'ROC_AUC_mean', 'AUPR_mean']]
print('summary same keys/order:', (a[['Model', 'Feature Set']].values == b[['Model', 'Feature Set']].values).all(),
      'max |diff|', np.abs(a[['ROC_AUC_mean', 'AUPR_mean']].values - b[['ROC_AUC_mean', 'AUPR_mean']].values).max())
