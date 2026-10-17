# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/vent_newwin_append.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Add the trajectory features computed for the added eICU ventilation windows to the analysis inputs.
Only added windows that have no cached value are filled; every cached value is kept unchanged."""
from pathlib import Path
import pandas as pd

STAGE = Path('/tmp/tw_rerun'); OUT = STAGE / 'eicu_vent_newwin'; B = STAGE / 'base/eicu/ventilator'
OLD = Path('/home/gaga/data/physionet/eicu/ventilator')
key = ['stay_id', 'time_day']
t = pd.read_csv(OUT / 'added_windows.csv')

def write(df, p):
    if p.is_symlink():
        p.unlink()
    df.to_csv(p, index=False)

# MCMC cache (per measurement rows) and the merged file the loader reads first
nb = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('new_ventilator_trajectory_probs_bayes_cohort*.csv'))], ignore_index=True)
nb = nb.merge(t, on=key)
cache = pd.read_csv(OLD / 'ventilator_trajectory_probs_bayes.csv')
assert set(nb.columns) == set(cache.columns), set(nb.columns) ^ set(cache.columns)
nb = nb.merge(cache[key].drop_duplicates().assign(_h=1), on=key, how='left'); nb = nb[nb._h.isna()].drop(columns='_h')
write(pd.concat([cache, nb[cache.columns]], ignore_index=True), B / 'ventilator_trajectory_probs_bayes.csv')
tc = ['prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement']
m = pd.read_csv(B / 'ventilator_trajectory_probs.csv', low_memory=False)
before = m.merge(t, on=key)[tc[0]].notna().mean()
w = nb[key + tc].drop_duplicates(key, keep='first')
m2 = m.merge(w, on=key, how='left', suffixes=('', '_new'))
for c in tc:
    m2[c] = m2[c].fillna(m2[f'{c}_new'])
m2 = m2.drop(columns=[f'{c}_new' for c in tc])
assert len(m2) == len(m) and (m2[key].values == m[key].values).all()
write(m2, B / 'ventilator_trajectory_probs.csv')
after = m2.merge(t, on=key)[tc[0]].notna().mean()
print(f'MCMC: {nb.groupby(key).ngroups:,} added windows filled; coverage of added windows {before:.1%} -> {after:.1%}; '
      f'all windows {m[tc[0]].notna().mean():.1%} -> {m2[tc[0]].notna().mean():.1%}')

# Bootstrap cache (one row per window), read directly by the loader
ob = pd.read_csv(OLD / 'pf_ratio_trajectory_probs_bootstrap.csv')
nbb = pd.read_csv(OUT / 'new_pf_ratio_trajectory_probs_bootstrap_all.csv').merge(t, on=key)
nbb = nbb.merge(ob[key].assign(_h=1), on=key, how='left'); nbb = nbb[nbb._h.isna()].drop(columns='_h')
aug = pd.concat([ob, nbb[ob.columns]], ignore_index=True)
assert not aug.duplicated(key).any()
write(aug, B / 'pf_ratio_trajectory_probs_bootstrap.csv')
cov = lambda d: t.merge(d[key + ['pf_ratio_stable_boot']], on=key, how='left')['pf_ratio_stable_boot'].notna().mean()
print(f'bootstrap: {len(nbb):,} added windows filled; coverage of added windows {cov(ob):.1%} -> {cov(aug):.1%}')
