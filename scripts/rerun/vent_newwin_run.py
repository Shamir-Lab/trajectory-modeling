# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/vent_newwin_run.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Trajectory features (MCMC + bootstrap) for the eICU ventilation windows added by the corrected cohort.

Each affected stay gets its full P/F-ratio history up to its last added day, so the added windows are computed
exactly as in a full run; only the added windows are kept afterwards (see vent_newwin_append.py)."""
import subprocess, sys
from pathlib import Path
import pandas as pd

STAGE = Path('/tmp/tw_rerun'); REPO = Path('/home/gaga/tamarw1/trajectory-modeling')
OUT = STAGE / 'eicu_vent_newwin'; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / 'src'))
from traj_features.backends.bootstrap import BootstrapTraj, BootstrapConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj

key = ['stay_id', 'time_day']
new = pd.read_csv(STAGE / 'base/eicu/ventilator/ventilator_prediction_dataset.csv', usecols=key)
old = pd.read_csv('/home/gaga/data/physionet/eicu/ventilator/ventilator_prediction_dataset.csv', usecols=key).drop_duplicates()
t = new.merge(old.assign(_o=1), on=key, how='left'); t = t[t._o.isna()][key].drop_duplicates()
t.to_csv(OUT / 'added_windows.csv', index=False)
ts = pd.read_csv(STAGE / 'data_root/eicu/ventilator/pf_ratio_timeseries.csv')
mx = t.groupby('stay_id').time_day.max().rename('_maxd')
sub = ts.merge(mx, left_on='stay_id', right_index=True); sub = sub[sub.time_day <= sub._maxd].drop(columns='_maxd')
sub.to_csv(OUT / 'pf_ratio_timeseries.csv', index=False)
print(f'added windows: {len(t)} in {t.stay_id.nunique()} stays; history rows: {len(sub)}, stay-days: {sub.groupby(key).ngroups}', flush=True)

# Bootstrap: same backend and settings as the analysis template (eICUVentilatorConfig, BOOTSTRAP_N=200)
traj_types = ('prolonged_nonprogression', 'linear_decline', 'nonlinear')
inp = sub[['stay_id', 'time_days', 'time_day', 'pf_ratio']].rename(
    columns={'stay_id': 'patientid', 'time_days': 'time', 'time_day': 'time_window', 'pf_ratio': 'lab_value'})
inp = inp.dropna(subset=['lab_value', 'time', 'time_window']).sort_values(['patientid', 'time'])
model = BootstrapTraj(BootstrapConfig(
    window_years=3.0, n_bootstrap=200, flat_thr=10.0, decline_thr=30.0, nonlinear_gap=20.0,
    class_func=pos_flags_from_traj, traj_types=traj_types,
    label_map={'nonprogression': traj_types[0], 'linear': traj_types[1], 'nonlinear': traj_types[2]},
    pids='patientid', values='lab_value', time_col='time', windowing_col='time_window', n_jobs=24, progressbar=False))
boot = model.embed(inp).rename(columns={'patientid': 'stay_id', 'time_window': 'time_day'})
dst = ['pf_ratio_stable_boot', 'pf_ratio_gradual_boot', 'pf_ratio_rapid_boot']
boot = boot.rename(columns=dict(zip([f'trajtype_{x}_prob' for x in traj_types], dst)))
for c in dst:
    if c not in boot.columns: boot[c] = float('nan')
boot = boot[key + dst].drop_duplicates(key, keep='last')
boot.to_csv(OUT / 'new_pf_ratio_trajectory_probs_bootstrap_all.csv', index=False)
print(f'bootstrap windows computed: {len(boot)}', flush=True)

# MCMC: the production script on the same history, 4 sub-cohorts in parallel
procs = [subprocess.Popen([sys.executable, '-u', str(REPO / 'scripts/trajectory/eicu/ventilator_trajs.py'),
    '--input', str(OUT / 'pf_ratio_timeseries.csv'),
    '--pred-dataset', str(STAGE / 'base/eicu/ventilator/ventilator_prediction_dataset.csv'),
    '--output', str(OUT / f'merged_unused_{k}.csv'),
    '--probs-output', str(OUT / 'new_ventilator_trajectory_probs_bayes.csv'),
    '--n-batches', '1', '--cohort-splits', '4', '--cohort-index', str(k)], cwd=str(REPO),
    stdout=open(OUT / f'mcmc_{k}.log', 'w'), stderr=subprocess.STDOUT) for k in range(4)]
rcs = [p.wait() for p in procs]
print('ventilator_trajs rc', rcs, flush=True)
