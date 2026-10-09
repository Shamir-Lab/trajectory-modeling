# Trajectory pipeline: extracted mathematics, logic and configuration

Raw extraction from the implementation. No manuscript prose, no LaTeX formatting.
Companion to `docs/thesis/methods_trajectory_modeling.tex`, which is the written-up
version of the same material.

Terminology: the code says "biomarker" throughout. This document says **clinical
parameter** or **physiological indicator** for the same object. Code identifiers are
quoted verbatim where they need to be looked up.

Source of truth, all paths relative to the repository root:

| Concern | File |
| --- | --- |
| Window construction, parallel driver | `src/traj_features/backends/bayes/pipeline.py` |
| Normalization, basis, model, sampling | `src/traj_features/backends/bayes/sampling.py` |
| Decision rules, probability aggregation | `src/traj_features/backends/bayes/classify.py` |
| Config dataclass and entry point | `src/traj_features/backends/bayes/model.py` |
| Bootstrap replicates and priority collapse | `src/traj_features/backends/bootstrap/classify.py` |
| Bootstrap window construction | `src/traj_features/backends/bootstrap/pipeline.py` |
| Per-task settings | `scripts/trajectory/{mimic,eicu,hirid}/*.py` |

Verified 2026-09-05 against the working tree on branch `stupidmistakes`.

---

## Part 1 — Answers to the three questions raised

### 1.1 Time normalization to the unit interval

Computed per window in `_scale_vals_and_grid` and its caller `_sample_post_trajs_scaled`,
before any fitting.

```
t_min = min{ t_j : j in window }
t_max = max{ t_j : j in window }
Delta = max(1e-8, t_max - t_min)          # variable named "span" in the code
s_j   = (t_j - t_min) / Delta             # s_j in [0, 1]
```

Properties that matter:

- The map is anchored to the **observed support** of the window, not to the nominal
  horizon `[a - W, a]`. `s = 0` is the first available measurement inside the window
  and `s = 1` is the last, whatever fraction of `W` they actually cover.
- A window whose measurements span less than `W` is stretched to the full unit
  interval. Slopes stay correct in native units because `Delta` is carried through the
  inverse map, but the effective resolution of the fit follows the observed span.
- The floor at `1e-8` prevents division by zero when all measurements in a window
  share one timestamp.
- Inverse map, applied after sampling: `g = t_min + u * Delta` for a normalized grid
  point `u`.

### 1.2 MCMC hyperparameters

**The code does not run 4 chains / 500 tuning / 1000 draws.** That combination exists
only as the dataclass default in `BayesConfig` and no script uses it. There are three
layers of defaults; every production script overrides the sampling ones.

| Layer | draws per chain | tuning | chains | total draws S | target_accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BayesConfig` dataclass default | 1000 | 500 | 4 | 4000 | 0.95 |
| `compute_time_varying_trajectory_covariates_parallel` signature | 1000 | 1000 | 4 | 4000 | 0.95 |
| HiRID circulatory failure | 300 | 300 | 4 | 1200 | 0.995 |
| All other task scripts | 200 | 300 | 4 | 800 | 0.99 |

The last row covers MIMIC-IV and eICU circulatory failure, plus AKI, sepsis, liver,
ventilator and the eICU all-parameter sepsis script.

- Every script sets `chains=4`.
- `cores=1` is the dataclass default, so the four chains run sequentially rather than
  in parallel. All four are still retained.
- Retained sample size is `S = chains * draws_per_chain`, because the posterior is
  stacked over both the chain and draw dimensions:
  `trace.posterior["beta"].stack(sample=("chain","draw"))`.
- Report `S = 1200` for HiRID and `S = 800` elsewhere.
- `df_basis` is 4 in the dataclass but 5 in the pipeline signature and in every script.
- `min_points_per_window` is 5 in the dataclass, 4 in most scripts, 6 for AKI.
- `grid_freq` is 12 in the dataclass, 3 for HiRID circulatory failure, 2 everywhere else.

### 1.3 Bootstrap priority logic

**Yes, the bootstrap backend forces a single label per replicate.** The multi-label
rule is evaluated first, then collapsed by strict priority in
`classify_trajectory_from_spline`:

```
flags = class_func(spline(time_grid), time_grid, flat_thr, decline_thr, nonlinear_gap, label_map)

if   flags[nonlinear_label] -> "nonlinear"
elif flags[linear_label]    -> "linear"
else                        -> "nonprogression"
```

The `else` branch is unconditional, so `nonprogression` is the fallback bucket and is
assigned even when the stability test itself returned false. Probabilities are relative
frequencies over replicates that produced a label:

```
p_c = count(c) / total
total = number of replicates that produced a label
p_nonprogression + p_linear + p_nonlinear = 1
```

Two degenerate paths return `(1/3, 1/3, 1/3)`: fewer than four observations in the
window, and every replicate failing. The denominator excludes replicates skipped for
having fewer than four distinct times and those whose spline fit raised, so it is
generally smaller than the requested replicate count `n_bootstrap`.

This is a genuine structural difference from the Bayesian backend, which emits three
independent marginal probabilities that do not sum to one. The two backends are not on
the same scale and their values must not be compared one for one.

---

## Part 2 — Stage-by-stage extraction

Let `B` be the set of clinical parameters, indexed by `b`. Let `W` be the lookback
horizon expressed in the units of the windowing column. The parameter is named
`window_years` in the code regardless of the actual unit: it is hours for circulatory
failure and days for sepsis, AKI, liver and ventilator. All stages run independently
per pair of stay and clinical parameter.

### Stage 1 — Windowing

Each measurement carries an exact time `t_j` from `time_col` and a coarse alignment
index `theta_j` from `windowing_col`. Anchors are the distinct values of `theta` for
that stay.

```
Window(a) = { (t_j, y_j) : a - W <= theta_j <= a }
n_a       = |Window(a)|
retained  iff n_a >= n_min
```

Right-closed at `a`, so no information after the anchor enters the window. Anchors
failing the count test are dropped and emit nothing.

### Stage 2 — Normalization and standardization

```
s_j   = (t_j - t_min) / Delta                 # see 1.1
mu    = mean(y)
sigma = std(y, ddof=0)                        # sigma := 1 if std == 0
z_j   = (y_j - mu) / sigma
```

### Stage 3 — Basis construction

Cubic B-spline, degree 3, order 4, built by `patsy` with `include_intercept=True`.

```
d_eff      = min( d, max(4, n_a - 1), 8 )
n_interior = d_eff - 4
interior knots = quantiles of {s_j} at levels r / (n_interior + 1),  r = 1..n_interior
boundary knots = 0 and 1, each with multiplicity 4
X = [ 1 | B_1(s) ... B_d_eff(s) ]             # shape (n_a, d_eff + 1)
```

The leading column of ones is added by the formula parser on top of a basis that
already spans the constant. The basis is a partition of unity, so that column equals
the sum of the others:

```
rank(X) = d_eff
X @ beta invariant along v = (1, -1, ..., -1)
```

The coefficient vector is unidentified along `v`. The fitted mean is identified, and
only the fitted mean propagates forward, so no reported feature is affected. The
proper Gaussian prior keeps the posterior proper.

With `d = 5`: `n_interior = 1` once the window holds at least six observations, giving
one interior knot at the median normalized time. With `d_eff = 4`: no interior knots,
a single cubic segment.

### Stage 4 — Model

```
beta  ~ Normal(0, 1),  shape = d_eff + 1
sigma ~ HalfNormal(1)
z     ~ Normal(X @ beta, sigma)
```

Priors are weakly informative on the standardized scale: the response has unit
empirical variance by construction and the basis is a partition of unity, so a
coefficient of magnitude one is a deviation of one within-window standard deviation.

### Stage 5 — Sampling

NUTS. Backend selection order is nutpie, then numpyro or blackjax under JAX, then the
reference PyMC sampler, each falling back to the next on exception. All three target
the identical posterior. Seed is the module constant 920, fixed for every window, so
the Bayesian feature matrix is deterministic given the data and the settings.

### Stage 6 — Evaluation grid and back-transform

`f` is `grid_freq`, a density in points per unit of the time column.

```
n_g = max( 3, f, round(f * Delta) )
u_k = (k - 1) / (n_g - 1),   k = 1..n_g
X~  = basis specification re-evaluated at {u_k}
y_k^(s) = sigma * (X~ @ beta^(s))_k + mu
g_k     = t_min + u_k * Delta
```

Two points:

- These are draws of the **conditional mean**, not posterior predictive draws.
  `sigma` is marginalized out but observation noise is not re-injected, so the
  classification acts on smooth curves.
- `X~` is a fresh design whose interior knots are quantiles of the uniform grid, not
  of `{s_j}`. Fitting and prediction bases coincide only when `n_interior = 0` or the
  observed normalized times are close to uniform.

Worked values of the grid formula:

| f | Delta | n_g |
| ---: | ---: | ---: |
| 2 | 12 | 24 |
| 3 | 12 | 36 |
| 2 | 0.5 | 3 |
| 12 | 2 | 24 |

### Stage 7 — Slope field

```
K       = n_g - 1
delta   = Delta / (n_g - 1)                       # uniform spacing, native units
m_k^(s) = (y_{k+1}^(s) - y_k^(s)) / delta         # parameter units per time unit
m_bar^(s) = mean_k m_k^(s)
```

### Stage 8 — Decision rules

`rho = 0.8` is a literal inside the classifier, not a config field.

Decreasing direction, `flags_from_traj`, used where deterioration is a falling value:

```
pi_flat  = mean_k [ m_k >= tau_dec ]
pi_dec   = mean_k [ m_k <  tau_dec ]              # = 1 - pi_flat
F_stable = (pi_flat >= rho) AND (m_bar >= tau_flat)
F_linear = (pi_dec  >= rho)
```

Increasing direction, `pos_flags_from_traj`, the exact reflection under
`m -> -m`, `tau -> -tau`:

```
pi_flat  = mean_k [ m_k <= tau_dec ]
pi_inc   = mean_k [ m_k >  tau_dec ]
F_stable = (pi_flat >= rho) AND (m_bar <= tau_flat)
F_linear = (pi_inc  >= rho)
```

Because the two interval fractions are complementary, `F_stable` and `F_linear` cannot
both hold at `rho = 0.8`. They could only co-occur if `rho <= 0.5`.

Nonlinearity, shared by both rules:

```
sort m ascending -> m_(1) <= ... <= m_(K)
h    = floor(K / 2)
low  = mean of m_(1..h)
high = mean of m_(h+1..K)
F_nonlinear = |high - low| > gamma
```

This is a contrast between halves of the empirical slope distribution and is invariant
to the order of the intervals. It measures spread in the rate of change, not curvature:
an accelerating monotone curve, a decelerating one and an oscillating one with the same
slope multiset all score identically. It requires `K >= 2`, hence `n_g >= 3`.

A third rule, `mmse_flag_from_traj`, adds a fourth threshold hardcoded at `-0.5`,
partitions the slope axis into four regions, and takes the linear flag as a disjunction
over the two change regions. No intensive-care task uses it.

### Stage 9 — Aggregation

```
p_c = (1 / S) * sum_s F_c^(s)      for c in {stable, linear, nonlinear}
```

Monte Carlo estimate of the posterior probability that the indicator holds. Monte Carlo
standard error is approximately `sqrt( p (1 - p) / S_eff )` with `S_eff` the effective
sample size of the indicator sequence, bounded above by `1 / sqrt(4 S_eff)`.

The three indicators are separate functionals of the same curve, so in general the
three probabilities do not sum to one. Generic labels are renamed per task through
`label_map`, and only keys listed in `traj_types` are emitted, as
`trajtype_<label>_prob`. A vectorized path computes all draws at once for the three
known rules; a per-draw loop is the fallback for a custom rule. The two are
algebraically identical.

### Stage 10 — Failure handling

A window whose sampling raises under all backends returns an empty result and is
emitted with missing feature values rather than dropped. A batch exceeding
`batch_timeout_s`, default 14400 seconds, is recorded as missing in its entirety. The
resulting missingness is not guaranteed independent of trajectory shape, since
degenerate and ill-conditioned windows are the ones most likely to fail.

### Stage 11 — Bootstrap backend

Window construction differs from the Bayesian path: window ids are
`floor(windowing_col)` and the window is `[window_id - W, window_id]`.

Per replicate:

```
1. resample the n_a indices uniformly with replacement
2. sort by time
3. average values sharing a timestamp   -> n_u distinct times
4. skip the replicate if n_u < 4
5. fit UnivariateSpline with k = min(3, n_u - 1), smoothing s
6. evaluate on time_grid, apply the Stage 8 rules, collapse by priority
```

With `smoothing=None`, which every script passes, SciPy sets the smoothing budget to
`n_u`, so the fit satisfies:

```
sum_j ( y_j - f(t_j) )^2 <= n_u
```

The evaluation grid is `linspace(t_min, t_max, grid_freq)` in **native units**, taken
from the full window rather than from the replicate, so replicates that drop an
endpoint are extrapolated under the SciPy default `ext=0`. Neither time normalization
nor value standardization is applied anywhere in this backend.

---

## Part 3 — Configuration surface

### 3.1 Structural settings per task

| Task | Window units | W default | n_min | grid_freq | df_basis | draws | tuning | target_accept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Circulatory failure, HiRID | hours | 12 | 4 | 3 | 5 | 300 | 300 | 0.995 |
| Circulatory failure, MIMIC-IV and eICU | hours | 12 | 4 | 2 | 5 | 200 | 300 | 0.99 |
| Sepsis | days | 3 | 4 | 2 | 5 | 200 | 300 | 0.99 |
| AKI | days | 7 | 6 | 2 | 5 | 200 | 300 | 0.99 |
| Liver | days | 5 | 4 | 2 | 5 | 200 | 300 | 0.99 |
| Ventilator | days | 3 | 4 | 2 | 5 | 200 | 300 | 0.99 |

### 3.2 Thresholds per clinical parameter

Native units per unit of time, hours for circulatory failure and days elsewhere. Sign
follows the direction of deterioration.

| Task and cohort | Parameter | Deterioration | tau_flat | tau_dec | gamma | Rule |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Circulatory failure, HiRID | lactate | increasing | 0.03 | 0.12 | 0.18 | pos |
| Circulatory failure, HiRID | heart rate | increasing | 1.2 | 4.5 | 7.0 | pos |
| Circulatory failure, HiRID | systolic pressure | decreasing | -1.2 | -4.5 | 7.0 | neg |
| Circulatory failure, MIMIC-IV and eICU | lactate | increasing | 0.04 | 0.15 | 0.2 | pos |
| Circulatory failure, MIMIC-IV and eICU | heart rate | increasing | 1.5 | 6.0 | 10.0 | pos |
| Circulatory failure, MIMIC-IV and eICU | systolic pressure | decreasing | -1.5 | -6.0 | 8.0 | neg |
| Sepsis, MIMIC-IV and eICU | lactate | increasing | 0.1 | 0.3 | 0.5 | pos |
| Sepsis, MIMIC-IV and eICU | white cell count | increasing | 1.0 | 3.0 | 2.0 | pos |
| Sepsis, MIMIC-IV and eICU | platelets | decreasing | -20.0 | -50.0 | 30.0 | neg |
| AKI, MIMIC-IV and eICU | creatinine | increasing | 0.1 | 0.3 | 0.5 | pos |
| Liver, MIMIC-IV and eICU | bilirubin | increasing | 0.5 | 1.0 | 3.0 | pos |
| Ventilator, MIMIC-IV and eICU | P/F ratio | see defect 4.3 | 10.0 | 30.0 | 20.0 | pos |

"pos" is `pos_flags_from_traj`, "neg" is `flags_from_traj`. All thresholds for AKI,
liver and ventilator are command-line defaults rather than table entries, so they can
be overridden per run.

### 3.3 Bootstrap settings

| Setting | Value in the circulatory-failure bootstrap scripts |
| --- | --- |
| `n_bootstrap` | 1000, command-line default |
| `smoothing` | `None`, so the budget is `n_u` |
| `min_points_per_window` | 4 |
| `grid_freq` | 2, meaning two grid points, see defect 4.1 |
| seeding | none, see defect 4.2 |

---

## Part 4 — Defects found during extraction

Listed most severe first. Each was confirmed by running the relevant code path, not by
reading alone.

### 4.1 The bootstrap nonlinear class can never fire

All six circulatory-failure bootstrap scripts pass `grid_freq=2`. In
`src/traj_features/backends/bootstrap/pipeline.py` the grid is
`np.linspace(t_min, t_max, grid_freq)`, so `grid_freq` is the literal **number of grid
points**, not a density as in the Bayesian backend. Two grid points give:

```
K    = 1
h    = 1 // 2 = 0
low  = mean of an empty slice = nan
F_nonlinear = |high - nan| > gamma  ->  False, always
```

Warnings are suppressed at module import in `classify.py`, so the empty-slice warning
is never printed. Confirmed numerically: a trajectory dropping from 120 to 20 over the
window returns `nonlin=False` with `fast=nan`.

Consequence beyond the nonlinear class: with two grid points the curve is evaluated
only at the window endpoints, so the fitted spline shape is discarded entirely and the
whole classification reduces to thresholding one secant slope,
`(y(t_max) - y(t_min)) / (t_max - t_min)`. Stable and linear then partition the
outcome space and the bootstrap result is a two-point difference classifier.

The Bayesian backend is not affected. It enforces `n_g = max(3, f, round(f * Delta))`
and the vectorized aggregation raises explicitly when the half is empty.

### 4.2 The bootstrap backend is unseeded

`process_patient_window` never passes `random_state` to `bootstrap_trajectory_probs`,
which then constructs a fresh unseeded `np.random.RandomState()` per window. Bootstrap
results are not reproducible across reruns. The Bayesian backend is seeded with the
fixed constant 920.

### 4.3 AKI, liver and ventilator emit all-zero probabilities

Six scripts pass `class_func=pos_flags_from_traj` together with
`traj_types=('prolonged_nonprogression', 'linear_decline', 'nonlinear')` and **no**
`label_map`:

```
scripts/trajectory/mimic/aki_trajs.py
scripts/trajectory/mimic/liver_trajs.py
scripts/trajectory/mimic/ventilator_trajs.py
scripts/trajectory/eicu/aki_trajs.py
scripts/trajectory/eicu/liver_trajs.py
scripts/trajectory/eicu/ventilator_trajs.py
```

With `label_map=None`, `_posterior_feature_probs_from_samples` falls back to the
default map registered for `pos_flags_from_traj`, whose values are `stable`,
`slow_decline` and `rapid_decline`. None of those match the three requested type names,
so the guard `if mapped_label in counts` never passes and every count stays at zero:

```
_posterior_feature_probs_from_samples(rising_curves, grid, 0.1, 0.3, 0.5,
    pos_flags_from_traj,
    ('prolonged_nonprogression', 'linear_decline', 'nonlinear'), label_map=None)

-> {'trajtype_prolonged_nonprogression_prob': 0.0,
    'trajtype_linear_decline_prob':           0.0,
    'trajtype_nonlinear_prob':                0.0}
```

The same call with a matching `label_map` returns a stable probability of 1.0 on the
same input, so the curves classify correctly and only the label lookup fails. The
circulatory-failure and sepsis scripts all pass an explicit `label_map` and are
unaffected.

Two fixes are available: pass the matching `label_map` in those six scripts, or change
`traj_types` to the positive-direction names. Either makes the guard pass. A defensive
change in `classify.py` to raise when no requested type is ever matched would surface
this class of mistake instead of silently emitting zeros.

Separately, the ventilator task treats a rising P/F ratio as the progression class,
since it uses the increasing-direction rule with positive thresholds. A rising P/F
ratio is clinical improvement, so the direction convention there should be confirmed
against intent before the outputs are used.

### 4.4 Prediction basis is re-derived on the grid

Covered in Stage 6. Knots at prediction time come from quantiles of the uniform grid,
not from the observed times, so the fitting and prediction bases differ whenever the
window has an interior knot and the sampling is irregular. Small misspecification,
vanishes for regularly sampled indicators.

### 4.5 Corrections owed to the Methods draft

`docs/thesis/methods_trajectory_modeling.tex` describes the bootstrap backend as
seeded and reproducible, which contradicts 4.2, and describes the nonlinear class as
reachable in that backend, which contradicts 4.1. Those two passages need editing, or
the underlying settings need fixing first. Not yet applied.
