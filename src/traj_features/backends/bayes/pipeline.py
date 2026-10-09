from __future__ import annotations
import numpy as np, pandas as pd
import os
import tempfile
from joblib import Parallel, delayed
from .sampling import _sample_post_trajs_scaled
from .classify import _posterior_feature_probs_from_samples, flags_from_traj


def _patch_pytensor_compile_function_src() -> None:
    """Redirect pytensor's NamedTemporaryFile(delete=False) calls to TMPDIR,
    using content-hash filenames so identical Numba dispatch sources reuse the
    same file instead of creating a new one per invocation.  Files land in
    TMPDIR (routed to /dev/shm by SLURM scripts) so they never touch NFS quota.
    The SLURM EXIT trap removes the /dev/shm dir on job end."""
    try:
        import pytensor.link.utils as _ptu
        if getattr(_ptu.compile_function_src, "_patched_tmpdir", False):
            return
        import hashlib as _hashlib
        from typing import cast as _cast, Callable as _Callable, Any as _Any

        def _redirect(
            src: str,
            function_name: str,
            global_env: "dict[_Any, _Any] | None" = None,
            local_env: "dict[_Any, _Any] | None" = None,
        ) -> "_Callable":
            tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
            # Deterministic name: same source → same file → no duplicates across workers
            src_hash = _hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()
            fname = os.path.join(tmpdir, f"pytensor_{src_hash}.py")
            if not os.path.exists(fname):
                with open(fname, "w") as _f:
                    _f.write(src)
            if global_env is None:
                global_env = {}
            if local_env is None:
                local_env = {}
            mod_code = compile(src, fname, mode="exec")
            exec(mod_code, global_env, local_env)
            res = _cast(_Callable, local_env[function_name])
            res.__source__ = src  # type: ignore
            return res

        _redirect._patched_tmpdir = True  # type: ignore
        _ptu.compile_function_src = _redirect
    except Exception:
        pass


_patch_pytensor_compile_function_src()


def _apply_worker_process_patches() -> None:
    """PyTensor/loky worker-safety patches + BLAS thread limits.

    Must be called inside every joblib worker process (loky spawns fresh
    interpreters, so patches applied only in the parent process don't apply
    here) before any PyMC/PyTensor sampling call. Shared by `_window_worker`
    and any other per-window Bayesian worker (e.g. a raw-slope cache builder)
    that needs the same SLURM-array-job race-condition fixes.
    """
    # Patch add_key to be idempotent: Loky reuses worker processes so a key
    # compiled in batch N is still in memory when batch N+1 runs.
    # Workers share the compiledir (set by PYTENSOR_FLAGS) so compiled Numba
    # dispatch functions are cached once and reused across all workers.
    try:
        from pytensor.link.c.cmodule import KeyData as _KD
        if not getattr(_KD.add_key, "_idempotent", False):
            _orig_add_key = _KD.add_key
            def _idempotent_add_key(self, key, save_pkl=True):
                if key not in self.keys:
                    _orig_add_key(self, key, save_pkl=save_pkl)
            _idempotent_add_key._idempotent = True
            _KD.add_key = _idempotent_add_key
    except Exception:
        pass

    # Patch _get_from_hash to treat a corrupted/stale key.pkl as a cache miss
    # rather than raising AssertionError.  When multiple loky workers race to
    # write the same compiledir entry, the losing worker may read a partially-
    # written key.pkl whose key doesn't match — returning None here causes
    # module_from_key to fall through to recompile under a proper file lock.
    try:
        from pytensor.link.c.cmodule import ModuleCache as _MC
        if not getattr(_MC._get_from_hash, "_safe", False):
            _orig_gfh = _MC._get_from_hash
            def _safe_gfh(self, module_hash, key):
                try:
                    return _orig_gfh(self, module_hash, key)
                except (AssertionError, EOFError, OSError):
                    # Treat any failure to read key.pkl as a cache miss so
                    # module_from_key falls through to recompile under a file
                    # lock. EOFError happens when concurrent workers truncate
                    # the pickle; OSError covers other fs-level failures.
                    return None
            _safe_gfh._safe = True
            _MC._get_from_hash = _safe_gfh
    except Exception:
        pass

    # Disable the broken-eq check that triggers the AssertionError above;
    # it exists only to catch Op.__hash__/__eq__ bugs, not needed at runtime.
    try:
        from pytensor.link.c.cmodule import get_module_cache as _gmc
        _gmc().check_for_broken_eq = False
    except Exception:
        pass

    # Re-apply the compile_function_src patch in case this worker was spawned
    # fresh (spawn/forkserver start method) and imported a clean module state.
    _patch_pytensor_compile_function_src()
    # Keep BLAS single-threaded inside each worker to avoid oversubscription.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _window_worker(
    window_df: pd.DataFrame,
    flat_thr=-1.0, decline_thr=-2.0, nonlinear_gap=3.0,
    df_basis=5, n_samples=1000, tune=1000, min_points=5, grid_freq=12,
    class_func=flags_from_traj, values='lab_value', time_col='time',
    traj_types=('prolonged_nonprogression','linear_decline','nonlinear'),
    label_map=None,
    *,
    sampler: str = 'pymc',
    chains: int = 4,
    cores: int = 1,
    progressbar: bool = False,
    chain_method: str = "vectorized",
    target_accept: float = 0.95,

):
    _apply_worker_process_patches()

    try:
        tg, ys = _sample_post_trajs_scaled(
            window_df, df_basis, n_samples, tune, min_points, grid_freq,
            target_accept, values, time_col,
            sampler=sampler, chains=chains, cores=cores, progressbar=progressbar,
            chain_method=chain_method
        )
        return _posterior_feature_probs_from_samples(
            ys, tg, flat_thr, decline_thr, nonlinear_gap, class_func, traj_types, label_map=label_map
        )
    except Exception as e:
        print(f"[WARNING] Window worker failed, skipping patient: {type(e).__name__}: {e}")
        return {}


def _window_worker_gpu_pinned(
    window_df: pd.DataFrame,
    gpu_id: int,
    *args, sampler: str, chains: int, cores: int, progressbar: bool, chain_method: str, target_accept: float, **kwargs
):
    # Pin this child process to a specific GPU, then import JAX lazily
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if sampler in ("numpyro", "blackjax"):
        import jax  # ensures CUDA_VISIBLE_DEVICES is honored in this process
        jax.config.update("jax_enable_x64", True)
    return _window_worker(
        window_df, *args, sampler=sampler, chains=chains, cores=cores,
        progressbar=progressbar, chain_method=chain_method, target_accept=target_accept, **kwargs
    )


def compute_time_varying_trajectory_covariates_parallel(
    lab_df: pd.DataFrame,
    window_years: float = 2.0,
    flat_thr=-1.0, decline_thr=-2.0, nonlinear_gap=3.0,
    df_basis: int = 5, n_samples: int = 1000, tune: int = 1000,
    n_jobs: int = -1, min_points_per_window: int = 5, grid_freq: int = 12,
    batch_size: int = 500,
    class_func=flags_from_traj, pids='patient_id', values='lab_value',
    time_col='time', traj_types=('prolonged_nonprogression','linear_decline','nonlinear'),
    label_map=None,
    verbose: int = 0,
    *,
    sampler: str = 'pymc',
    chains: int = 4,
    cores: int = 1,
    progressbar: bool = False,
    chain_method: str = "vectorized",
    target_accept: float = 0.95,
    available_gpus: list[int]|None = None,
    windowing_col: str|None = None,  # allows windows in different granullarity than the measuring frequency
    batch_timeout_s: float = 14400.0,
) -> pd.DataFrame:
    """
    For each (pid, anchor time), use the previous window_years to compute
    posterior probabilities for each trajectory type. Returns a long DF:
      [pids, time_col, traj_prob_* ...]
    """

    if label_map is None:
        label_map = {
            'nonprogression': 'prolonged_nonprogression',
            'linear': 'linear_decline',
            'nonlinear': 'nonlinear'
        }

    windowing_col_effective = time_col if windowing_col is None else windowing_col

    def _process_batch(batch):
        if not batch:
            return []

        # A per-batch timeout bounds a genuinely pathological hang (e.g. a stuck
        # worker under proot that never returns) to batch_timeout_s instead of an
        # unbounded wait. Set generously (4h): real HiRID cohorts legitimately take
        # 27-71h end-to-end across many biomarkers/batches, so a tight timeout risks
        # silently discarding valid slow-but-healthy batches as failed. On timeout
        # the whole batch is marked failed (matches _window_worker's existing
        # per-window except-and-skip contract) rather than hanging indefinitely.
        try:
            if sampler in ("numpyro", "blackjax") and available_gpus and len(available_gpus) > 1:
                nj_eff = min(len(available_gpus), len(batch))
                print(f"[INFO] Using {nj_eff} GPUs in parallel: {available_gpus[:nj_eff]}")
                gpu_rr = [available_gpus[i % len(available_gpus)] for i in range(len(batch))]
                results = Parallel(n_jobs=nj_eff, backend="loky", verbose=verbose, timeout=batch_timeout_s)(
                    delayed(_window_worker_gpu_pinned)(
                        win_df, gpu_id,
                        flat_thr, decline_thr, nonlinear_gap, df_basis, n_samples, tune,
                        min_points_per_window, grid_freq, class_func, values, time_col, traj_types, label_map=label_map,
                        sampler=sampler, chains=chains, cores=cores, progressbar=progressbar, chain_method=chain_method,
                        target_accept=target_accept
                    )
                    for (_, _, win_df), gpu_id in zip(batch, gpu_rr)
                )
            else:
                nj_eff = n_jobs if sampler == "nutpie" else 1 if sampler in ("numpyro", "blackjax") else n_jobs
                results = Parallel(n_jobs=nj_eff, backend="loky", verbose=verbose, timeout=batch_timeout_s)(
                    delayed(_window_worker)(
                        win_df, flat_thr, decline_thr, nonlinear_gap, df_basis, n_samples, tune,
                        min_points_per_window, grid_freq, class_func, values, time_col, traj_types, label_map=label_map,
                        sampler=sampler, chains=chains, cores=cores, progressbar=progressbar, chain_method=chain_method,
                        target_accept=target_accept
                    )
                    for (_, _, win_df) in batch
                )
        except TimeoutError:
            print(f"[WARNING] Batch of {len(batch)} windows exceeded batch_timeout_s={batch_timeout_s}s "
                  f"(stuck worker/poisoned window) — skipping this batch and continuing.")
            results = [{} for _ in batch]

        out_rows = []
        for (pid, t, _), probs in zip(batch, results):
            rec = {pids: pid, windowing_col_effective: t}
            rec.update(probs)
            out_rows.append(rec)
        return out_rows

    if available_gpus is None and sampler in ("numpyro", "blackjax"):
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            # Parse CUDA_VISIBLE_DEVICES (e.g., "0,1,2" -> [0, 1, 2])
            available_gpus = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
            print(f"[INFO] Auto-detected GPUs: {available_gpus}")

    rows = []
    batch = []
    for pid, g in lab_df.groupby(pids):
        times = np.asarray(sorted(g[windowing_col_effective].unique()))
        for t in times:
            win = g[(g[windowing_col_effective] >= t - window_years) & (g[windowing_col_effective] <= t)]
            if len(win) >= min_points_per_window:
                batch.append((pid, t, win[[pids, time_col, values]].copy()))

            if len(batch) >= batch_size:
                rows.extend(_process_batch(batch))
                batch = []

    if batch:
        rows.extend(_process_batch(batch))

    return pd.DataFrame(rows)

