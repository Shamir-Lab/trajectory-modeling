# Container Guide: Running Trajectories with g++ Acceleration

This guide covers building and running the `trajectory-modeling` container, which provides
PyMC with g++ (via PyTensor C compilation) for 10–50× faster MCMC sampling.

## Why a container?

The university HPC server does not allow installing compilers (g++) system-wide — it requires
root privileges we don't have. Without g++, PyTensor runs in slower pure-Python mode.
The container bakes g++ in at build time, so it is always available at runtime.

External users running Docker get the same environment without modification.

---

## Build (requires Docker — do this on any machine with Docker)

```bash
# From the repo root:
docker build -t trajectory-modeling -f docker/Dockerfile .
```

Build takes ~5–10 minutes (mostly downloading Python packages).
The resulting image is ~1.5 GB.

### Export for UDocker (university server)

UDocker cannot build images. Export from Docker and transfer:

```bash
# Export image to a tar file
docker save trajectory-modeling > trajectory-modeling.tar

# Transfer to university server (example using scp)
scp trajectory-modeling.tar <user>@<server>:/path/to/storage/
```

---

## University server setup (UDocker)

UDocker is installed at `/usr/local/bin/udocker` on the university server.

```bash
# One-time: load the image
udocker load -i /path/to/storage/trajectory-modeling.tar

# One-time: create a container from the image
# (UDocker containers are persistent; you create once and reuse)
udocker create --name=trajectory-modeling trajectory-modeling

# Verify g++ is available in the container
udocker run trajectory-modeling g++ --version
# Expected: g++ (Debian ...) 12.x.x ...

# Verify the package is installed
udocker run trajectory-modeling python -c "import traj_features; print(traj_features.__version__)"
```

### UDocker limitations (what NOT to do)

| Limitation | Detail |
|---|---|
| No root | Container runs as your OS user (UDocker fakes root inside, but you still can't do privileged ops) |
| No `--privileged` | Not supported |
| No named volumes | Use `-v /host/path:/container/path` bind mounts only |
| No `--rm` | UDocker keeps containers; use `udocker rm <name>` to remove |
| No `--user` | Unnecessary — UDocker always runs as the invoking user |

---

## Running trajectory scripts

### University server (UDocker)

```bash
# HiRiD circulatory failure trajectories
udocker run \
    -v /home/gaga/data/physionet:/data:ro \
    -v /home/gaga/tamarw1/results:/out \
    trajectory-modeling \
    python /traj/scripts/trajectory/hirid/circulatory_failure_trajs.py \
        --sampler pymc \
        --input-dir /data/hirid/circulatory_failure \
        --output-dir /out/hirid

# MIMIC circulatory failure trajectories
udocker run \
    -v /home/gaga/data/physionet:/data:ro \
    -v /home/gaga/tamarw1/results:/out \
    trajectory-modeling \
    python /traj/scripts/trajectory/mimic/circulatory_failure_trajs.py \
        --sampler pymc \
        --input-dir /data/mimic

# eICU trajectories
udocker run \
    -v /home/gaga/data/physionet:/data:ro \
    -v /home/gaga/tamarw1/results:/out \
    trajectory-modeling \
    python /traj/scripts/trajectory/eicu/circulatory_failure_trajs.py \
        --sampler pymc \
        --input-dir /data/eicu
```

### External users (Docker)

```bash
# Same scripts, same flags — just replace `udocker run` with `docker run --rm`
docker run --rm \
    -v /your/data:/data:ro \
    -v /your/results:/out \
    trajectory-modeling \
    python /traj/scripts/trajectory/hirid/circulatory_failure_trajs.py \
        --sampler pymc \
        --input-dir /data/hirid/circulatory_failure \
        --output-dir /out/hirid
```

### Convenience wrapper

`docker/run_trajectory.sh` wraps both runtimes and auto-detects which is available:

```bash
# Auto-detect (prefers udocker if found, else docker)
bash docker/run_trajectory.sh \
    --data-dir /home/gaga/data/physionet \
    --out-dir /home/gaga/tamarw1/results \
    python /traj/scripts/trajectory/hirid/circulatory_failure_trajs.py \
        --sampler pymc

# Force udocker
bash docker/run_trajectory.sh --udocker ...

# Force docker
bash docker/run_trajectory.sh --docker ...
```

---

## Running in SLURM (inside container)

The SLURM scripts are designed for the conda environment, not the container. To run
container-based trajectories under SLURM, replace the Python invocation:

```bash
# Instead of:
conda activate pymc_env && python scripts/...py --sampler nutpie

# Use:
udocker run -v /data:/data -v /out:/out trajectory-modeling \
    python /traj/scripts/...py --sampler pymc
```

Or wrap the `udocker run` call in a minimal SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=hirid_container
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00

udocker run \
    -v /home/gaga/data/physionet:/data:ro \
    -v /home/gaga/tamarw1/results:/out \
    -e OMP_NUM_THREADS=1 \
    trajectory-modeling \
    python /traj/scripts/trajectory/hirid/circulatory_failure_trajs.py \
        --sampler pymc \
        --n-jobs ${SLURM_CPUS_PER_TASK}
```

---

## Validating g++ acceleration

Run the validation script inside the container to confirm C compilation is active
and measure the speedup over pure-Python mode:

```bash
# University server
udocker run trajectory-modeling \
    python /traj/scripts/validate_gpp_acceleration.py --n-patients 4 --n-samples 50

# Docker
docker run --rm trajectory-modeling \
    python /traj/scripts/validate_gpp_acceleration.py --n-patients 4 --n-samples 50
```

Expected output:
```
g++ in PATH : YES — C compilation will be active
...
[WITH g++]  elapsed: X.Xs ...
[NO  g++]   elapsed: X.Xs ...
Speedup: X.Xx (with g++ vs without)
g++ acceleration: ACTIVE
```

Typical speedup is **5–30×** depending on model size. The first run in a fresh container
is slower (PyTensor compiles C ops to `/tmp/pytensor_cache`); subsequent runs reuse
the cache and are faster.

---

## Sampler note

The container enforces `--sampler pymc`. `numpyro` and `nutpie` are intentionally
NOT installed:

- `numpyro` requires JAX — not installed (container is CPU-only; UDocker doesn't handle CUDA)
- `nutpie` is a Rust-based sampler — not installed to keep the container minimal

`pymc` with g++ (via PyTensor) is the correct sampler for this container and gives
excellent performance on CPU.

---

## Updating the container

When the `traj_features` package changes, rebuild and re-export:

```bash
docker build --no-cache -t trajectory-modeling -f docker/Dockerfile .
docker save trajectory-modeling > trajectory-modeling.tar
# Transfer and reload on server:
udocker rm trajectory-modeling          # remove old container
udocker rmi trajectory-modeling         # remove old image
udocker load -i trajectory-modeling.tar
udocker create --name=trajectory-modeling trajectory-modeling
```

---

## Troubleshooting

**`udocker: command not found`**
Check: `which udocker` or try `/usr/local/bin/udocker`.

**`ImportError: No module named 'traj_features'`**
The package was not installed in the image. Rebuild. Or check `pip show traj_features`
inside the container.

**`g++ not found` inside container**
Should not happen with this image. Verify: `udocker run trajectory-modeling g++ --version`.
If missing, the Dockerfile's `build-essential` install may have failed — rebuild.

**PyTensor slow (no C compilation)**
If PYTENSOR_FLAGS contains `cxx=` (empty), C compilation is disabled. Remove the `cxx=`
flag or set `cxx=g++` to re-enable it.

**`/tmp/pytensor_cache` permission denied**
This happens if the container was run as a different user. Clear the cache:
`udocker run trajectory-modeling rm -rf /tmp/pytensor_cache` then rerun.

**Out-of-memory on SLURM**
Reduce `--n-jobs` or `--mem` per worker. The PyMC sampler itself is single-threaded
per chain; parallelism comes from joblib distributing windows across workers.
