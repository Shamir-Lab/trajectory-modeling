# syntax=docker/dockerfile:1
#
# Trajectory Modeling Container
# ------------------------------
# Provides PyMC with g++ (via build-essential) so PyTensor can compile C ops
# at runtime, giving 10–50× speedup over pure-Python MCMC sampling.
#
# This container is the solution for university HPC servers (UDocker) where
# installing compilers system-wide requires root permissions we don't have.
# The same image runs on Docker for external users without modification.
#
# Build:  docker build -t trajectory-modeling -f docker/Dockerfile .
# Run:    docker run -v /host/data:/data trajectory-modeling python /traj/scripts/...py [args]
# UDocker (after export/load):
#         udocker run -v /host/data:/data trajectory-modeling python /traj/scripts/...py [args]

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
# build-essential provides: g++, gcc, make
# g++ is required by PyTensor for C op compilation.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── PyTensor environment ──────────────────────────────────────────────────────
# base_compiledir: where PyTensor writes compiled C ops (must be writable at runtime)
# cxx=g++: explicit compiler — auto-detected anyway but made explicit for clarity
# optimizer=fast_compile: fast compilation (vs full optimization; good for MCMC)
#
# Note: trajectory scripts override PYTENSOR_FLAGS via os.environ at startup.
# PyTensor finds g++ by scanning PATH regardless of the cxx flag value.
ENV PYTENSOR_FLAGS="base_compiledir=/tmp/pytensor_cache,optimizer=fast_compile,cxx=g++"

# Thread limits: prevent oversubscription when many joblib workers are active.
# Individual scripts may tighten these further.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

WORKDIR /traj

# ── Python package ───────────────────────────────────────────────────────────
# Copy only what pip needs first (layer-cache friendly: package changes rarely)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install traj_features[bayes,bootstrap,evaluation] from source.
# Intentionally NOT installing numpyro or nutpie so pymc is the only sampler.
RUN pip install --no-cache-dir -e ".[bayes,bootstrap,evaluation]"

# ── Trajectory scripts ───────────────────────────────────────────────────────
# Scripts are included for convenience. Users can also mount custom scripts
# with -v /host/scripts:/traj/scripts.
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# ── Entrypoint ───────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# UDocker compatibility notes:
#   - No USER instruction: UDocker runs the container as the invoking OS user.
#     A USER instruction here would be ignored and could cause permission errors.
#   - No VOLUME declaration: UDocker does not handle Dockerfile VOLUME well.
#     Bind-mount data at runtime with: -v /host/path:/container/path
#   - ENTRYPOINT + CMD is preferred over bare ENTRYPOINT: CMD can be overridden
#     by appending a command to `udocker run IMAGE <command>`.

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-c", \
     "import traj_features, subprocess, shutil; \
      gpp = shutil.which('g++'); \
      print('traj_features', traj_features.__version__, 'ready'); \
      print('g++:', gpp or 'NOT FOUND'); \
      r = subprocess.run(['g++', '--version'], capture_output=True, text=True) if gpp else None; \
      print(r.stdout.strip() if r else ''); \
      print(); \
      print('Usage: python /traj/scripts/trajectory/<dataset>/<script>.py --sampler pymc [args]')"]
