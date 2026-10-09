#!/bin/bash
# Container entrypoint for trajectory-modeling.
#
# Responsibilities:
#   1. Ensure the PyTensor compile cache directory exists (writable at runtime)
#   2. Apply thread limits if not already set
#   3. Execute the user's command
#
# The trajectory scripts override PYTENSOR_FLAGS themselves (they set a
# job-specific compiledir). This entrypoint sets a sane fallback for scripts
# that do not.
#
# Usage (via Docker or UDocker):
#   python /traj/scripts/trajectory/<dataset>/<script>.py --sampler pymc [args]

set -e

# Ensure the default PyTensor compile cache is writable.
# Scripts that set their own PYTENSOR_FLAGS will use a different path.
mkdir -p /tmp/pytensor_cache

# Apply thread limits (prevent oversubscription inside container).
# ENV vars set in the Dockerfile serve as defaults; scripts may override.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

exec "$@"
