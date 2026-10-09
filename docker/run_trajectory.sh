#!/usr/bin/env bash
# run_trajectory.sh — convenience wrapper for running trajectory scripts inside
# the trajectory-modeling container on both Docker and UDocker.
#
# Usage:
#   bash docker/run_trajectory.sh [OPTIONS] SCRIPT [SCRIPT_ARGS...]
#
# Options:
#   --docker          Use Docker (default if docker command found)
#   --udocker         Use UDocker (default on university server)
#   --image NAME      Container image name (default: trajectory-modeling)
#   --data-dir DIR    Host path to data root, mounted as /data (default: $TRAJ_DATA_ROOT or .)
#   --out-dir DIR     Host path for outputs, mounted as /out (default: ./results)
#   --help            Show this help
#
# Examples:
#   # HiRiD circulatory failure trajectories (UDocker, university server):
#   bash docker/run_trajectory.sh --udocker \
#       python /traj/scripts/trajectory/hirid/circulatory_failure_trajs.py \
#       --sampler pymc --input-dir /data/hirid/circulatory_failure
#
#   # MIMIC trajectories (Docker, external machine):
#   bash docker/run_trajectory.sh --docker \
#       python /traj/scripts/trajectory/mimic/circulatory_failure_trajs.py \
#       --sampler pymc --input-dir /data/mimic
#
#   # Validate g++ acceleration:
#   bash docker/run_trajectory.sh python /traj/scripts/validate_gpp_acceleration.py

set -euo pipefail

IMAGE="${TRAJ_IMAGE:-trajectory-modeling}"
DATA_DIR="${TRAJ_DATA_ROOT:-$(pwd)}"
OUT_DIR="${TRAJ_OUT_DIR:-$(pwd)/results}"
RUNTIME=""

# ── Argument parsing ──────────────────────────────────────────────────────────
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker)   RUNTIME=docker;  shift ;;
        --udocker)  RUNTIME=udocker; shift ;;
        --image)    IMAGE="$2";      shift 2 ;;
        --data-dir) DATA_DIR="$2";   shift 2 ;;
        --out-dir)  OUT_DIR="$2";    shift 2 ;;
        --help|-h)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#PASS_ARGS[@]} -eq 0 ]]; then
    echo "Error: no script specified." >&2
    echo "Usage: bash docker/run_trajectory.sh [OPTIONS] python /traj/scripts/.../script.py [args]" >&2
    exit 1
fi

# ── Auto-detect runtime ────────────────────────────────────────────────────────
if [[ -z "$RUNTIME" ]]; then
    if command -v udocker &>/dev/null; then
        RUNTIME=udocker
    elif command -v docker &>/dev/null; then
        RUNTIME=docker
    else
        echo "Error: neither 'docker' nor 'udocker' found in PATH." >&2
        exit 1
    fi
fi

echo "Runtime : $RUNTIME"
echo "Image   : $IMAGE"
echo "Data dir: $DATA_DIR → /data"
echo "Out dir : $OUT_DIR  → /out"
echo "Command : ${PASS_ARGS[*]}"
echo ""

mkdir -p "$OUT_DIR"

# ── Build common volume / env flags ───────────────────────────────────────────
COMMON_ARGS=(
    -v "${DATA_DIR}:/data:ro"
    -v "${OUT_DIR}:/out"
    -e "TRAJ_DATA_ROOT=/data"
)

# ── Run ────────────────────────────────────────────────────────────────────────
case "$RUNTIME" in
    docker)
        docker run --rm \
            "${COMMON_ARGS[@]}" \
            "$IMAGE" \
            "${PASS_ARGS[@]}"
        ;;
    udocker)
        # UDocker notes:
        #   - No --rm flag (UDocker keeps containers; remove manually with udocker rm)
        #   - No --user flag needed: UDocker always runs as the invoking OS user
        #   - -v syntax is identical to Docker
        udocker run \
            "${COMMON_ARGS[@]}" \
            "$IMAGE" \
            "${PASS_ARGS[@]}"
        ;;
esac
