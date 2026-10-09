#!/usr/bin/env bash
# cf_batch_watchdog.sh — cron-driven watchdog for docker/run_cf_batches.sh
#
# Why this exists: on 2026-08-10 all 3 CF runs (hirid, mimic, eicu) were killed
# simultaneously when their compute host(s) went down for what looks like
# unannounced cluster maintenance (rack-shamir8's own boot record shows a ~22h
# outage: down 2026-08-10 11:20 -> back 2026-08-11 09:35). Nothing was watching,
# so all 3 runs sat dead for 5 days before anyone noticed. `nohup ... &` alone
# does not survive a host reboot — cron does, because the cron *daemon* is a
# system service that comes back up on its own after a reboot and re-fires this
# script on its next scheduled tick.
#
# Design: delegates ALL "is there remaining work" and "is it safe to resume"
# logic to run_cf_batches.sh itself — it already skips completed cohorts/
# biomarkers and exits 0 immediately when a dataset is fully done. This script's
# only job is to notice a dataset's run is NOT currently holding its lock
# (i.e. genuinely not running, not just between cohorts) and relaunch it. It
# reuses run_cf_batches.sh's own per-dataset flock (.{dataset}_cf_batches.lock)
# as the single source of truth for "is one already running", so this can
# never double-launch a dataset that's mid-run.
#
# Install (crontab -e):
#   */20 * * * * /home/gaga/tamarw1/trajectory-modeling/scripts/watchdog/cf_batch_watchdog.sh
#
# Pause without touching crontab:
#   touch /home/gaga/tamarw1/trajectory-modeling/.watchdog_disable
# Resume:
#   rm /home/gaga/tamarw1/trajectory-modeling/.watchdog_disable
#
# Per-dataset logs: logs/watchdog/<dataset>.log (relaunch decisions only —
# the actual run output already goes to logs/container_runs/<dataset>_cf/...
# via run_cf_batches.sh's own tee, so we deliberately do NOT duplicate that
# here by piping the child's stdout through this script).
#
# NOTE on --lock-suffix sharding: this watchdog only probes each dataset's
# DEFAULT (no-suffix) lock via `nohup run_cf_batches.sh --dataset <ds>` with
# no --cohorts/--lock-suffix, i.e. "run whatever's left, auto-detected". If
# you're manually running a dataset as multiple concurrent --lock-suffix
# shards (see run_cf_batches.sh --help), this watchdog does NOT know about
# those shard locks and will NOT resume them after an outage — you're on the
# hook for that dataset yourself. Don't mix watchdog-managed and manually-
# sharded runs for the same dataset at the same time.

set -uo pipefail  # no -e: one dataset failing to probe/relaunch must not stop the others

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATASETS=("hirid" "mimic" "eicu")
LOG_DIR="${REPO_ROOT}/logs/watchdog"
mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

# ── Don't let overlapping cron ticks (e.g. this script itself hanging on a
#    stuck NFS mount) run concurrently with each other ─────────────────────
SELF_LOCK="${REPO_ROOT}/.cf_batch_watchdog.lock"
exec 8>"$SELF_LOCK"
if ! flock -n 8; then
    echo "$(ts) [watchdog] previous invocation still running — skipping this tick" >> "${LOG_DIR}/watchdog.log"
    exit 0
fi

if [[ -f "${REPO_ROOT}/.watchdog_disable" ]]; then
    echo "$(ts) [watchdog] .watchdog_disable present — skipping this tick" >> "${LOG_DIR}/watchdog.log"
    exit 0
fi

for ds in "${DATASETS[@]}"; do
    LOCK_FILE="${REPO_ROOT}/.${ds}_cf_batches.lock"

    # Non-blocking probe of run_cf_batches.sh's OWN lock. If we can grab it,
    # nothing is running for this dataset right now.
    exec 9>"$LOCK_FILE"
    if flock -n 9; then
        flock -u 9
        exec 9>&-   # release the fd fully before launching the child, so it
                     # can acquire a clean lock of its own
        echo "$(ts) [$ds] not running — relaunching run_cf_batches.sh --dataset $ds" >> "${LOG_DIR}/${ds}.log"
        nohup bash "${REPO_ROOT}/docker/run_cf_batches.sh" --dataset "$ds" \
            > /dev/null 2>&1 &
        disown
    else
        exec 9>&-
        echo "$(ts) [$ds] already running — nothing to do" >> "${LOG_DIR}/${ds}.log"
    fi
done
