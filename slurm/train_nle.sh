#!/bin/bash
# train_nle.sh — local (no-SLURM) equivalent of slurm/train_nle.sbatch,
# for tara (Linux, CUDA) and tucana (macOS, MPS). One script: the only
# differences are the memory guard, which applies to tara alone, and how
# each platform is asked about its hardware.
#
#   ./slurm/train_nle.sh configs/nle_8p_v4.py              # detached (default)
#   ./slurm/train_nle.sh -f configs/nle_8p_v4.py           # stay in this shell
#   ./slurm/train_nle.sh --time 12h configs/nle_8p_v4.py   # wall-clock limit
#   ./slurm/train_nle.sh configs/nle_8p_v4.py --config.batch_size=256
#
# Mirrors the sbatch script's behaviour: a per-run directory with a manifest,
# a config snapshot, and separate stdout/stderr logs. Unlike SLURM there is no
# queue -- the run starts immediately and competes for the single GB10 GPU.

set -uo pipefail

JOB_NAME="${JOB_NAME:-nle_train}"
NLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# tara is a shared Linux box with a CUDA GPU and a memory budget;
# tucana is a macOS workstation with 256 GB of unified memory and no
# limit to respect. Everything else below is common to both.
case "$(uname -s)" in
    Darwin) PLATFORM=mac ;;
    *)      PLATFORM=linux ;;
esac

LOG_ROOT="${LOG_ROOT:-$HOME/scratch/local_logs/nle}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch}"
BACKGROUND=1          # detached by default, like sbatch
TIME_LIMIT=""

usage() {
    cat >&2 <<'USAGE'
Usage: train_nle.sh [options] <path/to/config.py> [--config.field=value ...]

  -f, --foreground      run in this shell instead of detaching
  -b, --background      run detached (default)
  -t, --time DURATION   wall-clock limit, e.g. 12h, 90m, 3600s
  -h, --help            show this message

Detached is the default: the run survives logout, as an sbatch job would.
USAGE
    exit 1
}

# Hand-rolled parser: getopts cannot do long options, and we must not
# consume the --config.* overrides meant for train_nle.py.
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--foreground) BACKGROUND=0; shift ;;
        -b|--background) BACKGROUND=1; shift ;;
        -t|--time)
            [[ $# -ge 2 ]] || { echo "$1 needs a DURATION" >&2; usage; }
            TIME_LIMIT="$2"; shift 2 ;;
        --time=*)        TIME_LIMIT="${1#*=}"; shift ;;
        -t=*)            TIME_LIMIT="${1#*=}"; shift ;;
        -h|--help)       usage ;;
        --)              shift; break ;;
        *)               break ;;   # config path; the rest passes through
    esac
done
[[ $# -lt 1 ]] && usage

CONFIG="$1"; shift
EXTRA_ARGS=("$@")

# ml_collections allows "<file>.py:<arg>" to pass a string to get_config();
# split it off before resolving the file, re-append it after.
CONFIG_ARG=""
if [[ "$CONFIG" == *.py:* ]]; then
    CONFIG_ARG="${CONFIG##*.py:}"
    CONFIG="${CONFIG%:"$CONFIG_ARG"}"
fi

# Resolve the config relative to nle_dsph/ if not found as given.
if [[ ! -f "$CONFIG" ]]; then
    if [[ -f "$NLE_DIR/$CONFIG" ]]; then
        CONFIG="$NLE_DIR/$CONFIG"
    else
        echo "Config not found: $CONFIG" >&2
        exit 1
    fi
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
CONFIG_FILE="$CONFIG"
[[ -n "$CONFIG_ARG" ]] && CONFIG="$CONFIG:$CONFIG_ARG"

# Memory guard. tara is shared and has no scheduler, and a 57-shard
# device-resident run holds ~45 GB of unified memory; two of them plus
# the loading peak of a third filled the 121 GB on 2026-10-07. Refuse to
# start unless at least MIN_FREE_GB is available (override with
# MIN_FREE_GB=... or FORCE=1), and never start a second full-data run.
# tucana has no such limit and is skipped entirely.
if [[ "$PLATFORM" == linux ]]; then
    MIN_FREE_GB="${MIN_FREE_GB:-60}"
    avail_gb=$(free -g | awk '/^Mem:/{print $7}')
    if [[ "${FORCE:-0}" != 1 && "$avail_gb" -lt "$MIN_FREE_GB" ]]; then
        echo "Refusing to start: ${avail_gb} GB available < ${MIN_FREE_GB} GB" \
             "(other users share tara; set MIN_FREE_GB or FORCE=1)" >&2
        exit 1
    fi
fi

# Stand-in for $SLURM_JOB_ID: timestamp + PID is unique and sorts
# chronologically. The detached child inherits the parent's directory
# through RUN_DIR_PRESET and must not mint a second one from its own
# PID, which is how an empty run dir was left behind on every
# detached run.
if [[ -n "${RUN_DIR_PRESET:-}" ]]; then
    RUN_DIR="$RUN_DIR_PRESET"
    RUN_ID="${RUN_DIR##*/${JOB_NAME}_}"
else
    RUN_ID="$(date +%Y%m%d-%H%M%S)_$$"
    RUN_DIR="$LOG_ROOT/${JOB_NAME}_${RUN_ID}"
fi
MANIFEST="$RUN_DIR/manifest.txt"
mkdir -p "$RUN_DIR" || { echo "Cannot create $RUN_DIR" >&2; exit 1; }

# Re-exec detached once the run dir exists, so the caller gets the real paths.
if [[ "$BACKGROUND" == 1 ]]; then
    export JOB_NAME LOG_ROOT VENV_DIR
    RUN_DIR_PRESET="$RUN_DIR" \
        nohup "$0" --foreground ${TIME_LIMIT:+--time "$TIME_LIMIT"} "$CONFIG" "${EXTRA_ARGS[@]}" \
        > "$RUN_DIR/nohup.out" 2>&1 &
    echo "Started detached: PID $!"
    echo "  run dir: $RUN_DIR"
    echo "  follow:  tail -f $RUN_DIR/stdout.log"
    exit 0
fi

# Caches that would otherwise land in $HOME. Kept on scratch to match trillium,
# where $HOME was read-only on compute nodes.
export XDG_CACHE_HOME="$HOME/scratch/cache"
export MPLCONFIGDIR="$HOME/scratch/cache/matplotlib"
export WANDB_CONFIG_DIR="$HOME/scratch/cache/wandb_config"
export TORCH_HOME="$HOME/scratch/cache/torch"
export NUMBA_CACHE_DIR="$HOME/scratch/cache/numba"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$HOME/scratch/wandb_data}"
export WANDB_X_SERVICE_WAIT=300
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_CONFIG_DIR" "$TORCH_HOME" "$NUMBA_CACHE_DIR" "$WANDB_DATA_DIR"

# One GPU, and threads capped so a run does not take every core.
[[ "$PLATFORM" == linux ]] && \
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${PYTHONPATH:-}:$HOME/my_modules"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MANIFEST"; }

log "===== run start ====="
log "run_id:       $RUN_ID"
log "job_name:     $JOB_NAME"
log "host:         $(hostname)"
log "pid:          $$"
if [[ "$PLATFORM" == linux ]]; then
    log "gpus:         $(nvidia-smi -L 2>/dev/null | tr '\n' ';')"
    log "cpus:         $(nproc --all) total (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
    log "mem:          $(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
else
    log "gpus:         $(sysctl -n machdep.cpu.brand_string) (unified memory)"
    log "cpus:         $(sysctl -n hw.ncpu) total (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
    log "mem:          $(( $(sysctl -n hw.memsize) / 1073741824 )) GB unified"
fi
log "config:       $CONFIG"
log "extra_args:   ${EXTRA_ARGS[*]:-<none>}"
log "time_limit:   ${TIME_LIMIT:-<none>}"
log "wandb_data:   $WANDB_DATA_DIR (mode=${WANDB_MODE:-online})"

cp -- "${CONFIG_FILE:-$CONFIG}" "$RUN_DIR/config_snapshot.py" 2>/dev/null || true

# Activate directly: ~/.bashrc's venv-activate is not defined in a
# non-interactive shell, same reason as on trillium.
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    log "FATAL: no venv at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"
log "python:       $(command -v python3) ($(python3 --version 2>&1))"
log "torch:        $(python3 -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available(), "mps:", torch.backends.mps.is_available())' 2>/dev/null)"

on_exit() {
    local code=$?
    log "exit_code:    $code"
    log "===== run end ====="
    exit "$code"
}
trap on_exit EXIT
trap 'log "received TERM -- shutting down"' TERM
trap 'log "received INT (Ctrl-C)"' INT

cd "$NLE_DIR" || exit 1

RUNNER=(python3 train_nle.py --config "$CONFIG" "${EXTRA_ARGS[@]}" --config.enable_progress_bar=False)

# The wall-clock limit is enforced by a watchdog rather than timeout(1),
# which macOS does not ship: TERM first so Lightning can write its
# checkpoint, KILL a minute later if that did not take.
TIME_LIMIT_SECONDS=""
if [[ -n "$TIME_LIMIT" ]]; then
    number="${TIME_LIMIT%[smhd]}"
    unit="${TIME_LIMIT: -1}"
    case "$unit" in
        s) TIME_LIMIT_SECONDS="$number" ;;
        m) TIME_LIMIT_SECONDS=$((number * 60)) ;;
        h) TIME_LIMIT_SECONDS=$((number * 3600)) ;;
        d) TIME_LIMIT_SECONDS=$((number * 86400)) ;;
        *) TIME_LIMIT_SECONDS="$TIME_LIMIT" ;;   # bare seconds
    esac
    log "watchdog:     TERM after ${TIME_LIMIT_SECONDS}s, KILL 60s later"
fi

log "command:      ${RUNNER[*]}"
"${RUNNER[@]}" > >(tee -a "$RUN_DIR/stdout.log") 2> >(tee -a "$RUN_DIR/stderr.log" >&2) &
RUN_PID=$!

WATCHDOG_PID=""
if [[ -n "$TIME_LIMIT_SECONDS" ]]; then
    (
        sleep "$TIME_LIMIT_SECONDS"
        kill -TERM "$RUN_PID" 2>/dev/null
        sleep 60
        kill -KILL "$RUN_PID" 2>/dev/null
    ) &
    WATCHDOG_PID=$!
fi

wait "$RUN_PID"
run_code=$?
[[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null
exit "$run_code"
