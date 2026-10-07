#!/bin/bash
# queue_nle.sh — run train_nle.sh configs one after another (tara has no
# scheduler and is shared, so runs are queued rather than stacked).
#
#   ./slurm/queue_nle.sh [--after PID] <config[:arm]> [<config[:arm]> ...]
#
# --after PID: wait for that process to exit before the first run.
# Each run goes through train_nle.sh in the foreground, so its memory
# guard applies; a run that fails does not stop the queue.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--after" ]]; then
    while kill -0 "$2" 2>/dev/null; do sleep 30; done
    shift 2
fi
for cfg in "$@"; do
    name="$(basename "${cfg%%.py*}")_${cfg##*:}"
    echo "[$(date '+%F %T')] queue: starting $cfg"
    JOB_NAME="$name" "$HERE/train_nle.sh" --foreground "$cfg"
    echo "[$(date '+%F %T')] queue: $cfg exited with $?"
done
