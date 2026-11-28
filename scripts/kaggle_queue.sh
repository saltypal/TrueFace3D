#!/usr/bin/env bash
# Overnight Kaggle queue: push each staged kernel when its account has a free GPU slot (2 per account),
# then download every finished kernel's output. State lives in $STATE (one line per slug: account status).
#   scripts/kaggle_queue.sh a:tf3d-k10-vits-e3-19 b:tf3d-k10-deca ...
# Kernels must already be staged in kaggle_staging/<slug>/ (scripts/kaggle_push.py --dry-run stages them).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUTDIR=${KAGGLE_OUT:-/mnt/DataBunker/kaggle_outputs}
STATE=${KAGGLE_QUEUE_STATE:-/mnt/DataBunker/logs/kaggle_queue.state}
LOG=/mnt/DataBunker/logs/kaggle_queue.log
declare -A OWNER=([a]=satyapaladugu [b]=satyapaladuguplus)
mkdir -p "$OUTDIR"; touch "$STATE"
for item in "$@"; do grep -q "^${item#*:} " "$STATE" || echo "${item#*:} ${item%%:*} pending" >> "$STATE"; done
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
setst() { sed -i "s#^$1 .*#$1 $2 $3#" "$STATE"; }

while true; do
  busy=0
  while read -r slug acc st; do
    case "$st" in
      pending)
        busy=1
        msg=$(kaggle-$acc kernels push -p "$ROOT/kaggle_staging/$slug" 2>&1 | tail -1)
        if echo "$msg" | grep -qi "successfully pushed"; then setst "$slug" "$acc" running; say "pushed $acc/$slug"
        elif ! echo "$msg" | grep -qi "Maximum batch GPU session"; then say "push $slug: $msg"; fi ;;
      running)
        busy=1
        s=$(kaggle-$acc kernels status "${OWNER[$acc]}/$slug" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | cut -d. -f2)
        case "$s" in
          COMPLETE|ERROR|CANCEL_ACKNOWLEDGED|CANCELLED)
            say "$slug finished: $s; downloading"
            mkdir -p "$OUTDIR/$slug"
            kaggle-$acc kernels output "${OWNER[$acc]}/$slug" -p "$OUTDIR/$slug" -o >> "$LOG" 2>&1
            say "$slug downloaded: $(du -sh "$OUTDIR/$slug" | cut -f1)"
            setst "$slug" "$acc" "done_$s" ;;
        esac ;;
    esac
  done < "$STATE"
  [ "$busy" = 0 ] && { say "queue empty"; exit 0; }
  sleep 180
done
