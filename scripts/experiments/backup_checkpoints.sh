#!/usr/bin/env bash
# Periodic backup of Tinker checkpoint indices (checkpoints.jsonl files)
# and run-info files. Tinker state itself lives server-side and is not
# deleted; what we protect here is the local index that maps step → URL.
#
# Strategy:
#   - "latest/"  : a mirror of the current checkpoint index files (always
#                   overwritten in-place; cheap diff vs source).
#   - "history/<TS>/": a periodic snapshot every BACKUP_INTERVAL_SEC. We
#                   keep a rolling window of HISTORY_KEEP snapshots.
#
# Run it as `bash backup_checkpoints.sh &` (or in a tmux session). It
# loops forever. Safe to run multiple times — uses a pidfile.

set -euo pipefail

PROJECT=.
SRC_ROOT="$PROJECT/tinker_logs"
DEST_ROOT="$PROJECT/backups/tinker_checkpoints"
LATEST="$DEST_ROOT/latest"
HISTORY="$DEST_ROOT/history"
BACKUP_INTERVAL_SEC=${BACKUP_INTERVAL_SEC:-600}    # 10 min
HISTORY_KEEP=${HISTORY_KEEP:-72}                    # last 12h at 10-min cadence
PIDFILE="$DEST_ROOT/backup.pid"
LOG="$DEST_ROOT/backup.log"

mkdir -p "$LATEST" "$HISTORY"

# Single-instance lock
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$(date '+%F %T') already running as pid $(cat "$PIDFILE")" >> "$LOG"
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

backup_once() {
    local ts files=0 size_kb=0
    ts=$(date '+%Y%m%d_%H%M%S')
    local snapshot="$HISTORY/$ts"
    mkdir -p "$snapshot"

    # Find every checkpoints.jsonl and *_info.json inside tinker_logs/.
    # rsync preserves the relative tree under tinker_logs/.
    if command -v rsync >/dev/null 2>&1; then
        rsync -aR --include='*/' \
            --include='checkpoints.jsonl' \
            --include='*_info.json' \
            --exclude='*' \
            "$SRC_ROOT/./" "$LATEST/" >/dev/null
        rsync -a "$LATEST/" "$snapshot/" >/dev/null
    else
        # Fallback if rsync isn't available
        cd "$SRC_ROOT"
        while IFS= read -r f; do
            local rel="${f#./}"
            mkdir -p "$LATEST/$(dirname "$rel")"
            cp -p "$f" "$LATEST/$rel"
        done < <(find . -type f \( -name 'checkpoints.jsonl' -o -name '*_info.json' \))
        cp -rp "$LATEST"/. "$snapshot/"
    fi

    files=$(find "$snapshot" -type f | wc -l)
    size_kb=$(du -sk "$snapshot" 2>/dev/null | awk '{print $1}')
    log "snapshot $ts: $files files, ${size_kb} KB"

    # Trim history (keep newest HISTORY_KEEP)
    local kept=0
    for d in $(ls -1tr "$HISTORY" 2>/dev/null); do
        kept=$((kept + 1))
    done
    if [ "$kept" -gt "$HISTORY_KEEP" ]; then
        local to_remove=$((kept - HISTORY_KEEP))
        ls -1tr "$HISTORY" | head -n "$to_remove" | while read -r d; do
            rm -rf "$HISTORY/$d"
        done
        log "trimmed $to_remove old snapshots; keeping newest $HISTORY_KEEP"
    fi
}

log "backup daemon starting (interval=${BACKUP_INTERVAL_SEC}s, keep=${HISTORY_KEEP})"
log "source=$SRC_ROOT"
log "dest=$DEST_ROOT"

while true; do
    backup_once || log "backup_once failed: $?"
    sleep "$BACKUP_INTERVAL_SEC"
done
