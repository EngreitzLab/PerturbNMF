#!/bin/bash
# Keep running the blinded dispatch until every prompt has a COMPLETE answer to its CURRENT prompt.
#
# `claude -p` fails nonzero with an EMPTY stderr when the account is usage-limited, which is
# indistinguishable from a real error. So this just retries: each pass skips completed prompts,
# so a pass during a limited window costs almost nothing and a pass after it finishes the work.
# Meant to run detached (nohup ... &). The verdict is <dispatch_root>/DISPATCH_STATUS, not the
# exit code — read it.
#
# Usage: dispatch_until_complete.sh <dispatch_root> <dir_glob> [concurrency] [max_passes] [sleep_s]
#   e.g. nohup bash dispatch_until_complete.sh dispatch "v3_p*" 4 60 600 > dispatch.log 2>&1 &
# Env:   PYTHON (default: python), ANNOTATOR_MODEL (default: sonnet)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DISPATCH="${1:?dispatch root}"
DIR_GLOB="${2:?directory glob, e.g. v3_p*}"
CONCURRENCY="${3:-4}"
MAX_PASSES="${4:-60}"
SLEEP_BETWEEN="${5:-600}"
PYTHON="${PYTHON:-python}"
STATUS="$DISPATCH/DISPATCH_STATUS"

remaining() {
    local n=0 d
    for d in "$DISPATCH"/$DIR_GLOB; do
        [ -d "$d" ] || continue
        "$PYTHON" "$HERE/answer_one_prompt.py" needs-work "$d" && n=$((n+1))
    done
    echo "$n"
}

for pass in $(seq 1 "$MAX_PASSES"); do
    left="$(remaining)"
    echo "=== pass $pass — $left prompt(s) outstanding — $(date '+%H:%M:%S') ==="
    [ "$left" -eq 0 ] && { echo "ALL COMPLETE"; echo "complete $(date)" > "$STATUS"; exit 0; }
    echo "pass $pass, $left outstanding, $(date)" > "$STATUS"
    bash "$HERE/run_blinded_annotations.sh" "$DISPATCH" "$CONCURRENCY" "$DIR_GLOB" 2>&1 | grep -E "^(ok|FAIL)" | tail -5
    [ "$(remaining)" -eq 0 ] && { echo "ALL COMPLETE"; echo "complete $(date)" > "$STATUS"; exit 0; }
    echo "--- sleeping ${SLEEP_BETWEEN}s before retry ---"
    sleep "$SLEEP_BETWEEN"
done
echo "GAVE UP after $MAX_PASSES passes with $(remaining) outstanding"
echo "INCOMPLETE: $(remaining) outstanding after $MAX_PASSES passes $(date)" > "$STATUS"
exit 1
