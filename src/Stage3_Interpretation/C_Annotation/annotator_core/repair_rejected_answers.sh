#!/bin/bash
# Repair answers a validator rejected, instead of re-running the whole prompt.
#
# Run the arm's validator with --write-problems first: it leaves problems.json in each failing
# directory. This sends each such prompt again with the previous answer and the problems, and the
# model returns only the fields that change (answer_one_prompt.py, "repair"). Then run the
# validator again; repeat once if needed (at most 2 repairs per directory).
#
# Usage: repair_rejected_answers.sh <dispatch_root> <concurrency> <dir_glob>
# Env:   PYTHON (default: python), ANNOTATOR_MODEL (default: sonnet)
set -uo pipefail

DISPATCH="${1:?dispatch root}"
CONCURRENCY="${2:-4}"
DIR_GLOB="${3:?directory glob, e.g. v3_p*}"
export PYTHON="${PYTHON:-python}"
export ANNOTATOR_MODEL="${ANNOTATOR_MODEL:-sonnet}"
export ANSWER_ONE="$(cd "$(dirname "$0")" && pwd)/answer_one_prompt.py"

find "$DISPATCH" -mindepth 2 -maxdepth 2 -name problems.json -path "*/$DIR_GLOB/*" | xargs -n1 dirname | sort |
    xargs -P "$CONCURRENCY" -I{} bash -c '"$PYTHON" "$ANSWER_ONE" repair "$1"' _ {}

echo "=== repair pass finished — re-run the validator with --write-problems ==="
