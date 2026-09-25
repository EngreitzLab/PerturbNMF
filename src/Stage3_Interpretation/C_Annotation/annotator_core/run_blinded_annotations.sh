#!/bin/bash
# Answer every un-answered prompt under a dispatch root with a headless `claude -p` session.
#
# Why headless rather than subagents: each prompt gets its own process with NO tools and NO
# filesystem access — the prompt arrives on stdin and the answer leaves on stdout. An annotator
# therefore cannot reach an answer key, another program's prompt or anything else, regardless of
# what it decides to do. Blinding is a property of the harness, not of an instruction the model
# has to obey.
#
# "Answered" means check_answer_complete.py accepts the file (the JSON must PARSE), not merely
# that it is non-empty: `claude -p` can return a truncated answer and exit 0. An incomplete
# answer is moved aside to answer.invalid.<n>.json and the prompt is retried on the next pass.
#
# `claude -p` needs keychain and network access: run outside any sandbox.
#
# Usage: run_blinded_annotations.sh <dispatch_root> <concurrency> <dir_glob>
#   e.g. run_blinded_annotations.sh dispatch 4 "v3_p*"
# Env:   PYTHON (default: python), ANNOTATOR_MODEL (default: sonnet)
set -uo pipefail

DISPATCH="${1:?dispatch root}"
CONCURRENCY="${2:-4}"
DIR_GLOB="${3:?directory glob, e.g. v3_p*}"
export PYTHON="${PYTHON:-python}"
export ANNOTATOR_MODEL="${ANNOTATOR_MODEL:-sonnet}"
export CHECK="$(cd "$(dirname "$0")" && pwd)/check_answer_complete.py"

answer_one() {
    directory="$1"
    out="$directory/answer.json"
    "$PYTHON" "$CHECK" "$out" && { echo "skip  $(basename "$directory") (already answered)"; return 0; }
    if [ -s "$out" ]; then
        n=1; while [ -e "$directory/answer.invalid.$n.json" ]; do n=$((n+1)); done
        mv "$out" "$directory/answer.invalid.$n.json"
    fi

    tmp="$(mktemp)"
    if ! claude -p --model "$ANNOTATOR_MODEL" --allowed-tools "" < "$directory/prompt.md" > "$tmp" 2>"$tmp.err"; then
        echo "FAIL  $(basename "$directory") — $(head -c 200 "$tmp.err")"
        rm -f "$tmp" "$tmp.err"; return 1
    fi
    # Strip a stray markdown code fence (models add one even when told not to), and a leading
    # "+" on a JSON number ("log2fc": +1.008) — prompts print signed log2FCs and models copy the
    # sign, which JSON forbids.
    sed -e '1{/^```/d;}' -e '${/^```$/d;}' "$tmp" | sed -E 's/(": *)\+([0-9.])/\1\2/g' > "$out"
    rm -f "$tmp" "$tmp.err"

    if ! "$PYTHON" "$CHECK" "$out"; then
        echo "FAIL  $(basename "$directory") — incomplete answer ($(wc -c < "$out") bytes), will retry"
        return 1
    fi
    echo "ok    $(basename "$directory") ($(wc -c < "$out") bytes)"
}
export -f answer_one

find "$DISPATCH" -mindepth 1 -maxdepth 1 -type d -name "$DIR_GLOB" | sort |
    xargs -P "$CONCURRENCY" -I{} bash -c 'answer_one "$@"' _ {}

echo "=== dispatch pass finished ==="
# NOTE: `claude -p` exits NONZERO WITH EMPTY STDERR when the account hits its usage limit. Many
# bare "FAIL  <name> — " lines are almost certainly a usage limit, not a bug. Re-running is
# safe: completed prompts are skipped. dispatch_until_complete.sh does the waiting for you.
