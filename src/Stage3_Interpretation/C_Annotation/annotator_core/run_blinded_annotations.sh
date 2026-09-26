#!/bin/bash
# Answer every un-answered prompt under a dispatch root with a headless `claude -p` session.
#
# Why headless rather than subagents: each prompt gets its own process with NO tools, NO Claude
# Code context and a neutral working directory — the prompt arrives on stdin and the answer leaves
# on stdout. An annotator therefore cannot reach an answer key, another program's prompt or
# anything else. Blinding is a property of the harness, not of an instruction the model has to
# obey. The per-directory work (skip check, prompt hash, the call, usage log) is in
# answer_one_prompt.py, whose docstring explains each flag.
#
# "Answered" means the answer is complete (check_answer_complete.py: the JSON must PARSE) AND was
# produced from the current prompt (answer.prompt_sha256). A changed prompt is re-dispatched.
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
export ANSWER_ONE="$(cd "$(dirname "$0")" && pwd)/answer_one_prompt.py"

find "$DISPATCH" -mindepth 1 -maxdepth 1 -type d -name "$DIR_GLOB" | sort |
    xargs -P "$CONCURRENCY" -I{} bash -c '"$PYTHON" "$ANSWER_ONE" answer "$1"' _ {}

echo "=== dispatch pass finished ==="
# NOTE: `claude -p` exits NONZERO WITH EMPTY STDERR when the account hits its usage limit. Many
# bare "FAIL  <name> — claude exited 1:" lines are almost certainly a usage limit, not a bug.
# Re-running is safe: completed prompts are skipped. dispatch_until_complete.sh does the waiting.
# Cost of the pass: summarize_dispatch_usage.py <dispatch_root>.
