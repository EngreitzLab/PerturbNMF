#!/usr/bin/env bash
# Build a sterile directory for a blind perturbseq-hypotheses run.
#
# A blind run must not be able to reach prior results on the same screen: earlier analyses,
# git history, the answer key, auto-memory, or a skill copy that names findings. v1 runs were
# contaminated through all of these. This script builds a directory that holds only the inputs,
# a scrubbed skill copy and a settings file that blocks the rest, then prints how to launch it.
#
#   setup_sterile_run.sh --run-dir ~/ph-sterile/my_study_v2 \
#       --inputs inputs_manifest.tsv \
#       --terms ~/held_out/my_study_terms.txt \
#       [--study study.yaml] [--skill-src DIR] [--analysis-report-src DIR] [--style-src DIR]
#
# inputs_manifest.tsv: <name><TAB><path>, one per line. A local path is copied into inputs/<name>.
# A path written as remote:/path/on/host is added to fetch_inputs.sh (scp over SSH, which may ask
# for a password — run it yourself). fetch_inputs.sh reads the host from STERILE_RUN_REMOTE
# (user@host, e.g. a cluster's data-transfer node).
#
# --terms: held-out terms (answer-key genes, predecessor paper title, pathway names), one per line,
# kept OUTSIDE the run directory. The run fails to build if any appears in the text it will read.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="$(dirname "$SCRIPT_DIR")"
ANALYSIS_REPORT_SRC="$HOME/Claude/projects/EngreitzLabAgents/skills/analysis-report"
STYLE_SRC="$HOME/Claude/projects/EngreitzLabAgents/style"
RUN_DIR="" INPUTS="" TERMS="" STUDY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --inputs) INPUTS="$2"; shift 2 ;;
    --terms) TERMS="$2"; shift 2 ;;
    --study) STUDY="$2"; shift 2 ;;
    --skill-src) SKILL_SRC="$2"; shift 2 ;;
    --analysis-report-src) ANALYSIS_REPORT_SRC="$2"; shift 2 ;;
    --style-src) STYLE_SRC="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_DIR" && -n "$INPUTS" ]] || { echo "need --run-dir and --inputs" >&2; exit 2; }
[[ -n "$TERMS" ]] || echo "WARNING: no --terms file; the skill copy is not checked for held-out terms" >&2

RUN_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$RUN_DIR")"
case "$RUN_DIR/" in
  "$HOME/Claude/"*) echo "run dir must be outside ~/Claude: CLAUDE.md files walk up the tree, and permission deny rules on ~/Claude/projects would also block the run itself" >&2; exit 2 ;;
esac
if [[ -e "$RUN_DIR" && -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]]; then
  echo "run dir $RUN_DIR exists and is not empty; a sterile run starts from nothing" >&2; exit 2
fi
if [[ -n "$TERMS" ]]; then
  TERMS_ABS="$(cd "$(dirname "$TERMS")" && pwd)/$(basename "$TERMS")"
  case "$TERMS_ABS" in "$RUN_DIR"/*) echo "--terms must live outside the run dir" >&2; exit 2 ;; esac
fi

# Resolve every source path to absolute BEFORE the cd below, and remember the
# manifest's own directory so relative paths inside it resolve against the
# manifest, not against $RUN_DIR.
INPUTS="$(cd "$(dirname "$INPUTS")" && pwd)/$(basename "$INPUTS")"
MANIFEST_DIR="$(dirname "$INPUTS")"
[[ -n "$STUDY" ]] && STUDY="$(cd "$(dirname "$STUDY")" && pwd)/$(basename "$STUDY")"
SKILL_SRC="$(cd "$SKILL_SRC" && pwd)"
ANALYSIS_REPORT_SRC="$(cd "$ANALYSIS_REPORT_SRC" && pwd)"
STYLE_SRC="$(cd "$STYLE_SRC" && pwd)"

mkdir -p "$RUN_DIR"/{inputs,explore,ledger/snapshots,traces,literature,report,src,.claude/skills}
cd "$RUN_DIR"
git init -q

# 1. Scrubbed skill copies. examples/ holds prior reports on real screens; the archived v1
#    templates carry the old process-first report shape; DESIGN.md is a human-facing design
#    record describing prior runs on real screens.
rsync -a --exclude examples --exclude 'references/v1-*' --exclude '__pycache__' --exclude 'DESIGN.md' \
  "$SKILL_SRC/" .claude/skills/perturbseq-hypotheses/
rsync -a --exclude '__pycache__' --exclude '.pixi' "$ANALYSIS_REPORT_SRC/" .claude/skills/analysis-report/
mkdir -p style && rsync -a --exclude '__pycache__' "$STYLE_SRC/" style/

# 2. Inputs.
cat > fetch_inputs.sh <<'FETCH'
#!/usr/bin/env bash
# Copies the remote inputs over SSH. One shared connection, so the password is asked once.
set -euo pipefail
cd "$(dirname "$0")"
REMOTE="${STERILE_RUN_REMOTE:?set STERILE_RUN_REMOTE to user@host for scp}"
# %C is a short hash: macOS $TMPDIR paths exceed the Unix-socket length limit.
mkdir -p ~/.ssh/cm
SOCKET="$HOME/.ssh/cm/%C"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$SOCKET" -o ControlPersist=15m)
ssh "${SSH_OPTS[@]}" -fN "$REMOTE"
trap 'ssh "${SSH_OPTS[@]}" -O exit "$REMOTE" 2>/dev/null || true' EXIT
FETCH
n_remote=0
while IFS=$'\t' read -r name path || [[ -n "$name" ]]; do
  [[ -z "${name// }" || "$name" == \#* ]] && continue
  if [[ "$path" == remote:* ]]; then
    echo "scp -r \"\${SSH_OPTS[@]}\" \"\$REMOTE:${path#remote:}\" 'inputs/$name'" >> fetch_inputs.sh
    n_remote=$((n_remote + 1))
  else
    case "$path" in
      /*) : ;;                          # already absolute
      *) path="$MANIFEST_DIR/$path" ;;  # relative: resolve against the manifest's directory
    esac
    cp -R "$path" "inputs/$name"
  fi
done < "$INPUTS"
if [[ $n_remote -gt 0 ]]; then chmod +x fetch_inputs.sh; else rm fetch_inputs.sh; fi
cut -f1 "$INPUTS" | grep -v '^#' | grep -v '^\s*$' > inputs/MANIFEST.txt || true
[[ -n "$STUDY" ]] && cp "$STUDY" study.yaml

# 3. Settings: no auto-memory, no reads of prior work. Permission rules cover Claude's tools;
#    the sandbox rule covers Bash.
cat > .claude/settings.json <<JSON
{
  "autoMemoryEnabled": false,
  "permissions": {
    "deny": [
      "Read(~/Claude/projects/**)", "Read(~/Claude/tasks/**)", "Read(~/Claude/PROJECTS.md)",
      "Read(~/Claude/AGENTS.md)", "Read(~/Claude/Setup History/**)", "Read(~/.claude/projects/**)",
      "Read(~/.claude/skills/perturbseq-hypotheses/**)"
    ]
  },
  "sandbox": {
    "enabled": true,
    "filesystem": {
      "denyRead": ["~/Claude/projects", "~/Claude/tasks", "~/Claude/PROJECTS.md", "~/.claude/projects"]
    }
  }
}
JSON

cat > README.md <<'MD'
# Blind hypothesis run

Inputs are in `inputs/` (see `inputs/MANIFEST.txt`). Run `/perturbseq-hypotheses` using the skill
copy in `.claude/skills/perturbseq-hypotheses/`. Work only from this directory.
MD
cat > .gitignore <<'GI'
inputs/
.pixi/
*.html
GI

# 4. Leak check over everything the run will read except the data.
if [[ -n "$TERMS" ]]; then
  scan_paths=(.claude README.md style inputs/MANIFEST.txt)
  [[ -f study.yaml ]] && scan_paths+=(study.yaml)
  [[ -f fetch_inputs.sh ]] && scan_paths+=(fetch_inputs.sh)
  python3 "$SCRIPT_DIR/check_for_leaked_terms.py" --terms "$TERMS_ABS" --paths "${scan_paths[@]}" || {
    echo "held-out terms found in the text the run will read; scrub them, then rebuild" >&2; exit 1; }
fi

git -c user.name=sterile-run -c user.email=sterile-run@localhost add -A
git -c user.name=sterile-run -c user.email=sterile-run@localhost commit -qm "Sterile run scaffold" && echo "committed scaffold in $RUN_DIR"

cat <<EOF

Sterile run ready: $RUN_DIR
$( [[ -f fetch_inputs.sh ]] && echo "1. Fetch the remote inputs yourself (export STERILE_RUN_REMOTE=user@host first; SSH may ask for a password):  $RUN_DIR/fetch_inputs.sh" )

Launch from a new terminal (not from a session started in ~/Claude):

  cd "$RUN_DIR" && CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 claude

Paste this as the FIRST message, before invoking the skill:

  Context audit before anything else. (1) Try to read
  ~/Claude/projects/PerturbNMF-paper-internal/CLAUDE.md and report only whether it was blocked.
  (2) List every CLAUDE.md and memory file loaded into your context. (3) List the skills you
  can see and the path each perturbseq-hypotheses skill loads from. Do not read anything else.

Expected: (1) blocked, (2) none, (3) the copy under $RUN_DIR/.claude/skills. If any check fails,
stop and fix the settings before running the skill.

Residual risk: large inputs on the cluster are read over SSH, and the cluster account can see
all of the shared filesystem. The skill tells the Analyst to read only the paths in inputs/MANIFEST.txt; that is
an instruction, not a block.
EOF
