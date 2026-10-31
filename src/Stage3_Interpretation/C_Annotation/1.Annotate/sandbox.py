"""Bubblewrap sandbox for one program agent: it sees only what it needs and can write only its own program folder.

run_program_agent.py points the Claude Agent SDK at a per-program wrapper (ClaudeAgentOptions(cli_path=...)), so
the whole agent — its Read / Write / Bash tools and every skill script it runs — lives inside the sandbox:

    read-only   the OS (/usr /bin /lib /lib64 /etc), the Python env running this script (python, skill
                dependencies, the SDK's bundled Claude CLI), C_Annotation/{agent, 0.Prepare, 1.Annotate, api},
                <run>/.claude/skills, <run>/Programs/meta.json, and from every other program only
                regulators_sig.tsv and label.json (regulator specificity, section C4)
    read-write  <run>/Programs/P<k>/; a fresh HOME (/sandbox-home) and /tmp; with --auth login the real
                ~/.claude/.credentials.json (token refreshes must reach the real file)
    hidden      everything else: C_Annotation/.env, other runs, other programs' outputs, the real HOME, /oak ...
    network     shared (the skills call public APIs); the environment is reduced to an allowlist

The namespaces follow DistributedAgents' runtime/sandbox.py, without the network and UTS namespaces (some HPC
kernels allow none: max_net_namespaces = max_uts_namespaces = 0, where --unshare-all fails). The root is remounted read-only, so a write outside the
writable binds fails instead of landing in a throwaway tmpfs.

    python sandbox.py --program_dir <run>/Programs/P21 --check          # spec + canary, no model call
    python sandbox.py --program_dir <run>/Programs/P21 -- bash -c '...'  # run a command inside (tests)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OS_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc")
CODE_DIRS = ("agent", "0.Prepare", "1.Annotate", "api")
SIBLING_FILES = ("regulators_sig.tsv", "label.json")
FAKE_HOME = "/sandbox-home"
ENV_KEEP = {
    "LANG",
    "LC_ALL",
    "TERM",
    "TZ",
    "OPENALEX_API_KEY",
    "OPENALEX_EMAIL",
    "OPENALEX_MAILTO",
    "NCBI_API_KEY",
    "PUBMED_EMAIL",
    "NCBI_EMAIL",
    "UNPAYWALL_EMAIL",
    "CROSSREF_MAILTO",
    "OPENAI_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    # set by the Agent SDK for the CLI it starts (claude_agent_sdk/_internal/transport/subprocess_cli.py)
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SDK_READS_SESSION_STATE",
    "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING",
    "CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS",
    "TRACEPARENT",
    "TRACESTATE",
}
ENV_PREFIXES = (
    "CLAUDE_AGENT_SDK_",
)  # not CLAUDE_CODE_*: a host Claude Code session's ids and tokens stay out


def find_bwrap(explicit=None):
    """--bwrap, else $BWRAP, else bwrap on PATH."""
    for candidate in (explicit, os.environ.get("BWRAP"), shutil.which("bwrap")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(
        "bwrap (bubblewrap) not found: pass --bwrap or set BWRAP, or run with --no_sandbox inside "
        "another enclosure"
    )


def bundled_cli():
    """The Claude CLI the Agent SDK would start (its bundled binary, else `claude` on PATH)."""
    try:
        import claude_agent_sdk

        path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
        if path.is_file():
            return str(path)
    except ImportError:
        pass
    found = shutil.which("claude")
    if not found:
        raise RuntimeError(
            "no Claude CLI found (claude_agent_sdk bundle or `claude` on PATH)"
        )
    return str(Path(found).resolve())


def sandbox_spec(program_dir, run_dir, auth="apikey", bwrap=None):
    """The mounts, environment policy and paths of one program's sandbox (a JSON-serializable dict)."""
    program_dir, run_dir = Path(program_dir).resolve(), Path(run_dir).resolve()
    cli = bundled_cli()
    ro = [r for r in OS_ROOTS if Path(r).exists()]
    ro.append(sys.prefix)  # python, the skills' dependencies, the bundled CLI
    if not Path(cli).resolve().is_relative_to(Path(sys.prefix).resolve()):
        ro.append(str(Path(cli).resolve().parent))
    ro += [str(ROOT / d) for d in CODE_DIRS]
    for p in (run_dir / ".claude" / "skills", run_dir / "Programs" / "meta.json"):
        if p.exists():
            ro.append(str(p))
    for other in sorted((run_dir / "Programs").glob("P*")):
        if other.is_dir() and other.resolve() != program_dir:
            ro += [str(other / f) for f in SIBLING_FILES if (other / f).is_file()]
    rw = [str(program_dir)]
    credentials = Path.home() / ".claude" / ".credentials.json"
    return {
        "bwrap": find_bwrap(bwrap),
        "cli": cli,
        "program_dir": str(program_dir),
        "run_dir": str(run_dir),
        "auth": auth,
        "ro": ro,
        "rw": rw,
        "credentials": str(credentials) if auth == "login" else None,
        "path": f"{sys.prefix}/bin:/usr/bin:/bin",
    }


def bwrap_args(spec, home):
    """bwrap arguments (without the command) for spec, with home as the writable fake HOME."""
    args = [
        spec["bwrap"],
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-cgroup-try",
        "--die-with-parent",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for p in spec["ro"]:
        args += ["--ro-bind", p, p]
    for p in spec["rw"]:
        args += ["--bind", p, p]
    Path(home, ".claude").mkdir(parents=True, exist_ok=True)
    args += ["--bind", str(home), FAKE_HOME]
    if spec["credentials"]:
        target = Path(home, ".claude", ".credentials.json")
        target.touch()
        args += [
            "--bind",
            spec["credentials"],
            f"{FAKE_HOME}/.claude/.credentials.json",
        ]
    args += ["--remount-ro", "/", "--chdir", spec["run_dir"], "--clearenv"]
    return args


def sandbox_env(spec, environ=None):
    """--setenv pairs: the allowlisted variables of environ, plus HOME / PATH / TMPDIR of the sandbox."""
    environ = os.environ if environ is None else environ
    keep = {
        k: v for k, v in environ.items() if k in ENV_KEEP or k.startswith(ENV_PREFIXES)
    }
    if spec["auth"] == "login":
        keep.pop("ANTHROPIC_API_KEY", None)
    keep.update({"HOME": FAKE_HOME, "PATH": spec["path"], "TMPDIR": "/tmp"})
    return [x for k, v in sorted(keep.items()) for x in ("--setenv", k, v)]


def command(spec, cmd, home, environ=None):
    return [*bwrap_args(spec, home), *sandbox_env(spec, environ), "--", *cmd]


def sandbox_dir(spec):
    """<run>/.sandbox/P<k>/: wrapper, spec and HOME of one program, outside everything the sandbox mounts (the agent
    cannot rewrite its own rules)."""
    return Path(spec["run_dir"]) / ".sandbox" / Path(spec["program_dir"]).name


def write_cli_wrapper(spec):
    """<run>/.sandbox/P<k>/claude: the CLI the SDK starts, inside the sandbox; plus sandbox.json and home/."""
    out_dir = sandbox_dir(spec)
    (out_dir / "home").mkdir(parents=True, exist_ok=True)
    (out_dir / "sandbox.json").write_text(json.dumps(spec, indent=2))
    wrapper = out_dir / "claude"
    wrapper.write_text(f"""#!{sys.executable}
# Generated by 1.Annotate/sandbox.py: start the Claude CLI inside the program's bubblewrap sandbox.
import json, os, sys
sys.path.insert(0, {str(HERE)!r})
from sandbox import command
spec = json.load(open({str(out_dir / "sandbox.json")!r}))
args = command(spec, [spec["cli"], *sys.argv[1:]], {str(out_dir / "home")!r})
os.execv(args[0], args)
""")
    wrapper.chmod(0o755)
    return wrapper


CANARY = r"""
import json, os, socket, sys
spec = json.loads(sys.argv[1]); forbidden = json.loads(sys.argv[2]); report = {"visible": [], "writable_outside": [], "errors": []}
for p in forbidden:
    if os.path.exists(p):
        report["visible"].append(p)
probe = os.path.join(spec["program_dir"], ".sandbox_canary")
try:
    open(probe, "w").write("ok"); os.remove(probe)
except OSError as e:
    report["errors"].append(f"program folder not writable: {e}")
for p in [os.path.join(spec["run_dir"], "Programs", ".x"), os.path.join(spec["run_dir"], ".x"), "/etc/.x", "/.x"]:
    try:
        open(p, "w").write("x"); report["writable_outside"].append(p)
    except OSError:
        pass
try:
    import requests, pymupdf  # noqa: F401
except Exception as e:
    report["errors"].append(f"python dependencies: {e}")
if not os.path.exists(os.path.join(spec["run_dir"], ".claude", "skills", "program-annotator-skill", "SKILL.md")):
    report["errors"].append("skills not visible")
try:
    socket.getaddrinfo("eutils.ncbi.nlm.nih.gov", 443)
except OSError as e:
    report["errors"].append(f"no DNS: {e}")
print(json.dumps(report))
"""


def forbidden_paths(spec):
    """What the agent must not see: secrets, the real HOME, other runs, other programs' outputs and evidence."""
    run_dir, program_dir = Path(spec["run_dir"]), Path(spec["program_dir"])
    paths = [ROOT / ".env", Path.home(), ROOT / "2.Synthesis", run_dir / ".sandbox"]
    paths += [
        p for p in run_dir.parent.iterdir() if p.resolve() != run_dir
    ]  # other runs / folders
    for other in (run_dir / "Programs").glob("P*"):
        if other.is_dir() and other.resolve() != program_dir:
            paths += [
                other / f
                for f in (
                    "interpretation.json",
                    "literature.jsonl",
                    "evidence.md",
                    "papers",
                )
                if (other / f).exists()
            ]
    return [str(p) for p in paths]


def canary(spec):
    """Run the checks inside the sandbox; returns (ok, report). Fails closed on any visible secret or extra write."""
    with tempfile.TemporaryDirectory(prefix="sandbox-home-") as home:
        cmd = command(
            spec,
            [
                "python",
                "-c",
                CANARY,
                json.dumps(spec),
                json.dumps(forbidden_paths(spec)),
            ],
            home,
        )
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if done.returncode:
        return False, {
            "errors": [f"sandbox did not start: {done.stderr.strip()[-500:]}"]
        }
    report = json.loads(done.stdout.strip().splitlines()[-1])
    ok = not (report["visible"] or report["writable_outside"] or report["errors"])
    return ok, report


def main():
    p = argparse.ArgumentParser(
        description="Check a program's sandbox, or run a command inside it."
    )
    p.add_argument("--program_dir", required=True)
    p.add_argument("--run_dir", default=None, help="Default: <program_dir>/../..")
    p.add_argument("--auth", default="apikey", choices=["apikey", "login"])
    p.add_argument("--bwrap", default=None)
    p.add_argument(
        "--check", action="store_true", help="Print the spec and run the canary."
    )
    p.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="-- command to run inside the sandbox"
    )
    args = p.parse_args()
    program_dir = Path(args.program_dir).resolve()
    spec = sandbox_spec(
        program_dir, args.run_dir or program_dir.parent.parent, args.auth, args.bwrap
    )
    if args.check:
        print(
            f"bwrap {spec['bwrap']}\ncli   {spec['cli']}\nrw    {spec['rw']}\nro    {len(spec['ro'])} paths"
        )
        ok, report = canary(spec)
        print(
            ("canary: ok" if ok else "canary: FAILED")
            + "\n"
            + json.dumps(report, indent=2)
        )
        return 0 if ok else 1
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        p.error("give --check or a command after --")
    with tempfile.TemporaryDirectory(prefix="sandbox-home-") as home:
        return subprocess.run(command(spec, cmd, home)).returncode


if __name__ == "__main__":
    sys.exit(main())
