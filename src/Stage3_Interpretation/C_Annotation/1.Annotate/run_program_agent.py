"""One Claude agent per program: each works Programs/P<k>/plan.md with the program-annotator skill.

    python run_program_agent.py --run_dir <out_dir>/<run_name>/Annotation --programs 10 11 --workers 2

Before the agents start, the skills in C_Annotation/agent/*-skill are linked into <run_dir>/.claude/skills, so the
agents (Claude Agent SDK, cwd = run_dir) and an interactive Claude Code session opened in the run folder see the
same skills. --link_only does only that and prints how to annotate a program by hand in Claude Code (no API cost
beyond your Claude Code plan).

Each agent runs in a Bubblewrap sandbox (sandbox.py, on by default): it sees the OS, the Python env, the skills, its own
program folder (the only writable path) and two small tables of every other program, and nothing else — not
C_Annotation/.env, other runs, other programs' outputs or the real HOME. A canary checks that before any model call
and the program is skipped when it fails. --no_sandbox turns it off (a PreToolUse hook still denies Write / Edit
outside the program folder, but Bash is then unrestricted). Its messages go to Programs/P<k>/_runs/agent_transcript.jsonl and its outcome (cost, turns, plan
status before and after) to Programs/P<k>/_runs/agent.json. Programs whose required steps are all done are skipped.
Keys come from C_Annotation/.env (ANTHROPIC_API_KEY for --auth apikey, OPENALEX_API_KEY, OPENAI_API_KEY, ...).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "0.Prepare"), str(HERE)]
from program_dir import list_programs, mark, read_plan  # noqa: E402

AGENT_DIR = ROOT / "agent"
DEFAULT_EXCLUDE = ("perturbseq-hypotheses-skill",)  # screen-wide, not per program
TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"]
TASK = (
    "Annotate program {pid} of this PerturbNMF screen with the program-annotator-skill: work through "
    "Programs/{pid}/plan.md until every required step is done. The run folder is your working directory; "
    "write only inside Programs/{pid}/. Use `python` for the skill scripts.{extra}"
)
OPTIONAL_TASK = (
    " Also do the optional step(s) {steps} (they are open in plan.md): follow the program-annotator-skill's "
    "references for them and close each with plan.py done."
)


def load_env(path):
    env = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def link_skills(run_dir, exclude=DEFAULT_EXCLUDE):
    """<run_dir>/.claude/skills/<name> -> C_Annotation/agent/<name>; returns the skill names."""
    target = Path(run_dir) / ".claude" / "skills"
    target.mkdir(parents=True, exist_ok=True)
    names = []
    for skill in sorted(AGENT_DIR.glob("*-skill")):
        if skill.name in exclude or not (skill / "SKILL.md").exists():
            continue
        link = target / skill.name
        if link.is_symlink() and link.resolve() != skill.resolve():
            link.unlink()
        if not link.exists():
            link.symlink_to(skill)
        names.append(skill.name)
    return names


# a reopened step also reopens the steps built on its outputs
DEPENDENTS = {
    "gene_info": ("regulator_context", "interpretation"),
    "interactions": ("regulator_context", "interpretation"),
    "literature": ("interpretation",),
    "regulator_context": ("interpretation",),
    "label": ("interpretation",),
}


def reopen_stale(program_dir):
    """Re-run the checks of every done required step; reopen ([ ]) the failing ones and their dependents.
    Returns {step: reason}."""
    from validate_program import validate_step

    plan = read_plan(program_dir)
    reopened = {}
    for s in plan.steps:
        if s.optional or s.status != "done" or s.id in reopened:
            continue
        errors = validate_step(program_dir, s.id)
        if errors:
            reopened[s.id] = "; ".join(errors)[:200]
            for dep in DEPENDENTS.get(s.id, ()):
                reopened.setdefault(dep, f"reopened with {s.id}")
    for step, why in reopened.items():
        if plan.step(step).status == "done":
            mark(
                program_dir,
                step,
                "todo",
                note=f"reopened: {why}",
                by="run_program_agent.py --redo_stale",
            )
    return reopened


def plan_summary(program_dir):
    return {s.id: s.status for s in read_plan(program_dir).steps}


def write_guard(program_dir):
    """PreToolUse hook: deny Write / Edit outside the program folder."""
    allowed = Path(program_dir).resolve()

    async def hook(input_data, tool_use_id, context):
        path = (input_data.get("tool_input") or {}).get("file_path")
        if path and not Path(path).resolve().is_relative_to(allowed):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"write only inside {allowed}",
                }
            }
        return {}

    return hook


async def run_program(program_dir, run_dir, skills, env, args, sem):
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        HookMatcher,
        ResultMessage,
        ToolUseBlock,
        query,
    )

    pid = program_dir.name
    runs = program_dir / "_runs"
    runs.mkdir(exist_ok=True)
    before = plan_summary(program_dir)
    sandbox_record, cli_path = {"enabled": False}, None
    if not args.no_sandbox:
        from sandbox import canary, sandbox_spec, write_cli_wrapper

        spec = sandbox_spec(program_dir, run_dir, args.auth, args.bwrap)
        ok, report = canary(spec)
        sandbox_record = {
            "enabled": True,
            "spec": spec,
            "canary": report,
            "canary_ok": ok,
        }
        if not ok:
            print(f"[{pid}] sandbox canary FAILED; skipped: {report}")
            (runs / "agent.json").write_text(
                json.dumps(
                    {
                        "program": pid,
                        "skipped": "sandbox canary failed",
                        "sandbox": sandbox_record,
                    },
                    indent=2,
                )
            )
            return {"program": pid, "cost_usd": None, "skipped": True}
        cli_path = str(write_cli_wrapper(spec))
    options = ClaudeAgentOptions(
        cwd=str(run_dir),
        skills=skills,
        setting_sources=["project"],
        allowed_tools=TOOLS,
        permission_mode="bypassPermissions",
        model=args.model,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        env=env,
        cli_path=cli_path,
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Write|Edit", hooks=[write_guard(program_dir)])
            ]
        },
    )
    async with sem:
        t0, result, n_assistant, n_tools = time.time(), None, 0, 0
        print(f"[{pid}] agent started")
        with (runs / "agent_transcript.jsonl").open("a") as log:
            try:
                async for msg in query(
                    prompt=TASK.format(
                        pid=pid,
                        extra=OPTIONAL_TASK.format(steps=", ".join(args.do_optional))
                        if args.do_optional
                        else "",
                    ),
                    options=options,
                ):
                    log.write(
                        json.dumps(
                            {"type": type(msg).__name__, "message": repr(msg)[:20000]}
                        )
                        + "\n"
                    )
                    log.flush()
                    if isinstance(
                        msg, AssistantMessage
                    ):  # counted here: a sub-agent can end with its own
                        n_assistant += (
                            1  # ResultMessage, whose num_turns is not the session's
                        )
                        n_tools += sum(isinstance(b, ToolUseBlock) for b in msg.content)
                    if isinstance(msg, ResultMessage) and (
                        result is None
                        or (msg.total_cost_usd or 0) >= (result.total_cost_usd or 0)
                    ):
                        result = msg  # the cumulative (largest-cost) result
            except Exception as e:  # one failing program must not stop the others
                log.write(
                    json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"})
                    + "\n"
                )
                print(f"[{pid}] agent failed: {type(e).__name__}: {e}")
    after = plan_summary(program_dir)
    record = {
        "program": pid,
        "model": args.model,
        "started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t0, 1),
        "cost_usd": getattr(result, "total_cost_usd", None),
        "num_turns": n_assistant,
        "tool_calls": n_tools,
        "is_error": getattr(result, "is_error", True),
        "stop": getattr(result, "subtype", None),
        "summary": (getattr(result, "result", None) or "")[:2000],
        "plan_before": before,
        "plan_after": after,
        "sandbox": {k: v for k, v in sandbox_record.items() if k != "spec"}
        | ({"spec": str(Path(cli_path).parent / "sandbox.json")} if cli_path else {}),
    }
    if (
        runs / "agent.json"
    ).exists():  # keep the earlier sessions' records (agent.<started>.json)
        prev = json.loads((runs / "agent.json").read_text())
        (runs / "agent.json").rename(
            runs / f"agent.{str(prev.get('started', 'earlier')).replace(':', '')}.json"
        )
    (runs / "agent.json").write_text(json.dumps(record, indent=2))
    todo = [s for s, st in after.items() if st not in ("done", "skipped")]
    print(
        f"[{pid}] agent finished: ${record['cost_usd'] or 0:.2f}, {record['num_turns']} turns, "
        + (
            "all required steps done"
            if not [s for s in todo if s != "driver_genes"]
            else f"not done: {todo}"
        )
        + "".join(f"; {s}: {after.get(s)}" for s in args.do_optional)
    )
    return record


async def main_async(args):
    run_dir = Path(args.run_dir).resolve()
    skills = link_skills(run_dir)
    if args.link_only:
        print(
            f"[skills] {len(skills)} skills linked into {run_dir / '.claude' / 'skills'}"
        )
        print(
            f"Annotate by hand: cd {run_dir} && claude, then ask: annotate P<k> with the program-annotator-skill"
        )
        return 0
    folders = {int(d.name[1:]): d for d in list_programs(run_dir)}
    pids = args.programs or sorted(folders)
    if args.redo_stale:
        for p in pids:
            if p in folders:
                again = reopen_stale(folders[p])
                if again:
                    print(f"[redo] P{p}: reopened {', '.join(again)}")
    for p in pids if args.do_optional else ():
        if (
            p in folders
        ):  # reopen the requested optional steps (e.g. driver_genes skipped earlier)
            plan = read_plan(folders[p])
            for step in args.do_optional:
                if plan.step(step).status != "done":
                    mark(
                        folders[p],
                        step,
                        "todo",
                        note="requested (--do_optional)",
                        by="run_program_agent.py",
                    )

    def has_work(d):
        plan = read_plan(d)
        return plan.next_todo() is not None or any(
            plan.step(s).status == "todo" for s in args.do_optional
        )

    todo = [folders[p] for p in pids if p in folders and has_work(folders[p])]
    skipped = sorted(set(pids) - {int(d.name[1:]) for d in todo})
    if skipped:
        print(
            f"[skip] no required step left (or no folder): {', '.join(f'P{p}' for p in skipped)}"
        )
    file_env = load_env(args.env_file)
    env = {
        k: v
        for k, v in file_env.items()
        if k != "ANTHROPIC_API_KEY" or args.auth == "apikey"
    }
    if args.auth == "login":
        env["ANTHROPIC_API_KEY"] = (
            ""  # use the Claude Code login instead of the API key
        )
    sem = asyncio.Semaphore(args.workers)
    records = await asyncio.gather(
        *(run_program(d, run_dir, skills, env, args, sem) for d in todo)
    )
    total = sum(r["cost_usd"] or 0 for r in records)
    print(
        f"[done] {len(records)} program(s), ${total:.2f}; check with: python status.py --run_dir {run_dir} --recheck"
    )
    return 0


def main():
    p = argparse.ArgumentParser(
        description="One Claude agent per program, working its plan.md with the program-annotator skill."
    )
    p.add_argument(
        "--run_dir", required=True, help="Annotation run folder (holds Programs/)."
    )
    p.add_argument(
        "--programs",
        type=int,
        nargs="+",
        default=None,
        help="Program ids. Default: every program with a required step left.",
    )
    p.add_argument(
        "--workers", type=int, default=2, help="Programs annotated at the same time."
    )
    p.add_argument(
        "--model", default="claude-sonnet-5", help="Claude model of the agents."
    )
    p.add_argument(
        "--max_turns", type=int, default=250, help="Max agent turns per program."
    )
    p.add_argument(
        "--max_budget_usd", type=float, default=3.0, help="Spend cap per program (USD)."
    )
    p.add_argument(
        "--auth",
        default="apikey",
        choices=["apikey", "login"],
        help="apikey: ANTHROPIC_API_KEY from --env_file; login: the Claude Code login of this machine.",
    )
    p.add_argument(
        "--env_file",
        default=str(ROOT / ".env"),
        help="KEY=value file loaded into the agents' environment.",
    )
    p.add_argument(
        "--redo_stale",
        action="store_true",
        help="Before starting, re-run the checks of every done step and reopen the failing ones (and the steps built on them), e.g. after a step's checks were tightened.",
    )
    p.add_argument(
        "--do_optional",
        nargs="+",
        default=[],
        choices=["driver_genes"],
        help="Also do these optional steps (reopened when skipped before), e.g. driver_genes for programs already annotated.",
    )
    p.add_argument(
        "--no_sandbox",
        action="store_true",
        help="Run the agents without the Bubblewrap sandbox (only inside another enclosure).",
    )
    p.add_argument(
        "--bwrap",
        default=None,
        help="bubblewrap executable. Default: $BWRAP, else bwrap on PATH.",
    )
    p.add_argument(
        "--link_only",
        action="store_true",
        help="Only link the skills into <run_dir>/.claude/skills and print how to annotate in Claude Code.",
    )
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
