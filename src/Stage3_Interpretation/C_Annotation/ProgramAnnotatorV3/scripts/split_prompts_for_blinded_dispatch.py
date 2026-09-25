"""Explode a batch_request.json into one isolated directory per (arm, program).

Blinding is the point. Each annotator subagent is pointed at exactly one directory that
contains its own prompt and nothing else — no gold labels, no other arm's prompt, no other
arm's answers. The comparison is worthless if an annotator can see what it is being compared
against.

Usage:
    python split_prompts_for_blinded_dispatch.py --batch arms/v1/batch_request.json \
        --arm v1 --dispatch-root dispatch
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--dispatch-root", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.batch.read_text())
    written = []
    for request in payload["requests"]:
        match = re.match(r"topic_(\d+)", request["custom_id"])
        if not match:
            raise SystemExit(f"unexpected custom_id: {request['custom_id']}")
        program_id = match.group(1)

        params = request["params"]
        blocks = []
        if params.get("system"):
            blocks.append(
                "=== SYSTEM INSTRUCTIONS (your operating rules for this task) ===\n"
                + params["system"]
            )
        blocks.append(
            "=== TASK (answer exactly this) ===\n" + params["messages"][0]["content"]
        )

        directory = args.dispatch_root / f"{args.arm}_p{program_id}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prompt.md").write_text("\n\n".join(blocks), encoding="utf-8")
        written.append(directory)

    print(f"{args.arm}: wrote {len(written)} isolated prompt directories under {args.dispatch_root}")
    for directory in written:
        print(f"  {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
