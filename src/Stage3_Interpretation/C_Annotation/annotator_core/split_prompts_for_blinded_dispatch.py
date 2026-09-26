"""Explode a batch_request.json into one isolated directory per (arm, program).

Blinding is the point. Each annotator call is pointed at exactly one directory that
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
        directory = args.dispatch_root / f"{args.arm}_p{program_id}"
        directory.mkdir(parents=True, exist_ok=True)
        # The static instructions go to system.md, sent as the system prompt and cached across
        # the batch; the per-item data goes to prompt.md. See answer_one_prompt.py.
        system_path = directory / "system.md"
        if params.get("system"):
            system_path.write_text(params["system"], encoding="utf-8")
        elif system_path.exists():
            system_path.unlink()
        (directory / "prompt.md").write_text(params["messages"][0]["content"], encoding="utf-8")
        written.append(directory)

    print(f"{args.arm}: wrote {len(written)} isolated prompt directories under {args.dispatch_root}")
    for directory in written:
        print(f"  {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
