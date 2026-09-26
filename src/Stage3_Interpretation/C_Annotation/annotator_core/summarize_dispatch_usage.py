"""Sum the cost and tokens of a dispatch root from the usage.jsonl files answer_one_prompt.py writes.

Usage: python summarize_dispatch_usage.py <dispatch_root> [<dispatch_root> ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    for root in map(Path, sys.argv[1:]):
        records = [json.loads(line) for path in sorted(root.glob("*/usage.jsonl"))
                   for line in path.read_text().splitlines() if line.strip()]
        if not records:
            print(f"{root}: no usage.jsonl (answered before usage logging, or not yet dispatched)")
            continue
        for kind in sorted({record["kind"] for record in records}):
            subset = [record for record in records if record["kind"] == kind]
            costs = [record.get("cost_usd") or 0 for record in subset]
            tokens = {key: sum(record["usage"].get(key) or 0 for record in subset)
                      for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
            print(f"{root} [{kind}]: {len(subset)} calls, ${sum(costs):.2f} total, ${statistics.median(costs):.3f} median; "
                  + ", ".join(f"{key.replace('_input_tokens', '').replace('_tokens', '')} {value:,}" for key, value in tokens.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
