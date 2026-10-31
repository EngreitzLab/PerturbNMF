"""Per-program API spend: a ledger file per program shared by every pipeline step, and a cap.

    <budget_dir>/P<k>.json   {"program", "cap_usd", "spent_usd", "by_step": {step: usd}, "events": [...]}

Steps run as separate processes, so the ledger is re-read before every check and every charge. A call is
allowed only when its worst-case cost (the full prompt plus max_tokens of output, or a fixed estimate for
paper-qa) still fits under cap_usd - reserve_usd, where the reserve is what later steps of the same program
still need. Charges are the real costs: Claude usage x PRICES_PER_M, paper-qa's own session cost.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# USD per million tokens (input, output), Anthropic first-party API rates. Cache writes bill at 1.25x
# input, cache reads at 0.1x input.
PRICES_PER_M = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}
EMBEDDING_PER_M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
CHARS_PER_TOKEN = (
    3.0  # conservative (more tokens than the usual ~4 chars) for worst-case estimates
)


def prices(model):
    if model not in PRICES_PER_M:
        raise ValueError(
            f"no price for model {model!r}; add it to api/budget.py PRICES_PER_M"
        )
    return PRICES_PER_M[model]


def claude_cost(model, usage):
    """USD of one Claude response from its usage (input, cache write / read, output incl. thinking)."""
    p_in, p_out = prices(model)
    n_in = getattr(usage, "input_tokens", 0) or 0
    n_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    n_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    n_out = getattr(usage, "output_tokens", 0) or 0
    return (
        n_in * p_in + n_write * p_in * 1.25 + n_read * p_in * 0.1 + n_out * p_out
    ) / 1e6


def claude_worst_case(model, prompt_chars, max_tokens):
    """Upper bound of one call: the whole prompt as input, max_tokens of output."""
    p_in, p_out = prices(model)
    return (prompt_chars / CHARS_PER_TOKEN * p_in + max_tokens * p_out) / 1e6


def embedding_cost(model, n_chars):
    return n_chars / CHARS_PER_TOKEN * EMBEDDING_PER_M.get(model, 0.13) / 1e6


class ProgramBudget:
    def __init__(self, budget_dir, program, cap_usd):
        self.path = Path(budget_dir) / f"{program}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.program, self.cap = program, float(cap_usd)
        if not self.path.exists():
            self._write(
                {
                    "program": program,
                    "cap_usd": self.cap,
                    "spent_usd": 0.0,
                    "by_step": {},
                    "events": [],
                }
            )

    def _read(self):
        return json.loads(self.path.read_text())

    def _write(self, data):
        tmp = self.path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)

    @property
    def spent(self):
        return self._read()["spent_usd"]

    def allows(self, estimate_usd, reserve_usd=0.0):
        return self.spent + estimate_usd <= self.cap - reserve_usd + 1e-9

    def charge(self, step, cost_usd, what="", **detail):
        data = self._read()
        data["spent_usd"] = round(data["spent_usd"] + cost_usd, 6)
        data["by_step"][step] = round(data["by_step"].get(step, 0.0) + cost_usd, 6)
        data["events"].append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "step": step,
                "what": what,
                "usd": round(cost_usd, 6),
                **detail,
            }
        )
        self._write(data)

    def skip(self, step, what, estimate_usd):
        """Record a call that was not made because it did not fit the budget."""
        data = self._read()
        data.setdefault("skipped", []).append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "step": step,
                "what": what,
                "estimate_usd": round(estimate_usd, 4),
                "spent_usd": data["spent_usd"],
            }
        )
        self._write(data)
        print(
            f"  [budget] {self.program} {step}: skipped {what} (spent ${data['spent_usd']:.3f} of ${self.cap:.2f}, "
            f"worst case ${estimate_usd:.3f})"
        )


def add_budget_args(p, reserve_default):
    """--budget_dir / --budget_usd / --budget_reserve_usd on a step's parser."""
    p.add_argument(
        "--budget_dir",
        default=None,
        help="Per-program API spend ledgers (P<k>.json) shared by every step; calls stop once a program would pass --budget_usd. Default: <run folder>/Budget, the folder next to the bundle folders.",
    )
    p.add_argument(
        "--budget_usd",
        type=float,
        default=2.0,
        help="Spend cap per program in USD, over all steps.",
    )
    p.add_argument(
        "--budget_reserve_usd",
        type=float,
        default=reserve_default,
        help="USD of the cap this step leaves for the program's later steps.",
    )
    p.add_argument(
        "--no_budget",
        action="store_true",
        help="No ledger and no cap (not recommended).",
    )


def resolve_budget_dir(args, run_dir):
    """--budget_dir, else <run_dir>/Budget; None with --no_budget. run_dir is the folder holding the bundle folders."""
    args.budget_dir = (
        None if args.no_budget else Path(args.budget_dir or Path(run_dir) / "Budget")
    )
    if args.budget_dir:
        print(
            f"[budget] ledgers in {args.budget_dir} (cap ${args.budget_usd:.2f} per program)"
        )


def attach(handler, args, program, step):
    """Point handler (api/LLM_interface.claude_api) at program's ledger; no-op without --budget_dir."""
    if getattr(args, "budget_dir", None):
        handler.budget = ProgramBudget(args.budget_dir, program, args.budget_usd)
        handler.budget_step, handler.budget_reserve = step, args.budget_reserve_usd
    return getattr(handler, "budget", None)
