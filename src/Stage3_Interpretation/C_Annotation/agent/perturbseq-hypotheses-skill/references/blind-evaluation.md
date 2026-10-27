# Blind evaluation — scoring a run against known answers

Used only when a run is being evaluated against a known-answer list: a predecessor paper on the
same screen, or the parent manuscript's results. **This happens after the run, in a separate
session, and the result goes in a separate file**, `eval/<run>_score.md`, next to the key and
never inside the run directory. The report itself never mentions blindness, leaks, predecessors
or scores.

## Order

1. The run finishes and its report is committed in the sterile directory. That commit is the
   freeze.
2. A new session, started outside the sterile directory, reads the frozen report at the freeze
   commit and the key.
3. It writes the score file.

## What to report

The key (a predecessor paper on the same screen, or the parent manuscript's own results) is
opened only after the report is frozen. Score the report's `supported` hypotheses against it:

- **Recovered**: a key item appears as a member of a supported hypothesis's claim (the entry's
  `members` and the claim sentence both name it). A gene mentioned only in a card's evidence
  bullets, the killed appendix or the positive controls does not count: the report does not
  assert anything about it. Nor does an entry tagged `contrast`, whose claim is that the screen
  does *not* reproduce the key item: scoring it as recovered counts a miss as a hit.
- **Missed**: key items no supported hypothesis claims. For each, say by inspection why: never
  generated, killed (quote the `kill_reason`), shelved, or filtered out by the Explorer.
- **Novel**: supported hypotheses whose members are not on the key. This is the actual output.

**Anything that changes after the freeze is disclosed in the score file**: edits to the report,
the ledger or its vocabularies (statuses, confidence labels, tags). A definition change crosses
the audit boundary as surely as a claim change does.

**Unpublished parent-manuscript claims are the one exception**, because no literature search can
find them. Supply them, but only *after* the freeze, only as deduplication ("these are already
ours, do not re-nominate them as new"), and record them separately from anything the skill
derived — so the evaluation can still measure what it would have found unaided.


## Also rate quality, not just recall

Recovering the predecessor's hits measures recall, not whether the new hypotheses are worth
doing. For each of the top N, the user fills a short rating: **interest**, **believability**,
**would fund the experiment** (1–3 each), plus one free-text line. Store it in the score file. This
is the benchmark for tuning the skill.

## Stability (planned, not yet run)

Run the loop 2–3 times independently on the same inputs and report how many of the top N recur.
Recurring hypotheses are more robust. This follows FutureHouse Robin, which ran its data agent
repeatedly and kept only findings most runs agreed on.
