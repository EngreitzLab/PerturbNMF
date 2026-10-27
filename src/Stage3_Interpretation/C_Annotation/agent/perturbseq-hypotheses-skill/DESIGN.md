# Design record — `perturbseq-hypotheses` v2

v2 replaced the linear v1 pipeline on 2026-09-23. The v1 design record is in git history
(`git log -- .claude/skills/perturbseq-hypotheses/DESIGN.md`).

## 1. What v1 got wrong

Reviewed on the first blind run, on an endothelial genome-wide screen (Jesse, 2026-09-23):

- **Literature was a label, not a filter.** Hypotheses were generated, one literature pass ran,
  and each hypothesis got a verdict (`keep / reframe / downgrade / already published`). Nothing
  was revised or killed. The report printed the original claim beside a verdict saying to reframe
  it. The final list was hand-typed into a selector script.
- **The report was about the process.** The template required Summary, Goals, Procedure, Datasets
  and a gate funnel before the ranked list, which started on line 363 of 770. The title was a
  blind-scoring result. The blindness-disclosure machinery pushed "leaked", "uncontaminated" and
  "predecessor" into headings.
- **No per-hypothesis figures.** The skill said reports could be figure-free.
- **Hypotheses were observations** ("loss of <core complex> moves sixteen programs"), used
  undefined coined terms ("runs against the <X> model", with the model never stated), and leaned
  on generic core-machinery responses.
- **Not actually blind.** The skill text, the task brief, git history, sibling analyses and a
  neighbouring checkout all exposed the answers.

## 2. What v2 borrows, and from where

A literature survey of AI-scientist systems (2026-09-23):

| System | Borrowed | Source |
|---|---|---|
| Kosmos (FutureHouse/Edison) | One structured shared state between data and literature agents. Every report claim links to code or a paper. Synthesis claims are the least accurate (58%, against 85% for data claims), so they need a critic. | [arXiv 2511.02824](https://arxiv.org/abs/2511.02824) |
| POPPER | Tests designed to *falsify* a hypothesis's measurable consequences, pre-registered | [arXiv 2502.09858](https://arxiv.org/abs/2502.09858) |
| Google AI co-scientist | Generate → reflect → evolve (rewrite winners, drop losers); pairwise ranking | [arXiv 2502.18864](https://arxiv.org/abs/2502.18864) |
| Robin (FutureHouse) | Rerun the data agent; keep only effects that reproduce | [arXiv 2505.13400](https://arxiv.org/abs/2505.13400) |
| Virtual Lab | A dedicated scientific critic is essential against hallucination | [Nature 2025](https://www.nature.com/articles/s41586-025-09442-9) |
| CellVoyager | The analysis agent looks at its own figures and revises | [Nat Methods](https://www.nature.com/articles/s41592-026-03029-6) |
| Failing to Falsify (2026) | LLMs pick confirming tests; requiring a counter-example test helps | [arXiv 2604.02485](https://arxiv.org/abs/2604.02485) |

## 3. Design choices

1. **The ledger is the only shared memory.** Agents never read each other's transcripts, and the
   Writer never sees the process. That is what keeps process out of the report.
2. **Every round must change every open hypothesis.** The validator rejects a round where a
   hypothesis only gained a verdict. This is the fix for annotation without revision.
3. **At least one counter-example test per hypothesis**, registered before it runs, to counter
   confirmation bias.
4. **Survival is earned in code** (`validate_hypothesis_ledger.py`), not asserted in prose.
5. **Default N = 10**, from about 30 candidates. Fewer and better; expand once 10 looks good.
6. **Ranking is a judgement.** Interest, support and tractability, then pairwise comparison. A
   composite score is not evidence; v1's lesson on this still holds.
7. **The data-hygiene rules survive unchanged** (`references/data-hygiene.md`). They were right;
   they were just in the wrong place.
8. **Blindness is a property of the environment, not the report.** A sterile directory, a
   held-out-terms check, and scoring in a separate session. The literature agent may use prior
   papers on the same screen: the goal is new hypotheses beyond them.
9. **Plots per hypothesis**, produced by the Analyst as traces. The report shows one or two per
   card and links the rest.
10. **Cost is set by agent length, not by the skill's prose** (audit of the first two v2 runs,
    2026-09-25). Analysts were about half the cost: long agents (~95 turns) re-reading a growing
    context every turn. Next came the orchestrator's own context (to 731k, never compacted), and
    work lost when a usage limit stopped ten parallel Analysts. Reference files were 1–2%. Hence:
    short agents (one entry each), concurrency caps, resumable units, per-role brief files the
    agent reads itself, five-line hand-backs, a ledger view instead of ledger dumps, shared loaders
    written once by the Explorer, Literature only on changed claims, reruns only of the test a
    would-be survivor rests on, and Sonnet for mechanical fixes (`agent-roles.md` "Dispatching").

## 4. Open

- **Stability**: run 2–3 independent loops and compare the top 10. Planned once a single run
  looks good.
- **Quality benchmark**: the user's interest / believability / would-fund ratings on the top 10
  (`references/blind-evaluation.md`).
- **The `analysis-report` rewrite** (in progress elsewhere) may supersede parts of
  `report-template.md`.
- **Cell-type specificity** depends on reaching an external genome-wide screen. The Explorer
  records `not_tested` when it cannot.
