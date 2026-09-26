# Lessons

- **Present the plan before editing when the request comes from outside feedback** (reviewer
  comments, Slack threads). Each point hides a scope choice (general workflow vs. one-off
  viewer, show vs. remove) that only Jesse can make. Map feedback -> proposed change -> open
  decisions, then wait. (2026-09-25: started editing the viewer before review; comparators
  and family/distinguisher went the opposite way from what I'd have built.)

- **Before editing or rebuilding an artifact, check for unmerged branches/worktrees that touched
  it** (`git branch --no-merged main`, `git worktree list`). 2026-09-25: regenerated the benchmark
  viewers from main while the earlier section-order/volcano feedback sat on
  `claude/jolly-lehmann-ef4477`, unmerged. Jesse had to ask where it went.

## Annotator cost: cut the harness, not the prompt (2026-09-26)
- The expensive part of each `claude -p` call was Claude Code's own context (~45k tokens), not the
  prompt. The minimal call (5fa73bc) halved program cost with no quality change.
- Trimming the program prompt (TASK/OUTPUT into the cached system block, terse confounder
  evidence, high/medium regulators only, compacted evidence) lost a 50-program blind Opus A/B
  39-6-5 (biology 27-1) for ~$2 saved per 50 programs. Abandoned; record, data and scripts on
  branch `annotator-cost-cut` and in ~/Claude/projects/annotator-cost-cut-ab50/.
- Rule: validate any prompt-content change with a blinded pairwise judge on the full set before
  adopting it; 5-item spot checks passed the gate and still hid the regression.
