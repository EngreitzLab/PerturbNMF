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
