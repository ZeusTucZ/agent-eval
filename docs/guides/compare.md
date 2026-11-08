# Compare versions and gate CI

A "version" is any change to the agent: its prompt, model, settings, tools or code.
`agenteval compare` answers one question: **is B really better than A, or did it get lucky?**

```bash
agenteval run --name before ...
# change the agent
agenteval run --name after ...
agenteval compare 1 2
```

```
Differs in:   prompt

Pass rate:    87.3% -> 88.2%   +0.9 points   95% CI [-9.1, +10.9]
McNemar:      3 improved, 3 regressed   p = 1.000   (7 more changed only partly)
Latency:      1663ms -> 1548ms per trial, excluding waits   -115ms   95% CI [-301, +72]

Categories that moved (leads to investigate, not proof):
  long_chain        66.7% -> 41.7%  (6 cases)
  no_tool           66.7% -> 100.0%  (6 cases)

Verdict: Inconclusive. this could be chance: the true change is plausibly between -9 and +11 points.
```

In the dashboard, tick two experiments in the list and press **Compare**. In Python,
`after.compare(before)`.

## How it decides

- **Fairness first.** Different datasets are refused (a content hash catches an edited
  dataset), and so is an interrupted run. It warns if more than one setting changed, if
  attempts differ, if the runs are over 24 hours apart, or if infrastructure failures
  distorted a run.
- **Paired, case by case.** Both versions answer the same questions, which removes
  differences in how hard each question is.
- **McNemar's exact test** on the cases that cleanly flipped. A case that ties (1 of 2
  attempts) has no majority, so it isn't counted as a flip: use an odd number of attempts.
- **Bootstrap confidence intervals** resample cases, not trials, because attempts at the same
  question aren't independent.
- **Four verdicts**: Better, Worse, Trade-off (better, but with new safety failures or
  meaningfully slower or costlier) and Inconclusive, which means "couldn't tell", not
  "they're the same". `agenteval power` tells you how many cases you need to tell.

## Gate pull requests

`agenteval gate baseline.db candidate.db` turns a comparison into an exit code CI acts on:

| Result | Exit code |
|---|---|
| Better, Inconclusive, or better but slower or costlier (pass, with a warning) | 0 |
| Worse, or any new safety failure | 1 |
| Different datasets, an interrupted run, or over 10% infrastructure failures (can't judge) | 2 |

Stricter options: `--block-inconclusive` (small drops can hide in Inconclusive) and
`--min-pass-rate 0.85`.

### With the GitHub Action

Run your agent on the pull request, download a baseline database saved from `main`, then:

```yaml
- uses: ZeusTucZ/agent-eval@v1
  with:
    baseline-db: baseline.db
    candidate-db: candidate.db
    block-inconclusive: "false"
```

The job fails on a block or can't-judge, and the Markdown report goes to the job summary
(and to `steps.<id>.outputs.report`, to post as a PR comment).

### Budgeting CI runs

A run of the agent costs tokens, and free tiers are small: on Groq's, one 55-case run with 3
attempts uses about 140,000 of the 200,000 tokens allowed per day. This repository's own
workflows therefore run the gate only on pull requests labelled `run-agent-eval`, and store
the `main` baseline once instead of re-running it for every pull request. Copy them from
`.github/workflows/agent-eval*.yml`. For a cheaper check on every pull request, gate on a
fixed `--sample` of the dataset, run the same way for the baseline.
