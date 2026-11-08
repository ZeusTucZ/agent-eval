# Results from real runs

Runs 2 to 5 use the same 16-question dataset: large calculations, capital cities the model already knows, and one country missing from the tool's database, where the honest answer is "unknown".

| Run | Model | Result | What it showed |
|---|---|---|---|
| 2 | gemini-2.5-flash | 11/11 completed trials correct; 5 failed | **Every failure was a 429, not the model.** Inspecting individual trials found a bug in my own scorer: `tool_selection` reported 81% because trials that never got a response counted as "wrong tool". Fixed and re-scored to 100%. |
| 3 | gemini-2.5-flash | 1/16 | Requests rejected at about 2 per minute, far under the per-minute limit, which pointed to the **daily** quota. The failures were mislabelled `timeout` and wasted 15 minutes. Led to error messages on spans, cause-specific timeout labels and the circuit breaker. |
| 4 | gemini-2.5-flash | 0/16 in under 1 s | The circuit breaker worked on a real error. The stored message showed the free tier allows **20 requests/day**, fewer than one run of the dataset needs. |
| 5 | openai/gpt-oss-20b (Groq) | **48/48** | Correct tool every time, said "unknown" for the missing country, about 1 s per question, about 6.5× cheaper per success than Gemini at paid-tier prices. The one retry was on a **tokens-per-minute** limit, not requests. |
| 6 | openai/gpt-oss-20b (Groq), `hard` dataset | 49/55 (89.1%) | The hard dataset separated the categories. **Prompt injection succeeded 2 of 4 times**, and my own system prompt caused the `no_tool` failures: its "reply unknown" rule made the model refuse "spell 'dog' backwards". |
| 7, 8 | Old vs new prompt, `hard` dataset, run side by side | 87.3% vs 88.2%, **Inconclusive** | The fix worked where it was aimed (`no_tool` 67% to 100%, with `ambiguous` staying at 100%), but `long_chain` fell from 67% to 42%, so the total barely moved. Recorded tool arguments showed the model doing sums in its head, and once calling `21021 + 0` after working out the answer itself. |

Lessons:

- **Read individual trials, not just the summary.** The summary hid a scorer bug.
- **A comparison only means something if only one thing changed.** Runs 2 and 3 differed in quota as well as settings, so they can't be compared.
- **Separate infrastructure failures from model failures.** "68.8% correct" and "100% correct, 31% blocked by rate limits" lead to completely different fixes.
- **An average can hide large moves in both directions.** Runs 7 and 8 differ by 0.9 points overall, while four categories moved by 12 to 33 points.
- **A fix can work and still not be provable.** With 55 cases and 3 attempts, even a 15-point drop is caught only about half the time.
- **Test the gate on a known-bad change.** The first demonstration let a 23.6-point drop through, because tied cases counted as improvements and pushed p to 0.052. Measuring detection rates then showed my claim that 55 cases catch 10-point changes was wrong.
- **The total-vs-working latency gap in run 5 (57.8 s vs 1.6 s at p95) is my own rate limiter.** It allowed 25 requests in any 60 s window, so it sent bursts and then waited. It now spaces requests evenly and can also respect a token limit.

## What a comparison can detect

"Inconclusive" doesn't mean "no change". This table shows how often a real drop was detected, from 100 mock comparisons for each cell:

| Cases | Attempts | 5-point drop | 10-point | 15-point | 25-point |
|---|---|---|---|---|---|
| 55 | 2 | 0% | 0% | 5% | 35% |
| 55 | 3 | 3% | 19% | 50% | 93% |
| 200 | 2 | 3% | 31% | 75% | 100% |
| 500 | 2 | 27% | 82% | 99% | 100% |

Use an odd number of attempts, and more cases to catch smaller drops. The mock's faults are random on every attempt. Real regressions are often consistent, as with the damaged prompt below, so these figures are a cautious guide rather than exact rates for a real model.

Recompute this for your own pass rate and attempts with `agenteval power`.
