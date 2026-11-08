# FAQ

**Why run each question several times?**
LLMs don't answer the same way every time, even at temperature 0. One attempt per question
mistakes luck for quality. Three is the default; use an odd number so a case can't tie.

**The comparison says "Inconclusive". Is the change safe?**
Not necessarily: it means a drop of that size can't be told apart from chance with this many
cases. `agenteval power` shows how many cases you'd need. The gate can block Inconclusive
results with `--block-inconclusive`.

**Why does a right answer fail `tool_selection` or `tool_faithfulness`?**
Because it wasn't backed by the tools: the model answered from memory or worked it out in its
head. In an agent that's a risk even when the answer happens to be right.

**A run stopped halfway (Ctrl+C, a crash, the quota). Is it lost?**
No. Every finished trial is saved as it finishes. Run the same command with
`--resume <experiment id>`; only the missing trials run.

**Most trials failed with `quota_exhausted` or `rate_limit`.**
That's the provider, not your agent. Reports and comparisons label these as infrastructure
failures, and the gate won't judge a run where over 10% of trials failed this way. Use
`--rpm`/`--tpm` to stay under limits, `--max-cost` to cap spending, and run again after the
quota resets.

**My model isn't in the price table.**
`agenteval prices --update`, or pass `--price INPUT OUTPUT` (dollars per million tokens).

**Can the LLM judge be trusted?**
Only as far as you've checked it: label about 50 answers (`agenteval calibrate label`, or the
dashboard) and look at the kappa. Below 0.6, read its scores as leads, not measurements.

**Does it work in Jupyter?**
Yes: `evaluate()` works inside a running event loop, and `Results` displays as an HTML report.

**Does agenteval send my data anywhere?**
No telemetry. Only the calls you configure (your agent, the judge, `generate`) send data to
their providers. See [Privacy](guides/privacy.md).

**Why is it `pip install agenteval-kit` but `import agenteval`?**
The shorter names were taken on PyPI.
