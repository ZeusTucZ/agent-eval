# Security

## Reporting a problem

Please report security problems privately, through GitHub's
[private vulnerability reporting](https://github.com/ZeusTucZ/agent-eval/security/advisories/new),
not in a public issue. Include what's affected, how to reproduce it and the impact you see.
You'll get a reply within a week; fixes for confirmed problems ship as a patch release, with
credit if you'd like it.

## Supported versions

The latest 1.x release gets security fixes.

## Running it safely

- `agenteval ui` listens on 127.0.0.1 only and makes up a token for each session.
- A deployed API server refuses every change without its `API_TOKEN`, and all of them with
  `READ_ONLY=1`; keep the token secret and use read-only for public demos.
- `--agent module:function` and `--scorer module:function` import and run Python code, as
  importing any module does: only point them at code you trust.
- Databases hold the questions, answers and tool data of every run. Use `--redact` and keep
  them out of public places. See [Privacy](docs/guides/privacy.md).
