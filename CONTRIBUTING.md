# Contributing

## The five rules a change here must obey

1. **Refuse, never degrade.** Default is RAISE. A bare `except: pass`, a silent
   `return None`, or a `continue` over a contract violation is forbidden. A
   deviation carries a written rationale naming the failure mode swallowed, why
   the primary deliverable survives, and the concrete recording surface — plus
   an inline comment naming the first and the third.
2. **The harness is called, never re-implemented.** Champion refusal, attestation
   checking and feed validation live in `crucible` and are imported. A second
   implementation of any of them in this repository is a defect.
3. **Trading days always.** Every key, window, count and horizon is a NYSE
   trading day from `krepis.trading_calendar`. One week is five trading days; a
   holiday week is still one week. `calendar_date` is recorded for provenance and
   never used as a key.
4. **No LLM calls.** This repository has no LLM dependency and makes no model
   call. Adding one is a policy change, not a code change.
5. **Nothing tier-1 in the tree.** No credentials, account identifiers, ARNs,
   instance ids or tuned risk constants — even though the repository is private.
   Real values come from SSM or the strategy tree at runtime.

## Before you open a PR

```
uv sync --frozen
uv lock --check
uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen pytest -q --cov=crucible_trader --cov-config=pyproject.toml
```

All of it green locally. "Will verify in CI" is not acceptable.

A `pyproject.toml` dependency change lands with the regenerated `uv.lock` **in
the same commit** — `uv sync --frozen` does not catch a stale lockfile, and
`uv lock --check` is the thing that cannot be fooled.

## Tests

Test-driven: the failing test first, then the code that makes it pass. For a
contract, **the negative cases are the test** — a champion whose manifest is
`failed`, absent, or unreadable, and an S champion without `attestation: PASS`,
each stopping the trader dead. A happy-path-only suite over a refusal contract
grades nothing.

## Tracker and PR shape

- The tracker is `nousergon/alpha-engine-config`. Reference issues as
  `alpha-engine-config-I<N>` and PRs as `<repo>-PR<N>`.
- A closing keyword (`Closes #N`) only works within this repository. The tracker
  issue is closed **by hand**, in the same turn as the merge, with a comment
  naming the PR and stating what was verified.
- A PR must be deployable by the merge button alone. A "run this after merging"
  instruction in a PR body is not a deploy mechanism.

## Review

`@cipher813` reviews and merges. Opening the PR is not merge authority.
