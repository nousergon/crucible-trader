# crucible-trader

The **trader**. It reads one contract, sizes a book from it, and refuses to
trade when the contract does not hold.

This is not the experiment harness. The harness is
[`nousergon/crucible`](https://github.com/nousergon/crucible) — public, AGPL,
and the durable product. The two systems are coupled by **two documents and
nothing else**:

| Document | Key | Written by | Read here |
|---|---|---|---|
| Champion pointer | `champions/{slot}/current.json` | `crucible experiment.grade` | every session |
| Predictions feed | `predictions/{trading_day}.json` | `crucible serving.publish` | every session |

The harness must complete every acceptance test with the trader switched off,
and the trader must run a week on a frozen champion with the harness switched
off. **That mutual independence is the test that they are separate systems**,
and it is why nothing in this repository reaches into the harness's job
machinery, and nothing in the harness reaches into this one.

## Why it exists

A champion that nothing trades is a verdict. A trader that trades a champion
nobody checked is an incident. This repository is the one place where the
second is structurally impossible:

- **It refuses an unattested champion.** `crucible.champion.read_champion`
  re-derives the producing run's manifest `status` on every read and raises
  `ChampionUnusableError` when it is not `ok`; an S-slot champion whose
  `pit_parity` attestation is not `PASS` raises the same way. The trader
  **calls that reader** — a second implementation of the refusals here would be
  the defect, not the deliverable.
- **It distinguishes "no champion" from "a champion I must not serve."** An
  absent pointer is a hold and, past the absence deadline, a page. An unusable
  pointer is a page **now** and a book that trades nothing. Collapsing the two
  would make a refused champion look like a quiet day.
- **There is no fallback.** It never serves yesterday's feed, never falls back
  to a different slot, and never sizes on a partial read. A refusal flattens or
  holds the book and exits non-zero.

## Zero LLM calls, ever

The fleet's standing rule confines LLM calls to research. This repository makes
none and has no LLM dependency — direct or transitive through `crucible`'s own
optional extras. The control is the absence itself: there is no call site to
register in `alpha-engine-config/private-docs/LLM_CALLSITE_REGISTRY.yaml`, which
declares call sites and has no row shape for a repository that has none. Adding a
model dependency here is a policy change, not a code change.

## Run it

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```
uv sync --frozen
uv run --frozen python -c "from crucible_trader.settings import Settings; print(Settings.from_env())"
```

Every command is `uv run --frozen`. A bare `python3 -m pytest` here measures the
laptop's global interpreter, not this repository, and produces false failures.

## Test it

```
uv run --frozen ruff check
uv run --frozen ruff format --check
uv run --frozen pytest -q --cov=crucible_trader --cov-config=pyproject.toml
```

The coverage floor in `pyproject.toml` is a **ratchet**: set at the measured
value, raised as coverage improves, never lowered to make a change pass. It is
measured over the whole `crucible_trader` package (`[tool.coverage.run] source`),
not over the files a test happened to import, and `omit` and `exclude_lines` are
both empty so the denominator cannot be narrowed into a flattering number.

## How it is deployed

The trader runs **the same wheel as the harness** (plan §4.11) and **pins a
release**: it reads `trader/release_pin` from the store and installs
`releases/{sha}/`. It never follows `releases/current`. Promotion of a release to
the trader is an explicit, off-market-hours action gated by the trader's own
smoke (connect to the paper gateway, read the book, place no order).

In this repository's own dependency graph, `crucible` is pinned by git sha in
`pyproject.toml` and locked in `uv.lock` — the same sha the wheel is built from,
so what CI grades and what the box installs are the same code.

## Where the rest lives

| You need | Go to |
|---|---|
| Repo conventions for agents | `AGENTS.md` (symlinked from `nous-ergon-ops/agent-instructions/crucible-trader/`) |
| The binding plan | `alpha-engine-config/private-docs/crucible_v2_rebuild_plan_260901.md` §3, §4.11, §9.5 |
| The tracker | `nousergon/alpha-engine-config` — issues are `alpha-engine-config-I<N>` |
| The epic | `alpha-engine-config-I9751` |
| The harness | [`nousergon/crucible`](https://github.com/nousergon/crucible) |

## Security

See [SECURITY.md](SECURITY.md). This repository is on the money path: treat every
credential, account identifier and tolerance in it as tier-1 content.
