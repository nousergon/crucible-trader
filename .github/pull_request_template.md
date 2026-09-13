## What & why

<!-- Reference the tracked issue (`alpha-engine-config-I<N>` — that repo is this
     one's tracker), a concrete observed defect, or a named ruling. A closing
     keyword only works for an issue in THIS repository; the tracker issue is
     closed by hand after merge. -->

## Measured

<!-- What was measured, against what, and when. Not "should work". -->

## Test plan

<!-- Confirm the suite passed BEFORE opening this PR. "Will verify in CI" is not
     acceptable.
       uv lock --check
       uv run --frozen ruff check && uv run --frozen ruff format --check
       uv run --frozen pytest -q --cov=crucible_trader --cov-config=pyproject.toml
     For a contract change, name the NEGATIVE cases covered. -->

## Money-path impact

**Money-path:** <!-- `none`, or name what changes about orders, sizing, the
kill switch, the account guard, reconciliation, or the refusal path. A change
that can alter an order is never bundled with anything else. -->

## Deploy mechanism

**Deploy-mechanism:** <!-- the workflow that fires on merge, or `N/A`. This
repository pins a release explicitly; a merge here does not move
`trader/release_pin`. -->

## Out of scope

<!-- Findings this PR does not fix. Each one filed on
     `nousergon/alpha-engine-config` with its issue number named here. -->

---

Prepared by: <model-name> via [Claude Code](https://claude.com/claude-code)
