"""The trader.

It reads one contract — `champions/{slot}/current.json` plus
`predictions/{trading_day}.json` — and refuses to trade when the contract does
not hold. It is NOT the experiment harness: the harness is `nousergon/crucible`,
consumed here as a pinned library whose readers this package calls and never
re-implements.

The mutual independence is the design: the harness completes every acceptance
test with the trader switched off, and the trader runs a week on a frozen
champion with the harness switched off (plan §3).
"""

from __future__ import annotations

#: Kept in step with `[project] version` in `pyproject.toml`. Not read from
#: package metadata: a checkout that has not been installed has no metadata, and
#: `importlib.metadata.version` raises `PackageNotFoundError` there — which is
#: how a bare `python3 -m pytest` produced eight false failures in a `crucible`
#: worktree (2026-09-11). A literal cannot fail that way.
__version__ = "0.1.0"

__all__ = ["__version__"]
