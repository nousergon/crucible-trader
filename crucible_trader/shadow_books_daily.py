"""Advance every active S arm's shadow book, one session behind -- the box entry point.

`alpha-engine-config-I11545` (the v2 trader runs on the executor box) over
`alpha-engine-config-I10653`'s producer: `shadow_inputs.resolve_shadow_inputs`
assembles each ACTIVE arm's session the way the S grade does, and
`shadow_books.run_shadow_books` advances, writes and refuses. This module only
decides WHICH days, from the NYSE calendar, and runs the two.

**Fills at close, one session behind.** A decision day's book is held through
its successor session, so it can be advanced only after that successor has
closed. ``T`` is the LAST CLOSED session at run time
(`crucible.calendar.resolve_trading_day`, the binding every harness job uses):

    as_of          = T                          (the successor, now closed)
    decision_day   = the session before T       (the book being advanced)
    previous       = the session before that    (the document it advances from)

**It runs in the MORNING, not after the close.** The inputs for ``T`` --
`data/{T}/panel.parquet` above all -- are published by the v2 `data-daily`
schedule at 18:30 New York time, and the executor box is stopped by the v1
postclose pipeline at 17:10-17:50 New York time (CloudTrail, 2026-09-11 to
-24). An after-close run on the box would read a panel that does not exist
yet. Binding ``T`` to the last closed session makes the run correct at any
hour, so the box timer fires it the next morning, after the session.

and `trader/shadow_books/{decision_day}.json` is written. The phase-4 clause
`shadow_books_cover_every_active_arm` reads that document for the window's
last session, checked against the trader's own `days_served`.

Not a `crucible.runner` job, as `crucible.models.TRADER_JOB_VALUES` says of
this producer: its artifact carries its own schema and failure record. A book
that failed is written as failed and the process then exits non-zero
(`ShadowBookFailure`), which is the systemd unit's failure.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from crucible.calendar import is_trading_day, resolve_trading_day
from krepis.trading_calendar import previous_trading_day

from crucible_trader.kill_switch import utcnow
from crucible_trader.paper_smoke import open_configured_store
from crucible_trader.settings import Settings
from crucible_trader.shadow_books import run_shadow_books
from crucible_trader.shadow_inputs import resolve_shadow_inputs

__all__ = ["main", "sessions_for"]

NYSE_ZONE = ZoneInfo("America/New_York")


def sessions_for(as_of: dt.date) -> tuple[str, str, str]:
    """``(as_of, decision_day, previous)`` as ISO days, for a closed session ``as_of``."""
    decision_day = previous_trading_day(as_of)
    return (
        as_of.isoformat(),
        decision_day.isoformat(),
        previous_trading_day(decision_day).isoformat(),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], dt.datetime] = utcnow,
    printer: Callable[[str], None] = print,
) -> int:
    """Box entry point (`scripts/trader_pinned.sh shadow_books_daily`). Takes no arguments."""
    if argv:
        raise SystemExit(
            f"crucible_trader.shadow_books_daily takes no arguments (got {list(argv)})"
        )
    now = clock()
    today = now.astimezone(NYSE_ZONE).date()
    if not is_trading_day(today):
        # A holiday's run would repeat the previous morning's; the next
        # session's run picks up the same last closed session instead.
        printer(f"SKIP: {today} is not an NYSE session; the next session's run advances the books")
        return 0
    as_of, decision_day, previous = sessions_for(resolve_trading_day(now))
    store = open_configured_store(Settings.from_env(None if environ is None else dict(environ)))
    resolved = resolve_shadow_inputs(
        store, decision_day=decision_day, as_of=as_of, previous_trading_day=previous
    )
    document = run_shadow_books(
        store,
        trading_day=decision_day,
        previous_trading_day=previous,
        inputs=resolved.inputs,
        unresolved=resolved.unresolved,
        now=now,
    )
    printer(
        f"shadow books {decision_day} (as of {as_of}): {len(document['books'])} active arm(s), "
        "every book advanced"
    )
    return 0
