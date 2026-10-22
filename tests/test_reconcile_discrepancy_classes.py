"""The three discrepancy classes measured on the 2026-10-19..21 trader
reconciles, pinned end to end (`alpha-engine-config-I12376`).

The nightly classification of those runs found no genuine broker-vs-ledger
drift. Every finding fell into one of three classes, and each one has a
distinct signature this module pins. A regression in any of them (a reader
that drops fills, a fill costed without its commission, a corporate-action
default that reads as a pass or hides a break) changes a signature here.

1. **Timing/snapshot boundary** (10-19 and 10-20). The reconcile ran after
   00:00Z and the gateway answered none of the session's executions, so zero
   fills were applied. Signature: one `share_delta` per missed fill, equal to
   its net shares, and one `cash_delta` equal to the missed fills' net cash,
   commissions included. The cause is the gateway's day boundary, not this
   package: an execution is assigned to the day its own timestamp names, never
   the day the reconcile happens to run. The fix is scheduling
   (nousergon-data#2244 drains the reconcile before the trading box stops).
2. **Commission costed at zero** (10-21, `cash_delta` -1.00). Fixed by
   crucible-trader#47. Signature: shares agree and cash is off by exactly the
   commission. Through `fills_from_ib` the same fill reconciles to the cent.
3. **No corporate-action source** (16 rows each night, `alpha-engine-config-
   I11076`). Signature: exactly one `corporate_action_check_incomplete` per
   instrument in the universe, carrying no numbers. These rows fail the run
   and never stand in for, or hide, a share or cash break.

The fills, prices, deltas and cash residuals below are the measured 10-20 and
10-21 values. The baseline holdings, the anchor cash, and the twelve
untraded instruments are synthetic. Only the 5.01 commission total of 10-20
was measured, so its per-fill split here is illustrative. No live gateway is
exercised: `FakeIB` answers `reqExecutions()` as `ib_async` 2.1.0 does.
"""

from __future__ import annotations

import datetime as dt

import pytest
from ib_fakes import FakeClock, FakeIB, broker_fill

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.reconcile_entrypoint import fills_from_ib, live_corporate_actions
from crucible_trader.reconciliation import (
    CASH_INSTRUMENT,
    CorporateActionSet,
    Fill,
    ReconciliationInputs,
    reconcile,
)

ACCOUNT = "DU0000001"
D19, D20, D21 = "2026-10-19", "2026-10-20", "2026-10-21"

#: Twelve held, untraded instruments, so each night's universe is the measured 16.
UNTRADED = {f"X{n:02d}": 10 * n for n in range(1, 13)}

#: 10-20's v1 fills: (symbol, side, shares, avg price, commission, time).
FILLS_1020 = (
    ("ANF", "SLD", 46, 161.14, 1.00, dt.datetime(2026, 10, 20, 13, 30, tzinfo=dt.UTC)),
    ("DECK", "SLD", 60, 87.50, 1.00, dt.datetime(2026, 10, 20, 13, 30, tzinfo=dt.UTC)),
    ("NBIX", "SLD", 195, 138.21, 1.00, dt.datetime(2026, 10, 20, 13, 30, tzinfo=dt.UTC)),
    ("DUOL", "BOT", 238, 140.53, 2.01, dt.datetime(2026, 10, 20, 15, 38, tzinfo=dt.UTC)),
)
#: 10-21's one v1 fill. Its time of day is illustrative.
FILL_1021 = ("DUOL", "BOT", 39, 142.44, 1.00, dt.datetime(2026, 10, 21, 13, 45, tzinfo=dt.UTC))

ANCHOR_1019 = BrokerStatement(
    account=ACCOUNT,
    trading_day=D19,
    positions={"ANF": 146, "DECK": 160, "NBIX": 495, "DUOL": 279, **UNTRADED},
    cash_usd=50_000.00,
)
BROKER_1020 = BrokerStatement(
    account=ACCOUNT,
    trading_day=D20,
    positions={"ANF": 100, "DECK": 100, "NBIX": 300, "DUOL": 517, **UNTRADED},
    cash_usd=56_162.24,  # measured: +6,162.24 over the anchor
)
BROKER_1021 = BrokerStatement(
    account=ACCOUNT,
    trading_day=D21,
    positions={**BROKER_1020.positions, "DUOL": 556},
    cash_usd=round(56_162.24 - 39 * 142.44 - 1.00, 2),
)


def records(*rows):
    return [
        broker_fill(sym, side=side, shares=n, avg_price=px, commission=fee, time=at)
        for sym, side, n, px, fee, at in rows
    ]


def everything_resolved(*statements):
    """A corporate-action set resolving the whole universe with no actions,
    so a test isolates the share and cash checks."""
    return CorporateActionSet(resolved=frozenset().union(*(s.positions for s in statements)))


def run(day, prev, anchor, broker, fills, corporate_actions):
    return reconcile(
        ReconciliationInputs(
            trading_day=day,
            broker=broker,
            anchor=anchor,
            fills=tuple(fills),
            corporate_actions=corporate_actions,
            previous_trading_day=prev,
        )
    )


def findings(result, *klasses):
    return [
        (d.klass, d.instrument, d.delta, d.side)
        for d in result.discrepancies
        if not klasses or d.klass in klasses
    ]


class TestTimingBoundary:
    """10-20, reconciled at 00:36Z 10-21: the gateway answered no executions."""

    def test_a_gateway_answering_no_executions_grades_each_missed_fill_as_a_delta(self):
        after_midnight = FakeIB(clock=FakeClock(dt.datetime(2026, 10, 21, 0, 36, tzinfo=dt.UTC)))
        fills = fills_from_ib(after_midnight, D20)
        assert fills == ()

        result = run(D20, D19, ANCHOR_1019, BROKER_1020, fills, everything_resolved(BROKER_1020))

        assert findings(result) == [
            ("share_delta", "ANF", -46.0, "broker_under"),
            ("share_delta", "DECK", -60.0, "broker_under"),
            ("share_delta", "DUOL", 238.0, "broker_over"),
            ("share_delta", "NBIX", -195.0, "broker_under"),
            ("cash_delta", CASH_INSTRUMENT, 6_162.24, "broker_over"),
        ]

    def test_every_delta_traces_to_a_missed_fill_and_its_commission(self):
        """The class's signature: what the broker moved is exactly what the
        reconcile did not apply, cash net of the 5.01 commission total."""
        missed = fills_from_ib(FakeIB(broker_fills=records(*FILLS_1020)), D20)
        result = run(D20, D19, ANCHOR_1019, BROKER_1020, (), everything_resolved(BROKER_1020))

        share = {d.instrument: d.delta for d in result.discrepancies if d.klass == "share_delta"}
        assert share == {f.ticker: float(f.shares) for f in missed}
        assert result.cash_residual_usd == pytest.approx(sum(f.cash_usd for f in missed), abs=0.005)
        notional = 7_412.44 + 5_250.00 + 26_950.95 - 33_446.14
        assert result.cash_residual_usd == pytest.approx(notional - 5.01, abs=0.005)

    def test_executions_read_after_utc_midnight_still_belong_to_their_session_day(self):
        """Nothing in this package keys a fill to the run clock: the same
        executions, answered at 00:36Z the next day, reconcile 10-20 clean.
        So the class lives at the gateway's day boundary, which is why the
        fix is to run the reconcile before the box stops (nousergon-data#2244)."""
        ib = FakeIB(
            clock=FakeClock(dt.datetime(2026, 10, 21, 0, 36, tzinfo=dt.UTC)),
            broker_fills=records(*FILLS_1020),
        )
        fills = fills_from_ib(ib, D20)
        assert len(fills) == 4

        result = run(D20, D19, ANCHOR_1019, BROKER_1020, fills, everything_resolved(BROKER_1020))
        assert result.clean, [d.page_line() for d in result.discrepancies]


class TestCommissionCostedAtZero:
    """10-21: DUOL BOT 39 @ 142.44 with a 1.00 commission, graded -1.00 cash."""

    def test_a_fill_costed_without_its_commission_is_a_cash_delta_of_exactly_that(self):
        costed_at_zero = Fill(ticker="DUOL", shares=39, cash_usd=-39 * 142.44)
        result = run(
            D21, D20, BROKER_1020, BROKER_1021, (costed_at_zero,), everything_resolved(BROKER_1021)
        )
        assert findings(result) == [("cash_delta", CASH_INSTRUMENT, -1.0, "broker_under")]

    def test_the_fill_read_through_fills_from_ib_reconciles_to_the_cent(self):
        """`FakeIB.reqExecutions()` answers with the SDK's unattached default
        report (commission 0.0), as `ib_async` 2.1.0 does; the commission is
        read from the session store's report for the same execId (#47)."""
        fills = fills_from_ib(FakeIB(broker_fills=records(FILL_1021)), D21)
        assert fills == (Fill(ticker="DUOL", shares=39, cash_usd=pytest.approx(-5_556.16)),)

        result = run(D21, D20, BROKER_1020, BROKER_1021, fills, everything_resolved(BROKER_1021))
        assert result.clean, [d.page_line() for d in result.discrepancies]


class TestNoCorporateActionSource:
    """16 rows each night: no live corporate-action source (I11076)."""

    def test_each_instrument_is_one_check_incomplete_row_and_the_run_is_not_clean(self):
        fills = fills_from_ib(FakeIB(broker_fills=records(FILL_1021)), D21)
        result = run(D21, D20, BROKER_1020, BROKER_1021, fills, live_corporate_actions())

        universe = sorted(set(BROKER_1020.positions) | set(BROKER_1021.positions))
        assert len(universe) == 16
        assert findings(result) == [
            ("corporate_action_check_incomplete", ticker, None, "trader") for ticker in universe
        ]
        assert all((d.expected, d.actual) == (None, None) for d in result.discrepancies), (
            "a missing source is not a measured action"
        )
        assert not result.clean
        assert (result.n_comparable, result.match_rate) == (0, None)

    def test_check_incomplete_rows_never_hide_a_share_or_cash_break(self):
        """10-20 as it ran: 4 share deltas, 1 cash delta and 16 check-incomplete
        rows, 21 findings, with the breaks listed ahead of the missing source."""
        result = run(D20, D19, ANCHOR_1019, BROKER_1020, (), live_corporate_actions())

        order = [d.klass for d in result.discrepancies]
        assert order == (
            ["share_delta"] * 4 + ["cash_delta"] + ["corporate_action_check_incomplete"] * 16
        )
        assert findings(result, "share_delta", "cash_delta") == findings(
            run(D20, D19, ANCHOR_1019, BROKER_1020, (), everything_resolved(BROKER_1020))
        )
