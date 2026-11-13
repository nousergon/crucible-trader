"""The month-end close (back office 1e, `alpha-engine-config-I13122`) and the
performance statement and simulated fee bill built on it (1f, `-I13123`)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from decimal import Decimal

import pytest
from crucible.keys import data_panel_key, trader_broker_statement_key, trader_reconciliation_key
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.exception_decision import SCORE_CURRENT_KEY, load_proposals
from crucible_trader.exception_proposal import close_proposals_key
from crucible_trader.exception_record import (
    PAPER_FUND_TENANT,
    SOURCE_MONTH_END_CLOSE,
    validate_exception_record,
)
from crucible_trader.ledger import (
    CASH,
    DECLARED_FLOWS,
    DIVIDEND_INCOME,
    OPENING,
    RECEIVABLE,
    SECURITIES,
    SUSPENSE_CASH,
    SUSPENSE_POSITION,
    Entry,
    Posting,
    close_suspense_line_ids,
    journal_document,
    journal_key,
    load_journals,
    period_lock_key,
    post_daily_ledger,
    read_period_locks,
)
from crucible_trader.month_end_close import (
    CloseError,
    CloseInputs,
    _Lot,  # pyright: ignore[reportPrivateUsage]
    _move,  # pyright: ignore[reportPrivateUsage]
    close_attempt_key,
    close_key,
    close_month,
    close_prior_month,
    month_trading_days,
    panel_close_source,
    previous_month,
)
from crucible_trader.month_rebuild import (
    REBUILD_METRIC,
    _compare,  # pyright: ignore[reportPrivateUsage]
    rebuild_key,
    verify_month_rebuild,
)
from crucible_trader.performance_statement import (
    FEE_SCHEDULES,
    SIMULATED_FEE_SCHEDULE,
    StatementError,
    benchmark_returns,
    build_statement,
    file_statement,
    statement_key,
)
from crucible_trader.reconciliation import (
    CorporateActionSet,
    Fill,
    ReconciliationInputs,
    reconcile,
)
from crucible_trader.reconciliation_job import file_exception_score

T = PAPER_FUND_TENANT
NOW = "2026-12-01T21:00:00Z"
NONE = CorporateActionSet(resolved=frozenset({"AAA", "BBB"}))
CLOSES = {
    "2026-11-25": {"AAA": 10.0, "SPY": 500.0, "QQQ": 400.0},
    "2026-11-30": {"AAA": 11.0, "BBB": 21.0, "SPY": 510.0, "QQQ": 404.0},
}


def every_close(_day, tickers):
    """A settled-close source with a close for every name (I13129's daily check)."""
    return {t: 1.0 for t in tickers}


def closes(day, tickers):
    return {t: CLOSES.get(day, {})[t] for t in tickers if t in CLOSES.get(day, {})}


def stmt(day, positions, cash):
    return BrokerStatement("DU1", day, positions, cash)


def ctx_for(store, day):
    return RunContext(
        run_id="01JG0000000000000000000000",
        job="trader.reconcile",
        trading_day=dt.date.fromisoformat(day),
        calendar_date=dt.date.fromisoformat(day),
        store=store,
        seed=0,
        started=dt.datetime(2026, 12, 1, 21, 0, tzinfo=dt.UTC),
    )


def post(store, broker, anchor, fills=()):
    store.put_bytes(
        trader_broker_statement_key(broker.trading_day),
        json.dumps(broker.to_document()).encode(),
    )
    result = reconcile(
        ReconciliationInputs(
            trading_day=broker.trading_day,
            broker=broker,
            anchor=anchor,
            fills=tuple(fills),
            cash_flows=(),
            corporate_actions=NONE,
            genesis=anchor is None,
        )
    )
    document = {**result.to_document(), "control_arm": {"passed": True, "missed": []}}
    store.put_bytes(trader_reconciliation_key(broker.trading_day), json.dumps(document).encode())
    posting = post_daily_ledger(
        ctx_for(store, broker.trading_day),
        broker=broker,
        anchor=anchor,
        fills=fills,
        cash_flows=(),
        corporate_actions=NONE,
        reconciliation={**document, "void": False},
        closes=every_close,
        now="2026-11-30T21:00:00Z",
    )
    assert posting is not None
    return posting


def fill(ticker, shares, price, commission, exec_id):
    return Fill(
        ticker, shares, -shares * price - commission, commission_usd=commission, exec_id=exec_id
    )


def november(store, *, ib_cash_end=946.0, ib_realized_net=None):
    """Genesis 11-25 (10 AAA, $1,000); buy 5 BBB at 20 on 11-27; sell 4 AAA at
    12 on 11-30. Each with a $1 commission. ``ib_realized_net`` is IB's
    ``realizedPNL`` on the sale, when it states one."""
    d1 = stmt("2026-11-25", {"AAA": 10}, 1_000.0)
    post(store, d1, None)
    d2 = stmt("2026-11-27", {"AAA": 10, "BBB": 5}, 899.0)
    post(store, d2, d1, [fill("BBB", 5, 20.0, 1.0, "e1")])
    d3 = stmt("2026-11-30", {"AAA": 6, "BBB": 5}, ib_cash_end)
    sale = dataclasses.replace(fill("AAA", -4, 12.0, 1.0, "e2"), realized_pnl_usd=ib_realized_net)
    post(store, d3, d2, [sale])


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


def run(store, day="2026-12-01", **kw):
    ctx = ctx_for(store, day)
    out = close_prior_month(
        ctx,
        day=day,
        account="DU1",
        now=NOW,
        closes=closes,
        after_close=lambda c: file_statement(ctx, c, now=NOW, closes=closes),
        **kw,
    )
    return ctx, out


class TestClose:
    def test_a_sale_booked_at_ibs_realized_figure_leaves_the_close_unchanged(self, store):
        """The ledger books IB's realized gain on the sale beside a holding
        relieved at IB's cost; the close's average-cost P&L is the same."""
        november(store, ib_realized_net=19.0)  # IB's cost basis $28: gain $20 gross
        _, close = run(store)
        assert close is not None and close["status"] == "closed", close
        assert close["pnl"]["realised_cents"] == 800 and close["pnl"]["total_cents"] == 1_700

    def test_cash_rounded_to_ib_is_its_own_attribution_line_and_still_closes(self, store):
        november(store, ib_cash_end=946.03)  # 3 cents over: inside the band
        _, close = run(store)
        assert close is not None and close["status"] == "closed", close
        assert close["pnl"]["cash_rounded_to_broker_cents"] == 3
        assert close["pnl"]["total_cents"] == 1_703

    def test_a_month_that_ties_closes_locks_and_files_its_statement(self, store):
        november(store)
        ctx, close = run(store)
        assert close is not None and close["status"] == "closed", close
        assert close["nav_start_cents"] == 110_000  # $1,000 + 10 AAA at the opening close
        assert close["nav_end_cents"] == 94_600 + 6 * 1_100 + 5 * 2_100
        assert close["pnl"] == {
            "realised_cents": 800,  # 4 AAA sold for $48 net of nothing, cost $40
            "unrealised_change_cents": 600 + 500,
            "dividends_cents": 0,
            "interest_cents": 0,
            "interest_first_recorded_cents": 0,
            "declared_cash_flows_cents": 0,
            "commissions_cents": -200,
            "suspense_cents": 0,
            "reconciliation_adjustments_cents": 0,
            "cash_rounded_to_broker_cents": 0,
            "total_cents": 1_700,
        }
        assert close["positions"]["AAA"] == {
            "shares": 6,
            "close_usd": 11.0,
            "market_value_cents": 6_600,
            "cost_basis_cents": 6_000,
            "unrealised_cents": 600,
        }
        assert close["fees_accrued"] == {"commissions_cents": 200}
        assert close["exceptions"] == [] and close["trial_balance_cents"] == 0
        assert store.exists(close_key(T, "2026-11"))
        assert set(read_period_locks(store, T)["2026-11"]) == {
            "2026-11-25",
            "2026-11-27",
            "2026-11-30",
        }
        statement = json.loads(store.get_bytes(statement_key(T, "2026-11")))
        assert statement["time_weighted_return_gross"] == pytest.approx(1_700 / 110_000)
        assert statement["benchmarks"]["SPY"]["return"] == pytest.approx(0.02)
        assert statement["benchmarks"]["QQQ"]["return"] == pytest.approx(0.01)
        assert statement["simulated_fee_bill"]["booked"] is False
        assert [m["name"] for m in ctx.metrics] == ["month_end_close", "performance_statement"]
        assert [m["status"] for m in ctx.metrics] == ["OK", "OK"]

        # A later run is a no-op: nothing is rewritten and nothing recorded.
        again, same = run(store, "2026-12-02")
        assert same == {k: v for k, v in close.items() if k != "_key"} and again.metrics == []
        # And the locked month refuses a late journal (step 5d).
        locked = post_daily_ledger(
            ctx_for(store, "2026-11-30"),
            broker=stmt("2026-11-30", {"AAA": 6, "BBB": 5}, 946.0),
            anchor=None,
            fills=(),
            cash_flows=(),
            corporate_actions=NONE,
            reconciliation={},
            now=NOW,
        )
        assert locked is None

    def test_the_return_reproduces_from_raw_positions_and_closes_to_the_basis_point(self, store):
        november(store)
        _, close = run(store)
        assert close is not None
        start = 1_000.0 + 10 * CLOSES["2026-11-25"]["AAA"]
        end = 946.0 + 6 * CLOSES["2026-11-30"]["AAA"] + 5 * CLOSES["2026-11-30"]["BBB"]
        statement = json.loads(store.get_bytes(statement_key(T, "2026-11")))
        assert abs(statement["time_weighted_return_gross"] - (end / start - 1)) < 1e-4

    def test_a_suspense_balance_refuses_the_close_until_an_adjustment_clears_it(self, store):
        november(store, ib_cash_end=946.5)  # IB has 50 cents the ledger cannot explain
        ctx, refused = run(store)
        assert refused is not None and refused["status"] == "refused"
        classes = [e["class"] for e in refused["exceptions"]]
        assert classes == ["close_suspense_cash"]
        for record in refused["exceptions"]:
            validate_exception_record(record)
        assert store.exists(close_attempt_key(T, "2026-11", "2026-12-01"))
        assert not store.exists(close_key(T, "2026-11"))
        assert not store.exists(period_lock_key(T, "2026-11"))
        assert ctx.metrics[0]["status"] == "WATCH"
        # Still refused on the fourth trading day of the month: now it FAILs.
        late, _ = run(store, "2026-12-04")
        assert late.metrics[0]["status"] == "FAIL"
        # Each refused attempt files its lines for review (I13121); the queue
        # holds each line once however many times the close is refused.
        assert "1 proposal(s) filed for review" in ctx.metrics[0]["status_reason"]
        for run_day in ("2026-12-01", "2026-12-04"):
            assert store.exists(close_proposals_key(T, "2026-11", run_day))
        queued = load_proposals(store)
        assert [(p["source"], p["class"], p["trading_day"]) for p in queued] == [
            (SOURCE_MONTH_END_CLOSE, "close_suspense_cash", "2026-11-30")
        ]
        assert queued[0]["exception_id"] == refused["exceptions"][0]["exception_id"]
        assert queued[0]["proposed_at"] == NOW
        assert load_proposals(store, until="2026-11-30") == []

        # A prior-period adjustment dated in the open month clears it.
        gap = next(
            e
            for e in load_journals(store, T, before="2026-12-01")["2026-11-30"]
            if e.kind == "cash_gap"
        )
        adjustment = Entry(
            T,
            "DU1",
            "2026-12-02",
            "adjustment",
            "interest/2026-11",
            (Posting(SUSPENSE_CASH, 50), Posting(DECLARED_FLOWS, -50)),
            {"exception_ids": list(gap.evidence["exception_ids"]), "decision_id": "d1"},
        )
        store.put_bytes(
            journal_key(T, "2026-12-02"), journal_document(T, "2026-12-02", [adjustment])
        )
        _, closed = run(store, "2026-12-02")
        assert closed is not None and closed["status"] == "closed"
        assert closed["suspense_cleared_after_month_end"] == [gap.entry_id]
        assert closed["pnl"]["suspense_cents"] == 50
        assert closed["pnl"]["total_cents"] == closed["nav_end_cents"] - 110_000

    def test_a_missing_close_statement_and_ledger_tie_out_are_each_a_record(self, store):
        november(store)
        store.put_bytes(
            trader_broker_statement_key("2026-11-30"),
            json.dumps(stmt("2026-11-30", {"AAA": 7, "BBB": 5}, 940.0).to_document()).encode(),
        )
        CLOSES["2026-11-30"].pop("BBB")
        try:
            _, refused = run(store)
        finally:
            CLOSES["2026-11-30"]["BBB"] = 21.0
        assert refused is not None
        classes = sorted(str(e["class"]) for e in refused["exceptions"])
        assert classes == ["close_cash_delta", "close_input_missing", "close_share_delta"]

    def test_missing_inputs_refuse_the_close(self, store):
        november(store)
        journals = load_journals(store, T, before="2026-12-01")
        del journals["2026-11-27"]
        CLOSES["2026-11-25"].pop("AAA")
        try:
            document = close_month(
                CloseInputs(
                    T,
                    "DU1",
                    "2026-11",
                    "2026-12-01",
                    journals,
                    None,
                    None,
                    None,
                    closes,
                    NOW,
                )
            )
        finally:
            CLOSES["2026-11-25"]["AAA"] = 10.0
        details = [str(e["detail"]) for e in document["exceptions"]]
        assert document["status"] == "refused"
        assert any("opening day" in d for d in details)
        assert any("no ledger journal" in d for d in details)
        assert any("no IB statement" in d for d in details)
        assert any("no ledger view" in d for d in details)
        # Each line is its own record: one id, one proposal, one decision (I13121).
        ids = [e["exception_id"] for e in document["exceptions"]]
        assert len(set(ids)) == len(ids)

    def test_a_failed_ledger_check_on_the_last_day_refuses_the_close(self, store):
        november(store)
        view = {
            "checks": {
                "breaks": [
                    {"check": "completeness", "day": "2026-11-26", "detail": "x"},
                    {"check": "completeness", "day": "2026-11-27", "detail": "x"},
                    {"check": "roll_forward", "day": "2026-11-30", "detail": "y"},
                    {"check": "break_aging", "day": "2026-11-20", "detail": "z"},
                ]
            }
        }
        document = close_month(
            CloseInputs(
                T,
                "DU1",
                "2026-11",
                "2026-12-01",
                load_journals(store, T, before="2026-12-01"),
                stmt("2026-11-30", {"AAA": 6, "BBB": 5}, 946.0),
                view,
                None,
                closes,
                NOW,
            )
        )
        assert sorted(str(e["class"]) for e in document["exceptions"]) == [
            "close_checks_failed",
            "close_input_missing",
            "close_input_missing",
        ]
        # One check failing on two days is two lines with two ids (I13121).
        ids = [e["exception_id"] for e in document["exceptions"]]
        assert len(set(ids)) == len(ids)

    def test_income_suspense_shares_and_an_unattributed_flow_are_each_explained_or_refused(
        self, store
    ):
        november(store)
        journals = load_journals(store, T, before="2026-12-01")

        def entry(kind, ref, postings, **evidence):
            return Entry(T, "DU1", "2026-11-30", kind, ref, tuple(postings), evidence)

        journals["2026-11-30"] = (
            *journals["2026-11-30"],
            entry(
                "declared_cash_flow", "flow/x", [Posting(CASH, 25), Posting(DECLARED_FLOWS, -25)]
            ),
            entry(
                "dividend_receivable",
                "div/AAA",
                [Posting(RECEIVABLE + "AAA", 60), Posting(DIVIDEND_INCOME + "AAA", -60)],
            ),
            entry(
                "position_gap",
                "gap/BBB",
                [Posting(SECURITIES + "BBB", 0, 1), Posting(SUSPENSE_POSITION + "BBB", 0, -1)],
                exception_ids=["z"],
            ),
            # A capital contribution is an external flow the attribution has no
            # line for: the close must refuse it rather than call it P&L.
            entry(
                "adjustment",
                "capital",
                [Posting(CASH, 1_000), Posting(OPENING, -1_000)],
                exception_ids=["q"],
                decision_id="d",
            ),
        )
        document = close_month(
            CloseInputs(
                T,
                "DU1",
                "2026-11",
                "2026-12-01",
                journals,
                stmt("2026-11-30", {"AAA": 6, "BBB": 5}, 946.0),
                {"checks": "not a mapping"},
                None,
                closes,
                NOW,
            )
        )
        assert document["pnl"]["dividends_cents"] == 60
        assert document["pnl"]["declared_cash_flows_cents"] == 25
        assert document["receivables_cents"] == 60
        assert sorted(str(e["class"]) for e in document["exceptions"]) == [
            "close_attribution",
            "close_cash_delta",
            "close_share_delta",
            "close_suspense_shares",
        ]
        # The ledger derives the same id for the gap's close line, so a
        # decision on that line clears the gap (alpha-engine-config-I13160).
        (gap,) = [e for e in journals["2026-11-30"] if e.kind == "position_gap"]
        (line,) = [e for e in document["exceptions"] if e["class"] == "close_suspense_shares"]
        assert close_suspense_line_ids(gap, "2026-11-30") == [line["exception_id"]]

    def test_the_next_month_starts_from_the_prior_close(self, store):
        november(store)
        _, nov = run(store)
        assert nov is not None
        prior = json.loads(store.get_bytes(close_key(T, "2026-11")))
        anchor = stmt("2026-11-30", {"AAA": 6, "BBB": 5}, 946.0)
        for day in month_trading_days("2026-12"):
            fills = [fill("AAA", -6, 13.0, 0.0, "e3")] if day == "2026-12-15" else []
            positions = {"BBB": 5} if day >= "2026-12-15" else {"AAA": 6, "BBB": 5}
            cash = 946.0 + 78.0 if day >= "2026-12-15" else 946.0
            today = stmt(day, positions, cash)
            post(store, today, anchor, fills)
            anchor = today
        CLOSES["2026-12-31"] = {"BBB": 22.0, "SPY": 520.0}
        try:
            _, dec = run(store, "2027-01-04")
        finally:
            del CLOSES["2026-12-31"]
        assert dec is not None and dec["status"] == "closed", dec["exceptions"]
        assert dec["nav_start_cents"] == prior["nav_end_cents"]
        assert dec["pnl"]["realised_cents"] == 7_800 - 6_000
        statement = json.loads(store.get_bytes(statement_key(T, "2026-12")))
        assert statement["inputs"]["prior_statement"] == statement_key(T, "2026-11")
        assert statement["benchmarks"]["QQQ"]["return"] is None

    def test_a_refused_close_is_in_the_same_days_review_queue(self, store):
        november(store, ib_cash_end=946.5)
        ctx, refused = run(store)
        assert refused is not None
        file_exception_score(ctx, now=NOW)
        current = json.loads(store.get_bytes(SCORE_CURRENT_KEY))
        assert [(q["source"], q["class"], q["status"]) for q in current["queue"]] == [
            (SOURCE_MONTH_END_CLOSE, "close_suspense_cash", "undecided")
        ]

    def test_proposals_that_cannot_be_filed_cost_nothing_but_the_proposals(
        self, store, monkeypatch
    ):
        def broken(*_a, **_k):
            raise ValueError("proposer down")

        monkeypatch.setattr("crucible_trader.month_end_close.propose_records", broken)
        november(store, ib_cash_end=946.5)
        ctx, refused = run(store)
        assert refused is not None and refused["status"] == "refused"
        assert store.exists(close_attempt_key(T, "2026-11", "2026-12-01"))
        assert ctx.metrics[0]["status"] == "WATCH"
        assert "proposals NOT filed (ValueError: proposer down)" in ctx.metrics[0]["status_reason"]
        assert load_proposals(store) == []

    def test_the_oldest_unclosed_month_is_retried_before_advancing(self, store):
        # codex 9270d56e: November refused on 12-01 used to stay open forever,
        # because 2027-01-04 only ever attempted December (and refused on November).
        november(store, ib_cash_end=946.5)
        _, refused = run(store)
        assert refused is not None and refused["status"] == "refused"
        ctx, out = run(store, "2027-01-04")
        assert out is not None and out["month"] == "2026-11"
        assert "2026-11 close" in ctx.metrics[0]["status_reason"]
        # codex 3aac006d90588adc: November is a month overdue on 2027-01-04;
        # counting from January's first day read it as freshly refused.
        assert ctx.metrics[0]["status"] == "FAIL"

    def test_a_month_before_an_unclosed_month_is_refused_and_nothing_raises(self):
        with pytest.raises(CloseError, match="close in order"):
            close_month(
                CloseInputs(
                    tenant=T,
                    account="DU1",
                    month="2026-12",
                    run_day="2027-01-04",
                    journals={"2026-11-25": ()},
                    statement=None,
                    view=None,
                    prior=None,
                    closes=closes,
                    detected_at=NOW,
                )
            )

    def test_an_empty_ledger_has_nothing_to_close(self, store):
        ctx, out = run(store)
        assert out is None and ctx.metrics == []

    def test_a_close_written_with_other_bytes_is_a_failure_not_an_overwrite(self, store):
        november(store)
        store.put_bytes(period_lock_key(T, "2026-11"), b'{"schema_version": "x"}')
        ctx, out = run(store)
        assert out is None and ctx.metrics[0]["status"] == "FAIL"

    def test_a_metric_that_cannot_be_recorded_does_not_raise(self, store):
        class Broken:
            def __init__(self):
                self.store = store

            def record_metric(self, _):
                raise RuntimeError("no metrics today")

        november(store)
        assert close_prior_month(Broken(), day="2026-12-01", account="DU1", now=NOW) is None
        assert file_statement(Broken(), {"tenant": T}, now=NOW) is None

    def test_a_prior_close_of_the_wrong_month_is_refused(self):
        with pytest.raises(CloseError, match="finished close"):
            close_month(
                CloseInputs(
                    T,
                    "DU1",
                    "2026-12",
                    "2027-01-04",
                    {},
                    None,
                    None,
                    {"month": "2026-10", "status": "closed"},
                    closes,
                    NOW,
                )
            )

    def test_close_proposal_keys_end_in_the_day_they_were_filed(self):
        assert close_proposals_key(T, "2026-11", "2026-12-01") == (
            "trader/exception_proposals/month_end_close/nous-ergon-paper/2026-11/2026-12-01.json"
        )
        with pytest.raises(ValueError):
            close_proposals_key(T, "2026-13", "2026-12-01")

    def test_keys_and_months(self):
        assert previous_month("2026-01") == "2025-12"
        assert close_key(T, "2026-11") == "trader/ledger/close/nous-ergon-paper/2026-11.json"
        with pytest.raises(CloseError, match="YYYY-MM"):
            close_key(T, "2026-13")

    def test_lots_sell_past_zero_and_realise_leftover_cost(self):
        lots = {"AAA": _Lot(2, 2_000)}
        realised = _move(lots, "AAA", Posting("assets:securities:AAA", -3_300, -3), "fill")
        assert realised == 2_200 - 2_000 and lots["AAA"] == _Lot(-1, -1_100)

    def test_covering_part_of_a_short_realises_its_gain(self):
        # codex 416432a9d79c6d39: short 10 at $100, cover 5 at $90 realises $50.
        lots = {"AAA": _Lot(-10, -100_000)}
        realised = _move(lots, "AAA", Posting("assets:securities:AAA", 45_000, 5), "fill")
        assert realised == 50_000 - 45_000 and lots["AAA"] == _Lot(-5, -50_000)

    def test_buying_past_a_short_opens_a_long_at_its_own_price(self):
        lots = {"AAA": _Lot(-2, -20_000)}
        realised = _move(lots, "AAA", Posting("assets:securities:AAA", 27_000, 3), "fill")
        assert realised == 20_000 - 18_000 and lots["AAA"] == _Lot(1, 9_000)

    def test_the_panel_source_reads_close_raw_and_treats_a_bad_panel_as_missing(self, store):
        import pandas as pd

        frame = pd.DataFrame(
            {"trading_day": ["2026-11-30"], "ticker": ["SPY"], "close_raw": [510.0]}
        )
        store.put_bytes(data_panel_key("2026-11-30"), frame.to_parquet())
        source = panel_close_source(store)
        assert source("2026-11-30", ["SPY", "QQQ"]) == {"SPY": 510.0}
        assert source("2026-11-27", ["SPY"]) == {}


def closed(**over):
    base = {
        "status": "closed",
        "tenant": T,
        "account": "DU1",
        "month": "2026-11",
        "opened_in_month": True,
        "external_flows_cents": 0,
        "nav_start_cents": 1_000_000,
        "nav_end_cents": 1_100_000,
        "start_trading_day": "2026-11-25",
        "last_trading_day": "2026-11-30",
        "pnl": {},
    }
    return {**base, **over}


class TestStatement:
    def test_the_bill_charges_the_adviser_fee_and_performance_above_the_mark(self):
        statement = build_statement(closed(), None, {})
        bill = statement["simulated_fee_bill"]
        assert bill["aum_cents"] == 1_100_000
        # 1% / 12 of $11,000, prorated: the book opened 11-25, so 6 of 30 days.
        assert bill["adviser_fee_cents"] == 183
        assert (bill["billed_days"], bill["days_in_month"]) == (6, 30)
        after = 1_100_000 - 183
        assert bill["performance_fee_cents"] == int(
            (Decimal(after - 1_000_000) * Decimal("0.2")).quantize(Decimal(1))
        )
        assert statement["high_water_mark_cents"] == statement["shadow_net_nav_end_cents"]

    def test_a_book_open_from_the_first_trading_day_bills_the_whole_month(self):
        # alpha-engine-config-I13125: proration is for a book that opened late,
        # not for a month whose first calendar days were a weekend.
        statement = build_statement(
            closed(start_trading_day="2026-11-02", first_trading_day="2026-11-02"), None, {}
        )
        bill = statement["simulated_fee_bill"]
        assert bill["adviser_fee_cents"] == 917
        assert (bill["billed_days"], bill["days_in_month"]) == (30, 30)

    def test_an_opening_valued_in_the_prior_month_bills_the_whole_month(self):
        # aec#13237 cffb19f8: an anchor statement from 10-30 opens the book for
        # all of November; its day-of-month (30) must not prorate the bill.
        statement = build_statement(
            closed(start_trading_day="2026-10-30", first_trading_day="2026-11-02"), None, {}
        )
        bill = statement["simulated_fee_bill"]
        assert bill["adviser_fee_cents"] == 917
        assert (bill["billed_days"], bill["days_in_month"]) == (30, 30)

    def test_a_month_carried_from_a_prior_statement_bills_the_whole_month(self):
        prior = {
            "month": "2026-10",
            "tenant": T,
            "shadow_net_nav_end_cents": 1_000_000,
            "high_water_mark_cents": 1_000_000,
        }
        bill = build_statement(
            closed(opened_in_month=False, start_trading_day="2026-10-30"), prior, {}
        )["simulated_fee_bill"]
        assert bill["adviser_fee_cents"] == 917
        assert bill["billed_days"] == 30

    def test_the_first_schedule_still_bills_a_whole_month_and_says_nothing_of_days(self):
        bill = build_statement(closed(), None, {}, FEE_SCHEDULES["simulated.2026-11-03"])[
            "simulated_fee_bill"
        ]
        assert bill["adviser_fee_cents"] == 917
        assert "billed_days" not in bill and "days_in_month" not in bill

    def test_every_registered_schedule_is_keyed_by_its_own_id(self):
        assert all(k == v["schedule_id"] for k, v in FEE_SCHEDULES.items())
        assert SIMULATED_FEE_SCHEDULE in FEE_SCHEDULES.values()

    def test_a_month_below_the_mark_pays_no_performance_fee_and_keeps_the_mark(self):
        prior = {
            "month": "2026-11",
            "tenant": T,
            "shadow_net_nav_end_cents": 1_000_000,
            "high_water_mark_cents": 1_200_000,
        }
        statement = build_statement(
            closed(month="2026-12", opened_in_month=False), prior, {"SPY": {"return": 0.01}}
        )
        assert statement["simulated_fee_bill"]["performance_fee_cents"] == 0
        assert statement["high_water_mark_cents"] == 1_200_000
        assert statement["excess_return_gross"]["SPY"] == pytest.approx(0.09)

    @pytest.mark.parametrize(
        ("close", "prior", "match"),
        [
            (closed(status="refused"), None, "finished close"),
            (closed(external_flows_cents=5), None, "external flow"),
            (closed(nav_start_cents=0), None, "no return"),
            (closed(opened_in_month=False), None, "statement is missing"),
            (closed(), {"month": "2026-09", "tenant": T}, "prior statement"),
        ],
    )
    def test_refusals(self, close, prior, match):
        with pytest.raises(StatementError, match=match):
            build_statement(close, prior, {})

    def test_benchmarks_name_the_missing_day(self):
        out = benchmark_returns(closes, "2026-11-24", "2026-11-30")
        assert out["SPY"] == {"return": None, "reason": "no settled close for SPY on 2026-11-24"}

    def test_a_statement_filed_concurrently_is_not_rewritten(self, store):
        class Racing:
            def __init__(self):
                self.store = store
                self.metrics: list[object] = []

            def record_output_cas(self, *_):
                from crucible.store import PointerConflictError

                raise PointerConflictError("raced")

            def record_metric(self, m):
                self.metrics.append(m)

        ctx = Racing()
        assert file_statement(ctx, closed(), now=NOW, closes=closes) is None
        assert ctx.metrics == []

    def test_a_statement_that_cannot_be_built_is_a_fail_metric(self, store):
        ctx = ctx_for(store, "2026-12-01")
        assert file_statement(ctx, closed(status="refused"), now=NOW, closes=closes) is None
        assert ctx.metrics[0]["status"] == "FAIL"


def test_cash_account_name_is_the_ledgers():
    assert CASH == "assets:cash:USD"


class TestRebuildFromRaw:
    """1h's monthly check (`alpha-engine-config-I13129`): the close, the
    statement and the fee bill, rebuilt from stored inputs, match what was
    published; any difference is a break and a FAIL metric; it never raises."""

    def _closed(self, store):
        november(store)
        _, close = run(store)
        assert close is not None and close["status"] == "closed"

    def _verify(self, store, day="2026-12-02"):
        ctx = ctx_for(store, day)
        return ctx, verify_month_rebuild(ctx, day=day, account="DU1", now=NOW, closes=closes)

    def _rewrite(self, store, key, edit):
        document = json.loads(store.get_bytes(key))
        edit(document)
        store.put_bytes(key, json.dumps(document).encode())

    def test_a_published_month_rebuilds_exactly_and_only_once(self, store):
        self._closed(store)
        ctx, rebuilt = self._verify(store)
        assert rebuilt is not None and rebuilt["matches"] is True and rebuilt["breaks"] == []
        assert {n: d["matches"] for n, d in rebuilt["documents"].items()} == {
            "close": True,
            "statement": True,
            "fee_bill": True,
        }
        assert json.loads(store.get_bytes(rebuild_key(T, "2026-11")))["rebuilt_on"] == "2026-12-02"
        (metric,) = ctx.metrics
        assert (metric["name"], metric["status"], metric["value"]) == (REBUILD_METRIC, "OK", 0.0)
        again, none = self._verify(store, "2026-12-03")
        assert none is None and again.metrics == []

    def test_nothing_is_rebuilt_before_the_close_and_statement_are_published(self, store):
        november(store)
        ctx, rebuilt = self._verify(store, "2026-12-01")
        assert rebuilt is None and ctx.metrics == []
        assert not store.exists(rebuild_key(T, "2026-11"))

    def test_a_changed_fee_bill_is_a_break(self, store):
        self._closed(store)

        def bump(doc):
            doc["simulated_fee_bill"]["adviser_fee_cents"] += 1

        self._rewrite(store, statement_key(T, "2026-11"), bump)
        ctx, rebuilt = self._verify(store)
        assert rebuilt is not None and rebuilt["matches"] is False
        assert [(b["check"], b["document"]) for b in rebuilt["breaks"]] == [
            ("rebuild_from_raw", "fee_bill")
        ]
        assert rebuilt["documents"]["fee_bill"]["differs"] == ["adviser_fee_cents"]
        assert ctx.metrics[0]["status"] == "FAIL" and ctx.metrics[0]["value"] == 1.0

    def test_a_bill_filed_under_an_earlier_schedule_rebuilds_under_its_own_terms(self, store):
        # A schedule change must not turn every earlier month's rebuild red.
        self._closed(store)
        key = statement_key(T, "2026-11")
        published = json.loads(store.get_bytes(key))
        close = json.loads(store.get_bytes(close_key(T, "2026-11")))
        earlier = build_statement(
            close,
            None,
            published["benchmarks"],
            FEE_SCHEDULES["simulated.2026-11-03"],
        )
        store.put_bytes(key, json.dumps(earlier).encode())
        _, rebuilt = self._verify(store)
        assert rebuilt is not None and rebuilt["matches"] is True, rebuilt

    def test_a_bill_naming_an_unregistered_schedule_is_a_break(self, store):
        self._closed(store)

        def rename(doc):
            doc["simulated_fee_bill"]["schedule"]["schedule_id"] = "simulated.unregistered"

        self._rewrite(store, statement_key(T, "2026-11"), rename)
        _, rebuilt = self._verify(store)
        assert rebuilt is not None and rebuilt["matches"] is False
        assert rebuilt["documents"]["fee_bill"]["differs"] == ["schedule"]

    def test_a_changed_close_is_a_break_and_the_statement_rebuilds_from_raw(self, store):
        self._closed(store)

        def bump(doc):
            doc["nav_end_cents"] += 100

        self._rewrite(store, close_key(T, "2026-11"), bump)
        _, rebuilt = self._verify(store)
        assert rebuilt is not None
        assert [b["document"] for b in rebuilt["breaks"]] == ["close"]
        assert rebuilt["documents"]["close"]["differs"] == ["nav_end_cents"]

    def test_raw_inputs_that_no_longer_close_break_all_three(self, store):
        self._closed(store)

        def drift(doc):
            doc["cash_usd"] = 940.0

        self._rewrite(store, trader_broker_statement_key("2026-11-30"), drift)
        _, rebuilt = self._verify(store)
        assert rebuilt is not None
        assert [b["document"] for b in rebuilt["breaks"]] == ["close", "statement", "fee_bill"]
        assert "does not rebuild" in rebuilt["documents"]["statement"]["differs"][0]

    def test_a_concurrent_rebuild_is_not_rewritten(self, store):
        self._closed(store)

        class Racing:
            def __init__(self):
                self.store = store
                self.metrics: list[object] = []

            def record_output_cas(self, *_):
                from crucible.store import PointerConflictError

                raise PointerConflictError("raced")

            def record_metric(self, m):
                self.metrics.append(m)

        ctx = Racing()
        assert verify_month_rebuild(ctx, day="2026-12-02", account="DU1", now=NOW) is None
        assert ctx.metrics == []

    def test_a_failure_is_a_fail_metric_and_never_raises_even_when_the_metric_fails(self, store):
        class Broken:
            metrics: list[object] = []

            @property
            def store(self):
                raise OSError("store down")

            def record_metric(self, m):
                raise RuntimeError("metric sink down")

        assert verify_month_rebuild(Broken(), day="2026-12-02", account="DU1", now=NOW) is None

        class Exploding:
            def exists(self, key):
                raise OSError("exists down")

        class Down:
            def __init__(self):
                self.store = Exploding()
                self.metrics: list[dict[str, object]] = []

            def record_metric(self, m):
                self.metrics.append(m)

        ctx = Down()
        assert verify_month_rebuild(ctx, day="2026-12-02", account="DU1", now=NOW) is None
        assert ctx.metrics[0]["status"] == "FAIL"
        assert "rebuild NOT run" in ctx.metrics[0]["status_reason"]


def test_a_missing_field_does_not_match_an_explicit_null():
    # codex 165efe5b: {} and {"x": None} used to compare equal.
    assert _compare("k", {"a": 1}, {"a": 1, "x": None}) == {
        "key": "k",
        "matches": False,
        "differs": ["x"],
    }
    assert _compare("k", {"a": None}, {"a": None})["matches"] is True
