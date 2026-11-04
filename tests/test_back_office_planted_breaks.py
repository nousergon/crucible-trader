"""Back office stage 3: planted test breaks (`alpha-engine-config-I13125`).

nous-ergon-ops `plans/2026-11-02-back-office-implementation.md` § Schedule,
stage 3: "add planted test breaks to prove the scorer can fail"; gate
"planted breaks all caught". The Control standard's planted-controls row adds
"stage 3 extends plants to the close, the statement and the bill".

Every plant here is a KNOWN error injected into a synthetic November book run
through the real job body (:func:`reconcile_cycle`, steps 5 to 5g) over a local
store, never the live books. Each plant names the control that must catch it,
and :func:`test_every_planted_break_is_caught` reports caught / planted, which
must be 1. The plants reach the close (an unpriced holding), the statement (an
unpriced benchmark) and the bill (a publisher that overbills, caught by 5g's
rebuild from raw). The second half plants breaks with an ANSWER KEY and has a reviewer
who decides by the key alone: where the key and the proposer differ, the
scorer must show the disagreement (agreement below 1) -- a scorer that always
reads 1.0 proves nothing.

The book (same as the stage 1 gate): 11-25 genesis, 10 AAA and $1,000; 11-27
buy 5 BBB at $20 with a $1 commission; 11-30 sell 4 AAA at $12 with a $1
commission; the run on 12-01 closes November and files its statement.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
from collections.abc import Callable, Mapping

import pytest
from crucible.runner import RunContext
from crucible.store import LocalStore

from crucible_trader import (
    ledger,
    month_end_close,
    month_rebuild,
    performance_statement,
    reconciliation_job,
)
from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.exception_decision import (
    SCORE_CURRENT_KEY,
    decide,
    exception_score_key,
    record_decision,
)
from crucible_trader.exception_proposal import exception_proposals_key
from crucible_trader.exception_record import PAPER_FUND_TENANT, exceptions_key
from crucible_trader.ledger import ledger_reconciliation_key, load_journals
from crucible_trader.month_end_close import close_key
from crucible_trader.performance_statement import statement_key
from crucible_trader.reconciliation import CorporateActionSet, Fill
from crucible_trader.reconciliation_job import (
    RECONCILE_JOB,
    ReconciliationDiscrepancyError,
    reconcile_cycle,
)

T = PAPER_FUND_TENANT
DAYS = ("2026-11-25", "2026-11-27", "2026-11-30")
CLOSE_RUN = "2026-12-01"
ACTIONS = CorporateActionSet(resolved=frozenset({"AAA", "BBB"}))
CLOSES = {
    "2026-11-25": {"AAA": 10.0, "SPY": 500.0, "QQQ": 400.0},
    "2026-11-30": {"AAA": 11.0, "BBB": 21.0, "SPY": 510.0, "QQQ": 404.0},
}

Closes = Callable[[str, list[str]], Mapping[str, float]]


def closes_without(day: str, ticker: str) -> Closes:
    """The settled closes, with ``ticker`` missing on ``day`` (a planted gap)."""

    def closes(d: str, tickers: list[str]) -> Mapping[str, float]:
        marks = {k: v for k, v in CLOSES.get(d, {}).items() if not (d == day and k == ticker)}
        return {t: marks[t] for t in tickers if t in marks}

    return closes


def fill(ticker: str, shares: int, price: float, commission: float, exec_id: str) -> Fill:
    return Fill(
        ticker, shares, -shares * price - commission, commission_usd=commission, exec_id=exec_id
    )


class Ib:
    def __init__(self, books: Mapping[str, tuple[dict[str, int], float]]):
        self.books = books

    def read_statement(self, trading_day: str) -> BrokerStatement:
        positions, cash = self.books[trading_day]
        return BrokerStatement("DU1", trading_day, positions, cash)


def november_ib(*, extra_cash: float = 0.0, extra_bbb: int = 0) -> Ib:
    """IB's view of the book; a plant shifts it from 11-27 on."""
    bbb = 5 + extra_bbb
    return Ib(
        {
            DAYS[0]: ({"AAA": 10}, 1_000.0),
            DAYS[1]: ({"AAA": 10, "BBB": bbb}, 899.0 + extra_cash),
            DAYS[2]: ({"AAA": 6, "BBB": bbb}, 946.0 + extra_cash),
            CLOSE_RUN: ({"AAA": 6, "BBB": bbb}, 946.0 + extra_cash),
        }
    )


FILLS = {DAYS[1]: (fill("BBB", 5, 20.0, 1.0, "e1"),), DAYS[2]: (fill("AAA", -4, 12.0, 1.0, "e2"),)}


@pytest.fixture
def store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> LocalStore:
    use_closes(monkeypatch, closes_without("", ""))
    monkeypatch.setattr(
        ledger, "_panel_closes", lambda _store: lambda _day, names: dict.fromkeys(names, 1.0)
    )
    monkeypatch.setattr(reconciliation_job, "load_declared_entitlements", lambda: {})
    return LocalStore(tmp_path)


def use_closes(monkeypatch: pytest.MonkeyPatch, closes: Closes) -> None:
    for module in (month_end_close, performance_statement, month_rebuild):
        monkeypatch.setattr(module, "panel_close_source", lambda _store, c=closes: c)


@dataclasses.dataclass
class Run:
    ctx: RunContext
    paged: bool = False


def run_day(store: LocalStore, ib: Ib, day: str) -> Run:
    ctx = RunContext(
        run_id="01JG0000000000000000000000",
        job=RECONCILE_JOB,
        trading_day=dt.date.fromisoformat(day),
        calendar_date=dt.date.fromisoformat(day),
        store=store,
        seed=0,
        started=dt.datetime.fromisoformat(f"{day}T21:00:00+00:00"),
    )
    run = Run(ctx)
    try:
        reconcile_cycle(
            ctx,
            source=ib,
            fills=FILLS.get(day, ()),
            cash_flows=(),
            corporate_actions=ACTIONS,
            genesis=day == DAYS[0],
        )
    except ReconciliationDiscrepancyError:
        run.paged = True
    return run


def read(store: LocalStore, key: str) -> dict:
    return json.loads(store.get_bytes(key))


def records(store: LocalStore, day: str) -> list[dict]:
    return read(store, exceptions_key(day))["records"] if store.exists(exceptions_key(day)) else []


def metric(ctx: RunContext, name: str) -> dict:
    (found,) = [m for m in ctx.metrics if m["name"] == name]
    return found


# --------------------------------------------------------------------------- plants


@dataclasses.dataclass(frozen=True)
class Plant:
    """One known error and the control that must catch it."""

    name: str
    extra_cash: float = 0.0
    extra_bbb: int = 0
    closes: Closes | None = None
    #: What catching it looks like, checked by :func:`caught`.
    record_class: str | None = None
    ledger_gap: str | None = None
    pages: bool = False
    close_line: str | None = None
    statement_benchmark_missing: str | None = None
    #: A publisher defect: the statement filed in 5f bills this many cents more
    #: adviser fee than its inputs give. Caught by 5g's rebuild-from-raw.
    publisher_fee_error_cents: int = 0
    rebuild_break: str | None = None


PLANTS = (
    # The reconciler and the ledger: a cash difference in the review band.
    Plant(
        "cash_review",
        extra_cash=0.50,
        record_class="cash_delta",
        ledger_gap="cash_gap",
        close_line="close_suspense_cash",
    ),
    # The same, over the $1.00 page threshold: it must also page.
    Plant(
        "cash_page",
        extra_cash=9.00,
        record_class="cash_delta",
        ledger_gap="cash_gap",
        pages=True,
        close_line="close_suspense_cash",
    ),
    # A share IB holds that no fill explains.
    Plant(
        "share",
        extra_bbb=1,
        record_class="share_delta",
        ledger_gap="position_gap",
        pages=True,
        close_line="close_suspense_shares",
    ),
    # The close: a held name with no settled close on the month's last day.
    Plant(
        "close_unpriced",
        closes=closes_without(DAYS[2], "BBB"),
        close_line="close_input_missing",
    ),
    # The statement: a benchmark with no settled close. It is filed (the
    # bill does not depend on it), stated on its face, and the metric WATCHes.
    Plant(
        "statement_benchmark",
        closes=closes_without(DAYS[2], "SPY"),
        statement_benchmark_missing="SPY",
    ),
    # The bill: the publisher overbills one cent of adviser fee. The statement
    # is filed as published; the monthly rebuild from raw must disagree with it
    # (`alpha-engine-config-I13129` 1h, the plan's "plants to the bill").
    Plant("fee_bill", publisher_fee_error_cents=1, rebuild_break="fee_bill"),
)


def arm(monkeypatch: pytest.MonkeyPatch, plant: Plant) -> None:
    """Install ``plant``'s price gap and publisher defect, if it has them."""
    use_closes(monkeypatch, plant.closes or closes_without("", ""))
    if plant.publisher_fee_error_cents:
        honest = performance_statement.build_statement

        def overbilling(*args: object, **kwargs: object) -> dict[str, object]:
            statement = honest(*args, **kwargs)  # pyright: ignore[reportArgumentType]
            bill = statement["simulated_fee_bill"]
            bill["adviser_fee_cents"] += plant.publisher_fee_error_cents  # pyright: ignore
            return statement

        # The PUBLISHER's binding only: month_rebuild imported its own.
        monkeypatch.setattr(performance_statement, "build_statement", overbilling)


def run_month(store: LocalStore, plant: Plant) -> dict[str, Run]:
    ib = november_ib(extra_cash=plant.extra_cash, extra_bbb=plant.extra_bbb)
    return {day: run_day(store, ib, day) for day in (*DAYS, CLOSE_RUN)}


def caught(store: LocalStore, plant: Plant, runs: Mapping[str, Run]) -> list[str]:
    """The controls that caught ``plant``; a missing one is a missed plant."""
    misses: list[str] = []
    day = DAYS[1]
    if plant.record_class is not None:
        found = [r for r in records(store, day) if r["class"] == plant.record_class]
        proposals = read(store, exception_proposals_key(day))["proposals"] if found else []
        if not found or {p["exception_id"] for p in proposals} < {r["exception_id"] for r in found}:
            misses.append(f"record/proposal {plant.record_class}")
    if plant.ledger_gap is not None and not [
        e for e in load_journals(store, T, before=DAYS[2])[day] if e.kind == plant.ledger_gap
    ]:
        misses.append(f"ledger {plant.ledger_gap}")
    if plant.pages != runs[day].paged:
        misses.append("page" if plant.pages else "paged when it must not")
    # A finished close is filed; a refused one is not, and its metric names
    # every unexplained line's class.
    attempt = metric(runs[CLOSE_RUN].ctx, "month_end_close")
    closed = store.exists(close_key(T, "2026-11"))
    if plant.close_line is None:
        if not closed or attempt["status"] != "OK":
            misses.append(f"close refused: {attempt['status_reason']}")
    elif closed or attempt["status"] != "WATCH" or plant.close_line not in attempt["status_reason"]:
        misses.append(f"close {plant.close_line}")
    if plant.close_line is None:
        statement = metric(runs[CLOSE_RUN].ctx, "performance_statement")
        if not store.exists(statement_key(T, "2026-11")):
            misses.append("statement filed")
        elif plant.statement_benchmark_missing is None:
            if statement["status"] != "OK":
                misses.append(f"statement: {statement['status_reason']}")
        else:
            symbol = plant.statement_benchmark_missing
            row = read(store, statement_key(T, "2026-11"))["benchmarks"][symbol]
            if row["return"] is not None or statement["status"] != "WATCH":
                misses.append(f"statement benchmark {symbol}")
        rebuild = metric(runs[CLOSE_RUN].ctx, "month_rebuild")
        if plant.rebuild_break is None:
            if rebuild["status"] != "OK":
                misses.append(f"rebuild: {rebuild['status_reason']}")
        else:
            document = read(store, month_rebuild.rebuild_key(T, "2026-11"))
            if rebuild["status"] != "FAIL" or [b["document"] for b in document["breaks"]] != [
                plant.rebuild_break
            ]:
                misses.append(f"rebuild {plant.rebuild_break}")
    return misses


@pytest.mark.parametrize("plant", PLANTS, ids=[p.name for p in PLANTS])
def test_each_planted_break_is_caught_by_its_control(store, monkeypatch, plant):
    arm(monkeypatch, plant)
    assert caught(store, plant, run_month(store, plant)) == [], plant.name


def test_an_unplanted_month_closes_clean_so_every_catch_is_the_plant(store):
    """The negative control: with nothing planted nothing is caught, so each
    catch above is the plant's and not the harness's."""
    assert caught(store, Plant("none"), run_month(store, Plant("none"))) == []
    assert all(not records(store, d) for d in DAYS)
    view = read(store, ledger_reconciliation_key(T, DAYS[2]))
    assert view["clean"] is True and view["trial_balance_cents"] == 0


def test_every_planted_break_is_caught(tmp_path, monkeypatch):
    """The stage 3 gate metric: planted breaks caught / planted breaks set."""
    hits = 0
    for i, plant in enumerate(PLANTS):
        store = LocalStore(tmp_path / str(i))
        monkeypatch.undo()
        arm(monkeypatch, plant)
        monkeypatch.setattr(
            ledger, "_panel_closes", lambda _s: lambda _d, names: dict.fromkeys(names, 1.0)
        )
        monkeypatch.setattr(reconciliation_job, "load_declared_entitlements", lambda: {})
        hits += caught(store, plant, run_month(store, plant)) == []
    assert hits / len(PLANTS) == 1.0


# ------------------------------------------------------------ the scorer can fail


#: The answer key for the scorer plants: the (nature, action) a reviewer who
#: knows what was planted would record. The share plant's key deliberately
#: differs from what the deterministic proposer says, so a correct scorer
#: MUST count a disagreement there.
ANSWER_KEY = {
    "cash_delta": None,  # None: the proposer's own call is right; agree
    "share_delta": ("detector", "investigate_detector"),
}


def decide_by_key(proposal: Mapping[str, object]) -> dict[str, object]:
    truth = ANSWER_KEY[str(proposal["class"])]
    if truth is None or truth == (proposal["nature"], proposal["action"]):
        return decide(
            proposal, verdict="agree", decided_by="reviewer", decided_at="2026-11-28T15:00:00Z"
        )
    return decide(
        proposal,
        verdict="correct",
        decided_by="reviewer",
        decided_at="2026-11-28T15:00:00Z",
        nature=truth[0],
        action=truth[1],
        note="planted break: the answer key says otherwise",
    )


def test_the_scorer_counts_a_planted_disagreement(store):
    ib = november_ib(extra_cash=0.50, extra_bbb=1)
    run_day(store, ib, DAYS[0])
    run_day(store, ib, DAYS[1])
    proposals = read(store, exception_proposals_key(DAYS[1]))["proposals"]
    by_class = {p["class"]: p for p in proposals}
    assert {"cash_delta", "share_delta"} <= set(by_class)
    # The plant only proves something if the key really differs from the proposer.
    share = by_class["share_delta"]
    assert (share["nature"], share["action"]) != ANSWER_KEY["share_delta"]
    for proposal in proposals:
        record_decision(store, decide_by_key(proposal))

    run_day(store, ib, DAYS[2])
    scored = read(store, exception_score_key(DAYS[2]))
    assert scored == read(store, SCORE_CURRENT_KEY)
    classes = {c["class"]: c for c in scored["classes"]}
    assert (classes["share_delta"]["correct"], classes["share_delta"]["agreement"]) == (1, 0.0)
    assert classes["share_delta"]["consecutive_agree"] == 0
    assert classes["cash_delta"]["agreement"] == 1.0
    assert 0 < scored["totals"]["agreement"] < 1.0
