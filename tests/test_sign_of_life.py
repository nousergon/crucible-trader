"""The sign-of-life order: one order on a day whose plan has none (alpha-engine-config-I14646).

The 2026-11-25 book itself lives in the private store, so the replay below is a
SYNTHETIC equivalent built the way `test_order_router._book` builds a book: a
book 72% invested with 28% idle cash, the conviction gate engaged at
``conviction_budget_min_multiple: 0.05`` (a 1% budget against a 20% configured
cap), a 1% no-trade band, and every step the engine took held inside that band,
so the router plans nothing. All names, prices and sizes are invented.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from types import SimpleNamespace

import pytest
from crucible.hard_risk import LOCKOUT
from crucible.slots.strategy import BookUniverse

from crucible_trader.box import market_alert as m
from crucible_trader.construction import TargetBook, _aim_weights
from crucible_trader.hard_risk import HardRiskDecision
from crucible_trader.order_book import ShareBasis
from crucible_trader.order_router import plan_orders
from crucible_trader.sign_of_life import (
    SIGN_OF_LIFE_METRIC,
    apply_sign_of_life,
    metric_record,
    planned_order_count,
)

DAY = "2026-11-25"
NOTIONAL = 1_000_000.0
PRICES = {
    "AAA": 100.0,
    "BBB": 50.0,
    "CCC": 20.0,
    "DDD": 40.0,
    "EEE": 25.0,
    "FFF": 10.0,
    "GGG": 30.0,
}
HELD = {"AAA": 0.20, "BBB": 0.18, "CCC": 0.14, "DDD": 0.12, "EEE": 0.08}
CASH = "__CASH__"
#: A construction that recorded its band and no step toward the aim.
BAND_ONLY = {"settings": {"rebalance_band_pct": 0.01}}


def _held_weights() -> dict[str, float]:
    return {**HELD, CASH: 1.0 - sum(HELD.values())}


def _positions() -> dict[str, int]:
    return {t: round(w * NOTIONAL / PRICES[t]) for t, w in HELD.items()}


def _book(**overrides) -> TargetBook:
    tickers = (*PRICES, "SPY", CASH)
    weights = {t: HELD.get(t, 0.0) for t in PRICES}
    # The router's own band holds this one: $500 on a name held on both sides.
    weights["AAA"] = 0.2005
    weights = {**weights, "SPY": 0.0, CASH: 1.0 - sum(weights.values())}
    base = {
        "trading_day": DAY,
        "s_champion": "s:arm:def",
        "m_champion": "m:arm:abc",
        "u_champion": None,
        "universe": BookUniverse(
            tickers=tickers,
            sectors=tuple("x" for _ in tickers),
            benchmark_idx=len(tickers) - 2,
            cash_idx=len(tickers) - 1,
        ),
        "weights": weights,
        "turnover_one_way_ratio": 0.0005,
        "cost_bps": 1.0,
        "portfolio_notional_usd": NOTIONAL,
        "evidence": {},
        "metric": {"name": "portfolio_construction"},
        "session_inputs_key": "k",
        "alpha": dict.fromkeys(tickers, 0.002),
        "eligible": dict.fromkeys(tickers, True),
        "turnover_budget": 0.01,
        "turnover_drivers": {
            "turnover_budget_configured": 0.20,
            "turnover_budget_discretionary": 0.01,
            "conviction_budget_multiplier": 0.05,
            "conviction_gate_applied": True,
        },
        "optimizer": {
            "settings": {
                "rebalance_band_pct": 0.01,
                "max_position_pct": 0.25,
                "cash_sleeve_pct": 0.03,
                "max_daily_turnover": 0.20,
            },
            "solved": {
                "rebalance_band": {
                    "band": 0.01,
                    "held": [
                        {"ticker": "GGG", "delta": 0.008, "is_new_position": True},
                        {"ticker": "CCC", "delta": 0.006, "is_new_position": False},
                        {"ticker": "BBB", "delta": -0.004, "is_new_position": False},
                        {"ticker": "FFF", "delta": 0.002, "is_new_position": True},
                        {"ticker": "SPY", "delta": 0.005},
                        {"ticker": "EEE", "delta": None},
                        "not a row",
                    ],
                },
                "aim_adjustment": {"entries_deferred": ["FFF", "SPY", "ZZZ"]},
            },
        },
        "aim_weights": {"FFF": 0.04, "SPY": 0.10, **HELD},
    }
    base.update(overrides)
    return TargetBook(**base)


def _locked(*names: str, multiplier: float | None = None) -> HardRiskDecision:
    drawdown = None
    if multiplier is not None:
        drawdown = SimpleNamespace(multiplier=multiplier, circuit_breaker=multiplier == 0.0)
    return HardRiskDecision(
        trading_day=DAY,
        forced=dict.fromkeys(names, LOCKOUT),
        new_exits={},
        lockouts={},
        loss_from_cost={},
        peak_nav_usd=NOTIONAL,
        nav_usd=NOTIONAL,
        nav_source="test",
        drawdown=drawdown,
        carried_from=None,
    )


def _plan(book: TargetBook):
    sells, buys = plan_orders(
        book,
        positions=_positions(),
        quote=PRICES.get,
        clock=lambda: dt.datetime(2026, 11, 25, 14, 30, tzinfo=dt.UTC),
        max_order_notional_usd=50_000.0,
        max_session_notional_usd=100_000.0,
    )
    return [*sells, *buys]


def test_the_replayed_zero_order_day_places_exactly_one_order() -> None:
    book = _book()
    assert _plan(book) == []  # the day as it ran: every step banded away
    assert planned_order_count(book, _held_weights()) == 0

    stepped, outcome = apply_sign_of_life(book, _held_weights())

    (order,) = _plan(stepped)
    # The top deferred entry, at the band (1% of the book), not its 4% aim.
    assert (order.decision.symbol, order.action, order.decision.quantity) == ("FFF", "BUY", 1000)
    assert outcome.status == "fired" and outcome.order is not None
    assert outcome.order.record() == {
        "ticker": "FFF",
        "side": "buy",
        "weight": pytest.approx(0.01),
        "notional_usd": pytest.approx(10_000.0),
        "source": "deferred_entry",
        "full_step": 0.04,
        "reason": outcome.reason,
    }
    assert stepped.weights[CASH] == pytest.approx(book.weights[CASH] - 0.01)
    assert stepped.turnover_budget == pytest.approx(0.02)  # raised by the one order only
    assert stepped.turnover_one_way_ratio == pytest.approx(0.0105)
    assert stepped.unbanded == frozenset({"FFF"})


def test_a_banded_resize_is_not_banded_away_again_by_the_router() -> None:
    # FFF's alpha is negative and GGG is locked out: the next step is CCC's
    # banded +0.6%, a resize of a held name the router would otherwise band.
    alpha = {**_book().alpha, "FFF": -0.001}
    book = _book(alpha=alpha, hard_risk=_locked("GGG"))
    stepped, outcome = apply_sign_of_life(book, _held_weights())
    (order,) = _plan(stepped)
    assert (order.decision.symbol, order.action, order.decision.quantity) == ("CCC", "BUY", 300)
    assert outcome.order is not None and outcome.order.source == "banded_change"
    basis = ShareBasis(positions=_positions(), prices=PRICES, band_usd=10_000.0)
    assert basis.target_shares("CCC", stepped.weights["CCC"], NOTIONAL) == 7000  # banded display
    unbanded = dataclasses.replace(basis, unbanded=stepped.unbanded)
    assert unbanded.target_shares("CCC", stepped.weights["CCC"], NOTIONAL) == 7300


def test_a_book_that_already_has_orders_is_unchanged() -> None:
    book = _book()
    weights = {**book.weights, "DDD": 0.10, CASH: book.weights[CASH] + 0.02}
    trading = dataclasses.replace(book, weights=weights)
    stepped, outcome = apply_sign_of_life(trading, _held_weights())
    assert stepped is trading
    assert (outcome.status, outcome.planned_orders, outcome.order) == ("not_needed", 1, None)
    record = metric_record(outcome, trading_day=DAY, now_utc="2026-11-25T14:30:00Z")
    assert (record["status"], record["value"], record["unit"]) == ("OK", 0.0, "order")


def test_a_day_where_every_candidate_is_locked_out_records_unmeasurable() -> None:
    held = [{"ticker": t, "delta": 0.005} for t in ("CCC", "GGG")]
    optimizer = _book().optimizer
    assert optimizer is not None
    solved = {**optimizer["solved"], "rebalance_band": {"held": held}}
    book = _book(optimizer={**optimizer, "solved": solved}, hard_risk=_locked("FFF", "GGG", "CCC"))
    stepped, outcome = apply_sign_of_life(book, _held_weights())
    assert stepped is book and _plan(stepped) == []
    assert outcome.status == "unmeasurable" and outcome.order is None
    for name in ("FFF", "GGG", "CCC"):
        assert f"{name} hard risk forced it out ({LOCKOUT})" in outcome.reason
    record = metric_record(outcome, trading_day=DAY, now_utc="2026-11-25T14:30:00Z")
    assert record["name"] == SIGN_OF_LIFE_METRIC and record["status"] == "unmeasurable"
    assert "value" not in record and record[SIGN_OF_LIFE_METRIC]["status"] == "unmeasurable"


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"eligible": {}}, "ineligible"),
        ({"alpha": {}}, "below zero or unknown"),
        ({"hard_risk": "breaker"}, "no room under the cap at buy multiplier 0"),
        ({"cash": 0.03}, "no cash above the sleeve"),
    ],
)
def test_no_buy_qualifies_past_a_risk_check_and_the_sell_goes_instead(overrides, why) -> None:
    kwargs = dict(overrides)
    if kwargs.pop("hard_risk", None):
        kwargs["hard_risk"] = _locked(multiplier=0.0)
    if (cash := kwargs.pop("cash", None)) is not None:
        weights = _book().weights
        kwargs["weights"] = {**weights, CASH: cash, "EEE": weights["EEE"] + weights[CASH] - cash}
    book = _book(**kwargs)
    held = _held_weights() if "weights" not in kwargs else {**kwargs["weights"]}
    stepped, outcome = apply_sign_of_life(book, held)
    assert why in outcome.reason
    assert outcome.order is not None
    assert (outcome.order.ticker, outcome.order.side) == ("BBB", "sell")
    assert outcome.order.weight == pytest.approx(0.004)  # the full step: smaller than the band
    assert stepped.weights["BBB"] == pytest.approx(0.176)


def test_the_ladder_scales_a_buy_down_never_to_zero() -> None:
    book = _book(hard_risk=_locked(multiplier=0.25))
    weights = {**book.weights, "FFF": 0.22, CASH: book.weights[CASH] - 0.22}
    book = dataclasses.replace(book, weights=weights)
    stepped, outcome = apply_sign_of_life(book, {**_held_weights(), "FFF": 0.22, CASH: 0.06})
    assert outcome.order is not None and outcome.order.ticker == "FFF"
    assert outcome.order.weight == pytest.approx(0.25 * (0.25 - 0.22))  # under the 25% cap
    assert stepped.weights["FFF"] <= 0.25


def test_nothing_to_step_toward_records_unmeasurable() -> None:
    stepped, outcome = apply_sign_of_life(_book(optimizer=BAND_ONLY), _held_weights())
    assert stepped.weights == _book().weights
    assert outcome.status == "unmeasurable" and "no deferred entry or banded change" in (
        outcome.reason
    )


def test_a_book_over_its_budget_gets_no_order_on_top() -> None:
    _, outcome = apply_sign_of_life(_book(turnover_one_way_ratio=0.02), _held_weights())
    assert outcome.status == "unmeasurable" and "already over its budget" in outcome.reason


def test_a_candidate_outside_the_universe_or_with_nothing_left_is_refused() -> None:
    held = [{"ticker": "ZZZ", "delta": 0.005}, {"ticker": "GGG", "delta": -0.005}]
    book = _book(
        optimizer={
            "settings": {"rebalance_band_pct": 0.01},
            "solved": {"rebalance_band": {"held": held}},
        },
        aim_weights=None,
    )
    _, outcome = apply_sign_of_life(book, _held_weights())
    assert outcome.status == "unmeasurable"
    assert "ZZZ not in the book's universe" in outcome.reason
    assert "GGG nothing left to trade" in outcome.reason


def test_an_inception_book_with_entries_needs_no_sign_of_life() -> None:
    assert planned_order_count(_book(), None) == 5


def test_the_aim_weights_are_read_from_the_engines_diagnostics() -> None:
    assert _aim_weights({"aim_weights": {"AAA": 0.1}}) == {"AAA": 0.1}
    assert _aim_weights({"aim_weights": None}) is None


def _close(run: dict) -> list[m.Check]:
    facts = m.CloseFacts(
        today="2026-11-26",
        day=DAY,
        session=m.Unit({"ActiveState": "inactive", "Result": "success"}),
        broker=m.Broker(open_orders=(), done=()),
        run=run,
        shortfall=None,
        watch=None,
        timers={},
    )
    return m.close_checks(facts)


def test_the_close_alert_names_the_sign_of_life_order() -> None:
    _, outcome = apply_sign_of_life(_book(), _held_weights())
    record = metric_record(outcome, trading_day=DAY, now_utc="2026-11-25T14:30:00Z")
    run = {"status": "ok", "metrics": [{"name": "other"}, record]}
    texts = [c.text for c in _close(run)]
    assert "Sign-of-life order: buy FFF $10,000 (the plan had no other order)" in texts


def test_the_close_alert_fails_when_no_sign_of_life_order_qualified() -> None:
    _, outcome = apply_sign_of_life(_book(optimizer=BAND_ONLY), _held_weights())
    record = metric_record(outcome, trading_day=DAY, now_utc="2026-11-25T14:30:00Z")
    (check,) = [c for c in _close({"status": "ok", "metrics": [record]}) if "Sign" in c.text]
    assert check.mark == "fail" and check.text.startswith("Sign-of-life order: none placed (")


def test_the_close_alert_is_silent_when_the_plan_had_its_own_orders() -> None:
    record = metric_record(
        apply_sign_of_life(_book(), None)[1], trading_day=DAY, now_utc="2026-11-25T14:30:00Z"
    )
    assert not [c for c in _close({"status": "ok", "metrics": [record]}) if "Sign" in c.text]
