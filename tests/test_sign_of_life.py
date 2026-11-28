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
    Planner,
    apply_sign_of_life,
    metric_record,
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


def _positions(weights=HELD) -> dict[str, int]:
    return {t: round(w * NOTIONAL / PRICES[t]) for t, w in weights.items() if t in PRICES and w}


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


def _planner(positions=None, prices=PRICES, session_cap=100_000.0, order_cap=50_000.0):
    """The router's own plan (`plan_orders`) over ``positions`` (default: the held book)."""
    held = _positions() if positions is None else positions

    def plan(book: TargetBook):
        sells, buys = plan_orders(
            book,
            positions=held,
            quote=prices.get,
            clock=lambda: dt.datetime(2026, 11, 25, 14, 30, tzinfo=dt.UTC),
            max_order_notional_usd=order_cap,
            max_session_notional_usd=session_cap,
        )
        return [*sells, *buys]

    return plan


_plan = _planner()


def test_the_replayed_zero_order_day_places_exactly_one_order() -> None:
    book = _book()
    assert _plan(book) == []  # the day as it ran: every step banded away

    stepped, outcome = apply_sign_of_life(book, _plan)

    (order,) = _plan(stepped)
    # The top deferred entry, at the band (1% of the book), not its 4% aim.
    assert (order.decision.symbol, order.action, order.decision.quantity) == ("FFF", "BUY", 1000)
    assert outcome.status == "fired" and outcome.order is not None
    assert outcome.order.record() == {
        "ticker": "FFF",
        "side": "buy",
        "weight": pytest.approx(0.01),
        "shares": 1000,
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
    stepped, outcome = apply_sign_of_life(book, _plan)
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
    stepped, outcome = apply_sign_of_life(trading, _plan)
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
    stepped, outcome = apply_sign_of_life(book, _plan)
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
    held = _positions(kwargs.get("weights", HELD))
    stepped, outcome = apply_sign_of_life(book, _planner(held))
    assert why in outcome.reason
    assert outcome.order is not None
    assert (outcome.order.ticker, outcome.order.side) == ("BBB", "sell")
    assert outcome.order.weight == pytest.approx(0.004)  # the full step: smaller than the band
    assert stepped.weights["BBB"] == pytest.approx(0.176)


def test_the_ladder_scales_a_buy_down_never_to_zero() -> None:
    book = _book(hard_risk=_locked(multiplier=0.25))
    weights = {**book.weights, "FFF": 0.22, CASH: book.weights[CASH] - 0.22}
    book = dataclasses.replace(book, weights=weights)
    stepped, outcome = apply_sign_of_life(book, _planner(_positions({**HELD, "FFF": 0.22})))
    assert outcome.order is not None and outcome.order.ticker == "FFF"
    assert outcome.order.weight == pytest.approx(0.25 * (0.25 - 0.22))  # under the 25% cap
    assert stepped.weights["FFF"] <= 0.25


def test_nothing_to_step_toward_records_unmeasurable() -> None:
    stepped, outcome = apply_sign_of_life(_book(optimizer=BAND_ONLY), _plan)
    assert stepped.weights == _book().weights
    assert outcome.status == "unmeasurable" and "no deferred entry or banded change" in (
        outcome.reason
    )


def test_a_book_over_its_budget_gets_no_order_on_top() -> None:
    _, outcome = apply_sign_of_life(_book(turnover_one_way_ratio=0.02), _plan)
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
    _, outcome = apply_sign_of_life(book, _plan)
    assert outcome.status == "unmeasurable"
    assert "ZZZ not in the book's universe" in outcome.reason
    assert "GGG nothing left to trade" in outcome.reason


def test_an_inception_book_with_entries_needs_no_sign_of_life() -> None:
    _, outcome = apply_sign_of_life(_book(), _planner({}, session_cap=1e7))
    assert (outcome.status, outcome.planned_orders) == ("not_needed", 5)


def _one_name_book(weights: dict[str, float], solved: dict) -> TargetBook:
    """A $1m book holding only ``weights`` (the rest cash), with ``solved`` recorded."""
    full = {t: 0.0 for t in (*PRICES, "SPY")} | weights
    full[CASH] = 1.0 - sum(full.values())
    optimizer = _book().optimizer
    assert optimizer is not None
    return _book(weights=full, optimizer={**optimizer, "solved": solved}, aim_weights=None)


def test_a_step_the_router_sizes_to_no_whole_share_is_passed_over_for_the_next() -> None:
    # Codex on crucible-trader-PR187: 400 AAA at $500 and a +0.01% banded step
    # is a $100 buy, under one share. The router plans nothing for it, so the
    # rule must not report it and must try the next candidate (BBB's -0.008%).
    prices = {"AAA": 500.0, "BBB": 10.0, "FFF": 10.0}
    positions = {"AAA": 400, "BBB": 10_000}
    held = [{"ticker": "AAA", "delta": 0.0001}, {"ticker": "BBB", "delta": -0.00008}]
    book = _one_name_book({"AAA": 0.2, "BBB": 0.1}, {"rebalance_band": {"held": held}})
    plan = _planner(positions, prices)
    assert plan(book) == []

    stepped, outcome = apply_sign_of_life(book, plan)

    assert outcome.status == "fired" and outcome.order is not None
    assert "AAA the router plans no whole-share order for it" in outcome.reason
    (order,) = plan(stepped)
    assert (order.decision.symbol, order.action, order.decision.quantity) == ("BBB", "SELL", 8)
    assert (outcome.order.ticker, outcome.order.shares) == ("BBB", 8)
    assert outcome.order.notional_usd == pytest.approx(80.0)


def test_a_plan_the_router_bands_to_nothing_on_whole_shares_still_gets_its_order() -> None:
    # Codex on crucible-trader-PR187: 333 shares at $600 (19.98%) toward
    # 20.981% is a 1.001% change, over the 1% band by weight; but the router
    # floors it to 349 shares and holds the $9,600 resize under its $10,000
    # band, so the day has no order and the sign-of-life order must go.
    prices = {"AAA": 600.0, "FFF": 10.0}
    solved = {"aim_adjustment": {"entries_deferred": ["FFF"]}}
    book = dataclasses.replace(_one_name_book({"AAA": 0.20981}, solved), aim_weights={"FFF": 0.04})
    plan = _planner({"AAA": 333}, prices)
    assert plan(book) == []

    stepped, outcome = apply_sign_of_life(book, plan)

    assert outcome.status == "fired" and outcome.order is not None
    (order,) = plan(stepped)
    assert (order.decision.symbol, order.action, order.decision.quantity) == ("FFF", "BUY", 1000)


def test_a_step_the_router_refuses_is_passed_over_for_the_next() -> None:
    # FFF has no decision price: the router refuses that plan, so GGG goes.
    prices = {t: p for t, p in PRICES.items() if t != "FFF"}
    stepped, outcome = apply_sign_of_life(_book(), _planner(prices=prices))
    assert outcome.order is not None and outcome.order.ticker == "GGG"
    assert "FFF the router refused it (no usable decision price for FFF" in outcome.reason
    assert [o.decision.symbol for o in _planner(prices=prices)(stepped)] == ["GGG"]


def _bbb_drifted(**overrides) -> tuple[TargetBook, Planner]:
    """Codex on crucible-trader-PR187: a $1m book, a 1% band, BBB's target 18%,
    its held shares worth $175,000, and a recorded -0.4% banded step."""
    held = [{"ticker": "BBB", "delta": -0.004, "is_new_position": False}]
    book = _one_name_book({"BBB": 0.18}, {"rebalance_band": {"held": held}})
    book = dataclasses.replace(book, **overrides)
    plan = _planner({"BBB": 3_500}, {"BBB": 50.0})
    assert plan(book) == []  # the $5,000 resize to 18% is inside the router's band
    # Unbanded at 17.6%, the router would BUY $1,000 against the $175,000 held.
    unbanded = dataclasses.replace(
        book, weights={**book.weights, "BBB": 0.176}, unbanded=frozenset({"BBB"})
    )
    (order,) = plan(unbanded)
    assert (order.action, order.decision.quantity) == ("BUY", 20)
    return book, plan


def test_a_sell_step_the_router_would_route_as_a_buy_is_passed_over() -> None:
    book, plan = _bbb_drifted()
    stepped, outcome = apply_sign_of_life(book, plan)
    assert stepped is book and plan(stepped) == []
    assert outcome.status == "unmeasurable" and outcome.order is None
    assert "BBB the router plans a buy of it, not the sell its step asks for" in outcome.reason
    record = metric_record(outcome, trading_day=DAY, now_utc="2026-11-25T14:30:00Z")
    assert record["status"] == "unmeasurable" and "value" not in record


def test_a_routed_buy_passes_the_buy_checks_on_the_order_itself() -> None:
    # The same drift with BBB's alpha below zero: the routed BUY is refused by
    # the buy checks, which the sell step itself never ran.
    book, plan = _bbb_drifted(alpha={**_book().alpha, "BBB": -0.001})
    stepped, outcome = apply_sign_of_life(book, plan)
    assert stepped is book and outcome.order is None
    assert "BBB the router plans a buy of it, refused: alpha -0.001 is below zero" in (
        outcome.reason
    )


def test_a_routed_order_larger_than_the_rule_sized_is_passed_over() -> None:
    # 9,500 FFF held at $10 (9.5%) against a 10% book weight and a +0.4% step:
    # the router would buy 900 shares, 0.9% of the book, not the 0.4% sized.
    held = [{"ticker": "FFF", "delta": 0.004, "is_new_position": False}]
    book = _one_name_book({"FFF": 0.10}, {"rebalance_band": {"held": held}})
    plan = _planner({"FFF": 9_500}, {"FFF": 10.0})
    assert plan(book) == []
    stepped, outcome = apply_sign_of_life(book, plan)
    assert stepped is book and outcome.order is None
    assert "FFF the router's buy is 0.9000% of the book, over the 0.4000% the rule sized" in (
        outcome.reason
    )


def test_the_budget_rises_by_the_order_the_router_planned_after_its_cap() -> None:
    # Codex on crucible-trader-PR187: a 1% step on a $1m book under a $1,000
    # per-order cap plans $1,000, so the budget rises by $1,000, not $10,000.
    book = _book()
    plan = _planner(order_cap=1_000.0)
    assert plan(book) == []
    stepped, outcome = apply_sign_of_life(book, plan)
    (order,) = plan(stepped)
    assert (order.decision.symbol, order.action, order.decision.quantity) == ("FFF", "BUY", 100)
    assert outcome.order is not None
    assert outcome.order.notional_usd == pytest.approx(1_000.0)
    assert outcome.order.weight == pytest.approx(0.001)
    assert book.turnover_budget is not None and stepped.turnover_budget is not None
    assert (stepped.turnover_budget - book.turnover_budget) * NOTIONAL == pytest.approx(1_000.0)
    assert stepped.turnover_one_way_ratio == pytest.approx(book.turnover_one_way_ratio + 0.001)


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
    _, outcome = apply_sign_of_life(_book(), _plan)
    record = metric_record(outcome, trading_day=DAY, now_utc="2026-11-25T14:30:00Z")
    run = {"status": "ok", "metrics": [{"name": "other"}, record]}
    texts = [c.text for c in _close(run)]
    assert "Sign-of-life order: buy FFF $10,000 (the plan had no other order)" in texts


def test_the_close_alert_fails_when_no_sign_of_life_order_qualified() -> None:
    _, outcome = apply_sign_of_life(_book(optimizer=BAND_ONLY), _plan)
    record = metric_record(outcome, trading_day=DAY, now_utc="2026-11-25T14:30:00Z")
    (check,) = [c for c in _close({"status": "ok", "metrics": [record]}) if "Sign" in c.text]
    assert check.mark == "fail" and check.text.startswith("Sign-of-life order: none placed (")


def test_the_close_alert_is_silent_when_the_plan_had_its_own_orders() -> None:
    record = metric_record(
        apply_sign_of_life(_book(), _planner({}, session_cap=1e7))[1],
        trading_day=DAY,
        now_utc="2026-11-25T14:30:00Z",
    )
    assert not [c for c in _close({"status": "ok", "metrics": [record]}) if "Sign" in c.text]
