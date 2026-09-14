"""The hold-book safeguard, lifted from v1: its numbers, its decisions, its refusals."""

from __future__ import annotations

import datetime as dt
import math

import pytest
from crucible.models import MetricRecordRow
from crucible.serving import PredictionsFeed
from crucible.store import LocalStore
from ib_fakes import T0, FakeIB, FakeSdk, ctx_for

from crucible_trader.hold_book import (
    HOLD_BOOK_ALPHA_MODAL_FRACTION,
    HOLD_BOOK_MIN_BATCH,
    enforce_hold_book,
    served_alpha_diagnostics,
    should_hold_book,
)
from crucible_trader.kill_switch import (
    IbBrokerControl,
    TradingHaltedError,
    assert_trading_permitted,
    read_state,
)


def _feed(alphas: dict) -> PredictionsFeed:
    return PredictionsFeed(
        slot="m",
        trading_day="2026-09-08",
        champion="m:arm:abc",
        feature_version="v1",
        source_key="arms/m/x.json",
        predicted_alpha=alphas,
    )


HEALTHY = {f"T{i}": 0.001 * i for i in range(10)}


def test_the_v1_numbers_are_kept() -> None:
    """`crucible-executor/executor/main.py`: HOLD_BOOK_ALPHA_MODAL_FRACTION = 0.90,
    _HOLD_BOOK_MIN_BATCH = 5."""
    assert (HOLD_BOOK_ALPHA_MODAL_FRACTION, HOLD_BOOK_MIN_BATCH) == (0.90, 5)


class TestDecision:
    def test_a_healthy_cross_section_proceeds(self) -> None:
        decision = should_hold_book(_feed(HEALTHY))
        assert (decision.hold, decision.decision) == (False, "proceed_signal_healthy")

    def test_a_constant_batch_holds(self) -> None:
        decision = should_hold_book(_feed({f"T{i}": 0.02 for i in range(6)}))
        assert (decision.hold, decision.decision) == (True, "hold_signal_degenerate")
        assert decision.diagnostics["alpha_modal_fraction"] == 1.0

    def test_the_ceiling_itself_holds(self) -> None:
        alphas = {f"T{i}": 0.0 for i in range(9)} | {"T9": 0.5}
        assert should_hold_book(_feed(alphas)).decision == "hold_signal_degenerate"

    def test_just_under_the_ceiling_proceeds(self) -> None:
        alphas = {f"T{i}": 0.0 for i in range(8)} | {"T8": 0.5, "T9": 0.6}
        assert should_hold_book(_feed(alphas)).hold is False

    def test_a_batch_too_small_to_judge_holds(self) -> None:
        decision = should_hold_book(_feed({f"T{i}": 0.001 * i for i in range(4)}))
        assert decision.decision == "hold_signal_undeterminable"

    @pytest.mark.parametrize("bad", [math.nan, math.inf, True, "0.1"])
    def test_a_non_finite_or_non_numeric_alpha_holds(self, bad) -> None:
        decision = should_hold_book(_feed(HEALTHY | {"BAD": bad}))
        assert decision.decision == "hold_tradable_signal_failed"
        assert decision.diagnostics["n_nonfinite"] == 1

    def test_no_finite_alpha_carries_no_shape_stats(self) -> None:
        _, diag = served_alpha_diagnostics({"A": math.nan})
        assert diag == {"n_alpha": 0, "n_nonfinite": 1}


class TestEnforce:
    def _broker(self, ib):
        return IbBrokerControl(ib, FakeSdk(ib))

    def test_a_healthy_feed_engages_nothing(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        ctx = ctx_for(store, "trader.session")
        ib = FakeIB()
        decision, outcome = enforce_hold_book(ctx, _feed(HEALTHY), self._broker(ib))
        assert outcome is None and read_state(store) is None
        (metric,) = ctx.metrics
        MetricRecordRow.model_validate(metric)
        assert metric["value"] == 0.0

    def test_a_degenerate_feed_holds_the_book_for_the_session(self, tmp_path) -> None:
        store = LocalStore(tmp_path)
        ib = FakeIB(positions={"AAA": 5})
        ib.working_order("AAA", at=T0 - dt.timedelta(hours=1))
        ctx = ctx_for(store, "trader.session")
        decision, outcome = enforce_hold_book(
            ctx, _feed({f"T{i}": 0.0 for i in range(6)}), self._broker(ib), clock=ib.clock
        )
        assert outcome is not None and outcome.passed and outcome.mode == "hold"
        state = read_state(store)
        assert state["cause"] == "degenerate_predictions:hold_signal_degenerate"
        assert ib.cancel_calls == 0 and ib.order_calls == []
        with pytest.raises(TradingHaltedError):
            assert_trading_permitted(store, "2026-09-08")
        assert ctx.metrics[0]["value"] == 1.0
