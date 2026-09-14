"""Execution shortfall against the DECISION price — `alpha-engine-config-I10652`.

The producer half of the contract whose consumer is `crucible.report`'s fifth
attribution row. Every document built here is validated by
`crucible.execution.validate_execution_shortfall` as shipped in the pinned
wheel, and the last class reduces what this producer wrote through the
harness's own reducer — so a schema change the trader cannot satisfy fails
here, not on a report card.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import pytest
from crucible.execution import (
    DECISION_PRICE_BASIS,
    EXECUTION_METRIC_NAME,
    execution_shortfall_key,
    validate_execution_shortfall,
)
from crucible.report import build_attribution
from crucible.store import LocalStore

from crucible_trader.execution_shortfall import (
    SHORTFALL_BAND,
    Fill,
    ShortfallBand,
    SizingDecision,
    build_no_orders_document,
    build_not_computed_document,
    build_shortfall_document,
    decide,
    write_shortfall_document,
)

DAY = "2026-09-11"
CHAMPION = "m:ridge_21d:0123456789ab"
DECIDED = dt.datetime(2026, 9, 11, 19, 55, 0, tzinfo=dt.UTC)
SUBMITTED = DECIDED + dt.timedelta(seconds=3)
NOW = dt.datetime(2026, 9, 11, 21, 0, tzinfo=dt.UTC)


def _fill(
    oid: str = "o1",
    *,
    side: str = "buy",
    price: float = 100.0,
    fill: float | None = 100.05,
    qty: float = 100.0,
    filled: float | None = None,
) -> Fill:
    decision = decide("AAA", side, qty, price=price, now=DECIDED)  # type: ignore[arg-type]
    return Fill(
        order_id=oid,
        decision=decision,
        submitted_at=SUBMITTED,
        filled_quantity=(0.0 if fill is None else qty) if filled is None else filled,
        fill_price=fill,
    )


class TestTheDecisionPriceIsTheSizingInstant:
    def test_shortfall_is_measured_from_the_decision_price_not_the_arrival(self) -> None:
        """The market moved between the decision (100.00) and submission; the fill
        (100.05) is charged against the DECISION, so the 5 bps of delay is in the
        figure rather than excused by an arrival price nobody recorded."""
        doc = build_shortfall_document(trading_day=DAY, champion=CHAMPION, fills=[_fill()], now=NOW)
        order = doc["orders"][0]
        assert order["decision_price"] == 100.0
        assert order["shortfall_bps"] == pytest.approx(5.0)
        assert doc["decision_price_basis"] == DECISION_PRICE_BASIS == "sizing_decision"
        assert order["decided_at_utc"].startswith("2026-09-11T19:55:00")

    def test_submission_accepts_no_price(self) -> None:
        fields = {f.name for f in dataclasses.fields(Fill)}
        assert "decision_price" not in fields and "arrival_price" not in fields
        with pytest.raises(dataclasses.FrozenInstanceError):
            _fill().decision.decision_price = 101.0  # type: ignore[misc]

    def test_an_order_submitted_before_its_decision_is_refused(self) -> None:
        decision = decide("AAA", "buy", 10, price=100.0, now=DECIDED)
        with pytest.raises(ValueError, match="before its sizing decision"):
            Fill("o1", decision, DECIDED - dt.timedelta(seconds=1), 10, 100.0)

    def test_a_naive_decision_instant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="naive datetime"):
            decide("AAA", "buy", 10, price=100.0, now=dt.datetime(2026, 9, 11, 19, 55))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"symbol": ""}, "names its symbol"),
            ({"side": "short"}, "side must be"),
            ({"quantity": 0}, "quantity must be positive"),
            ({"decision_price": 0.0}, "decision price must be positive"),
        ],
    )
    def test_a_malformed_decision_is_refused(self, kwargs, match) -> None:
        base = {
            "symbol": "AAA",
            "side": "buy",
            "quantity": 1.0,
            "decision_price": 1.0,
            "decided_at": DECIDED,
        }
        with pytest.raises(ValueError, match=match):
            SizingDecision(**{**base, **kwargs})

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"order_id": ""}, "names its order_id"),
            ({"filled_quantity": 11.0}, "filled 11.0 of 10"),
            ({"filled_quantity": 5.0, "fill_price": None}, "present iff"),
            ({"filled_quantity": 0.0, "fill_price": 100.0}, "present iff"),
            ({"fill_price": -1.0}, "must be positive"),
        ],
    )
    def test_a_malformed_fill_is_refused(self, kwargs, match) -> None:
        base = {
            "order_id": "o1",
            "decision": decide("AAA", "buy", 10, price=100.0, now=DECIDED),
            "submitted_at": SUBMITTED,
            "filled_quantity": 10.0,
            "fill_price": 100.0,
        }
        with pytest.raises(ValueError, match=match):
            Fill(**{**base, **kwargs})


class TestTheThreeOutcomes:
    def test_a_sell_filled_below_its_decision_is_a_cost(self) -> None:
        doc = build_shortfall_document(
            trading_day=DAY, champion=CHAMPION, fills=[_fill(side="sell", fill=99.9)], now=NOW
        )
        assert doc["orders"][0]["shortfall_bps"] == pytest.approx(10.0)
        assert doc["summary"]["shortfall_usd"] == pytest.approx(10.0)

    def test_the_session_metric_equals_the_decision_weighted_summary(self) -> None:
        fills = [
            _fill("a", fill=100.10, qty=300),
            _fill("b", fill=100.0, qty=100),
            _fill("c", fill=None),
        ]
        doc = build_shortfall_document(trading_day=DAY, champion=CHAMPION, fills=fills, now=NOW)
        (metric,) = doc["metrics"]
        assert metric["name"] == EXECUTION_METRIC_NAME
        assert metric["value"] == pytest.approx(7.5)
        assert doc["summary"]["n_filled"] == 2 and doc["summary"]["n_orders"] == 3
        assert doc["orders"][2]["shortfall_bps"] is None
        assert doc["summary"]["notional_usd"] > 0, "the absolute companion rides with the ratio"

    def test_orders_placed_none_filled_computes_no_value(self) -> None:
        doc = build_shortfall_document(
            trading_day=DAY, champion=CHAMPION, fills=[_fill(fill=None)], now=NOW
        )
        assert doc["summary"]["shortfall_bps_notional_weighted"] is None
        assert doc["metrics"][0]["value"] is None
        assert doc["metrics"][0]["status"] == "N/A-LOW-N"

    def test_computed_with_no_fills_is_refused(self) -> None:
        with pytest.raises(ValueError, match="build_no_orders_document"):
            build_shortfall_document(trading_day=DAY, champion=CHAMPION, fills=[], now=NOW)

    def test_no_orders_is_a_fact_that_is_emitted(self) -> None:
        doc = build_no_orders_document(
            trading_day=DAY, champion=CHAMPION, reason="no rebalance required", now=NOW
        )
        assert doc["outcome"] == "no_orders" and doc["orders"] == []

    def test_not_computed_is_a_failure_that_is_emitted(self) -> None:
        doc = build_not_computed_document(
            trading_day=DAY,
            champion=CHAMPION,
            reason="fill report never arrived",
            fills=[_fill()],
            now=NOW,
        )
        assert doc["outcome"] == "not_computed" and doc["summary"] is None
        assert len(doc["orders"]) == 1

    def test_a_band_with_no_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            ShortfallBand(baseline_bps=10.0, upper_bps=10.0, placeholder=False)

    def test_the_declared_band_says_it_is_a_placeholder(self) -> None:
        assert SHORTFALL_BAND.placeholder is True
        assert SHORTFALL_BAND.upper_bps > SHORTFALL_BAND.baseline_bps


class TestTheHarnessReadsWhatTheTraderWrote:
    """The producer/consumer contract end to end: written here, reduced by
    `crucible.report` as shipped in the pinned wheel."""

    WEEK = ("2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14")

    def _store_with(self, tmp_path, builder) -> LocalStore:
        store = LocalStore(tmp_path)
        for day in self.WEEK:
            key = write_shortfall_document(store, builder(day))
            assert key == execution_shortfall_key(day)
            validate_execution_shortfall(json.loads(store.get_bytes(key)), origin=key)
        return store

    def _row(self, store: LocalStore) -> dict:
        document, _ = build_attribution(
            store, trading_day=dt.date(2026, 9, 14), now=NOW, run_id="R" * 26
        )
        return document["rows"][-1]

    def test_a_traded_week_is_a_graded_row(self, tmp_path) -> None:
        def traded(day: str) -> dict:
            fills = [_fill(f"{day}-{i}", fill=100.02) for i in range(5)]
            return build_shortfall_document(
                trading_day=day, champion=CHAMPION, fills=fills, now=NOW
            )

        row = self._row(self._store_with(tmp_path, traded))
        assert row["value"] == pytest.approx(2.0)
        assert row["status"] in ("GREEN", "WATCH")
        assert row["band"] == SHORTFALL_BAND.record()

    def test_a_week_with_no_orders_is_a_named_na(self, tmp_path) -> None:
        def idle(day: str) -> dict:
            return build_no_orders_document(
                trading_day=day, champion=CHAMPION, reason="no rebalance", now=NOW
            )

        row = self._row(self._store_with(tmp_path, idle))
        assert row["status"] == "N/A-LOW-N"
        assert "placed no filled order" in row["status_reason"]

    def test_a_not_computed_session_turns_the_row_red(self, tmp_path) -> None:
        def broken(day: str) -> dict:
            return build_not_computed_document(
                trading_day=day, champion=CHAMPION, reason="fill report never arrived", now=NOW
            )

        row = self._row(self._store_with(tmp_path, broken))
        assert row["status"] == "RED"
