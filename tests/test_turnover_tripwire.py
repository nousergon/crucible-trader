"""The rolling turnover tripwire (v1 parity audit 2026-11-09, row 16).

v1's `executor/turnover_tripwire.py` paged on a daily band and a rolling-sum
band and named the driver. These tests pin the v2 port: both bands read
recorded facts only, the window spans SERVED sessions read from their
`trader.session` manifests, the driver comes from the construction's own
diagnostics, an undeclared or broken tripwire is RECORDED rather than silent,
and nothing here ever raises into the session.

The band values below are FIXTURE values; the operating values live in the
private strategy tree.
"""

from __future__ import annotations

import json

import pytest
import yaml
from crucible.keys import manifest_key, strategy_slot_key
from crucible.store import LocalStore

from crucible_trader import turnover_tripwire as module
from crucible_trader.turnover_tripwire import (
    METRIC_NAME,
    TRIPWIRE_BLOCK,
    TripwireConfig,
    TripwireConfigError,
    check,
    metric_record,
)

DAY = "2026-11-09"  # a Monday: the window walks back over the weekend
BLOCK = {"daily_multiple": 1.25, "rolling_sessions": 5, "rolling_sum_band": 0.60}


def _slot_file(store: LocalStore, block: object | None) -> None:
    payload: dict[str, object] = {"portfolio": {"risk_aversion": 5.0}}
    if block is not None:
        payload[TRIPWIRE_BLOCK] = block
    store.put_bytes(strategy_slot_key("s"), yaml.safe_dump(payload).encode())


def _session(value: float, budget: float | None = 0.2, **drivers: object) -> dict[str, object]:
    return {
        "turnover_one_way": value,
        "turnover_budget": budget,
        "turnover_drivers": {
            "turnover_mandatory_floor": 0.0,
            "turnover_budget_discretionary": budget,
            "conviction_ir_xs": 1.0,
            "conviction_gate_applied": False,
            "turnover_constraint_binding": False,
            **drivers,
        },
    }


def _manifest(store: LocalStore, day: str, *, turnover: float | None, metric: dict | None = None):
    document: dict[str, object] = {"job": "trader.session", "trading_day": day, "metrics": []}
    if turnover is not None:
        document["turnover"] = {"value": turnover, "budget": 0.2, "breached": False}
    if metric is not None:
        document["metrics"] = [metric]
    store.put_bytes(manifest_key("trader.session", day), json.dumps(document).encode())


@pytest.fixture
def store(tmp_path) -> LocalStore:
    s = LocalStore(tmp_path / "store")
    _slot_file(s, BLOCK)
    return s


class TestTheBands:
    def test_a_quiet_week_is_ok(self, store) -> None:
        for day in ("2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"):
            _manifest(store, day, turnover=0.05)
        block = check(store, trading_day=DAY, result=_session(0.05))
        assert block["status"] == "ok"
        assert block["n_sessions_used"] == 5
        assert block["rolling_sum"] == pytest.approx(0.25)

    def test_a_session_above_its_own_budget_times_the_multiple_is_a_daily_breach(
        self, store
    ) -> None:
        block = check(store, trading_day=DAY, result=_session(0.26, budget=0.2))
        assert block["status"] == "breach_daily"
        assert block["daily_band"] == pytest.approx(0.25)

    def test_churn_by_small_moves_is_a_rolling_breach(self, store) -> None:
        # Every session under its budget; the week's sum is not.
        for day in ("2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"):
            _manifest(store, day, turnover=0.15)
        block = check(store, trading_day=DAY, result=_session(0.15))
        assert block["daily_breach"] is False
        assert block["status"] == "breach_rolling"
        assert block["rolling_sum"] == pytest.approx(0.75)

    def test_the_window_spans_served_sessions_not_calendar_days(self, store) -> None:
        _manifest(store, "2026-11-06", turnover=None)  # held: no turnover recorded
        _manifest(store, "2026-11-05", turnover=0.10)
        block = check(store, trading_day=DAY, result=_session(0.10))
        assert [row["date"] for row in block["attribution"]["per_day"]] == [
            DAY,
            "2026-11-05",
        ]

    def test_a_malformed_metric_on_a_prior_manifest_is_skipped_not_fatal(self, store) -> None:
        _manifest(store, "2026-11-06", turnover=0.10, metric="not a record")  # type: ignore[arg-type]
        block = check(store, trading_day=DAY, result=_session(0.10))
        assert block["status"] == "ok" and block["n_sessions_used"] == 2

    def test_a_disabled_budget_has_no_daily_band(self, store) -> None:
        block = check(store, trading_day=DAY, result=_session(0.9, budget=None))
        assert block["daily_band"] is None and block["daily_breach"] is False
        assert block["status"] == "breach_rolling"


class TestTheDriverIsNamed:
    def test_forced_exits_dominate(self, store) -> None:
        block = check(
            store, trading_day=DAY, result=_session(0.5, budget=0.9, turnover_mandatory_floor=0.4)
        )
        assert block["attribution"]["driver"] == "forced_exits"

    def test_a_throttling_gate(self, store) -> None:
        block = check(store, trading_day=DAY, result=_session(0.1, conviction_gate_applied=True))
        assert block["attribution"]["driver"] == "conviction_gate_throttling"

    def test_a_collapsed_cross_section(self, store) -> None:
        block = check(store, trading_day=DAY, result=_session(0.1, conviction_ir_xs=0.2))
        assert block["attribution"]["driver"] == "predictor_conviction_collapse"

    def test_a_saturated_budget(self, store) -> None:
        block = check(
            store, trading_day=DAY, result=_session(0.1, turnover_constraint_binding=True)
        )
        assert block["attribution"]["driver"] == "budget_saturation"

    def test_nothing_dominant_is_unattributed_not_rounded(self, store) -> None:
        block = check(store, trading_day=DAY, result=_session(0.1))
        assert block["attribution"]["driver"] == "unattributed"

    def test_sessions_with_no_recorded_drivers_are_counted_as_such(self, store) -> None:
        _manifest(store, "2026-11-06", turnover=0.3)
        result = {"turnover_one_way": 0.3, "turnover_budget": 0.2}
        block = check(store, trading_day=DAY, result=result)
        assert block["attribution"]["driver"] == "unattributed"
        assert block["attribution"]["n_unattributed_sessions"] == 2

    def test_a_prior_sessions_own_row_is_preferred_over_its_bare_turnover(self, store) -> None:
        prior = check(store, trading_day="2026-11-06", result=_session(0.3, budget=0.9))
        record = metric_record(prior, trading_day="2026-11-06", now_utc="t", module="m")
        _manifest(store, "2026-11-06", turnover=0.3, metric=record)
        block = check(
            store, trading_day=DAY, result=_session(0.1, budget=0.9, turnover_mandatory_floor=0.4)
        )
        (_, earlier) = block["attribution"]["per_day"]
        assert earlier["forced"] == 0.0  # read from the row, not unknown


class TestItIsNeverSilentAndNeverRaises:
    def test_an_undeclared_block_is_recorded_as_unmeasurable(self, tmp_path) -> None:
        store = LocalStore(tmp_path / "s")
        _slot_file(store, None)
        block = check(store, trading_day=DAY, result=_session(0.1))
        record = metric_record(block, trading_day=DAY, now_utc="t", module="m")
        assert block["status"] == "undeclared"
        assert record["status"] == "unmeasurable" and "value" not in record

    def test_an_unreadable_parameter_file_is_an_error_record_not_a_raise(self, tmp_path) -> None:
        block = check(LocalStore(tmp_path / "empty"), trading_day=DAY, result=_session(0.1))
        assert block["status"] == "error"

    def test_a_session_with_no_turnover_is_recorded(self, store) -> None:
        block = check(store, trading_day=DAY, result={"turnover_one_way": None})
        assert block["status"] == "no_turnover"

    def test_a_breach_is_a_breach_metric_naming_the_driver(self, store) -> None:
        block = check(store, trading_day=DAY, result=_session(0.9, budget=0.2))
        record = metric_record(block, trading_day=DAY, now_utc="t", module="m")
        assert record["name"] == METRIC_NAME and record["status"] == "BREACH"
        assert "DRIVER" in str(record["status_reason"])
        assert record["unit"] == "ratio" and record["value"] == pytest.approx(0.9)
        assert record["source_path"] == manifest_key("trader.session", DAY)


class TestTheParameters:
    def test_every_field_is_required(self) -> None:
        with pytest.raises(TripwireConfigError, match="missing"):
            TripwireConfig.from_mapping({"daily_multiple": 1.25}, source="x")

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(TripwireConfigError, match="unknown"):
            TripwireConfig.from_mapping({**BLOCK, "x": 1}, source="x")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("daily_multiple", 0.9),
            ("rolling_sessions", 1),
            ("rolling_sum_band", 0.0),
            ("rolling_sessions", 2.5),
            ("daily_multiple", "1"),
        ],
    )
    def test_out_of_range_or_mistyped_values_are_refused(self, field, value) -> None:
        with pytest.raises(TripwireConfigError):
            TripwireConfig.from_mapping({**BLOCK, field: value}, source="x")

    def test_a_non_mapping_block_is_refused(self) -> None:
        with pytest.raises(TripwireConfigError):
            TripwireConfig.from_mapping([1, 2], source="x")

    def test_a_non_mapping_parameter_file_is_refused(self, tmp_path) -> None:
        store = LocalStore(tmp_path / "s")
        store.put_bytes(strategy_slot_key("s"), b"- 1\n")
        with pytest.raises(TripwireConfigError):
            module.load_config(store)
