"""The read-only CR-06 decomposition report over stored v1 sessions."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from crucible.execution import execution_shortfall_key
from crucible.store import DryRunWriteRefusedError, LocalStore, read_only

from crucible_trader import execution_decomposition_report as rep

DAY = dt.date(2026, 10, 20)


def _order(
    order_id: str,
    *,
    qty: float = 10.0,
    filled: float = 10.0,
    fill: float | None = 101.0,
    side: str = "buy",
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "symbol": "AAA",
        "side": side,
        "quantity": qty,
        "decided_at_utc": "2026-10-20T13:30:00.000000Z",
        "decision_price": 100.0,
        "submitted_at_utc": "2026-10-20T13:30:01.000000Z",
        "filled_quantity": filled,
        "fill_price": fill,
    }


def _put(store: LocalStore, day: dt.date, doc: dict[str, object]) -> None:
    store.put_bytes(execution_shortfall_key(day.isoformat()), json.dumps(doc).encode())


def _computed(*orders: dict[str, object]) -> dict[str, object]:
    return {"outcome": "computed", "orders": list(orders)}


def test_a_filled_order_keeps_its_filled_half_and_names_what_v1_lacks(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    _put(store, DAY, _computed(_order("o1")))
    report = rep.build_report(store, DAY, DAY)
    session = report["sessions"][DAY.isoformat()]  # type: ignore[index]
    (order,) = session["orders"]
    assert order["filled_vs_decision_usd"] == pytest.approx(10.0)
    assert order["exec_ids"] == ["o1:v1-aggregate"]
    assert order["terminal_state"] == "filled"
    assert order["unavailable"] == ["arrival_price", "commission_usd"]
    assert order["total_usd"] is None and session["net_usd"] is None
    assert report["summary"]["complete"] is False  # type: ignore[index]
    assert report["read_only"] is True


def test_partial_and_unfilled_orders_name_the_terminal_mark(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    _put(
        store,
        DAY,
        _computed(
            _order("p", filled=4.0, side="sell"),
            _order("n", filled=0.0, fill=None),
        ),
    )
    orders = {
        o["order_id"]: o
        for o in rep.build_report(store, DAY, DAY)["sessions"][DAY.isoformat()]["orders"]
    }  # type: ignore[index]
    assert orders["p"]["terminal_state"] == "partial"
    assert "terminal_mark" in orders["p"]["unavailable"]
    assert orders["n"]["terminal_state"] == "no_fill"
    assert orders["n"]["unavailable"] == ["commission_usd", "terminal_mark"]


def test_no_order_and_missing_days_stay_out_of_the_summary(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    _put(store, DAY, {"outcome": "no_orders", "outcome_reason": "shadow session", "orders": []})
    report = rep.build_report(store, DAY, DAY + dt.timedelta(days=1))
    assert report["sessions"] == {}
    assert report["sessions_without_orders"] == [
        {"trading_day": "2026-10-20", "outcome": "no_orders", "outcome_reason": "shadow session"}
    ]
    assert report["days_without_document"] == ["2026-10-21"]
    assert report["summary"] == {
        "n_days": 0,
        "complete": True,
        "incomplete_days": [],
        "mean_net_bps": None,
        "se_net_bps": None,
        "mean_gross_bps": None,
    }


def test_a_bad_side_and_a_reversed_window_are_refused(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    _put(store, DAY, _computed(_order("x", side="hold")))
    with pytest.raises(ValueError, match="not buy or sell"):
        rep.build_report(store, DAY, DAY)
    with pytest.raises(ValueError, match="before"):
        rep.build_report(store, DAY, DAY - dt.timedelta(days=1))


def test_the_cli_prints_the_report_and_cannot_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = LocalStore(tmp_path)
    _put(store, DAY, _computed(_order("o1")))
    assert rep.main(["--store", str(tmp_path), "--start", "2026-10-20", "--end", "2026-10-20"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema"] == rep.REPORT_SCHEMA and "2026-10-20" in out["sessions"]
    with pytest.raises(DryRunWriteRefusedError):
        read_only(LocalStore(tmp_path), reason="t").put_bytes("k", b"v")


def test_the_module_runs_as_a_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    monkeypatch.setattr(
        sys, "argv", ["x", "--store", str(tmp_path), "--start", "2026-10-20", "--end", "2026-10-20"]
    )
    monkeypatch.delitem(
        sys.modules, "crucible_trader.execution_decomposition_report", raising=False
    )
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("crucible_trader.execution_decomposition_report", run_name="__main__")
    assert stop.value.code == 0
