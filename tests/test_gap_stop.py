"""The catastrophic gap stop at the session start (v1 → v2 parity audit row 2)."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest
from crucible.keys import data_panel_key
from crucible.store import LocalStore

from crucible_trader.entry_timing import entry_watch_key
from crucible_trader.gap_stop import (
    DEFAULT_GAP_STOP_PCT,
    GAP_STOP_EXIT,
    GAP_STOP_SCHEMA_VERSION,
    GapCheck,
    apply_gap_exits,
    check_gaps,
    gap_stop_document,
    gap_stop_key,
    write_gap_stop_document,
)
from crucible_trader.order_router import GAP_STOP_VAR, RouterConfig, RouterSettingsError
from tests.test_entry_timing import PATHS, TimedIB, _run_day, _timed_router
from tests.test_order_router import DAY, RoutingIB, _book, _config, _env, _router

T0 = dt.datetime(2026, 9, 9, 13, 35, tzinfo=dt.UTC)


def _closes(store, closes: dict[str, float], day: str = DAY) -> None:
    frame = pd.DataFrame(
        {
            "trading_day": [day] * len(closes),
            "ticker": list(closes),
            "close_raw": list(closes.values()),
        }
    )
    store.put_bytes(data_panel_key(day), frame.to_parquet(index=False))


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


# -- the rule ------------------------------------------------------------------------


def test_v1s_threshold_and_reason_are_the_defaults() -> None:
    assert DEFAULT_GAP_STOP_PCT == 0.15
    assert GAP_STOP_EXIT == "catastrophic_gap_stop"


def test_a_drop_of_the_threshold_or_more_fires_and_less_holds() -> None:
    refs = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}
    prices = {"AAA": 85.0, "BBB": 85.01, "CCC": 60.0}
    checks = check_gaps(
        {"AAA": 10, "BBB": 5, "CCC": 7},
        reference=refs.get,
        live_price=prices.get,
        threshold=0.15,
    )
    by = {c.symbol: c for c in checks}
    assert by["AAA"].fired and by["AAA"].outcome == "fired"
    assert by["AAA"].drop == pytest.approx(0.15)
    assert not by["BBB"].fired and by["BBB"].outcome == "held"
    assert by["CCC"].fired and by["CCC"].shares == 7


def test_a_name_with_no_live_price_or_no_reference_never_fires() -> None:
    checks = check_gaps(
        {"AAA": 10, "BBB": 5, "FLAT": 0},
        reference={"AAA": 100.0}.get,
        live_price={"BBB": 1.0}.get,
        threshold=0.15,
    )
    assert [(c.symbol, c.outcome, c.fired) for c in checks] == [
        ("AAA", "unpriced", False),
        ("BBB", "no_reference", False),
    ]


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1])
def test_a_threshold_outside_the_unit_interval_is_refused(threshold) -> None:
    with pytest.raises(ValueError, match="must be in"):
        check_gaps({}, reference=lambda _s: None, live_price=lambda _s: None, threshold=threshold)


def test_a_fired_name_is_sold_whole_and_never_bought() -> None:
    fired = GapCheck("AAA", 10, 100.0, 80.0, 0.2, True, "fired")
    kept = GapCheck("BBB", 4, 100.0, 99.0, 0.01, False, "held")
    gapped_held = GapCheck("CCC", 6, 50.0, 40.0, 0.2, True, "fired")
    sells, buys = apply_gap_exits(
        [("AAA", -3), ("DDD", -1)],
        [("BBB", 2), ("CCC", 1)],
        [fired, kept, gapped_held],
        symbol_of=lambda o: o[0],
        full_exit=lambda c: (c.symbol, -c.shares),
    )
    assert sells == (("AAA", -10), ("CCC", -6), ("DDD", -1))
    assert buys == (("BBB", 2),)


def test_with_nothing_fired_the_plan_is_unchanged() -> None:
    sells, buys = apply_gap_exits(
        [("AAA", -1)],
        [("BBB", 1)],
        [GapCheck("AAA", 1, 1.0, 1.0, 0.0, False, "held")],
        symbol_of=lambda o: o[0],
        full_exit=lambda c: (c.symbol, 0),
    )
    assert (sells, buys) == ((("AAA", -1),), (("BBB", 1),))


def test_the_record_names_each_exit_with_its_reason() -> None:
    checks = (
        GapCheck("AAA", 10, 100.0, 80.0, 0.2, True, "fired"),
        GapCheck("BBB", 4, 100.0, 99.0, 0.01, False, "held"),
    )
    document = gap_stop_document(DAY, checks, threshold=0.15, reference_source="p", now=T0)
    assert document["schema_version"] == GAP_STOP_SCHEMA_VERSION
    assert document["exits"] == {"AAA": {"reason": GAP_STOP_EXIT, "drop": 0.2}}
    assert len(document["checks"]) == 2  # pyright: ignore[reportArgumentType]
    assert document["checked_at_utc"] == "2026-09-09T13:35:00Z"
    assert gap_stop_key(DAY) == f"trader/gap_stop/{DAY}.json"
    with pytest.raises(ValueError):
        gap_stop_key("not-a-day")


def test_a_failed_record_write_never_stops_the_exit(capsys) -> None:
    class Broken:
        def put_bytes(self, key, body) -> None:
            raise OSError("disk")

    write_gap_stop_document(Broken(), DAY, {"x": 1})  # pyright: ignore[reportArgumentType]
    assert "was not written" in capsys.readouterr().err


# -- configuration --------------------------------------------------------------------


def test_the_gap_stop_is_on_at_v1s_threshold_unless_switched_off() -> None:
    assert RouterConfig.from_env(_env()).gap_stop_pct == 0.15
    assert RouterConfig.from_env(_env(**{GAP_STOP_VAR: "0.2"})).gap_stop_pct == 0.2
    assert RouterConfig.from_env(_env(**{GAP_STOP_VAR: "off"})).gap_stop_pct is None


@pytest.mark.parametrize(("value", "match"), [("abc", "not a number"), ("1.5", "fraction")])
def test_a_malformed_threshold_is_refused(value, match) -> None:
    with pytest.raises(RouterSettingsError, match=match):
        RouterConfig.from_env(_env(**{GAP_STOP_VAR: value}))


# -- through the router ----------------------------------------------------------------


def test_a_gapped_held_name_is_sold_whole_in_the_opening_batch(store, capsys) -> None:
    # BBB closed at 62.5 and opens at 50 (-20%): the book wants 400 of the 500
    # held, the gap stop sells all 500. CCC is held at its close: book as usual.
    _closes(store, {"AAA": 100.0, "BBB": 62.5, "CCC": 20.0})
    ib = RoutingIB(positions={"BBB": 500, "CCC": 100})
    router = _router(store, ib, config=_config(gap_stop_pct=0.15))
    batches = list(router.batches(_book({"AAA": 0.3, "BBB": 0.2, "CCC": 0.0})))
    assert [[(o.decision.symbol, o.decision.quantity) for o in b] for b in batches] == [
        [("BBB", 500), ("CCC", 100)],
        [("AAA", 300)],
    ]
    record = json.loads(store.get_bytes(gap_stop_key(DAY)))
    assert record["exits"] == {"BBB": {"reason": GAP_STOP_EXIT, "drop": pytest.approx(0.2)}}
    assert record["reference_source"] == data_panel_key(DAY)
    assert "GAP STOP BBB" in capsys.readouterr().err


def test_a_gapped_name_the_book_would_buy_more_of_is_not_bought(store) -> None:
    _closes(store, {"AAA": 125.0, "BBB": 50.0, "CCC": 20.0})
    ib = RoutingIB(positions={"AAA": 100})
    router = _router(store, ib, config=_config(gap_stop_pct=0.15))
    batches = list(router.batches(_book({"AAA": 0.3, "BBB": 0.2, "CCC": 0.0})))
    assert [[(o.decision.symbol, o.decision.side) for o in b] for b in batches] == [
        [("AAA", "sell")],
        [("BBB", "buy")],
    ]


def test_a_name_priced_only_from_its_close_is_unpriced_not_a_zero_drop(store) -> None:
    _closes(store, {"AAA": 100.0, "BBB": 62.5, "CCC": 20.0})
    ib = RoutingIB(positions={"BBB": 500})
    router = _router(store, ib, config=_config(gap_stop_pct=0.15), live_quote=lambda _symbol: None)
    list(router.batches(_book()))
    record = json.loads(store.get_bytes(gap_stop_key(DAY)))
    assert record["exits"] == {}
    assert record["checks"][0]["outcome"] == "unpriced"


def test_switched_off_it_checks_and_records_nothing(store) -> None:
    _closes(store, {"AAA": 100.0, "BBB": 62.5, "CCC": 20.0})
    ib = RoutingIB(positions={"BBB": 500})
    router = _router(store, ib, config=_config(gap_stop_pct=None))
    (sells, _buys) = list(router.batches(_book()))
    assert [(o.decision.symbol, o.decision.quantity) for o in sells] == [("BBB", 100)]
    assert not store.exists(gap_stop_key(DAY))


# -- with timed orders: hard risk outranks timing ---------------------------------------


def _timed(store, ib):
    return _timed_router(store, ib, gap_stop_pct=0.15)


def test_with_timed_orders_a_gap_exit_goes_at_once_and_the_other_sells_wait(store) -> None:
    # BBB closed at 62.5 and trades at 50 all day (-20%); DDD is a plain exit, flat
    # at its close, so it waits for its 15:30 graduated trigger as before.
    _closes(store, {"AAA": 100.0, "BBB": 62.5, "DDD": 10.0})
    paths = {"AAA": PATHS["AAA"], "BBB": [(dt.time(9, 30), 50.0)], "DDD": PATHS["DDD"]}
    ib = TimedIB(paths, positions={"BBB": 500, "DDD": 100})
    router = _timed(store, ib)
    _run_day(router, _book({"AAA": 0.1, "BBB": 0.2, "DDD": 0.0}))
    placed = ib.placed()
    assert placed[0] == ("BBB", "SELL", 500, "09:35")  # the first batch, no trigger awaited
    assert ("DDD", "SELL", 100, "15:30") in placed
    assert not any(symbol == "BBB" and side == "BUY" for symbol, side, *_ in placed)
    rows = {r["symbol"]: r for r in json.loads(store.get_bytes(entry_watch_key(DAY)))["entries"]}
    assert (rows["BBB"]["status"], rows["BBB"]["trigger"]) == ("sent", GAP_STOP_EXIT)
    # The exit goes before the session-start stops, so the dumped name never gets one.
    assert "BBB" not in {symbol for symbol, *_ in ib.stops()}
    assert router.protective_stop_events[0]["moment"] == "at the session start"


def test_a_rerun_adopts_a_gap_exit_already_at_the_broker(store) -> None:
    _closes(store, {"BBB": 62.5})
    ib = TimedIB({"BBB": [(dt.time(9, 30), 50.0)]}, positions={"BBB": 500}, stall=True)
    first = _timed(store, ib)
    batches = iter(first.batches(_book({"BBB": 0.0})))
    first.send(next(batches))  # the gap exit; it never fills and the session dies
    rerun = _timed(store, ib)
    _run_day(rerun, _book({"BBB": 0.0}))
    assert ib.placed() == [("BBB", "SELL", 500, "09:35")]  # adopted, not re-sent
    (record,) = rerun.records
    assert record.disposition == "adopted"
