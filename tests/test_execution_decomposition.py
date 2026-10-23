"""Shortfall decomposition — `alpha-engine-config-I12566` (audit CR-06).

The closes-when, test by test: partial fill, reject, cancel and no-fill
reconcile the full intended quantity to filled + unfilled; signed delay +
execution + fees + opportunity reconcile to the total; delay + execution
reconciles to the published v1 filled-quantity shortfall; unknown fees,
arrival or mark are named unavailable and make the total (and the session,
and the multi-day summary) incomplete; the multi-day summary clusters by day;
and the cash-fee tie-out names every disagreement.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from crucible_trader.execution_decomposition import (
    CONVENTIONS,
    ArrivalObservation,
    ExecutionPiece,
    OrderOutcome,
    TerminalMark,
    _assert_reconciles,
    day_clustered_summary,
    decompose,
    session_decomposition,
    tie_out_fees,
)
from crucible_trader.execution_shortfall import Fill, decide

T0 = dt.datetime(2026, 10, 22, 13, 35, tzinfo=dt.UTC)
ARRIVAL = ArrivalObservation(price=101.0, observed_at=T0 + dt.timedelta(seconds=30))
CLOSE = TerminalMark(price=104.0, basis="session_close", horizon="same_session")


def _buy(qty: float = 100, price: float = 100.0):
    return decide("AAA", "buy", qty, price=price, now=T0)


def _sell(qty: float = 100, price: float = 100.0):
    return decide("BBB", "sell", qty, price=price, now=T0)


def _outcome(**kw) -> OrderOutcome:
    base = {
        "order_id": "o1",
        "decision": _buy(),
        "terminal_state": "filled",
        "executions": (ExecutionPiece("e1", 60, 101.5), ExecutionPiece("e2", 40, 102.0)),
        "arrival": ARRIVAL,
        "commission_usd": 1.0,
        "mark": CLOSE,
    }
    base.update(kw)
    return OrderOutcome(**base)


def _total_reconciles(doc) -> None:
    comps = doc["components_usd"]
    assert doc["total_usd"] == pytest.approx(sum(comps.values()))


class TestComponentsReconcile:
    def test_a_full_fill_splits_into_delay_and_execution(self) -> None:
        doc = decompose(_outcome())
        vwap = (60 * 101.5 + 40 * 102.0) / 100
        assert doc["fill_vwap"] == pytest.approx(vwap)
        assert doc["components_usd"] == pytest.approx(
            {"delay": 100.0, "execution": (vwap - 101.0) * 100, "fees": 1.0, "opportunity": 0.0}
        )
        _total_reconciles(doc)
        # delay + execution is the v1 filled-quantity shortfall, exactly.
        v1 = Fill("o1", _buy(), T0, 100, vwap).record()["shortfall_usd"]
        assert doc["components_usd"]["delay"] + doc["components_usd"]["execution"] == (
            pytest.approx(v1)
        )
        assert doc["complete"] is True
        assert doc["total_bps"] == pytest.approx(doc["total_usd"] / 10_000 * 1e4)
        assert doc["exec_ids"] == ["e1", "e2"]
        assert doc["terminal_mark"] == {
            "price": 104.0,
            "basis": "session_close",
            "horizon": "same_session",
        }

    def test_a_partial_fill_carries_opportunity_on_the_remainder(self) -> None:
        doc = decompose(
            _outcome(terminal_state="partial", executions=(ExecutionPiece("e1", 30, 101.5),))
        )
        assert doc["filled_quantity"] + doc["unfilled_quantity"] == doc["intended_quantity"]
        assert doc["components_usd"]["opportunity"] == pytest.approx((104.0 - 100.0) * 70)
        _total_reconciles(doc)

    def test_a_sell_signs_a_falling_price_as_cost(self) -> None:
        doc = decompose(
            _outcome(
                decision=_sell(),
                terminal_state="cancelled",
                executions=(ExecutionPiece("e1", 50, 98.0),),
                arrival=ArrivalObservation(price=99.0, observed_at=T0),
                mark=TerminalMark(price=97.0, basis="session_close", horizon="same_session"),
            )
        )
        comps = doc["components_usd"]
        assert comps["delay"] == pytest.approx(50.0)  # 100 -> 99 before arrival, selling
        assert comps["execution"] == pytest.approx(50.0)  # 99 -> 98 while working
        assert comps["opportunity"] == pytest.approx(150.0)  # 50 unsold, 100 -> 97
        _total_reconciles(doc)

    @pytest.mark.parametrize("state", ["rejected", "no_fill"])
    def test_nothing_filled_is_all_opportunity(self, state: str) -> None:
        doc = decompose(
            _outcome(terminal_state=state, executions=(), arrival=None, commission_usd=0.0)
        )
        comps = doc["components_usd"]
        assert comps["delay"] == comps["execution"] == 0.0
        assert comps["opportunity"] == pytest.approx(400.0)
        assert doc["fill_vwap"] is None
        assert doc["filled_vs_decision_usd"] == 0.0
        # The denominator is the INTENDED notional, so a no-fill order weighs in full.
        assert doc["total_bps"] == pytest.approx(400.0 / 10_000 * 1e4)
        assert doc["complete"] is True

    def test_a_broken_identity_raises(self) -> None:
        with pytest.raises(ArithmeticError, match="delay \\+ execution"):
            _assert_reconciles("o1", {"delay": 1.0, "execution": 1.0}, 3.0)


class TestUnknownIsNeverGreen:
    @pytest.mark.parametrize(
        ("change", "missing", "nulled"),
        [
            ({"arrival": None}, "arrival_price", ("delay", "execution")),
            ({"commission_usd": None}, "commission_usd", ("fees",)),
            (
                {
                    "terminal_state": "partial",
                    "executions": (ExecutionPiece("e1", 30, 101),),
                    "mark": None,
                },
                "terminal_mark",
                ("opportunity",),
            ),
        ],
    )
    def test_a_missing_input_is_named_and_blocks_the_total(self, change, missing, nulled) -> None:
        doc = decompose(_outcome(**change))
        assert doc["unavailable"] == [missing]
        for c in nulled:
            assert doc["components_usd"][c] is None
            assert doc["components_bps"][c] is None
        assert doc["total_usd"] is None
        assert doc["total_bps"] is None
        assert doc["complete"] is False

    def test_one_incomplete_order_makes_the_session_incomplete(self) -> None:
        session = session_decomposition([_outcome(), _outcome(order_id="o2", commission_usd=None)])
        assert session["complete"] is False
        assert session["gross_usd"] is None
        assert session["net_usd"] is None
        assert session["gross_bps"] is None
        assert session["components_usd"]["fees"] is None
        assert session["components_usd"]["delay"] is not None
        assert session["unavailable"] == ["commission_usd"]


class TestSession:
    def test_gross_excludes_fees_and_net_includes_them(self) -> None:
        a = _outcome()
        b = _outcome(
            order_id="o2",
            terminal_state="no_fill",
            executions=(),
            commission_usd=0.0,
        )
        session = session_decomposition([a, b])
        da, db = decompose(a), decompose(b)
        assert session["conventions"] == CONVENTIONS
        assert session["intended_notional_usd"] == pytest.approx(20_000.0)
        assert session["net_usd"] == pytest.approx(da["total_usd"] + db["total_usd"])
        assert session["gross_usd"] == pytest.approx(session["net_usd"] - 1.0)
        assert session["net_bps"] == pytest.approx(session["net_usd"] / 20_000 * 1e4)
        assert session["complete"] is True

    def test_an_empty_session_has_no_ratio(self) -> None:
        session = session_decomposition([])
        assert session["n_orders"] == 0
        assert session["net_usd"] == 0.0
        assert session["net_bps"] is None
        assert session["gross_bps"] is None

    def test_a_duplicate_order_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="appears twice"):
            session_decomposition([_outcome(), _outcome()])

    def test_a_quantity_identity_failure_is_refused(self, monkeypatch) -> None:
        import crucible_trader.execution_decomposition as mod

        real = mod.decompose

        def skewed(outcome):
            doc = real(outcome)
            doc["unfilled_quantity"] = 1.0
            return doc

        monkeypatch.setattr(mod, "decompose", skewed)
        with pytest.raises(ArithmeticError, match="filled \\+ unfilled"):
            mod.session_decomposition([_outcome()])


class TestDayClustered:
    def _session(self, net_bps: float, gross_bps: float, complete: bool = True):
        return {"net_bps": net_bps, "gross_bps": gross_bps, "complete": complete}

    def test_the_cluster_is_the_day(self) -> None:
        out = day_clustered_summary(
            {"2026-10-20": self._session(10, 9), "2026-10-21": self._session(20, 19)}
        )
        assert out["n_days"] == 2
        assert out["mean_net_bps"] == pytest.approx(15.0)
        assert out["mean_gross_bps"] == pytest.approx(14.0)
        # sd of (10, 20) is 7.0711; se over two days is 5.0.
        assert out["se_net_bps"] == pytest.approx(5.0)
        assert out["complete"] is True

    def test_one_day_has_no_standard_error(self) -> None:
        out = day_clustered_summary({"2026-10-20": self._session(10, 9)})
        assert out["se_net_bps"] is None
        assert out["mean_net_bps"] == 10

    def test_no_days(self) -> None:
        out = day_clustered_summary({})
        assert out["mean_net_bps"] is None
        assert out["mean_gross_bps"] is None

    def test_an_incomplete_day_is_not_dropped(self) -> None:
        out = day_clustered_summary(
            {"2026-10-20": self._session(10, 9), "2026-10-21": self._session(0, 0, False)}
        )
        assert out["complete"] is False
        assert out["incomplete_days"] == ["2026-10-21"]
        assert out["mean_net_bps"] is None


class TestFeeTieOut:
    def test_agreement_and_every_kind_of_disagreement(self) -> None:
        session = session_decomposition(
            [
                _outcome(order_id="agree"),
                _outcome(order_id="differs", commission_usd=2.0),
                _outcome(order_id="unknown", commission_usd=None),
                _outcome(order_id="mine_only"),
            ]
        )
        out = tie_out_fees(
            session, {"agree": 1.0, "differs": 2.5, "unknown": 1.0, "ledger_only": 0.5}
        )
        assert out["agrees"] is False
        assert out["mismatched"] == {
            "differs": {"shortfall": 2.0, "ledger": 2.5},
            "unknown": {"shortfall": None, "ledger": 1.0},
        }
        assert out["only_in_shortfall"] == ["mine_only"]
        assert out["only_in_ledger"] == ["ledger_only"]

    def test_a_clean_tie_out(self) -> None:
        session = session_decomposition([_outcome()])
        assert tie_out_fees(session, {"o1": 1.0})["agrees"] is True


class TestRefusals:
    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"order_id": ""}, "names its order_id"),
            ({"terminal_state": "lost"}, "terminal_state is one of"),
            (
                {"executions": (ExecutionPiece("e1", 50, 100), ExecutionPiece("e1", 50, 100))},
                "appears twice",
            ),
            ({"executions": (ExecutionPiece("e1", 150, 100),)}, "filled 150"),
            ({"terminal_state": "partial"}, "contradicts"),
            ({"terminal_state": "rejected"}, "contradicts"),
            ({"commission_usd": -1.0}, "cannot be negative"),
        ],
    )
    def test_an_inconsistent_outcome_is_refused(self, change, match) -> None:
        with pytest.raises(ValueError, match=match):
            _outcome(**change)

    def test_bad_observations_are_refused(self) -> None:
        with pytest.raises(ValueError, match="arrival price"):
            ArrivalObservation(price=0.0, observed_at=T0)
        with pytest.raises(ValueError, match="exec_id"):
            ExecutionPiece("", 1, 1)
        with pytest.raises(ValueError, match="must be positive"):
            ExecutionPiece("e", 0, 1)
        with pytest.raises(ValueError, match="terminal mark"):
            TerminalMark(price=-1.0, basis="session_close", horizon="same_session")
        with pytest.raises(ValueError, match="basis and horizon"):
            TerminalMark(price=1.0, basis="", horizon="same_session")

    def test_outcomes_are_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _outcome().order_id = "x"  # type: ignore[misc]
