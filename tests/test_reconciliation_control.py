"""The control arm. `alpha-engine-config-I10413`.

The closes-when is "a deliberately blinded reconciler (a mutation test) fails
it". So most of this file is blinded reconcilers -- each one a plausible way a
reconciler goes quietly wrong -- and the assertion that the control arm voids
the cycle naming the class the blinding hid.
"""

from __future__ import annotations

import dataclasses

import pytest

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.reconciliation import (
    CorporateActionSet,
    Discrepancy,
    Fill,
    ReconciliationInputs,
    reconcile,
)
from crucible_trader.reconciliation_control import (
    PLANT_CLASSES,
    PLANT_PREFIX,
    ControlVerdict,
    plant,
    replay_base,
    run_control_arm,
)

DAY, PREV = "2026-09-11", "2026-09-10"


def live_inputs(**kw):
    anchor = kw.pop("anchor", BrokerStatement("DU1", PREV, {"AAA": 10}, 1_000.0))
    broker = kw.pop("broker", BrokerStatement("DU1", DAY, {"AAA": 12}, 800.0))
    kw.setdefault("fills", (Fill("AAA", 2, -200.0),))
    kw.setdefault("corporate_actions", CorporateActionSet(resolved=frozenset({"AAA"})))
    return ReconciliationInputs(
        trading_day=DAY, broker=broker, anchor=anchor, previous_trading_day=PREV, **kw
    )


def blinded(drop):
    """A reconciler that silently never reports findings matching ``drop``."""

    def reconciler(inputs):
        result = reconcile(inputs)
        kept = tuple(d for d in result.discrepancies if not drop(d))
        return dataclasses.replace(result, discrepancies=kept)

    return reconciler


class TestTheRealReconcilerPasses:
    def test_clean_live_book(self):
        verdict = run_control_arm(live_inputs())
        assert verdict.passed, verdict.reason()
        assert verdict.n_detected == 3
        assert [o.plant.klass for o in verdict.outcomes] == list(PLANT_CLASSES)

    def test_dirty_live_book_still_detects_every_plant(self):
        # A real cash break AND a real share break already present.
        broker = BrokerStatement("DU1", DAY, {"AAA": 11}, 700.0)
        verdict = run_control_arm(live_inputs(broker=broker))
        assert verdict.passed, verdict.reason()

    def test_genesis_day_is_exercised_too(self):
        inputs = ReconciliationInputs(
            trading_day=DAY,
            broker=BrokerStatement("DU1", DAY, {"AAA": 5}, 50.0),
            anchor=None,
            genesis=True,
        )
        base = replay_base(inputs)
        assert base.anchor is not None and not base.genesis
        assert run_control_arm(inputs).passed

    def test_plants_never_touch_the_live_inputs(self):
        inputs = live_inputs()
        snapshot = dataclasses.asdict(inputs)
        run_control_arm(inputs)
        assert dataclasses.asdict(inputs) == snapshot
        assert not any(t.startswith(PLANT_PREFIX) for t in inputs.broker.positions)


class TestABlindedReconcilerVoidsTheCycle:
    @pytest.mark.parametrize(
        ("blinding", "missed"),
        [
            (lambda d: d.klass == "cash_delta", ("cash_delta",)),
            (lambda d: d.klass == "share_delta", ("share_delta",)),
            (lambda d: d.klass == "corporate_action", ("corporate_action",)),
            (lambda d: True, PLANT_CLASSES),  # v1's shape: never negative
            (lambda d: d.instrument.startswith(PLANT_PREFIX), ("share_delta", "corporate_action")),
        ],
    )
    def test_missed_class_is_named(self, blinding, missed):
        verdict = run_control_arm(live_inputs(), reconciler=blinded(blinding))
        assert not verdict.passed
        assert verdict.missed == missed
        assert all(m in verdict.reason() for m in missed)
        assert "VOID" in verdict.reason()
        assert verdict.to_document()["missed"] == list(missed)

    def test_a_misclassifying_reconciler_fails(self):
        def reclassify(inputs):
            result = reconcile(inputs)
            return dataclasses.replace(
                result,
                discrepancies=tuple(
                    dataclasses.replace(d, klass="share_delta")
                    if d.klass == "corporate_action"
                    else d
                    for d in result.discrepancies
                ),
            )

        assert run_control_arm(live_inputs(), reconciler=reclassify).missed == ("corporate_action",)

    def test_a_reconciler_that_drops_a_real_finding_under_a_plant_fails(self):
        real_break = BrokerStatement("DU1", DAY, {"AAA": 11}, 800.0)

        def drops_real_when_planted(inputs):
            result = reconcile(inputs)
            if any(t.startswith(PLANT_PREFIX) for t in inputs.broker.positions):
                kept = tuple(d for d in result.discrepancies if d.instrument != "AAA")
                return dataclasses.replace(result, discrepancies=kept)
            return result

        verdict = run_control_arm(
            live_inputs(broker=real_break), reconciler=drops_real_when_planted
        )
        assert verdict.missed == ("share_delta", "corporate_action")

    def test_wrong_delta_is_not_detection(self):
        def off_by_one(inputs):
            result = reconcile(inputs)
            return dataclasses.replace(
                result,
                discrepancies=tuple(
                    dataclasses.replace(d, delta=(d.delta or 0) + 1)
                    if d.klass == "share_delta"
                    else d
                    for d in result.discrepancies
                ),
            )

        assert "share_delta" in run_control_arm(live_inputs(), reconciler=off_by_one).missed


class TestShape:
    def test_unknown_plant_class_refused(self):
        base = live_inputs()
        with pytest.raises(ValueError, match="no plant"):
            plant(base, "nav_delta", reconcile(base))

    def test_an_incomplete_outcome_set_does_not_pass(self):
        verdict = run_control_arm(live_inputs())
        assert not ControlVerdict(verdict.outcomes[:2]).passed

    def test_discrepancy_type_is_what_the_arm_diffs(self):
        assert {f.name for f in dataclasses.fields(Discrepancy)} >= {"klass", "instrument", "delta"}
