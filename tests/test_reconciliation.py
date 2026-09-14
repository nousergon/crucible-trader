"""The reconciler. `alpha-engine-config-I10651`.

**The negative cases are the deliverable** (the issue's own gotcha): v1's split
check and ledger replay both passed suites that only asserted a clean book
reconciles clean. Each v1 defect has a test class here that REPRODUCES it -- the
v1 inputs, fed to this reconciler -- and asserts the result is a finding.
"""

from __future__ import annotations

import pytest

from crucible_trader.broker_statement import BrokerStatement
from crucible_trader.reconciliation import (
    CASH_TOLERANCE_USD,
    DISCREPANCY_CLASSES,
    HEADLINE_CLASS,
    CashFlow,
    CorporateAction,
    CorporateActionSet,
    Fill,
    ReconciliationInputError,
    ReconciliationInputs,
    reconcile,
)

ACCOUNT = "DU0000001"
DAY = "2026-09-11"
PREV = "2026-09-10"


def statement(positions, cash=10_000.0, day=DAY, account=ACCOUNT):
    return BrokerStatement(account=account, trading_day=day, positions=positions, cash_usd=cash)


def resolved(*tickers, actions=()):
    return CorporateActionSet(resolved=frozenset(tickers), actions=tuple(actions))


def inputs(anchor, broker, **kw):
    kw.setdefault("previous_trading_day", PREV)
    tickers = set(broker.positions) | set(anchor.positions if anchor else {})
    tickers |= {f.ticker for f in kw.get("fills", ())}
    kw.setdefault("corporate_actions", resolved(*tickers))
    return ReconciliationInputs(trading_day=DAY, broker=broker, anchor=anchor, **kw)


def classes(result):
    return [d.klass for d in result.discrepancies]


class TestCleanBook:
    def test_anchor_plus_fills_equal_broker_reconciles_clean(self):
        result = reconcile(
            inputs(
                statement({"AAA": 10, "BBB": 5}, cash=1_000.0, day=PREV),
                statement({"AAA": 15, "CCC": 2}, cash=1_000.0 - 500 + 250 - 40 + 3.5),
                fills=(Fill("AAA", 5, -500.0), Fill("BBB", -5, 250.0), Fill("CCC", 2, -40.0)),
                cash_flows=(CashFlow("interest", 3.5),),
            )
        )
        assert result.clean
        assert result.match_rate == 1.0
        assert (result.n_comparable, result.n_matched) == (3, 3)
        assert result.to_document()["headline"] is None

    def test_split_and_dividend_applied_on_both_sides_reconcile_clean(self):
        result = reconcile(
            inputs(
                statement({"AAA": 10}, cash=100.0, day=PREV),
                statement({"AAA": 20}, cash=100.0 + 10 * 0.25),
                corporate_actions=resolved(
                    "AAA",
                    actions=(
                        CorporateAction("AAA", "split", ratio=2.0),
                        CorporateAction("AAA", "cash_dividend", amount_per_share=0.25),
                    ),
                ),
            )
        )
        assert result.clean, result.discrepancies


class TestV1FalseOneThousandMatchIsImpossible:
    """v1 `reconciliation_audit.py`: `rate = 1.0 if not universe`; a malformed
    split row skipped; an unresolved split check graded `OK_UNVERIFIED`."""

    def test_zero_comparable_rows_is_no_rate_not_one(self):
        result = reconcile(inputs(statement({}, day=PREV), statement({})))
        assert result.clean  # empty books with equal cash genuinely agree...
        assert result.n_comparable == 0
        assert result.match_rate is None  # ...but there is no 1.000 to report

    def test_unresolved_split_status_fails_and_is_not_counted_as_a_match(self):
        # v1 inputs: every name matched, split query failed -> 1.000 OK_UNVERIFIED.
        result = reconcile(
            inputs(
                statement({"AAA": 10, "BBB": 4}, day=PREV),
                statement({"AAA": 10, "BBB": 4}),
                corporate_actions=CorporateActionSet(resolved=frozenset()),
            )
        )
        assert classes(result) == ["corporate_action_check_incomplete"] * 2
        assert not result.clean
        assert result.match_rate is None  # no resolved row, so no rate at all

    def test_partially_resolved_rate_counts_only_resolved_rows(self):
        result = reconcile(
            inputs(
                statement({"AAA": 10, "BBB": 4}, day=PREV),
                statement({"AAA": 10, "BBB": 4}),
                corporate_actions=resolved("AAA"),
            )
        )
        assert (result.n_comparable, result.n_matched) == (1, 1)
        assert classes(result) == ["corporate_action_check_incomplete"]

    @pytest.mark.parametrize("kind", ["spinoff", "merger", "stock_dividend"])
    def test_an_unhandled_action_kind_fails_rather_than_skipping(self, kind):
        result = reconcile(
            inputs(
                statement({"AAA": 10}, day=PREV),
                statement({"AAA": 10}),
                corporate_actions=resolved("AAA", actions=(CorporateAction("AAA", kind),)),
            )
        )
        assert classes(result) == ["unhandled_corporate_action"]

    @pytest.mark.parametrize("ratio", [None, 0.0, -2.0, float("nan")])
    def test_a_malformed_split_ratio_fails_rather_than_skipping(self, ratio):
        result = reconcile(
            inputs(
                statement({"AAA": 10}, day=PREV),
                statement({"AAA": 10}),
                corporate_actions=resolved(
                    "AAA", actions=(CorporateAction("AAA", "split", ratio=ratio),)
                ),
            )
        )
        assert "unhandled_corporate_action" in classes(result)

    def test_a_split_producing_fractional_shares_is_not_rounded_into_agreement(self):
        result = reconcile(
            inputs(
                statement({"AAA": 15}, day=PREV),
                statement({"AAA": 1}),
                corporate_actions=resolved(
                    "AAA", actions=(CorporateAction("AAA", "split", ratio=0.1),)
                ),
            )
        )
        assert "unhandled_corporate_action" in classes(result)

    def test_a_perfect_position_rate_beside_a_cash_break_is_not_clean(self):
        result = reconcile(
            inputs(statement({"AAA": 10}, cash=500.0, day=PREV), statement({"AAA": 10}, cash=0.0))
        )
        assert result.match_rate == 1.0
        assert classes(result) == ["cash_delta"]
        assert not result.clean


class TestV1TwentyVersusSevenIsImpossible:
    """v1: the ledger-replay backfill reported 20 positions against a real 7."""

    def test_a_count_disagreement_is_the_headline_finding(self):
        replayed = {f"T{i:02d}": 10 for i in range(20)}
        real = {f"T{i:02d}": 10 for i in range(7)}
        # The replayed book offered as the anchor: every one of the 13 phantom
        # positions is a share finding, AND the count is the headline.
        result = reconcile(inputs(statement(replayed, day=PREV), statement(real)))
        assert result.discrepancies[0].klass == HEADLINE_CLASS
        headline = result.discrepancies[0]
        assert (headline.expected, headline.actual, headline.delta) == (20.0, 7.0, -13.0)
        assert headline.side == "broker_under"
        assert classes(result).count("share_delta") == 13
        assert result.to_document()["headline"].startswith("position_count")
        assert result.n_positions_broker == 7

    def test_no_anchor_is_a_finding_and_nothing_is_replayed(self):
        result = reconcile(inputs(None, statement({"AAA": 1})))
        assert classes(result) == ["no_anchor"]
        assert result.expected_positions == {}
        assert result.expected_cash_usd is None

    def test_declared_genesis_is_clean_with_no_rate(self):
        result = reconcile(inputs(None, statement({"AAA": 1}), genesis=True))
        assert result.clean and result.genesis and result.match_rate is None

    def test_broker_over_count_side(self):
        result = reconcile(inputs(statement({}, day=PREV), statement({"AAA": 3})))
        assert classes(result) == ["position_count", "share_delta"]
        assert {d.side for d in result.discrepancies} == {"broker_over"}


class TestClassification:
    def test_share_delta_names_instrument_delta_and_side(self):
        (finding,) = reconcile(
            inputs(statement({"AAA": 10}, day=PREV), statement({"AAA": 7}))
        ).discrepancies
        assert (finding.klass, finding.instrument, finding.delta, finding.side) == (
            "share_delta",
            "AAA",
            -3.0,
            "broker_under",
        )
        assert "AAA" in finding.page_line() and "delta=-3" in finding.page_line()

    def test_split_not_applied_by_broker_is_a_corporate_action(self):
        (finding,) = reconcile(
            inputs(
                statement({"AAA": 10}, day=PREV),
                statement({"AAA": 10}),
                corporate_actions=resolved(
                    "AAA", actions=(CorporateAction("AAA", "split", ratio=2.0),)
                ),
            )
        ).discrepancies
        assert (finding.klass, finding.delta, finding.side) == ("corporate_action", -10.0, "broker")

    def test_dividend_not_paid_by_broker_is_a_corporate_action(self):
        (finding,) = reconcile(
            inputs(
                statement({"AAA": 100}, cash=0.0, day=PREV),
                statement({"AAA": 100}, cash=0.0),
                corporate_actions=resolved(
                    "AAA", actions=(CorporateAction("AAA", "cash_dividend", amount_per_share=0.5),)
                ),
            )
        ).discrepancies
        assert (finding.klass, finding.instrument, finding.delta) == (
            "corporate_action",
            "AAA",
            -50.0,
        )

    def test_cash_within_tolerance_is_clean_and_over_is_a_finding(self):
        anchor = statement({}, cash=100.0, day=PREV)
        assert reconcile(inputs(anchor, statement({}, cash=100.0 + CASH_TOLERANCE_USD / 2))).clean
        (finding,) = reconcile(inputs(anchor, statement({}, cash=112.0))).discrepancies
        assert (finding.klass, finding.side, finding.delta) == ("cash_delta", "broker_over", 12.0)

    def test_stale_anchor_is_a_finding(self):
        result = reconcile(inputs(statement({}, day="2026-09-08"), statement({})))
        assert classes(result) == ["stale_anchor"]

    def test_the_class_vocabulary_is_closed_and_headline_first(self):
        assert DISCREPANCY_CLASSES[0] == HEADLINE_CLASS
        assert len(set(DISCREPANCY_CLASSES)) == len(DISCREPANCY_CLASSES)

    def test_result_document_round_trips_findings(self):
        document = reconcile(inputs(statement({"AAA": 1}, day=PREV), statement({}))).to_document()
        assert document["clean"] is False
        assert [d["klass"] for d in document["discrepancies"]] == ["position_count", "share_delta"]


class TestInputsRefuseWhatIsNotAReconciliation:
    def test_broker_statement_for_another_day(self):
        with pytest.raises(ReconciliationInputError, match="not 2026-09-11"):
            ReconciliationInputs(trading_day=DAY, broker=statement({}, day=PREV), anchor=None)

    def test_genesis_over_an_anchor(self):
        with pytest.raises(ReconciliationInputError, match="genesis"):
            inputs(statement({}, day=PREV), statement({}), genesis=True)

    def test_anchor_of_another_account(self):
        with pytest.raises(ReconciliationInputError, match="account"):
            inputs(statement({}, day=PREV, account="DU9"), statement({}))

    def test_anchor_not_before_the_day(self):
        with pytest.raises(ReconciliationInputError, match="not before"):
            inputs(statement({}, day=DAY), statement({}))
