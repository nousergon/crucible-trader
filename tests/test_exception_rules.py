"""Tier-2 rule clears, the audit sample and demotion (alpha-engine-config-I13747).

The fixtures are the three real breaks that waited for Brian on 2026-11-16,
cut down to the fields the rules read: 11-03 (stale anchor, R3), 11-04 (IB
posted October interest, R1) and 11-09 (the MU dividend he waived, R2).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest
from crucible.store import LocalStore

from crucible_trader import exception_rules as er
from crucible_trader.exception_decision import (
    DecisionError,
    audit_sample,
    decide,
    demoted_rules,
    load_proposals,
    load_rule_clears,
    record_decision,
    rule_clear,
    rule_clears_key,
    score,
    validate_decision,
)
from crucible_trader.exception_proposal import PROPOSER, RULESET_VERSION
from crucible_trader.reconciliation import DividendWaiver

TENANT = "nous-ergon-paper"
SOURCE = "crucible-trader.broker_reconciliation"
NOW = "2026-11-18T22:00:00Z"
AS_OF = "2026-11-17"


def rec(day, discrepancies=(), *, clean=None, **extra):
    return {
        "schema_version": "broker_reconciliation.v1",
        "trading_day": day,
        "clean": (not discrepancies) if clean is None else clean,
        "void": False,
        "discrepancies": list(discrepancies),
        "dividends_owed": [],
        "dividends_unpaid": [],
        "dividend_credits": [],
        **extra,
    }


def stmt(day, accrued):
    return {
        "schema_version": "broker_statement.v2",
        "trading_day": day,
        "accrued_cash_usd": accrued,
    }


def exception(
    n, day, klass, *, amount=None, instrument="USD", unit="usd", expected=None, actual=None
):
    return {
        "schema_version": "exception.v1",
        "exception_id": f"exc-{n}",
        "tenant": TENANT,
        "trading_day": day,
        "source": SOURCE,
        "class": klass,
        "instrument": instrument,
        "amount": amount,
        "unit": unit,
        "expected": {"record": f"trader/reconciliation/{day}.json", "value": expected},
        "actual": {"record": f"trader/broker_statements/{day}.json", "value": actual},
        "detected_by": {"document_key": f"trader/reconciliation/{day}.json"},
    }


def proposal(n, day, klass, nature="defect"):
    return {
        "schema_version": "exception_proposal.v1",
        "proposal_id": f"{n:032x}",
        "exception_id": f"exc-{n}",
        "tenant": TENANT,
        "trading_day": day,
        "source": SOURCE,
        "class": klass,
        "nature": nature,
        "action": "investigate_cash",
        "rationale": "why",
        "rule_id": f"x.{klass}",
        "ruleset_version": RULESET_VERSION,
        "promotable": False,
        "held": False,
        "proposer": PROPOSER,
        "proposed_at": f"{day}T21:00:00Z",
    }


MU_WAIVER = DividendWaiver(
    "MU", "2026-10-29", 11.85, "Brian 2026-11-09: waive the MU dividend", "2026-11-09", ("x",)
)

STALE = {"klass": "stale_anchor", "instrument": "ACCT", "detail": "anchor statement is 2026-10-30"}
INTEREST = {"klass": "cash_delta", "instrument": "USD", "delta": 107.69, "side": "broker_over"}
MU = {"klass": "corporate_action", "instrument": "MU", "delta": -11.85, "side": "broker"}
MU_OWED = {"ticker": "MU", "pay_date": "2026-10-29", "amount_usd": 11.85}


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    s = LocalStore(tmp_path / "store")
    put = lambda key, doc: s.put_bytes(key, json.dumps(doc).encode())  # noqa: E731
    put("trader/reconciliation/2026-11-03.json", rec("2026-11-03", [STALE]))
    put("trader/reconciliation/2026-11-04.json", rec("2026-11-04", [INTEREST]))
    put("trader/reconciliation/2026-11-05.json", rec("2026-11-05"))
    put(
        "trader/reconciliation/2026-11-09.json",
        rec("2026-11-09", [MU], dividends_owed=[MU_OWED]),
    )
    for day, accrued in (("2026-11-03", 109.28), ("2026-11-04", 3.43), ("2026-11-05", 5.27)):
        put(f"trader/broker_statements/{day}.json", stmt(day, accrued))
    records = {
        "2026-11-03": [exception(1, "2026-11-03", "stale_anchor", instrument="ACCT", unit=None)],
        "2026-11-04": [exception(2, "2026-11-04", "cash_delta", amount=107.69)],
        "2026-11-09": [
            exception(
                3, "2026-11-09", "corporate_action", amount=-11.85, instrument="MU", unit=None
            )
        ],
    }
    for day, items in records.items():
        put(
            f"trader/exceptions/{day}.json",
            {"schema_version": "exception_day.v1", "records": items},
        )
    for n, (day, klass) in enumerate(
        (
            ("2026-11-03", "stale_anchor"),
            ("2026-11-04", "cash_delta"),
            ("2026-11-09", "corporate_action"),
        ),
        start=1,
    ):
        put(
            f"trader/exception_proposals/{day}.json",
            {"schema_version": "exception_proposal.v1", "proposals": [proposal(n, day, klass)]},
        )
    return s


@pytest.fixture(autouse=True)
def mu_waiver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(er, "load_declared_waivers", lambda: (MU_WAIVER,))


def drop(store, key):
    (store.root / key).unlink()


def file_clears(store, as_of=AS_OF, now=NOW):
    key, document, misses = er.file_rule_clears(store, as_of=as_of, now=now)
    if document is not None:
        store.put_bytes(key, json.dumps(document).encode())
    return document, misses


class TestRegistry:
    def test_r1_to_r4_ship_approved_by_brians_ruling(self):
        rows = {r["rule_id"]: r for r in er.registry()}
        assert sorted(rows) == ["R1", "R2", "R3", "R4"]
        for row in rows.values():
            assert row["state"] == "approved"
            assert row["approved_by"] == "cipher813"
            assert row["approved_at"] == "2026-11-17T15:21:31Z"
            assert row["approval_ref"] == "alpha-engine-config#13791"


class TestTheThreeRealBreaks:
    def test_each_clears_through_its_rule_with_its_records(self, store):
        document, misses = file_clears(store)
        assert misses == []
        clears = {c["trading_day"]: c for c in document["clears"]}
        assert {d: c["decided_by"] for d, c in clears.items()} == {
            "2026-11-03": "rule:R3",
            "2026-11-04": "rule:R1",
            "2026-11-09": "rule:R2",
        }
        for clear in clears.values():
            validate_decision(clear)
            assert clear["verdict"] == "cleared" and clear["decided"] is None

    def test_r1_shows_the_interest_arithmetic(self, store):
        document, _ = file_clears(store)
        (r1,) = [c for c in document["clears"] if c["decided_by"] == "rule:R1"]
        assert r1["rule"]["arithmetic"] == {
            "accrued_before_usd": 109.28,
            "accrued_on_day_usd": 3.43,
            "accrued_next_day_usd": 5.27,
            "accrual_fall_usd": 105.85,
            "accrued_that_day_usd": 1.84,
            "interest_posted_usd": 107.69,
            "gap_usd": 107.69,
            "difference_usd": 0.0,
            "tolerance_usd": 0.05,
        }
        assert [e["record"] for e in r1["rule"]["evidence"]] == [
            "trader/reconciliation/2026-11-04.json",
            "trader/broker_statements/2026-11-03.json",
            "trader/broker_statements/2026-11-04.json",
            "trader/broker_statements/2026-11-05.json",
        ]

    def test_r3_cites_the_clean_reconcile_that_superseded_it(self, store):
        document, _ = file_clears(store)
        (r3,) = [c for c in document["clears"] if c["decided_by"] == "rule:R3"]
        assert r3["rule"]["arithmetic"] == {"stale_on": "2026-11-03", "clean_on": "2026-11-05"}

    def test_r2_cites_the_waiver(self, store):
        document, _ = file_clears(store)
        (r2,) = [c for c in document["clears"] if c["decided_by"] == "rule:R2"]
        assert {"role": "waiver", "record": er.WAIVERS_FILE} in r2["rule"]["evidence"]

    def test_a_rerun_carries_its_clears_and_never_re_clears(self, store):
        first, _ = file_clears(store)
        again, _ = file_clears(store, now="2026-11-18T23:00:00Z")
        assert again["clears"] == first["clears"]
        later, _ = file_clears(store, as_of="2026-11-18", now="2026-11-19T22:00:00Z")
        assert later is None
        assert len(load_rule_clears(store)) == 3

    def test_the_queue_drops_them_and_the_log_shows_them(self, store):
        file_clears(store)
        clears = load_rule_clears(store)
        sampled = audit_sample(clears)
        doc = score(load_proposals(store), [], as_of=AS_OF, generated_at=NOW, clears=clears)
        # Each rule's first clear is sampled, so all three return for audit.
        assert len(sampled) == 3
        assert {q["status"] for q in doc["queue"]} == {"audit"}
        assert doc["totals"]["cleared"] == 3 and doc["totals"]["audit"] == 3
        assert {c["cleared_by_rule"] for c in doc["rule_clears"]} == {"R1", "R2", "R3"}
        assert all(q["rule_clear"]["sampled"] for q in doc["queue"])


class TestWhenARuleDoesNotClear:
    def test_missing_next_statement_leaves_the_interest_item(self, store):
        drop(store, "trader/broker_statements/2026-11-05.json")
        _, misses = file_clears(store)
        (miss,) = [m for m in misses if m["class"] == "cash_delta"]
        assert "missing or has no accrual" in miss["why"]

    def test_too_small_an_accrual_fall_leaves_the_interest_item(self, store):
        store.put_bytes(
            "trader/broker_statements/2026-11-03.json",
            json.dumps(stmt("2026-11-03", 100.00)).encode(),
        )
        _, misses = file_clears(store)
        (miss,) = [m for m in misses if m["class"] == "cash_delta"]
        assert "more than 5 cents" in miss["why"]

    def test_no_waiver_and_no_late_credit_leaves_the_dividend(self, store, monkeypatch):
        monkeypatch.setattr(er, "load_declared_waivers", lambda: ())
        _, misses = file_clears(store)
        (miss,) = [m for m in misses if m["class"] == "corporate_action"]
        assert "neither waived nor credited" in miss["why"]

    def test_a_late_credit_clears_the_dividend(self, store, monkeypatch):
        monkeypatch.setattr(er, "load_declared_waivers", lambda: ())
        store.put_bytes(
            "trader/reconciliation/2026-11-10.json",
            json.dumps(rec("2026-11-10", dividend_credits=[MU_OWED])).encode(),
        )
        document, _ = file_clears(store)
        (r2,) = [c for c in document["clears"] if c["decided_by"] == "rule:R2"]
        assert r2["rule"]["arithmetic"]["credited_on"] == "2026-11-10"

    def test_no_later_clean_reconcile_leaves_the_stale_anchor(self, store):
        drop(store, "trader/reconciliation/2026-11-05.json")
        _, misses = file_clears(store)
        assert any(m["class"] == "stale_anchor" for m in misses)

    def test_a_proposed_rule_clears_nothing(self, store, monkeypatch):
        rules = tuple(dataclasses.replace(r, state="proposed", approved_by=None) for r in er.RULES)
        monkeypatch.setattr(er, "RULES", rules)
        document, misses = file_clears(store)
        assert document is None
        assert all("the rule is proposed" in m["why"] for m in misses)

    def test_an_item_a_person_decided_is_never_cleared(self, store):
        (p,) = [q for q in load_proposals(store) if q["class"] == "cash_delta"]
        d = decide(p, verdict="defer", decided_by="brian", decided_at=NOW, note="wait")
        record_decision(store, d)
        document, _ = file_clears(store)
        assert "rule:R1" not in {c["decided_by"] for c in document["clears"]}


class TestSplitPending:
    def split(self, day):
        return {
            "klass": "corporate_action",
            "instrument": "NVDA",
            "side": "broker",
            "detail": "broker shares equal the pre-action holding; the same-day split ...",
        }

    def run(self, tmp_path, days):
        s = LocalStore(tmp_path / "split")
        for day in days:
            s.put_bytes(
                f"trader/reconciliation/{day}.json",
                json.dumps(rec(day, [self.split(day)])).encode(),
            )
        record = exception(9, days[-1], "corporate_action", instrument="NVDA", unit=None)
        return er.evaluate(er.RULES[3], record, er.StoreRecords(s), as_of=days[-1])

    def test_clears_within_five_trading_days(self, tmp_path):
        found, _ = self.run(tmp_path, ["2026-11-10", "2026-11-11", "2026-11-12"])
        assert found is not None and found.arithmetic["first_seen"] == "2026-11-10"

    def test_returns_after_five_trading_days(self, tmp_path):
        days = ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13", "2026-11-16"]
        found, why = self.run(tmp_path, days)
        assert found is None and "returns for review" in why

    def test_a_dividend_break_is_not_a_split(self, store):
        record = exception(3, "2026-11-09", "corporate_action", amount=-11.85, instrument="MU")
        found, _ = er.evaluate(er.RULES[3], record, er.StoreRecords(store), as_of=AS_OF)
        assert found is None


class TestAuditAndDemotion:
    def test_one_in_ten_per_rule_returns_starting_with_the_first(self):
        p = proposal(1, "2026-11-04", "cash_delta")
        clears = [
            rule_clear(
                {**p, "proposal_id": f"{n:032x}"},
                rule_id="R1",
                evidence=[{"role": "r", "record": "k"}],
                arithmetic={},
                note="n",
                decided_at=f"2026-11-04T21:{n:02d}:00Z",
            )
            for n in range(21)
        ]
        sampled = audit_sample(clears)
        assert sampled == {clears[i]["decision_id"] for i in (0, 10, 20)}

    def test_confirm_takes_it_off_the_queue(self, store):
        file_clears(store)
        clears = load_rule_clears(store)
        r1 = next(c for c in clears if c["decided_by"] == "rule:R1")
        p = next(q for q in load_proposals(store) if q["proposal_id"] == r1["proposal_id"])
        ok = decide(
            p, verdict="confirm", decided_by="brian", decided_at=NOW, supersedes=r1["decision_id"]
        )
        record_decision(store, ok)
        doc = score(load_proposals(store), [ok], as_of=AS_OF, generated_at=NOW, clears=clears)
        assert r1["proposal_id"] not in {q["proposal_id"] for q in doc["queue"]}
        assert (
            next(c for c in doc["rule_clears"] if c["cleared_by_rule"] == "R1")["status"]
            == "confirmed"
        )

    def test_a_reject_demotes_the_rule_and_it_clears_nothing_more(self, store):
        file_clears(store)
        clears = load_rule_clears(store)
        r1 = next(c for c in clears if c["decided_by"] == "rule:R1")
        p = next(q for q in load_proposals(store) if q["proposal_id"] == r1["proposal_id"])
        no = decide(
            p,
            verdict="reject",
            decided_by="brian",
            decided_at=NOW,
            supersedes=r1["decision_id"],
            note="that was not interest",
        )
        record_decision(store, no)
        assert demoted_rules([*clears, no]) == {"R1": no["decision_id"]}
        doc = score(
            load_proposals(store),
            [no],
            as_of=AS_OF,
            generated_at=NOW,
            clears=clears,
            rules=er.registry(),
        )
        item = next(q for q in doc["queue"] if q["proposal_id"] == r1["proposal_id"])
        assert item["status"] == "rejected" and item["rule_clear"]["rule_demoted"]
        row = next(r for r in doc["rules"] if r["rule_id"] == "R1")
        assert row["effective_state"] == "demoted" and row["rejected"] == 1
        # A new interest break: the demoted rule would match and does not clear.
        store.put_bytes(
            "trader/exception_proposals/2026-11-05.json",
            json.dumps(
                {"proposals": [{**p, "proposal_id": f"{7:032x}", "exception_id": "exc-2"}]}
            ).encode(),
        )
        _, misses = file_clears(store, as_of="2026-11-18", now="2026-11-19T22:00:00Z")
        assert any("the rule is demoted" in m["why"] for m in misses)

    def test_a_clear_is_judged_only_by_confirm_or_reject(self, store):
        file_clears(store)
        r1 = next(c for c in load_rule_clears(store) if c["decided_by"] == "rule:R1")
        p = next(q for q in load_proposals(store) if q["proposal_id"] == r1["proposal_id"])
        agree = decide(
            p, verdict="agree", decided_by="brian", decided_at=NOW, supersedes=r1["decision_id"]
        )
        with pytest.raises(DecisionError, match="confirm or reject"):
            record_decision(store, agree)


class TestSeparationOfDuties:
    def test_no_person_files_a_clear_and_no_rule_files_a_call(self):
        p = proposal(1, "2026-11-04", "cash_delta")
        with pytest.raises(DecisionError):
            decide(p, verdict="cleared", decided_by="brian", decided_at=NOW)
        with pytest.raises(DecisionError):
            decide(p, verdict="agree", decided_by="rule:R1", decided_at=NOW)

    def test_a_clear_is_never_written_as_a_decision(self, store):
        p = proposal(1, "2026-11-04", "cash_delta")
        clear = rule_clear(
            p,
            rule_id="R1",
            evidence=[{"role": "r", "record": "k"}],
            arithmetic={},
            note="n",
            decided_at=NOW,
        )
        with pytest.raises(DecisionError, match="filed by the reconcile job"):
            record_decision(store, clear)

    def test_a_clear_without_evidence_is_refused(self):
        p = proposal(1, "2026-11-04", "cash_delta")
        with pytest.raises(DecisionError, match="cites at least one"):
            rule_clear(p, rule_id="R1", evidence=[], arithmetic={}, note="n", decided_at=NOW)

    def test_a_tampered_clear_does_not_validate(self):
        p = proposal(1, "2026-11-04", "cash_delta")
        clear = rule_clear(
            p,
            rule_id="R1",
            evidence=[{"role": "r", "record": "k"}],
            arithmetic={},
            note="n",
            decided_at=NOW,
        )
        with pytest.raises(DecisionError):
            validate_decision({**clear, "decided_by": "rule:R2"})

    def test_proposals_never_read_the_clears_file(self, store):
        file_clears(store)
        assert store.exists(rule_clears_key(AS_OF))
        assert len(load_proposals(store)) == 3


class TestTheReconcileStep:
    def ctx(self, store, day="2026-11-17"):
        import datetime as dt

        from crucible.runner import RunContext

        from crucible_trader.reconciliation_job import RECONCILE_JOB

        return RunContext(
            run_id="01JG0000000000000000000000",
            job=RECONCILE_JOB,
            trading_day=dt.date.fromisoformat(day),
            calendar_date=dt.date.fromisoformat(day),
            store=store,
            seed=0,
            started=dt.datetime(2026, 11, 17, 21, 0, tzinfo=dt.UTC),
        )

    def test_files_the_clears_and_the_score_counts_them(self, store):
        from crucible.models import MetricRecordRow

        from crucible_trader.reconciliation_job import (
            EXCEPTION_RULE_CLEARS_METRIC,
            EXCEPTION_SCORE_METRIC,
            apply_rule_clears,
            file_exception_score,
        )

        ctx = self.ctx(store)
        apply_rule_clears(ctx, now=NOW)
        file_exception_score(ctx, now=NOW)
        (metric,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_RULE_CLEARS_METRIC]
        MetricRecordRow.model_validate(metric)
        assert metric["status"] == "OK" and metric["value"] == 3.0
        assert [o["key"] for o in ctx.outputs][0] == rule_clears_key(AS_OF)
        (scored,) = [m for m in ctx.metrics if m["name"] == EXCEPTION_SCORE_METRIC]
        assert "3 cleared by approved rules, 3 back for audit" in scored["status_reason"]

    def test_a_failing_rule_never_reaches_the_job(self, store, monkeypatch):
        from crucible_trader import reconciliation_job as job

        def boom(*_a, **_k):
            raise RuntimeError("rules broke")

        monkeypatch.setattr(job, "file_rule_clears", boom)
        ctx = self.ctx(store)
        job.apply_rule_clears(ctx, now=NOW)
        (metric,) = [m for m in ctx.metrics if m["name"] == job.EXCEPTION_RULE_CLEARS_METRIC]
        assert metric["status"] == "FAIL" and "rules broke" in metric["status_reason"]
        assert ctx.outputs == []


class FakeRecords:
    def __init__(self, recs=None, stmts=None):
        self.recs, self.stmts = recs or {}, stmts or {}

    def reconciliation(self, day):
        return self.recs.get(day)

    def statement(self, day):
        return self.stmts.get(day)

    def reconciliation_days(self):
        return sorted(self.recs)


R1, R2 = er.RULES[0], er.RULES[1]
STMTS = {
    d: stmt(d, a) for d, a in (("2026-11-03", 109.28), ("2026-11-04", 3.43), ("2026-11-05", 5.27))
}
CASH = exception(2, "2026-11-04", "cash_delta", amount=107.69)


@pytest.mark.parametrize(
    ("record", "records", "why"),
    [
        (CASH, FakeRecords(stmts=STMTS), "no reconciliation"),
        (CASH, FakeRecords({"2026-11-04": {**rec("2026-11-04"), "void": True}}, STMTS), "void"),
        (
            {**CASH, "amount": -1.0},
            FakeRecords({"2026-11-04": rec("2026-11-04", [INTEREST])}, STMTS),
            "in the account's favour",
        ),
        (
            CASH,
            FakeRecords(
                {"2026-11-04": {**rec("2026-11-04"), "discrepancies": None}},
                {**STMTS, "2026-11-03": stmt("2026-11-03", 3.43)},
            ),
            "did not fall",
        ),
        (
            CASH,
            FakeRecords(
                {"2026-11-04": rec("2026-11-04", [INTEREST])},
                {**STMTS, "2026-11-05": stmt("2026-11-05", 1.00)},
            ),
            "fell again",
        ),
    ],
)
def test_r1_refusals(record, records, why):
    found, reason = er.evaluate(R1, record, records, as_of=AS_OF)
    assert found is None and why in reason


@pytest.mark.parametrize(
    ("record", "records", "why"),
    [
        (
            exception(3, "2026-11-09", "corporate_action", amount=-11.85, instrument="MU"),
            FakeRecords(),
            "no reconciliation",
        ),
        (
            exception(3, "2026-11-09", "corporate_action", amount=None, instrument="MU"),
            FakeRecords({"2026-11-09": {**rec("2026-11-09"), "dividends_owed": None}}),
            "not a dividend break",
        ),
        (
            exception(3, "2026-11-09", "corporate_action", amount=-11.85, instrument="MU"),
            FakeRecords({"2026-11-09": {**rec("2026-11-09"), "dividends_owed": None}}),
            "not a dividend break",
        ),
    ],
)
def test_r2_refusals(record, records, why):
    found, reason = er.evaluate(R2, record, records, as_of=AS_OF, waivers=(MU_WAIVER,))
    assert found is None and why in reason


def test_a_rule_covers_only_its_class_and_needs_a_condition():
    found, why = er.evaluate(
        R1, exception(1, "2026-11-03", "stale_anchor"), FakeRecords(), as_of=AS_OF
    )
    assert found is None and "covers cash_delta" in why
    ghost = dataclasses.replace(R1, rule_id="R9")
    found, why = er.evaluate(ghost, CASH, FakeRecords(), as_of=AS_OF)
    assert found is None and "no condition in code" in why


def test_an_item_with_no_exception_record_is_reported(store):
    (store.root / "trader/exceptions/2026-11-04.json").unlink()
    _, misses = file_clears(store)
    assert {"proposal_id": f"{2:032x}", "why": "no exception record"} in misses


class TestRefusals:
    P = proposal(1, "2026-11-04", "cash_delta")
    EV = ({"role": "r", "record": "k"},)

    def clear(self, **kw):
        args = {
            "rule_id": "R1",
            "evidence": self.EV,
            "arithmetic": {},
            "note": "n",
            "decided_at": NOW,
        }
        return rule_clear(self.P, **{**args, **kw})

    @pytest.mark.parametrize(
        ("kw", "why"),
        [
            ({"verdict": "confirm", "nature": "event", "supersedes": "a" * 32}, "names no nature"),
            ({"verdict": "confirm"}, "supersedes the rule clear"),
            ({"verdict": "reject", "supersedes": "a" * 32}, "carries a note"),
        ],
    )
    def test_audit_verdicts(self, kw, why):
        with pytest.raises(DecisionError, match=why):
            decide(self.P, decided_by="brian", decided_at=NOW, **kw)

    @pytest.mark.parametrize(
        ("kw", "why"),
        [
            ({"rule_id": "X1"}, "is not R<n>"),
            ({"decided_at": "yesterday"}, "is not YYYY"),
            ({"note": " "}, "says why"),
        ],
    )
    def test_rule_clear(self, kw, why):
        with pytest.raises(DecisionError, match=why):
            self.clear(**kw)

    @pytest.mark.parametrize(
        ("change", "why"),
        [
            (lambda d: {**d, "extra": 1}, "fields differ"),
            (lambda d: {**d, "schema_version": "v0"}, "not an exception_decision.v1"),
            (lambda d: {**d, "proposed": None}, "are objects"),
            (lambda d: {**d, "rule": {**d["rule"], "x": 1}}, "names rule_id, evidence"),
            (lambda d: {**d, "rule": {**d["rule"], "evidence": "k"}}, "is a list"),
            (lambda d: {**d, "decision_id": "0" * 32}, "does not rebuild"),
        ],
    )
    def test_validate(self, change, why):
        with pytest.raises(DecisionError, match=why):
            validate_decision(change(self.clear()))

    def put(self, store, name, document):
        store.put_bytes(
            f"trader/exception_proposals/rule_clears/{name}", json.dumps(document).encode()
        )

    def test_loading_skips_other_files_and_later_days(self, store):
        self.put(store, "notes.txt", {})
        self.put(
            store,
            "2026-11-20.json",
            {"schema_version": "exception_rule_clears.v1", "clears": [self.clear()]},
        )
        assert load_rule_clears(store, until="2026-11-19") == []

    @pytest.mark.parametrize(
        ("document", "why"),
        [
            ({"schema_version": "v0"}, "not an exception_rule_clears.v1"),
            (
                {"schema_version": "exception_rule_clears.v1", "clears": [{"verdict": "cleared"}]},
                "fields differ",
            ),
        ],
    )
    def test_loading_refuses_a_bad_file(self, store, document, why):
        self.put(store, "2026-11-04.json", document)
        with pytest.raises(DecisionError, match=why):
            load_rule_clears(store)

    def test_loading_refuses_a_persons_call_in_the_clears_file(self, store):
        call = decide(self.P, verdict="agree", decided_by="brian", decided_at=NOW)
        self.put(
            store,
            "2026-11-04.json",
            {"schema_version": "exception_rule_clears.v1", "clears": [call]},
        )
        with pytest.raises(DecisionError, match="is not a rule clear"):
            load_rule_clears(store)


class TestCli:
    def test_dry_run_prints_what_would_clear(self, store, capsys):
        assert er.main(["--store", str(store.root), "--as-of", "2026-11-17"]) == 0
        out = json.loads(capsys.readouterr().out)
        assert len(out["would_file"]["clears"]) == 3 and out["not_cleared"] == []
        assert not store.exists(rule_clears_key(AS_OF))

    def test_a_refusal_exits_2(self, store, capsys):
        TestRefusals().put(store, "2026-11-04.json", {"schema_version": "v0"})
        assert er.main(["--store", str(store.root), "--as-of", "2026-11-15"]) == 2
        assert "refused" in capsys.readouterr().err

    def test_module_runs_as_a_script(self, store, monkeypatch, capsys):
        import runpy
        import sys

        monkeypatch.setattr(sys, "argv", ["exception_rules", "--store", str(store.root)])
        monkeypatch.delitem(sys.modules, "crucible_trader.exception_rules")
        with pytest.raises(SystemExit) as exited:
            runpy.run_module("crucible_trader.exception_rules", run_name="__main__")
        assert exited.value.code == 0
        assert "not_cleared" in json.loads(capsys.readouterr().out)


def test_an_unrecordable_metric_never_reaches_the_job(store, monkeypatch):
    from crucible_trader import reconciliation_job as job

    ctx = TestTheReconcileStep().ctx(store)

    def refuse(_metric):
        raise RuntimeError("no metrics today")

    monkeypatch.setattr(ctx, "record_metric", refuse)
    job.apply_rule_clears(ctx, now=NOW)
    assert ctx.outputs


def test_an_unsampled_clear_leaves_the_queue():
    first, second = proposal(1, "2026-11-04", "cash_delta"), proposal(2, "2026-11-05", "cash_delta")
    clears = [
        rule_clear(
            p,
            rule_id="R1",
            evidence=[{"role": "r", "record": "k"}],
            arithmetic={},
            note="n",
            decided_at=f"{p['trading_day']}T22:00:00Z",
        )
        for p in (first, second)
    ]
    doc = score([first, second], [], as_of=AS_OF, generated_at=NOW, clears=clears)
    assert [q["proposal_id"] for q in doc["queue"]] == [first["proposal_id"]]
    assert [c["status"] for c in doc["rule_clears"]] == ["cleared", "audit"]
    assert doc["totals"]["cleared"] == 2 and doc["totals"]["audit"] == 1


def test_an_audit_item_ages_from_the_day_it_was_cleared():
    """alpha-engine-config-I14054: a 10-trading-day-old break cleared today is
    age 0 and not overdue; an undecided break keeps its break-day age."""
    from crucible_trader.exception_decision import _expires

    old, open_ = proposal(1, "2026-11-04", "cash_delta"), proposal(2, "2026-11-04", "stale_anchor")
    today = "2026-11-18"
    clear = rule_clear(
        old,
        rule_id="R1",
        evidence=[{"role": "r", "record": "k"}],
        arithmetic={},
        note="n",
        decided_at=f"{today}T22:00:00Z",
    )
    doc = score([old, open_], [], as_of=today, generated_at=NOW, clears=[clear])
    by_id = {q["proposal_id"]: q for q in doc["queue"]}
    audit, undecided = by_id[old["proposal_id"]], by_id[open_["proposal_id"]]
    assert audit["status"] == "audit"
    assert audit["age_trading_days"] == 0 and audit["overdue"] is False
    assert audit["expires"] == _expires(today)
    assert undecided["status"] == "undecided"
    assert undecided["age_trading_days"] == 10 and undecided["overdue"] is True
    assert undecided["expires"] == _expires("2026-11-04")
