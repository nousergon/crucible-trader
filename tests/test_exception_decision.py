"""`exception_decision.v1` and the agreement score (alpha-engine-config-I13120)."""

from __future__ import annotations

import json
import pathlib

import pytest
from crucible.store import LocalStore

from crucible_trader import exception_decision as ed
from crucible_trader.exception_decision import (
    DecisionError,
    decide,
    decision_key,
    exception_score_key,
    load_decisions,
    load_proposals,
    record_decision,
    score,
    validate_decision,
)
from crucible_trader.exception_proposal import PROPOSER, RULESET_VERSION

DAY = "2026-09-11"
AT = "2026-09-11T22:00:00Z"


def proposal(n: int = 1, *, day: str = DAY, klass: str = "share_delta", nature: str = "defect"):
    return {
        "schema_version": "exception_proposal.v1",
        "proposal_id": f"{n:032x}",
        "exception_id": f"exc-{n}",
        "tenant": "nous-ergon-paper",
        "trading_day": day,
        "source": "broker_reconciliation",
        "class": klass,
        "nature": nature,
        "action": "investigate_shares",
        "rationale": "why",
        "rule_id": "shares.delta",
        "ruleset_version": RULESET_VERSION,
        "promotable": nature == "event",
        "held": False,
        "proposer": PROPOSER,
        "proposed_at": "2026-09-11T21:00:00Z",
    }


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path / "store")


def put_proposals(store, *proposals, day=DAY):
    document = {"schema_version": "exception_proposal.v1", "proposals": list(proposals)}
    store.put_bytes(f"trader/exception_proposals/{day}.json", json.dumps(document).encode())


class TestDecide:
    def test_agree_copies_the_proposal_and_validates(self):
        d = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)
        assert d["decided"] == d["proposed"] == {"nature": "defect", "action": "investigate_shares"}
        assert d["supersedes"] is None and len(str(d["decision_id"])) == 32
        validate_decision(d)

    def test_the_id_is_stable_and_content_addressed(self):
        a = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)
        b = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)
        c = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT, note="ok")
        assert a == b and a["decision_id"] != c["decision_id"]

    def test_a_correction_names_nature_action_and_why(self):
        d = decide(
            proposal(),
            verdict="correct",
            decided_by="brian",
            decided_at=AT,
            nature="event",
            action="await_broker_corporate_action",
            note=" split pending ",
        )
        assert d["decided"] == {"nature": "event", "action": "await_broker_corporate_action"}
        assert d["note"] == "split pending"
        validate_decision(d)

    def test_defer_carries_a_note_and_no_call(self):
        d = decide(proposal(), verdict="defer", decided_by="brian", decided_at=AT, note="wait")
        assert d["decided"] is None
        validate_decision(d)

    @pytest.mark.parametrize(
        ("kw", "message"),
        [
            ({"verdict": "approve"}, "not one of"),
            ({"decided_by": PROPOSER}, "never the proposer"),
            ({"decided_by": "Brian Smith"}, "plain name"),
            ({"decided_at": "2026-09-11 22:00"}, "decided_at"),
            ({"note": "x" * 2001}, "longer than"),
            ({"supersedes": "nope"}, "supersedes"),
            ({"verdict": "correct", "note": "n"}, "a correction names"),
            (
                {
                    "verdict": "correct",
                    "nature": "defect",
                    "action": "investigate_shares",
                    "note": "n",
                },
                "repeats the proposal",
            ),
            ({"verdict": "correct", "nature": "event", "action": "accept_partial_fill"}, "note"),
            ({"verdict": "agree", "action": "accept_partial_fill"}, "names no nature"),
            ({"verdict": "defer"}, "deferral carries a note"),
        ],
    )
    def test_refusals(self, kw, message):
        args = {"verdict": "agree", "decided_by": "brian", "decided_at": AT, **kw}
        with pytest.raises(DecisionError, match=message):
            decide(proposal(), **args)

    def test_a_malformed_proposal_is_refused(self):
        with pytest.raises(DecisionError, match="not an exception_proposal.v1"):
            decide(
                {**proposal(), "schema_version": "x"},
                verdict="agree",
                decided_by="b",
                decided_at=AT,
            )
        with pytest.raises(DecisionError, match="'rule_id' is missing"):
            decide({**proposal(), "rule_id": ""}, verdict="agree", decided_by="b", decided_at=AT)


class TestValidate:
    def good(self):
        return decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)

    def test_extra_or_missing_fields(self):
        with pytest.raises(DecisionError, match="fields differ"):
            validate_decision({**self.good(), "extra": 1})

    def test_wrong_schema(self):
        with pytest.raises(DecisionError, match="not an exception_decision.v1"):
            validate_decision({**self.good(), "schema_version": "v0"})

    def test_proposed_must_be_an_object(self):
        with pytest.raises(DecisionError, match="proposed is not an object"):
            validate_decision({**self.good(), "proposed": "x"})

    def test_an_edited_record_does_not_rebuild(self):
        with pytest.raises(DecisionError, match="does not rebuild"):
            validate_decision({**self.good(), "decided": {"nature": "event", "action": "x"}})


class TestKeys:
    def test_keys_bind_to_trading_days(self):
        assert decision_key(DAY, "a" * 32) == f"trader/exception_decisions/{DAY}/{'a' * 32}.json"
        assert exception_score_key(DAY) == f"trader/exception_scores/{DAY}.json"

    @pytest.mark.parametrize("day", ["2026-09-12", "2026-9-11", "not-a-day", 7])
    def test_a_non_session_is_refused(self, day):
        with pytest.raises(DecisionError):
            exception_score_key(day)

    def test_a_bad_id_is_refused(self):
        with pytest.raises(DecisionError, match="32 lowercase hex"):
            decision_key(DAY, "A" * 32)


class TestRecord:
    def test_create_only_idempotent_and_loaded(self, store):
        d = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)
        key = record_decision(store, d)
        assert record_decision(store, d) == key
        assert load_decisions(store) == [d]

    def test_a_different_body_at_the_same_key_is_refused(self, store):
        d = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)
        store.put_bytes(decision_key(DAY, str(d["decision_id"])), b"{}")
        with pytest.raises(DecisionError, match="different decision"):
            record_decision(store, d)

    def test_a_second_call_must_supersede_the_standing_one(self, store):
        first = decide(proposal(), verdict="defer", decided_by="brian", decided_at=AT, note="wait")
        record_decision(store, first)
        again = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)
        with pytest.raises(DecisionError, match="must supersede"):
            record_decision(store, again)
        then = decide(
            proposal(),
            verdict="agree",
            decided_by="brian",
            decided_at=AT,
            supersedes=str(first["decision_id"]),
        )
        record_decision(store, then)
        assert len(load_decisions(store)) == 2

    def test_a_conflicted_proposal_is_refused(self, store):
        for note in ("a", "b"):
            d = decide(proposal(), verdict="defer", decided_by="brian", decided_at=AT, note=note)
            store.put_bytes(
                decision_key(DAY, str(d["decision_id"])), (json.dumps(d) + "\n").encode()
            )
        d = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)
        with pytest.raises(DecisionError, match="conflicting"):
            record_decision(store, d)

    def test_a_lost_race_with_identical_bytes_is_fine(self, store, monkeypatch):
        from crucible.store import PointerConflictError

        d = decide(proposal(), verdict="agree", decided_by="brian", decided_at=AT)
        payload = (json.dumps(d, indent=2, sort_keys=True) + "\n").encode()
        key = decision_key(DAY, str(d["decision_id"]))

        def racing(k, _expected, body):
            store.put_bytes(k, body if racing.same else b"{}")
            raise PointerConflictError("lost")

        monkeypatch.setattr(store, "compare_and_swap", racing)
        racing.same = True
        assert record_decision(store, d) == key and store.get_bytes(key) == payload
        other = decide(proposal(2), verdict="agree", decided_by="brian", decided_at=AT)
        racing.same = False
        with pytest.raises(DecisionError, match="concurrently"):
            record_decision(store, other)

    def test_loaders_skip_non_json_and_refuse_bad_documents(self, store):
        store.put_bytes("trader/exception_decisions/2026-09-11/readme.txt", b"x")
        store.put_bytes("trader/exception_proposals/readme.txt", b"x")
        put_proposals(store, proposal(1))
        put_proposals(store, proposal(2, day="2026-09-14"), day="2026-09-14")
        assert [p["proposal_id"] for p in load_proposals(store, until=DAY)] == [f"{1:032x}"]
        assert len(load_proposals(store)) == 2
        assert load_decisions(store) == []
        store.put_bytes(f"trader/exception_decisions/{DAY}/{'b' * 32}.json", b"[]")
        with pytest.raises(DecisionError, match="not a JSON object"):
            load_decisions(store)
        store.put_bytes(f"trader/exception_decisions/{DAY}/{'b' * 32}.json", b"{}")
        with pytest.raises(DecisionError, match="fields differ"):
            load_decisions(store)


class TestScore:
    def test_agreement_streak_queue_and_age(self):
        props = [
            proposal(1),
            proposal(2),
            proposal(3),
            proposal(4, day="2026-09-14"),
            proposal(5, klass="corporate_action", nature="event"),
            proposal(6, day="2026-09-21"),  # after as_of: not scored
        ]
        agree1 = decide(
            props[0], verdict="agree", decided_by="b", decided_at="2026-09-11T22:00:00Z"
        )
        correct2 = decide(
            props[1],
            verdict="correct",
            decided_by="b",
            decided_at="2026-09-11T22:01:00Z",
            nature="event",
            action="await_broker_corporate_action",
            note="split",
        )
        agree3 = decide(
            props[2], verdict="agree", decided_by="b", decided_at="2026-09-11T22:02:00Z"
        )
        defer4 = decide(props[3], verdict="defer", decided_by="b", decided_at=AT, note="wait")
        orphan = decide(proposal(9), verdict="agree", decided_by="b", decided_at=AT)
        doc = score(
            props,
            [agree1, correct2, agree3, defer4, orphan],
            as_of="2026-09-18",
            generated_at="2026-09-18T21:00:00Z",
        )
        ca, shares = doc["classes"]
        assert (ca["class"], ca["promotable"], ca["undecided"], ca["agreement"]) == (
            "corporate_action",
            True,
            1,
            None,
        )
        assert shares["agree"] == 2 and shares["correct"] == 1 and shares["defer"] == 1
        assert shares["agreement"] == 0.6667 and shares["consecutive_agree"] == 1
        assert shares["auto_apply"] is False and doc["auto_apply"] is False
        totals = doc["totals"]
        assert totals["queue"] == 2 and totals["orphan_decisions"] == 1
        assert totals["agreement"] == 0.6667 and totals["proposals"] == 5
        oldest, newer = doc["queue"]
        assert (oldest["proposal_id"], oldest["age_trading_days"], oldest["overdue"]) == (
            f"{5:032x}",
            5,
            True,
        )
        assert (newer["status"], newer["note"], newer["age_trading_days"]) == (
            "deferred",
            "wait",
            4,
        )
        assert totals["overdue"] == 2 and totals["oldest_age_trading_days"] == 5

    def test_conflicted_and_empty(self):
        a = decide(proposal(), verdict="defer", decided_by="b", decided_at=AT, note="a")
        b = decide(proposal(), verdict="defer", decided_by="b", decided_at=AT, note="b")
        doc = score([proposal()], [a, b], as_of=DAY, generated_at=AT)
        assert doc["queue"][0]["status"] == "conflicted"
        assert doc["classes"][0]["conflicted"] == 1
        empty = score([], [], as_of=DAY, generated_at=AT)
        assert empty["totals"]["agreement"] is None
        assert empty["totals"]["oldest_age_trading_days"] is None


class TestCli:
    def test_record_then_score(self, store, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(ed, "_now", lambda: AT)
        put_proposals(store, proposal(1))
        uri = str(tmp_path / "store")
        rc = ed.main(
            [
                "--store",
                uri,
                "record",
                "--proposal-id",
                f"{1:032x}",
                "--verdict",
                "agree",
                "--by",
                "brian",
            ]
        )
        assert rc == 0 and json.loads(capsys.readouterr().out)["verdict"] == "agree"
        out = tmp_path / "score.json"
        assert ed.main(["--store", uri, "score", "--as-of", "2026-09-13", "--out", str(out)]) == 0
        document = json.loads(out.read_text())
        assert document["trading_day"] == DAY and document["totals"]["agreement"] == 1.0
        assert ed.main(["--store", uri, "score"]) == 0
        assert json.loads(capsys.readouterr().out)["schema_version"] == "exception_score.v1"

    def test_a_refusal_exits_2(self, store, tmp_path, capsys):
        rc = ed.main(
            [
                "--store",
                str(tmp_path / "store"),
                "record",
                "--proposal-id",
                "f" * 32,
                "--verdict",
                "agree",
                "--by",
                "brian",
            ]
        )
        assert rc == 2 and "no stored proposal" in capsys.readouterr().err


def test_module_runs_as_a_script(tmp_path, monkeypatch, capsys):
    import runpy
    import sys

    root = tmp_path / "store"
    put_proposals(LocalStore(root), proposal(1))
    monkeypatch.setattr(sys, "argv", ["exception_decision", "--store", str(root), "score"])
    monkeypatch.delitem(sys.modules, "crucible_trader.exception_decision")
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("crucible_trader.exception_decision", run_name="__main__")
    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out)["generated_at"].endswith("Z")
