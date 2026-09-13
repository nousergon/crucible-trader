"""The two-document contract, and every way the trader refuses to trade.

`alpha-engine-config-I10648`, deliverable 4. **The negative cases are the
test.** A happy-path suite over a refusal contract grades nothing: the property
this module exists for is that a champion the harness could not attest stops the
trader dead, and only a failing case can show that.

The consumer-contract half is `TestTheFeedIsReadThroughItsPublishedSchema`: this
repository validates a feed against `crucible/schemas/predictions_feed.v1.json`
as shipped in the pinned wheel, so a schema change in the harness that this
trader cannot consume fails here rather than at 06:00 on a market day.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from crucible.champion import CHAMPION_SCHEMA_VERSION
from crucible.explain import ChainVerification
from crucible.keys import (
    TRADER_EVIDENCE_KEY,
    arm_predictions_key,
    champion_key,
    predictions_key,
)
from crucible.models import TraderEvidenceDocument
from crucible.serving import PREDICTIONS_FEED_SCHEMA_VERSION
from crucible.store import LocalStore
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from crucible_trader import contract as contract_module
from crucible_trader.contract import (
    ContractRefusal,
    ContractUnavailable,
    record_session,
    resolve,
)

DAY = "2026-09-11"
NEXT_DAY = "2026-09-14"
ARM = "m:ridge_21d:0123456789ab"
OTHER_ARM = "m:ridge_63d:ba9876543210"
MANIFEST_KEY = f"runs/experiment.grade/{DAY}/run.json"


@pytest.fixture
def store(tmp_path: pathlib.Path) -> LocalStore:
    return LocalStore(tmp_path)


def _put(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document).encode("utf-8"))


def _pointer(arm_id: str = ARM, *, slot: str = "m", **overrides: object) -> dict:
    document: dict = {
        "schema_version": CHAMPION_SCHEMA_VERSION,
        "slot": slot,
        "arm_id": arm_id,
        "as_of": DAY,
        "decided_at": "2026-09-12T02:00:00Z",
        "run_id": "01JG0000000000000000000000",
        "code_sha": "a" * 40,
        "promotion_source": "evidence",
        "manifest_key": MANIFEST_KEY,
        "evidence": {"status": "decided", "moved": True, "paired_dates": 40},
    }
    document.update(overrides)
    return document


def _feed(*, champion: str = ARM, trading_day: str = DAY) -> dict:
    return {
        "schema_version": PREDICTIONS_FEED_SCHEMA_VERSION,
        "slot": "m",
        "trading_day": trading_day,
        "champion": champion,
        "feature_version": "v3",
        "source_key": arm_predictions_key(champion, trading_day),
        "predicted_alpha": {"AAPL": 0.0031, "MSFT": -0.0012},
    }


def _serveable(store: LocalStore, *, champion: str = ARM, trading_day: str = DAY) -> None:
    """A store in which the contract holds: pointer, its producing manifest, feed."""
    _put(store, champion_key("m"), _pointer(champion))
    _put(store, MANIFEST_KEY, {"status": "ok", "job": "experiment.grade", "reason": ""})
    _put(store, predictions_key(trading_day), _feed(champion=champion, trading_day=trading_day))


class TestAServeableContractResolves:
    def test_both_halves_come_back_agreeing(self, store: LocalStore) -> None:
        _serveable(store)

        resolved = resolve(store, DAY)

        assert resolved.arm_id == ARM
        assert resolved.trading_day == DAY
        assert resolved.feed.predicted_alpha["AAPL"] == pytest.approx(0.0031)

    def test_the_resolved_contract_is_frozen(self, store: LocalStore) -> None:
        """What a session serves is decided once, before the first order."""
        _serveable(store)
        resolved = resolve(store, DAY)

        with pytest.raises(AttributeError):
            resolved.trading_day = NEXT_DAY  # type: ignore[misc]


class TestAnAbsentHalfIsAHoldNotAPage:
    def test_no_champion_pointer_yet(self, store: LocalStore) -> None:
        with pytest.raises(ContractUnavailable, match="no champion pointer"):
            resolve(store, DAY)

    def test_no_feed_published_for_this_session(self, store: LocalStore) -> None:
        _put(store, champion_key("m"), _pointer())
        _put(store, MANIFEST_KEY, {"status": "ok", "job": "experiment.grade", "reason": ""})

        with pytest.raises(ContractUnavailable, match="no predictions feed"):
            resolve(store, DAY)

    def test_an_absence_is_never_a_refusal(self, store: LocalStore) -> None:
        """The two are separate types precisely so a caller cannot handle one
        while believing it handled the other."""
        assert not issubclass(ContractUnavailable, ContractRefusal)
        assert not issubclass(ContractRefusal, ContractUnavailable)


class TestAnUnattestedChampionStopsTheTrader:
    def test_a_champion_whose_producing_run_failed_is_refused(self, store: LocalStore) -> None:
        _serveable(store)
        _put(
            store,
            MANIFEST_KEY,
            {"status": "failed", "job": "experiment.grade", "reason": "arena raised"},
        )

        with pytest.raises(ContractRefusal, match="exists and is unusable"):
            resolve(store, DAY)

    def test_a_champion_whose_manifest_is_absent_is_refused(self, store: LocalStore) -> None:
        """A pointer whose producing run cannot be found is indistinguishable
        from one no run ever wrote."""
        _put(store, champion_key("m"), _pointer())
        _put(store, predictions_key(DAY), _feed())

        with pytest.raises(ContractRefusal, match="exists and is unusable"):
            resolve(store, DAY)

    def test_a_pointer_that_is_not_readable_json_is_refused(self, store: LocalStore) -> None:
        store.put_bytes(champion_key("m"), b"{not json")

        with pytest.raises(ContractRefusal, match="exists and is unusable"):
            resolve(store, DAY)

    def test_an_s_champion_without_a_pass_attestation_is_refused(self, store: LocalStore) -> None:
        """The S slot is attested; the harness's reader refuses it and this
        module must surface that as a page, not as a hold."""
        s_manifest = f"runs/experiment.grade/{DAY}/s/run.json"
        _put(
            store,
            champion_key("s"),
            _pointer("s:momentum:0011223344ff", slot="s", manifest_key=s_manifest),
        )
        _put(store, s_manifest, {"status": "ok", "job": "experiment.grade", "reason": ""})

        with pytest.raises(ContractRefusal, match="exists and is unusable"):
            resolve(store, DAY, slot="s")


class TestAMalformedOrMisfiledFeedIsRefused:
    def test_a_feed_that_is_not_readable_json_is_refused(self, store: LocalStore) -> None:
        _serveable(store)
        store.put_bytes(predictions_key(DAY), b"[]")

        with pytest.raises(ContractRefusal, match="unserveable"):
            resolve(store, DAY)

    def test_a_feed_carrying_another_days_body_is_refused(self, store: LocalStore) -> None:
        """The key and the body are two statements of the same fact. A trader
        that trusted the key alone would size today on another day's
        cross-section."""
        _serveable(store)
        _put(store, predictions_key(DAY), _feed(trading_day=NEXT_DAY))

        with pytest.raises(ContractRefusal, match="unserveable"):
            resolve(store, DAY)

    def test_an_empty_cross_section_is_refused(self, store: LocalStore) -> None:
        """A feed with no names is not an empty opinion; it is a producer that
        failed and wrote anyway."""
        _serveable(store)
        document = _feed()
        document["predicted_alpha"] = {}
        _put(store, predictions_key(DAY), document)

        with pytest.raises(ContractRefusal, match="unserveable"):
            resolve(store, DAY)


class TestTheTwoHalvesMustNameTheSameArm:
    def test_a_feed_from_another_arm_is_refused(self, store: LocalStore) -> None:
        """Neither document is wrong on its own -- which is why only a reader of
        BOTH can catch it, and why this check lives here rather than in either
        of the harness's readers."""
        _serveable(store)
        _put(store, predictions_key(DAY), _feed(champion=OTHER_ARM))

        with pytest.raises(ContractRefusal, match="disagree about what is being served"):
            resolve(store, DAY)


class TestABrokenMoneyPathRefusesTheSession:
    def test_a_chain_that_does_not_verify_stops_the_trader(
        self, store: LocalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _serveable(store)
        monkeypatch.setattr(
            contract_module,
            "verify_money_path_chain",
            lambda _store: ChainVerification(
                status="failed",
                reason="record 3's prev_sha256 does not match its predecessor",
                records=(),
                unlinked=(),
            ),
        )

        with pytest.raises(ContractRefusal, match="does not verify"):
            resolve(store, DAY)

    def test_an_intact_chain_does_not_block_a_session(self, store: LocalStore) -> None:
        """An empty store has an intact (empty) history -- a true statement, and
        a different one from "verified 40 records"."""
        _serveable(store)

        assert resolve(store, DAY).arm_id == ARM


class TestTheConsumerEvidenceIsTheTradersOwnAccount:
    def test_a_first_session_files_one_day(self, store: LocalStore) -> None:
        _serveable(store)
        document = record_session(store, resolve(store, DAY), calendar_date="2026-09-11")

        assert document.trading_days == 1
        assert document.days_served == [DAY]
        assert document.champion == ARM

    def test_it_lands_at_the_key_the_gate_reads(self, store: LocalStore) -> None:
        _serveable(store)
        record_session(store, resolve(store, DAY), calendar_date="2026-09-11")

        stored = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY).decode("utf-8"))

        assert stored["trading_days"] == 1
        TraderEvidenceDocument.model_validate(stored)

    def test_a_second_session_on_the_same_champion_accumulates(self, store: LocalStore) -> None:
        _serveable(store)
        record_session(store, resolve(store, DAY), calendar_date="2026-09-11")
        _serveable(store, trading_day=NEXT_DAY)
        document = record_session(store, resolve(store, NEXT_DAY), calendar_date="2026-09-14")

        assert document.trading_days == 2
        assert document.days_served == [DAY, NEXT_DAY]

    def test_recording_the_same_day_twice_does_not_inflate_the_count(
        self, store: LocalStore
    ) -> None:
        """A retried session is one day. Without this a crash-and-retry loop
        would manufacture a week."""
        _serveable(store)
        record_session(store, resolve(store, DAY), calendar_date="2026-09-11")
        document = record_session(store, resolve(store, DAY), calendar_date="2026-09-11")

        assert document.trading_days == 1

    def test_a_promotion_restarts_the_count(self, store: LocalStore) -> None:
        """ "A week on the v2 champion" is a week on ONE champion. A rolling
        total across promotions would let five one-day champions read as a
        week."""
        _serveable(store)
        record_session(store, resolve(store, DAY), calendar_date="2026-09-11")
        _serveable(store, champion=OTHER_ARM, trading_day=NEXT_DAY)
        document = record_session(store, resolve(store, NEXT_DAY), calendar_date="2026-09-14")

        assert document.champion == OTHER_ARM
        assert document.trading_days == 1
        assert document.days_served == [NEXT_DAY]

    def test_an_existing_document_that_does_not_validate_raises(self, store: LocalStore) -> None:
        """Overwriting it with a fresh count would silently reset the evidence
        the gate grades -- the one write this module must never make by
        accident."""
        _serveable(store)
        _put(store, TRADER_EVIDENCE_KEY, {"schema_version": "trader_evidence.v1"})

        with pytest.raises(ValidationError):
            record_session(store, resolve(store, DAY), calendar_date="2026-09-11")

    def test_five_sessions_are_the_week_the_gate_asks_for(self, store: LocalStore) -> None:
        """Five TRADING days, not seven calendar days (§4.12)."""
        week = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11", "2026-09-14"]
        for day in week:
            _serveable(store, trading_day=day)
            document = record_session(store, resolve(store, day), calendar_date=day)

        assert document.trading_days == 5
        assert document.days_served == week


class TestTheFeedIsReadThroughItsPublishedSchema:
    """The consumer-contract half: this repository validates against the schema
    the harness generated and shipped, not against a copy of its own."""

    @staticmethod
    def _schema() -> dict:
        import crucible

        path = (
            pathlib.Path(crucible.__file__).resolve().parent
            / "schemas"
            / "predictions_feed.v1.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_a_feed_this_trader_serves_validates_against_the_shipped_schema(self) -> None:
        Draft202012Validator(self._schema()).validate(_feed())

    def test_the_shipped_schema_still_refuses_an_empty_cross_section(self) -> None:
        """A second implementation of this trader reads the schema and no
        Python. If that rule ever left the schema, the refusal above would
        become a Python-only guarantee -- and this test is what says so."""
        document = _feed()
        document["predicted_alpha"] = {}

        with pytest.raises(Exception, match="(?i)minproperties|too short|non-empty|0"):
            Draft202012Validator(self._schema()).validate(document)

    def test_the_evidence_schema_this_trader_writes_is_the_one_the_gate_reads(self) -> None:
        import crucible

        path = (
            pathlib.Path(crucible.__file__).resolve().parent / "schemas" / "trader_evidence.v1.json"
        )
        schema = json.loads(path.read_text(encoding="utf-8"))
        document = TraderEvidenceDocument(
            schema_version="trader_evidence.v1",
            slot="m",
            champion=ARM,
            trading_days=1,
            days_served=[DAY],
            calendar_date="2026-09-11",
        )

        Draft202012Validator(schema).validate(document.model_dump())
