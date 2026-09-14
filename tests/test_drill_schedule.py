"""The sealed fire-drill schedule (`alpha-engine-config-I10761`): seal before, open at the fire."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import pytest
from crucible.keys import trader_fire_drill_schedule_key
from crucible.models import (
    FIRE_DRILL_SEAL_TOLERANCE_SECONDS,
    FireDrillScheduleDocument,
    fire_drill_commitment,
)
from crucible.store import LocalStore, PointerConflictError
from ib_fakes import T0
from s3_fake import FakeSsm

from crucible_trader import drill_schedule
from crucible_trader.drill_schedule import (
    NONCE_PARAMETER_ROOT,
    SealDoesNotOpenError,
    SealOpening,
    SealRefusedError,
    open_seal,
    seal_schedule,
)

FIRE = "2026-09-08T15:00:00Z"
WINDOW = {"window_start": "2026-09-08", "window_end": "2026-09-11"}
SCHEDULE_ID = "a" * 32
NONCE = "b" * 64


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _seal(store, ssm, **overrides):
    kwargs = {
        **WINDOW,
        "fire_instant": FIRE,
        "clock": lambda: T0,
        "token": lambda: NONCE,
        "new_id": lambda: SCHEDULE_ID,
    } | overrides
    return seal_schedule(store, ssm, **kwargs)


def _opening() -> SealOpening:
    return SealOpening(SCHEDULE_ID, WINDOW["window_start"], WINDOW["window_end"], FIRE, NONCE)


class TestSeal:
    def test_the_nonce_goes_to_a_secure_string_and_only_the_commitment_to_the_store(
        self, store
    ) -> None:
        ssm = FakeSsm()
        seal = _seal(store, ssm)
        where = trader_fire_drill_schedule_key("2026-09-08", "2026-09-11", SCHEDULE_ID)
        assert seal.key == where
        assert seal.parameter == f"{NONCE_PARAMETER_ROOT}{SCHEDULE_ID}"
        (parameter,) = ssm.parameters.values()
        assert parameter["Type"] == "SecureString" and parameter["Overwrite"] is False
        assert parameter["Value"] == NONCE
        raw = store.get_bytes(where)
        document = FireDrillScheduleDocument.model_validate_json(raw)
        assert document.commitment == fire_drill_commitment(FIRE, NONCE)
        assert NONCE not in raw.decode() and FIRE not in raw.decode()
        assert NONCE not in json.dumps(dataclasses.asdict(seal))

    def test_the_default_nonce_and_id_are_fresh_and_well_formed(self, store) -> None:
        ssm = FakeSsm()
        seal = seal_schedule(store, ssm, **WINDOW, fire_instant=FIRE, clock=lambda: T0)
        (parameter,) = ssm.parameters.values()
        assert len(seal.schedule_id) == 32 and len(parameter["Value"]) == 64

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"fire_instant": "2026-09-08 15:00"}, "cannot seal"),
            ({"window_end": "2026-09-12"}, "not an NYSE trading day"),
            ({"fire_instant": "2026-09-14T15:00:00Z"}, "outside the window"),
            ({"fire_instant": "2026-09-08T22:00:00Z"}, "outside regular NYSE hours"),
            ({"clock": lambda: T0.replace(hour=15)}, "not in the future"),
        ],
    )
    def test_a_refused_seal_writes_nothing(self, store, overrides, match) -> None:
        ssm = FakeSsm()
        with pytest.raises(SealRefusedError, match=match):
            _seal(store, ssm, **overrides)
        assert ssm.parameters == {} and list(store.list_keys("")) == []

    def test_a_seal_is_created_never_overwritten(self, store) -> None:
        """An overwrite would move the store's write time past the fire and void the seal."""
        _seal(store, FakeSsm())
        with pytest.raises(PointerConflictError):
            _seal(store, FakeSsm())

    def test_the_nonce_root_is_the_one_the_operated_stack_denies(self) -> None:
        """Restated in nous-ergon-ops `tests/test_crucible_v2_trader_drill_seal_denied.py`
        (`NONCE_PARAMETER_ROOT`), whose TraderRole Deny names this path."""
        assert NONCE_PARAMETER_ROOT == "/crucible-v2/trader/fire-drill-nonce/"


class TestOpen:
    def test_a_matching_opening_at_the_sealed_instant_reveals(self, store) -> None:
        _seal(store, FakeSsm())
        reveal = open_seal(store, _opening(), now=drill_schedule._instant(FIRE))
        assert reveal == dataclasses.asdict(_opening())

    def test_a_malformed_opening_is_refused(self, store) -> None:
        with pytest.raises(SealDoesNotOpenError, match="not a seal opening"):
            open_seal(store, dataclasses.replace(_opening(), nonce="short"), now=T0)

    def test_no_seal_is_refused(self, store) -> None:
        with pytest.raises(SealDoesNotOpenError, match="no sealed schedule"):
            open_seal(store, _opening(), now=drill_schedule._instant(FIRE))

    def test_a_wrong_nonce_is_refused(self, store) -> None:
        _seal(store, FakeSsm())
        with pytest.raises(SealDoesNotOpenError, match="do not open the commitment"):
            open_seal(
                store,
                dataclasses.replace(_opening(), nonce="c" * 64),
                now=drill_schedule._instant(FIRE),
            )

    @pytest.mark.parametrize("offset", [-1, FIRE_DRILL_SEAL_TOLERANCE_SECONDS + 1])
    def test_firing_off_the_sealed_instant_is_refused(self, store, offset) -> None:
        _seal(store, FakeSsm())
        now = drill_schedule._instant(FIRE) + dt.timedelta(seconds=offset)
        with pytest.raises(SealDoesNotOpenError, match="not the sealed fire"):
            open_seal(store, _opening(), now=now)
