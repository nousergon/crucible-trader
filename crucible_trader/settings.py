"""Where the trader is pointed, resolved once and fail-loud.

Every value here names a piece of live infrastructure, so none of them is a
literal in this tree (`repository-tiering-policy` test 2; this repository is
public).
They arrive from the environment, and a missing or malformed one raises at
resolution time rather than surfacing as a confusing failure three calls later.

There is no default store, deliberately. A default would mean a
misconfigured trader reads SOMEWHERE — and the one place a trader must never
read by accident is a store nobody pointed it at.
"""

from __future__ import annotations

import dataclasses
import os

#: The store the contract is read from. An `s3://bucket/prefix` URI in
#: production; a `file://` path in a test or a laptop dry run.
STORE_URI_VAR = "CRUCIBLE_TRADER_STORE_URI"

#: Which slot's champion this trader serves. `m` is the predicted-alpha
#: cross-section the `predictions/{trading_day}.json` half of the contract is
#: keyed to (`crucible.serving.PREDICTIONS_FEED_SLOT`); the other slots serve
#: their own keys and are not this contract.
SLOT_VAR = "CRUCIBLE_TRADER_SLOT"

#: The only slot this trader serves today. Closed rather than free-form: a slot
#: string the contract has no feed for would resolve a champion and then fail to
#: find a feed, which reads as a missing artifact rather than as a
#: misconfiguration.
SERVED_SLOT = "m"

#: Accepted store schemes. `file://` is admitted so the contract can be
#: exercised against a `LocalStore` in tests and in a laptop dry run without a
#: second code path — the same reader, a different backend.
_ACCEPTED_SCHEMES = ("s3://", "file://")


class SettingsError(RuntimeError):
    """The trader is not pointed at anything it can safely read.

    A distinct type, not `ValueError`: the caller's correct response is to page
    and exit, never to retry or to substitute a default.
    """


@dataclasses.dataclass(frozen=True)
class Settings:
    """Resolved configuration for one trader session.

    Frozen: a session's store and slot are decided once, before the first read.
    A trader that could repoint itself mid-session could serve two different
    champions in one book and report one of them.
    """

    store_uri: str
    slot: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        """Resolve from `env` (default: the process environment).

        Raises `SettingsError` on anything it cannot resolve. It never returns a
        partially-populated object and never falls back to a default store.
        """
        source = os.environ if env is None else env

        store_uri = (source.get(STORE_URI_VAR) or "").strip()
        if not store_uri:
            raise SettingsError(
                f"{STORE_URI_VAR} is unset or empty, so the trader is pointed at no store. "
                "There is deliberately no default: a default store means a misconfigured "
                "trader reads somewhere, and reading the wrong store is how a stale champion "
                "gets traded."
            )
        if not store_uri.startswith(_ACCEPTED_SCHEMES):
            raise SettingsError(
                f"{STORE_URI_VAR}={store_uri!r} is not a store URI. Accepted schemes: "
                f"{', '.join(_ACCEPTED_SCHEMES)}. A bare path is refused rather than "
                "guessed at, because the guess that reads a local directory as the "
                "production store is silent."
            )

        slot = (source.get(SLOT_VAR) or SERVED_SLOT).strip().lower()
        if slot != SERVED_SLOT:
            raise SettingsError(
                f"{SLOT_VAR}={slot!r} — this trader serves slot {SERVED_SLOT!r} only. "
                "`predictions/{trading_day}.json` is the M contract specifically; the R and "
                "U champions serve their own keys and are not this contract. A trader "
                "pointed at another slot would resolve a champion and then find no feed, "
                "which reads as a missing artifact rather than as a misconfiguration."
            )

        return cls(store_uri=store_uri, slot=slot)
