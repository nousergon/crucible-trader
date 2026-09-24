"""A store the S slot can construct on — shared by the construction and shadow-input tests.

Everything is written where production writes it and read back through the
harness's own readers: the price panel at `data_panel_key`, the M champion's
`arm_predictions`, the U champion's cut, the S recipe and parameter set in the
synced strategy tree, the attested champion pointers with `ok` producing
manifests, and the `session_inputs.v1` document produced by the harness's own
`resolve_session` — the exact write `experiment.run --slot s` makes. A fixture
that injected arrays directly would test a function the trader does not call.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import random
from collections.abc import Callable

import pandas as pd
import pytest
from crucible.calendar import is_trading_day
from crucible.champion import CHAMPION_SCHEMA_VERSION
from crucible.features import DEFAULT_FEATURE_VERSION
from crucible.gate import Clause, evaluate
from crucible.keys import (
    arm_predictions_key,
    champion_key,
    data_panel_key,
    features_key,
    session_inputs_key,
    shadow_key,
    strategy_arm_key,
    strategy_slot_key,
)
from crucible.slots.arms import register_arms, write_register
from crucible.slots.strategy import load_strategy_slot, parse_strategy_document, resolve_session
from crucible.store import LocalStore
from nousergon_lib.arena.arms import ArmRegister

AS_OF = dt.date(2026, 9, 11)
D_PREV = dt.date(2026, 9, 10)
NEXT = dt.date(2026, 9, 14)
M_CHAMPION = "m:fixture_model:aaaaaaaaaaaa"
U_CHAMPION = "u:fixture_cut:bbbbbbbbbbbb"
TICKERS = [f"T{i:02d}" for i in range(10)]

FLAT_COST = """\
  cost_model:
    name: flat_bps_v0
    placeholder: true
    params:
      half_spread_bps: 2.5
      commission_bps: 0.5
      slippage_bps: 10.0
"""
IMPACT_COST = """\
  cost_model:
    name: sqrt_impact_v1
    placeholder: false
    params:
      half_spread_bps: 2.5
      commission_bps: 0.5
      impact_coef_bps: 12.0
      min_cost_bps: 1.0
"""
PARAMS_YAML = """\
portfolio:
  risk_aversion: 5.0
  cash_sleeve_pct: 0.03
  max_sector_pct: 0.25
  min_position_pct: 0.005
  covariance_shrinkage: ledoit_wolf
  sigma_horizon_days: 1
  ewma_lambda_decay: 0.94
  vol_target_annual: null
  alpha_uncertainty_penalty: 0.0
  alpha_uncertainty_min_cv: 0.01
  max_pct_adv: null
  max_daily_turnover: 0.15
  large_move_turnover_flag: 0.35
  conviction_budget_gate_enabled: true
  conviction_ir_floor: 0.35
  conviction_ir_full: 0.75
  conviction_budget_min_multiple: 0.05
  conviction_gate_min_names: 3
  book_notional_usd: 1000000.0
"""


def recipe_yaml(name: str, *, cost: str = FLAT_COST, registered_at: str = "2026-09-01") -> str:
    return (
        f"slot: s\nname: {name}\nregistered_at: '{registered_at}'\n"
        f"notes: fixture recipe for {name}\nspec:\n  benchmark: SPY\n{cost}"
        "  rules:\n    - rule_id: profit_take\n      params:\n        profit_take_pct: 0.25\n"
    )


def sessions_ending(end: dt.date, count: int) -> list[dt.date]:
    out: list[dt.date] = []
    day = end
    while len(out) < count:
        if is_trading_day(day):
            out.append(day)
        day -= dt.timedelta(days=1)
    return sorted(out)


def remove(store: LocalStore, key: str) -> None:
    """Take a key out of the store — the state before its producer ran."""
    store._path(key).unlink()


def put(store: LocalStore, key: str, document: dict) -> None:
    store.put_bytes(key, json.dumps(document, indent=2, sort_keys=True).encode("utf-8"))


def pointer(slot: str, arm_id: str, **overrides: object) -> dict:
    document: dict = {
        "schema_version": CHAMPION_SCHEMA_VERSION,
        "slot": slot,
        "arm_id": arm_id,
        "as_of": AS_OF.isoformat(),
        "decided_at": "2026-09-12T02:00:00Z",
        "run_id": "01JG0000000000000000000000",
        "code_sha": "a" * 40,
        "promotion_source": "evidence",
        "manifest_key": f"runs/experiment.grade/{AS_OF.isoformat()}/{slot}/run.json",
        "evidence": {"status": "decided", "moved": True, "paired_dates": 40},
    }
    if slot == "s":
        document["attestation"] = {"kind": "pit_parity", "status": "PASS"}
    document.update(overrides)
    return document


def seat_champion(store: LocalStore, slot: str, arm_id: str, **overrides: object) -> None:
    document = pointer(slot, arm_id, **overrides)
    put(store, champion_key(slot), document)
    put(store, document["manifest_key"], {"status": "ok", "job": "experiment.grade", "reason": ""})


def _panel(days: list[dt.date]) -> pd.DataFrame:
    rows = []
    for seed, ticker in enumerate([*TICKERS, "SPY"]):
        rng = random.Random(1000 + seed)
        price = 50.0 + 10 * seed
        drift = rng.gauss(0.0004, 0.0008)
        for day in days:
            price *= math.exp(rng.gauss(drift, 0.012))
            rows.append(
                {
                    "trading_day": day,
                    "ticker": ticker,
                    "open_raw": price,
                    "high_raw": price * 1.01,
                    "low_raw": price * 0.99,
                    "close_raw": price,
                    "volume_raw": 1e6,
                }
            )
    return pd.DataFrame(rows)


@dataclasses.dataclass
class World:
    store: LocalStore
    arm_id: str
    recipe_name: str

    def record_session(self, arm_id: str, day: dt.date = AS_OF) -> str:
        session = resolve_session(self.store, trading_day=day.isoformat())
        key = session_inputs_key(arm_id, day.isoformat())
        put(self.store, key, session.to_dict())
        return key

    def file_recipe(self, name: str, *, cost: str = FLAT_COST, **kwargs: str) -> str:
        payload = recipe_yaml(name, cost=cost, **kwargs).encode("utf-8")
        self.store.put_bytes(strategy_arm_key("s", name), payload)
        return parse_strategy_document(payload, origin=name).arm_id

    def register(self, arm_ids: list[str]) -> None:
        """Register each filed recipe through the harness's own `register_arms`,
        so the register carries the recipe's own id exactly as `experiment.run`
        writes it."""
        loaded = load_strategy_slot(store=self.store)
        by_id = {arm.arm_id: arm for arm in loaded.registered}
        register, _ = register_arms(
            ArmRegister(), [by_id[a] for a in arm_ids], filed_on=AS_OF.isoformat()
        )
        write_register(self.store, "s", register)

    def drop_prediction(self, day: dt.date, ticker: str) -> None:
        """The M champion stops pricing ``ticker`` on ``day`` (I10754's fixture)."""
        key = arm_predictions_key(M_CHAMPION, day.isoformat())
        document = json.loads(self.store.get_bytes(key))
        del document["predicted_alpha"][ticker]
        put(self.store, key, document)

    def write_adv(self, day: dt.date = AS_OF) -> None:
        frame = pd.DataFrame(
            {
                "trading_day": [day.isoformat()] * len(TICKERS),
                "ticker": TICKERS,
                "dollar_volume_20d_raw": [5e8] * len(TICKERS),
            }
        )
        self.store.put_bytes(
            features_key(DEFAULT_FEATURE_VERSION, day.isoformat()), frame.to_parquet(index=False)
        )


@pytest.fixture
def world(tmp_path) -> World:
    store = LocalStore(tmp_path / "store")
    days = sessions_ending(AS_OF, 80)
    assert days[-2] == D_PREV
    store.put_bytes(data_panel_key(D_PREV.isoformat()), _panel(days[:-1]).to_parquet(index=False))
    store.put_bytes(data_panel_key(AS_OF.isoformat()), _panel(days).to_parquet(index=False))
    store.put_bytes(data_panel_key(NEXT.isoformat()), _panel([*days, NEXT]).to_parquet(index=False))
    rng = random.Random(7)
    for day in days[-3:]:
        put(
            store,
            arm_predictions_key(M_CHAMPION, day.isoformat()),
            {
                "schema_version": "arm_predictions.v1",
                "arm_id": M_CHAMPION,
                "trading_day": day.isoformat(),
                "feature_version": "vfixture",
                "predicted_alpha": {t: rng.gauss(0.0005, 0.004) for t in TICKERS},
            },
        )
        put(
            store,
            shadow_key(U_CHAMPION, day.isoformat()),
            {
                "schema_version": "shadow.v1",
                "arm_id": U_CHAMPION,
                "trading_day": day.isoformat(),
                "selection": TICKERS[:8],
                "population": TICKERS,
                "ranker": "momentum_sleeve",
                "params": {},
                "feature_version": "vfixture",
            },
        )
    store.put_bytes(strategy_slot_key("s"), PARAMS_YAML.encode("utf-8"))
    w = World(store=store, arm_id="", recipe_name="stock_registry")
    w.arm_id = w.file_recipe("stock_registry")
    seat_champion(store, "m", M_CHAMPION)
    seat_champion(store, "u", U_CHAMPION)
    seat_champion(store, "s", w.arm_id)
    return w


@pytest.fixture
def phase4_clause(monkeypatch: pytest.MonkeyPatch) -> Callable[[LocalStore, str, str], Clause]:
    """Read one phase-4 clause the way the published gate reads it.

    The producer/consumer contract end to end: a test writes an artifact through
    this repository's own writer, and this reads it back through
    `crucible.gate.evaluate(gate="phase4")` -- the public entry point
    `crucible gate --gate phase4` calls -- as shipped in the pinned wheel. A
    rename of a key, a job, a schema field or a clause on EITHER side fails
    here, rather than as a phase-4 reading that stays UNMET for a reason no one
    connects to this repository (`alpha-engine-config-I9760`).

    The clause is picked out by name, so a clause the gate stops reading is a
    `StopIteration` here, not a silent pass. The AWS credential chain is emptied
    first: the gate's Cost Explorer clause must answer UNMEASURABLE offline
    rather than reach whatever account the test runner can see.
    """
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/credentials")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/config")

    def read(store: LocalStore, name: str, trading_day: str) -> Clause:
        result = evaluate(store, gate="phase4", trading_day=dt.date.fromisoformat(trading_day))
        return next(clause for clause in result.clauses if clause.name == name)

    return read
