"""The daily session on the executor box: a shadow session is a served day.

`alpha-engine-config-I11545`, Brian's rulings of 2026-09-24. The properties:

* a shadow session (routing off) records the day as `shadow`, files a
  `no_orders` shortfall, and places, cancels and closes nothing;
* a held day (nothing published yet) records nothing and does not fail;
* a halted day is not a served day;
* a routed session records `live` and files the router's own shortfall, even
  when it fails after orders reached the broker;
* what it writes is what the phase-4 gate reads.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from types import SimpleNamespace

import pytest
from crucible.execution import execution_shortfall_key
from crucible.keys import TRADER_EVIDENCE_KEY, manifest_key, predictions_key
from crucible.release import TRADER_PIN_KEY
from crucible.store import LocalStore
from ib_fakes import FakeIB, FakeSdk, ctx_for, environ_for
from test_contract import ARM, _feed, _put
from test_contract import _serveable as _contract_holds
from test_order_router import _book as order_book
from test_session import SHA, VERSION, _halt, _pin_document
from test_simulated_broker import write_panel

from crucible_trader import daily_session as module
from crucible_trader.broker_session import OrderRefusedError
from crucible_trader.contract import ContractRefusal, ContractUnavailable
from crucible_trader.daily_session import (
    SESSION_JOB,
    ReadOnlyHoldControl,
    daily_session_cycle,
    main,
    run_daily_session,
    run_staging_session,
)
from crucible_trader.execution_shortfall import build_no_orders_document
from crucible_trader.kill_switch import TradingHaltedError
from crucible_trader.order_router import (
    RouterConfig,
    ShadowRouter,
    StagingRoutingRefusedError,
    build_router,
)
from crucible_trader.simulated_broker import (
    SIMULATED_FILLS_SCHEMA_VERSION,
    SimulatedFillsRefusedError,
    StagingSessionConfig,
    simulated_fills_key,
)

DAY = dt.date(2026, 9, 11)
#: Friday 2026-09-11, 10:15 ET: the session runs pre-close and binds to the
#: last CLOSED session, Thursday 09-10 -- so the fixtures serve 09-10.
NOW = dt.datetime(2026, 9, 11, 14, 15, tzinfo=dt.UTC)
BOUND = "2026-09-10"


def _serveable(store: LocalStore, *, trading_day: str) -> None:
    """The contract holds, with a feed broad enough that the hold-book
    safeguard reads it healthy (`hold_book.HOLD_BOOK_MIN_BATCH`)."""
    _contract_holds(store, trading_day=trading_day)
    feed = _feed(trading_day=trading_day)
    feed["predicted_alpha"] = {f"T{i}": 0.001 * (i + 1) for i in range(10)}
    _put(store, predictions_key(trading_day), feed)


def _construct(store_, **kwargs):
    return SimpleNamespace(
        s_champion="s:arm:def", metric={"name": "portfolio_construction"}, alpha_bench=None
    )


@pytest.fixture
def store(tmp_path) -> LocalStore:
    s = LocalStore(tmp_path)
    s.put_bytes(TRADER_PIN_KEY, json.dumps(_pin_document(SHA)).encode())
    return s


class LiveRouter:
    """A routed session's router: records orders it 'placed', optionally fails."""

    enabled = True

    def __init__(self, *, records=("placed",), fail: BaseException | None = None) -> None:
        self.records = tuple(records)
        self.fail = fail
        self.closed = 0
        self._client = FakeIB()

    def batches(self, book):
        return [["order"]]

    def send(self, batch):
        if self.fail is not None:
            raise self.fail

    def shortfall_document(self, *, champion, trading_day, now):
        return build_no_orders_document(
            trading_day=trading_day, champion=champion, reason="fixture: routed", now=now
        )

    def close(self):
        self.closed += 1


def _cycle(store, router, *, day=DAY, resolver=None, construct=_construct):
    ib = FakeIB()
    ctx = ctx_for(store, SESSION_JOB, day=day)
    kwargs = {} if resolver is None else {"contract_resolver": resolver}
    result = daily_session_cycle(
        ctx,
        broker=ReadOnlyHoldControl(ib, FakeSdk(ib)),
        router=router,
        installed_version=VERSION,
        clock=ib.clock,
        construct=construct,
        **kwargs,
    )
    return ctx, result, ib


class TestAShadowSessionIsAServedDay:
    def test_it_records_the_day_as_shadow_and_files_no_orders(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        ctx, result, ib = _cycle(store, ShadowRouter())

        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        shortfall = json.loads(store.get_bytes(execution_shortfall_key(DAY.isoformat())))
        assert evidence["session_modes"] == {DAY.isoformat(): "shadow"}
        assert evidence["champion"] == ARM
        assert shortfall["outcome"] == "no_orders"
        assert result["session_mode"] == "shadow"
        assert result["outcome"] == "served"
        assert result["shortfall_outcome"] == "no_orders"
        assert (result["trading_days"], result["shadow_days"]) == (1, 1)
        assert result["batches_sent"] == 0

    def test_a_benched_session_carries_its_bench_into_the_evidence(self, store) -> None:
        """`alpha-engine-config-I12024`: the bench the book was built under
        (`TargetBook.alpha_bench`) reaches `record_session`, so the served day
        is counted under the bench's identity and the metric names it."""
        from crucible.keys import slot_bench_key
        from crucible.models import TraderEvidenceBench

        bench = TraderEvidenceBench(key=slot_bench_key("m"), digest="sha256:" + "a" * 64)

        def benched(store_, **kwargs):
            return SimpleNamespace(
                s_champion="s:arm:def",
                metric={"name": "portfolio_construction"},
                alpha_bench=bench,
            )

        _serveable(store, trading_day=DAY.isoformat())
        ctx, result, _ = _cycle(store, ShadowRouter(), construct=benched)

        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        assert evidence["alpha_bench"] == bench.model_dump()
        assert result["alpha_bench"] == bench.model_dump()
        assert f"benched at weight 0 ({bench.key})" in ctx.metrics[-1]["status_reason"]

    def test_a_forecast_session_files_no_bench(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        _, result, _ = _cycle(store, ShadowRouter())

        assert "alpha_bench" not in json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        assert result["alpha_bench"] is None

    def test_both_artifacts_are_outputs_of_the_run(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        ctx, _, _ = _cycle(store, ShadowRouter())

        assert [o["key"] for o in ctx.outputs] == [
            TRADER_EVIDENCE_KEY,
            execution_shortfall_key(DAY.isoformat()),
        ]
        assert ctx.metrics[-1]["name"] == "trader_session_served"
        assert ctx.metrics[-1]["value"] == 1.0

    def test_nothing_touches_the_book(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        _, _, ib = _cycle(store, ShadowRouter())

        assert ib.order_calls == [] and ib.cancel_calls == 0


class TestWhatIsNotAServedDay:
    def test_nothing_published_is_a_hold_that_records_nothing(self, store) -> None:
        def unavailable(store_, day):
            raise ContractUnavailable("no champion pointer for slot 'm'")

        ctx, result, _ = _cycle(store, ShadowRouter(), resolver=unavailable)

        assert result == {
            "trading_day": DAY.isoformat(),
            "session_mode": "shadow",
            "outcome": "held",
        }
        assert not store.exists(TRADER_EVIDENCE_KEY)
        assert not store.exists(execution_shortfall_key(DAY.isoformat()))
        assert ctx.metrics[-1]["value"] == 0.0
        assert "no champion pointer" in ctx.metrics[-1]["status_reason"]

    def test_a_refused_contract_fails_the_run_and_records_nothing(self, store) -> None:
        def refused(store_, day):
            raise ContractRefusal("the 'm' champion pointer exists and is unusable")

        with pytest.raises(ContractRefusal):
            _cycle(store, ShadowRouter(), resolver=refused)
        assert not store.exists(TRADER_EVIDENCE_KEY)

    @pytest.mark.parametrize("mode", ["freeze", "flatten"])
    def test_a_halted_day_is_not_served(self, store, mode) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        _halt(store, mode, day=DAY.isoformat())

        with pytest.raises(TradingHaltedError):
            _cycle(store, ShadowRouter())
        assert not store.exists(TRADER_EVIDENCE_KEY)


class TestARoutedSession:
    def test_it_records_the_day_as_live_with_the_routers_shortfall(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        _, result, _ = _cycle(store, LiveRouter())

        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        shortfall = json.loads(store.get_bytes(execution_shortfall_key(DAY.isoformat())))
        assert evidence["session_modes"] == {DAY.isoformat(): "live"}
        assert shortfall["outcome_reason"] == "fixture: routed"
        assert (result["session_mode"], result["shadow_days"]) == ("live", 0)

    def test_a_failure_after_orders_reached_the_broker_files_them_then_fails(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        with pytest.raises(RuntimeError, match="gateway dropped"):
            _cycle(store, LiveRouter(fail=RuntimeError("gateway dropped")))

        assert store.exists(execution_shortfall_key(DAY.isoformat()))
        assert not store.exists(TRADER_EVIDENCE_KEY)

    def test_a_failure_before_any_order_files_nothing(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        with pytest.raises(RuntimeError):
            _cycle(store, LiveRouter(records=(), fail=RuntimeError("refused")))

        assert not store.exists(execution_shortfall_key(DAY.isoformat()))


class TestTheHoldControlIsReadOnly:
    def test_it_refuses_to_cancel_or_close(self) -> None:
        ib = FakeIB()
        control = ReadOnlyHoldControl(ib, FakeSdk(ib))

        with pytest.raises(OrderRefusedError):
            control.cancel_all()
        with pytest.raises(OrderRefusedError):
            control.submit_close("AAPL", 10)
        assert ib.order_calls == [] and ib.cancel_calls == 0


class TestTheJob:
    def test_a_shadow_run_files_an_ok_manifest_and_disconnects(self, store) -> None:
        _serveable(store, trading_day=BOUND)
        ib = FakeIB()
        ctx = run_daily_session(
            store,
            connector=lambda: ib,
            sdk=FakeSdk(ib),
            router_config=RouterConfig(enabled=False),
            installed_version=VERSION,
            clock=lambda: NOW,
            run_mode="live",
            router_factory=lambda config, **kw: build_router(config, **kw),
        )

        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, BOUND)))
        assert manifest["status"] == "ok", manifest.get("reason")
        assert ctx.trading_day.isoformat() == BOUND
        assert ib.disconnected == 1
        assert ib.order_calls == []

    def test_a_routed_run_reads_through_the_routers_connection_and_closes_it(
        self, store, monkeypatch
    ) -> None:
        _serveable(store, trading_day=BOUND)
        monkeypatch.setattr(module, "construct_target_book", _construct)
        router = LiveRouter()
        opened: list = []

        run_daily_session(
            store,
            connector=lambda: opened.append("second connection"),
            sdk=FakeSdk(router._client),
            router_config=RouterConfig(enabled=False),
            installed_version=VERSION,
            clock=lambda: NOW,
            run_mode="live",
            router_factory=lambda config, **kw: router,
        )

        assert opened == []
        assert router.closed == 1
        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        assert evidence["session_modes"] == {BOUND: "live"}


class TestTheEntryPoint:
    def test_a_non_trading_day_is_skipped_before_anything_opens(self, tmp_path) -> None:
        said: list[str] = []
        saturday = dt.datetime(2026, 9, 12, 14, 15, tzinfo=dt.UTC)

        assert main([], environ={}, clock=lambda: saturday, printer=said.append) == 0
        assert said and said[0].startswith("SKIP: 2026-09-12")

    def test_routing_stays_off_unless_the_environment_turns_it_on(self, store) -> None:
        """Nothing is published yet, so the session holds -- and it held as a
        SHADOW session: the router the entry point built was the shadow one."""
        ib = FakeIB()
        said: list[str] = []
        seen: list[RouterConfig] = []

        def factory(config, **kw):
            seen.append(config)
            return build_router(config, **kw)

        code = main(
            [],
            environ=environ_for(store.root),
            sdk_loader=FakeSdk(ib),
            version_reader=lambda: VERSION,
            clock=lambda: NOW,
            printer=said.append,
            router_factory=factory,
        )

        assert code == 0
        assert seen == [RouterConfig(enabled=False)]
        assert ib.connected_with["readonly"] is True
        assert said[-1].endswith("shadow session filed")
        assert not store.exists(TRADER_EVIDENCE_KEY)

    def test_a_routed_entry_point_says_so(self, store, monkeypatch) -> None:
        ib = FakeIB()
        said: list[str] = []
        env = environ_for(
            store.root,
            CRUCIBLE_TRADER_ORDER_ROUTING="ib_paper",
            CRUCIBLE_TRADER_IB_ACCOUNT="DU1",
            CRUCIBLE_TRADER_IB_PORT="4002",
            CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD="1000",
            CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD="5000",
        )

        main(
            [],
            environ=env,
            sdk_loader=FakeSdk(ib),
            version_reader=lambda: VERSION,
            clock=lambda: NOW,
            printer=said.append,
            router_factory=lambda config, **kw: LiveRouter(),
        )

        assert said[-1].endswith("routed session filed")


# -- staging (alpha-engine-config-I12507, S-19) --------------------------------------


def _no_sdk():
    pytest.fail("a staging session loaded the broker SDK")


def _staging_env(store, **extra):
    return environ_for(
        store.root,
        NE_ENV="staging",
        CRUCIBLE_TRADER_SIMULATED_FILLS="on",
        CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD="50000",
        CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD="100000",
        **extra,
    )


def _simulated_world(store, monkeypatch) -> None:
    _serveable(store, trading_day=BOUND)
    write_panel(store, day=BOUND)
    book = dataclasses.replace(
        order_book(day=BOUND),
        metric={
            "name": "portfolio_construction",
            "module": "fixture",
            "metric_type": "operational",
            "value": 1.0,
            "unit": "flag",
            "n_floor": 1,
            "status": "OK",
            "status_reason": "fixture book",
            "source_path": "fixture",
            "last_updated_utc": "2026-09-11T14:15:00Z",
        },
    )
    monkeypatch.setattr(module, "construct_target_book", lambda store_, **kw: book)


class TestAStagingSession:
    def test_it_writes_a_complete_run_and_fills_by_simulation_with_no_broker(
        self, store, monkeypatch
    ) -> None:
        _simulated_world(store, monkeypatch)
        said: list[str] = []

        code = main(
            [],
            environ=_staging_env(store),
            sdk_loader=_no_sdk,
            version_reader=lambda: VERSION,
            clock=lambda: NOW,
            printer=said.append,
            router_factory=lambda config, **kw: pytest.fail("the prod router was built"),
        )

        assert code == 0
        assert said[-1].endswith("staging simulated session filed (no broker)")
        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, BOUND)))
        assert manifest["status"] == "ok", manifest.get("reason")
        outputs = {o["key"]: o["schema_version"] for o in manifest["outputs"]}
        assert outputs[simulated_fills_key(BOUND)] == SIMULATED_FILLS_SCHEMA_VERSION
        assert execution_shortfall_key(BOUND) in outputs
        assert TRADER_EVIDENCE_KEY in outputs

        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        assert evidence["session_modes"] == {BOUND: "shadow"}
        shortfall = json.loads(store.get_bytes(execution_shortfall_key(BOUND)))
        assert shortfall["outcome"] == "no_orders"
        assert "filled by simulation" in shortfall["outcome_reason"]
        fills = json.loads(store.get_bytes(simulated_fills_key(BOUND)))
        assert [(o["symbol"], o["filled_quantity"], o["fill_price"]) for o in fills["orders"]] == [
            ("AAA", 300.0, 100.0),
            ("BBB", 400.0, 50.0),
        ]
        assert fills["fill_source"] == "simulated_close"
        (simulated,) = [
            m for m in manifest["metrics"] if m["name"] == "trader_session_simulated_fills"
        ]
        assert simulated["value"] == 2.0
        assert "no broker" in simulated["status_reason"]

    def test_the_session_document_says_simulated(self, store, monkeypatch) -> None:
        _simulated_world(store, monkeypatch)
        captured: list[dict] = []
        real = module.daily_session_cycle

        def spy(ctx, **kw):
            result = real(ctx, **kw)
            captured.append(result)
            return result

        monkeypatch.setattr(module, "daily_session_cycle", spy)
        config = StagingSessionConfig.from_env(_staging_env(store))
        assert config is not None
        run_staging_session(
            store, config=config, installed_version=VERSION, clock=lambda: NOW, run_mode="live"
        )

        assert captured[0]["session_mode"] == "shadow"
        # An all-cash simulated book has no sells: one batch, the buys.
        assert captured[0]["batches_sent"] == 1

    @pytest.mark.parametrize("value", ["ib_paper", "off"])
    def test_staging_can_never_route(self, store, value) -> None:
        with pytest.raises(StagingRoutingRefusedError):
            main(
                [],
                environ=_staging_env(store, CRUCIBLE_TRADER_ORDER_ROUTING=value),
                sdk_loader=_no_sdk,
                version_reader=lambda: VERSION,
                clock=lambda: NOW,
                printer=lambda line: None,
            )
        assert not store.exists(manifest_key(SESSION_JOB, BOUND))

    def test_prod_never_simulates(self, store) -> None:
        env = _staging_env(store)
        del env["NE_ENV"]
        with pytest.raises(SimulatedFillsRefusedError):
            main(
                [],
                environ=env,
                sdk_loader=_no_sdk,
                version_reader=lambda: VERSION,
                clock=lambda: NOW,
                printer=lambda line: None,
            )
        assert not store.exists(manifest_key(SESSION_JOB, BOUND))

    def test_a_staging_session_without_simulated_fills_is_shadow_with_no_broker(
        self, store, monkeypatch
    ) -> None:
        _serveable(store, trading_day=BOUND)
        monkeypatch.setattr(module, "construct_target_book", _construct)
        env = _staging_env(store)
        del env["CRUCIBLE_TRADER_SIMULATED_FILLS"]
        said: list[str] = []

        main(
            [],
            environ=env,
            sdk_loader=_no_sdk,
            version_reader=lambda: VERSION,
            clock=lambda: NOW,
            printer=said.append,
        )

        assert said[-1].endswith("staging shadow session filed (no broker)")
        assert json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))["session_modes"] == {
            BOUND: "shadow"
        }
        assert not store.exists(simulated_fills_key(BOUND))

    def test_a_held_staging_day_files_no_fills(self, store) -> None:
        config = StagingSessionConfig.from_env(_staging_env(store))
        assert config is not None
        ctx = run_staging_session(
            store, config=config, installed_version=VERSION, clock=lambda: NOW, run_mode="live"
        )

        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, BOUND)))
        assert manifest["status"] == "ok", manifest.get("reason")
        assert ctx.metrics[-1]["value"] == 0.0
        assert not store.exists(simulated_fills_key(BOUND))
        assert not store.exists(TRADER_EVIDENCE_KEY)


class TestTheProdPathIsUnchanged:
    """With the flag off, the prod entry point writes byte-identical artifacts."""

    @staticmethod
    def _run(tmp_path, name, extra, monkeypatch):
        s = LocalStore(tmp_path / name)
        s.put_bytes(TRADER_PIN_KEY, json.dumps(_pin_document(SHA)).encode())
        _serveable(s, trading_day=BOUND)
        monkeypatch.setattr(module, "construct_target_book", _construct)
        ib = FakeIB()
        seen: list[RouterConfig] = []
        said: list[str] = []

        def factory(config, **kw):
            seen.append(config)
            return build_router(config, **kw)

        main(
            [],
            environ=environ_for(s.root, **extra),
            sdk_loader=FakeSdk(ib),
            version_reader=lambda: VERSION,
            clock=lambda: NOW,
            printer=said.append,
            router_factory=factory,
        )
        artifacts = {
            key: s.get_bytes(key) for key in (TRADER_EVIDENCE_KEY, execution_shortfall_key(BOUND))
        }
        return seen, ib.connected_with, ib.order_calls, said[-1].split(": ")[-1], artifacts

    def test_unset_off_and_prod_are_the_same_session(self, tmp_path, monkeypatch) -> None:
        runs = [
            self._run(tmp_path, name, extra, monkeypatch)
            for name, extra in (
                ("unset", {}),
                ("off", {"CRUCIBLE_TRADER_SIMULATED_FILLS": "off"}),
                ("prod", {"NE_ENV": "prod"}),
            )
        ]
        baseline = runs[0]
        assert baseline[0] == [RouterConfig(enabled=False)]
        assert baseline[1]["readonly"] is True
        assert baseline[2] == []
        assert baseline[3] == "shadow session filed"
        for run in runs[1:]:
            assert run == baseline
