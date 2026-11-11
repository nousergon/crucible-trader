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
from test_order_router import _config as routed_config
from test_session import SHA, VERSION, _halt, _pin_document
from test_simulated_broker import write_panel

from crucible_trader import daily_session as module
from crucible_trader import simulated_broker as simulated_broker_module
from crucible_trader.broker_session import OrderRefusedError
from crucible_trader.construction import SessionSizing
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
    simulated_positions_key,
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


class WatchingRouter(LiveRouter):
    """A routed router that held buys for their triggers (`entry_timing`)."""

    def entry_watch_payload(self, now):
        return "trader/entry_watch/2026-09-11.json", b'{"state": "complete"}'


class TestTheEntryWatchIsOnTheRecord:
    def test_a_served_session_records_the_final_watch(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        ctx, _, _ = _cycle(store, WatchingRouter())
        assert "trader/entry_watch/2026-09-11.json" in {o["key"] for o in ctx.outputs}

    def test_a_watch_that_cannot_be_recorded_never_fails_the_session(self, store, capsys) -> None:
        ctx = SimpleNamespace(record_output=lambda *a: (_ for _ in ()).throw(PermissionError("no")))
        module._record_entry_watch(ctx, WatchingRouter(), clock=lambda: NOW)
        assert "not recorded (PermissionError" in capsys.readouterr().err

    def test_a_router_that_held_nothing_records_nothing(self, store) -> None:
        ctx = ctx_for(store, SESSION_JOB, day=DAY)
        module._record_entry_watch(ctx, LiveRouter(), clock=lambda: NOW)
        router = WatchingRouter()
        router.entry_watch_payload = lambda now: None
        module._record_entry_watch(ctx, router, clock=lambda: NOW)
        assert ctx.outputs == []

    def test_a_failure_after_orders_records_the_watch_too(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        router = WatchingRouter(fail=RuntimeError("gateway dropped"))
        with pytest.raises(RuntimeError, match="gateway dropped"):
            _cycle(store, router)
        assert store.exists("trader/entry_watch/2026-09-11.json")


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
    monkeypatch.setattr(
        module,
        "load_portfolio_params_from_store",
        lambda slot, *, store: SimpleNamespace(book_notional_usd=1_000_000.0),
    )


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
        assert evidence["session_modes"] == {BOUND: "simulated"}
        shortfall = json.loads(store.get_bytes(execution_shortfall_key(BOUND)))
        assert shortfall["outcome"] == "computed"
        assert shortfall["fill_source"] == "simulated_close"
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
        env = _staging_env(store)
        config = StagingSessionConfig.from_env(env)
        assert config is not None
        run_staging_session(
            store,
            config=config,
            installed_version=VERSION,
            clock=lambda: NOW,
            run_mode="live",
            environ=env,
        )

        assert captured[0]["session_mode"] == "simulated"
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

    def test_a_held_staging_day_files_no_fills(self, store, monkeypatch) -> None:
        monkeypatch.setattr(
            module,
            "load_portfolio_params_from_store",
            lambda slot, *, store: SimpleNamespace(book_notional_usd=1_000_000.0),
        )
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
        assert not store.exists(simulated_positions_key(BOUND))


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

    def test_prod_files_no_simulated_mode_and_no_fill_source(self, tmp_path, monkeypatch) -> None:
        """`alpha-engine-config-I12685`: the staging mode and the shortfall's
        `fill_source` never reach a prod artifact, and the prod session never
        reads the fleet registry."""
        monkeypatch.setattr(
            simulated_broker_module,
            "assert_registry_forbids_routing",
            lambda: pytest.fail("the prod path read the registry"),
        )
        for name, extra in (("unset", {}), ("prod", {"NE_ENV": "prod"})):
            *_, artifacts = self._run(tmp_path, name, extra, monkeypatch)
            evidence = json.loads(artifacts[TRADER_EVIDENCE_KEY])
            shortfall = json.loads(artifacts[execution_shortfall_key(BOUND)])
            assert evidence["session_modes"] == {BOUND: "shadow"}
            assert "fill_source" not in shortfall
            assert b"simulated" not in artifacts[TRADER_EVIDENCE_KEY]


class HoldingRouter(LiveRouter):
    """A live router whose account holds a book (2026-11-05: rebuilt from cash daily)."""

    def __init__(self, held: dict[str, float]) -> None:
        super().__init__()
        self.held = held
        self.notionals: list[float] = []

    def held_weights(self, notional: float) -> dict[str, float]:
        self.notionals.append(notional)
        return self.held


class TestTheBookHeldIntoTheSession:
    def _capture(self):
        seen: list[object] = []

        def construct(store_, **kwargs):
            seen.append(kwargs["previous_weights"])
            return _construct(store_, **kwargs)

        return seen, construct

    def test_a_live_session_constructs_from_the_accounts_positions(
        self, store, monkeypatch
    ) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        monkeypatch.setattr(
            module,
            "load_portfolio_params_from_store",
            lambda slot, *, store: SimpleNamespace(book_notional_usd=1_000_000.0),
        )
        seen, construct = self._capture()
        router = HoldingRouter({"PLTR": 0.25, "__CASH__": 0.75})
        _cycle(store, router, construct=construct)
        assert seen == [{"PLTR": 0.25, "__CASH__": 0.75}]
        assert router.notionals == [1_000_000.0]

    def test_a_shadow_session_still_starts_from_cash(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        seen, construct = self._capture()
        _cycle(store, ShadowRouter(), construct=construct)
        assert seen == [None]

    def test_an_unusable_parameter_set_refuses_the_session(self, store, monkeypatch) -> None:
        from crucible.portfolio import PortfolioParamsError

        _serveable(store, trading_day=DAY.isoformat())

        def broken(slot, *, store):
            raise PortfolioParamsError("no params")

        monkeypatch.setattr(module, "load_portfolio_params_from_store", broken)
        with pytest.raises(ContractRefusal, match="parameter set is unusable"):
            _cycle(store, HoldingRouter({}))


class SizingRouter(HoldingRouter):
    """A live router that sizes on its NAV and protects what it holds."""

    def __init__(self, held, *, nav: float, protect_fail: BaseException | None = None) -> None:
        super().__init__(held)
        self.nav = nav
        self.protect_fail = protect_fail
        self.protected = 0

    def session_sizing(self, ceiling: float) -> SessionSizing:
        return SessionSizing(ceiling_usd=ceiling, nav_usd=self.nav)

    def protect_holdings(self):
        self.protected += 1
        if self.protect_fail is not None:
            raise self.protect_fail
        return ({"symbol": "PLTR", "disposition": "placed"},)


class TestALiveSessionSizesOnNavAndProtectsItsHoldings:
    """2026-11-09 parity audit rows 5 and 1."""

    @pytest.fixture(autouse=True)
    def _params(self, monkeypatch) -> None:
        monkeypatch.setattr(
            module,
            "load_portfolio_params_from_store",
            lambda slot, *, store: SimpleNamespace(book_notional_usd=1_000_000.0),
        )

    def test_the_book_and_the_held_weights_are_sized_on_the_nav(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        sizings: list[object] = []

        def construct(store_, **kwargs):
            sizings.append(kwargs.get("sizing"))
            return _construct(store_, **kwargs)

        router = SizingRouter({"PLTR": 0.25, "__CASH__": 0.75}, nav=850_000.0)
        _, result, _ = _cycle(store, router, construct=construct)
        assert router.notionals == [850_000.0]
        assert sizings == [SessionSizing(ceiling_usd=1_000_000.0, nav_usd=850_000.0)]
        assert result["sizing"] == {
            "basis": "broker_net_liquidation",
            "nav_usd": 850_000.0,
            "ceiling_usd": 1_000_000.0,
            "notional_usd": 850_000.0,
        }
        assert router.protected == 1
        assert result["protective_stops"] == [{"symbol": "PLTR", "disposition": "placed"}]

    def test_a_router_without_nav_or_stops_keeps_the_declared_notional(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        _, result, _ = _cycle(store, LiveRouter())
        assert result["sizing"] is None and result["protective_stops"] is None
        assert result["protective_stops_in_session"] == []

    def test_the_stops_the_all_day_session_placed_are_on_the_record(self, store) -> None:
        _serveable(store, trading_day=DAY.isoformat())
        router = SizingRouter({"PLTR": 0.25, "__CASH__": 0.75}, nav=850_000.0)
        event = {"moment": "after the timed buy of PLTR", "symbol": "PLTR"}
        router.protective_stop_events = (event,)
        _, result, _ = _cycle(store, router)
        assert result["protective_stops_in_session"] == [event]

    def test_an_unreadable_nav_fails_the_session_before_anything_is_built(self, store) -> None:
        from crucible_trader.order_router import NavUnavailableError

        _serveable(store, trading_day=DAY.isoformat())

        class Blind(SizingRouter):
            def session_sizing(self, ceiling: float) -> SessionSizing:
                raise NavUnavailableError("no NetLiquidation")

        router = Blind({}, nav=1.0)
        with pytest.raises(NavUnavailableError):
            _cycle(store, router)
        assert router.notionals == [] and router.protected == 0
        assert not store.exists(TRADER_EVIDENCE_KEY)

    def test_a_name_left_unprotected_fails_the_run_after_filing_the_orders(self, store) -> None:
        from crucible_trader.order_router import ProtectiveStopError

        _serveable(store, trading_day=DAY.isoformat())
        router = SizingRouter({}, nav=1e6, protect_fail=ProtectiveStopError("AAA unprotected"))
        with pytest.raises(ProtectiveStopError):
            _cycle(store, router)
        assert store.exists(execution_shortfall_key(DAY.isoformat()))
        assert not store.exists(TRADER_EVIDENCE_KEY)


class TestTheSessionRecordsItsTurnover:
    """alpha-engine-config-I13305: the run record carries the book's turnover
    against its budget, through the harness's own `record_turnover`."""

    class _Ctx:
        def __init__(self) -> None:
            self.calls: list[dict[str, float]] = []

        def record_turnover(self, *, value: float, budget: float) -> None:
            self.calls.append({"value": value, "budget": budget})

    def test_the_constructed_turnover_and_cap_are_recorded(self) -> None:
        ctx = self._Ctx()
        module._record_turnover(ctx, {"turnover_one_way": 0.18, "turnover_budget": 0.2})
        assert ctx.calls == [{"value": 0.18, "budget": 0.2}]

    def test_a_disabled_budget_is_recorded_as_the_whole_book(self) -> None:
        ctx = self._Ctx()
        module._record_turnover(ctx, {"turnover_one_way": 0.4, "turnover_budget": None})
        assert ctx.calls == [{"value": 0.4, "budget": module.UNBOUNDED_TURNOVER_BUDGET}]

    def test_a_harness_without_the_method_files_nothing_and_does_not_fail(self) -> None:
        module._record_turnover(SimpleNamespace(), {"turnover_one_way": 0.4})

    def test_a_book_with_no_turnover_records_nothing(self) -> None:
        ctx = self._Ctx()
        module._record_turnover(ctx, {"turnover_one_way": None})
        assert ctx.calls == []

    def test_a_simulated_session_files_its_closing_book(self, store, monkeypatch) -> None:
        _simulated_world(store, monkeypatch)
        env = _staging_env(store)
        config = StagingSessionConfig.from_env(env)
        assert config is not None
        run_staging_session(
            store,
            config=config,
            installed_version=VERSION,
            clock=lambda: NOW,
            run_mode="live",
            environ=env,
        )
        document = json.loads(store.get_bytes(simulated_positions_key(BOUND)))
        assert document["carried_from"] is None
        assert document["positions"] == {"AAA": 300, "BBB": 400}


# -- a routed session starts only in market hours (alpha-engine-config-I13447) --

#: 2026-11-10 01:36:16Z is Monday 2026-11-09 20:36:16 ET (EST, UTC-5). A routed
#: session started then on the box. It binds to Monday's session, the last one
#: closed, and its feed was published nine minutes later.
INCIDENT = dt.datetime(2026, 11, 10, 1, 36, 16, tzinfo=dt.UTC)
INCIDENT_BOUND = "2026-11-09"


def _routed_job(store, *, clock, router=None, connector=None):
    """Run a ROUTED session job; return the context and how far it got."""
    built: list[object] = []
    connected: list[object] = []
    live = router if router is not None else LiveRouter()

    def factory(config, **kw):
        built.append(config)
        return live

    def connect():
        connected.append("connection")
        return FakeIB()

    ctx = run_daily_session(
        store,
        connector=connector or connect,
        sdk=FakeSdk(live._client),
        router_config=routed_config(),
        installed_version=VERSION,
        clock=clock,
        run_mode="live",
        router_factory=factory,
    )
    return ctx, built, connected, live


def _full_book(store_, **kwargs):
    """A constructed book whose metric record validates, so the manifest is `ok`."""
    return dataclasses.replace(
        order_book(day=INCIDENT_BOUND),
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
            "last_updated_utc": "2026-11-10T14:35:00Z",
        },
    )


def _served_metric(manifest: dict) -> dict:
    (metric,) = [m for m in manifest["metrics"] if m["name"] == "trader_session_served"]
    return metric


class TestARoutedSessionStartsOnlyInMarketHours:
    def test_the_2036_et_start_is_held_even_with_a_feed_and_routes_nothing(
        self, store, monkeypatch
    ) -> None:
        """The incident, replayed with the feed already published (the case
        that would have routed DAY market orders after hours)."""
        _serveable(store, trading_day=INCIDENT_BOUND)
        monkeypatch.setattr(module, "construct_target_book", _full_book)

        ctx, built, connected, router = _routed_job(store, clock=lambda: INCIDENT)

        assert built == [] and connected == []
        assert router._client.order_calls == []
        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, INCIDENT_BOUND)))
        assert manifest["status"] == "ok", manifest.get("reason")
        assert manifest["trading_day"] == INCIDENT_BOUND
        # The New York date (Monday), not the UTC date (Tuesday).
        assert manifest["calendar_date"] == "2026-11-09"
        assert ctx.calendar_date == dt.date(2026, 11, 9)
        metric = _served_metric(manifest)
        assert metric["value"] == 0.0
        assert metric["status_reason"] == (
            "held: outside market hours: 2026-11-09 20:36:16 ET is outside the regular "
            "session (09:30-16:00 ET on 2026-11-09)"
        )
        assert not store.exists(TRADER_EVIDENCE_KEY)
        assert not store.exists(execution_shortfall_key(INCIDENT_BOUND))

    def test_the_held_document_names_the_reason(self, store, monkeypatch) -> None:
        captured: list[dict] = []
        real = module.run_job

        def spy(job, fn, **kw):
            return real(job, lambda ctx: captured.append(fn(ctx)) or captured[-1], **kw)

        monkeypatch.setattr(module, "run_job", spy)
        _routed_job(store, clock=lambda: INCIDENT)

        (document,) = captured
        assert document["outcome"] == "held"
        assert document["session_mode"] == "live"
        assert document["held_reason"].startswith("outside market hours")

    def test_a_0935_et_start_is_served(self, store, monkeypatch) -> None:
        """Draft #139's start time. 09:35 EST is 14:35Z; it binds to Monday."""
        _serveable(store, trading_day=INCIDENT_BOUND)
        monkeypatch.setattr(module, "construct_target_book", _full_book)
        start = dt.datetime(2026, 11, 10, 14, 35, tzinfo=dt.UTC)

        ctx, built, _, router = _routed_job(store, clock=lambda: start)

        assert built == [routed_config()]
        assert router.closed == 1
        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, INCIDENT_BOUND)))
        assert manifest["status"] == "ok", manifest.get("reason")
        assert (manifest["trading_day"], manifest["calendar_date"]) == (
            INCIDENT_BOUND,
            "2026-11-10",
        )
        assert _served_metric(manifest)["value"] == 1.0
        evidence = json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))
        assert evidence["session_modes"] == {INCIDENT_BOUND: "live"}
        assert evidence["calendar_date"] == "2026-11-10"
        assert ctx.trading_day.isoformat() == INCIDENT_BOUND

    def test_a_session_that_started_in_hours_may_run_past_the_close(
        self, store, monkeypatch
    ) -> None:
        """The guard is on START. Draft #139's all-day session keeps its clock
        running after the start, and later past the close."""
        _serveable(store, trading_day=INCIDENT_BOUND)
        monkeypatch.setattr(module, "construct_target_book", _full_book)
        start = dt.datetime(2026, 11, 10, 20, 59, 59, tzinfo=dt.UTC)  # 15:59:59 ET
        later = dt.datetime(2026, 11, 10, 22, 0, tzinfo=dt.UTC)  # 17:00 ET
        ticks = iter([start])

        _routed_job(store, clock=lambda: next(ticks, later))

        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, INCIDENT_BOUND)))
        assert manifest["status"] == "ok", manifest.get("reason")
        assert _served_metric(manifest)["value"] == 1.0

    def test_a_shadow_session_after_hours_is_not_checked(self, store, monkeypatch) -> None:
        """Shadow routes nothing, so it runs as before. Its calendar date is
        New York's too."""
        _serveable(store, trading_day=INCIDENT_BOUND)
        monkeypatch.setattr(module, "construct_target_book", _full_book)
        ib = FakeIB()

        run_daily_session(
            store,
            connector=lambda: ib,
            sdk=FakeSdk(ib),
            router_config=RouterConfig(enabled=False),
            installed_version=VERSION,
            clock=lambda: INCIDENT,
            run_mode="live",
            router_factory=lambda config, **kw: build_router(config, **kw),
        )

        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, INCIDENT_BOUND)))
        assert manifest["calendar_date"] == "2026-11-09"
        assert json.loads(store.get_bytes(TRADER_EVIDENCE_KEY))["session_modes"] == {
            INCIDENT_BOUND: "shadow"
        }

    def test_a_staging_simulated_session_after_hours_is_not_checked(
        self, store, monkeypatch
    ) -> None:
        _simulated_world(store, monkeypatch)
        late = dt.datetime(2026, 9, 12, 0, 30, tzinfo=dt.UTC)  # Fri 09-11 20:30 ET

        code = main(
            [],
            environ=_staging_env(store),
            sdk_loader=_no_sdk,
            version_reader=lambda: VERSION,
            clock=lambda: late,
            printer=lambda line: None,
        )

        assert code == 0
        late_bound = "2026-09-11"
        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, late_bound)))
        assert manifest["calendar_date"] == "2026-09-11"
        assert _served_metric(manifest)["value"] == 0.0  # no feed for 09-11: held as before
        assert "outside market hours" not in _served_metric(manifest)["status_reason"]

    def test_the_entry_point_holds_an_after_hours_routed_start(self, store) -> None:
        said: list[str] = []
        env = environ_for(
            store.root,
            CRUCIBLE_TRADER_ORDER_ROUTING="ib_paper",
            CRUCIBLE_TRADER_IB_ACCOUNT="DU1",
            CRUCIBLE_TRADER_IB_PORT="4002",
            CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD="1000",
            CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD="5000",
        )

        code = main(
            [],
            environ=env,
            sdk_loader=FakeSdk(FakeIB()),
            version_reader=lambda: VERSION,
            clock=lambda: INCIDENT,
            printer=said.append,
            router_factory=lambda config, **kw: pytest.fail("a router was built after hours"),
        )

        assert code == 0
        manifest = json.loads(store.get_bytes(manifest_key(SESSION_JOB, INCIDENT_BOUND)))
        assert manifest["status"] == "ok"
        assert "outside market hours" in _served_metric(manifest)["status_reason"]


def _et(year, month, day, hour, minute, second=0):
    return dt.datetime(year, month, day, hour, minute, second, tzinfo=module.NYSE_ZONE)


class TestOutsideMarketHours:
    @pytest.mark.parametrize(
        "now",
        [
            _et(2026, 11, 10, 9, 30),  # the open
            _et(2026, 11, 10, 9, 35),  # draft #139's start
            _et(2026, 11, 10, 14, 0),  # today's interim start
            _et(2026, 11, 10, 15, 59, 59),
            _et(2026, 11, 27, 12, 59, 59),  # day after Thanksgiving, closes 13:00
            dt.datetime(2026, 7, 1, 13, 35, tzinfo=dt.UTC),  # 09:35 EDT
        ],
    )
    def test_inside_the_regular_session_a_start_is_allowed(self, now) -> None:
        assert module.outside_market_hours(now) is None

    @pytest.mark.parametrize(
        ("now", "says"),
        [
            (INCIDENT, "2026-11-09 20:36:16 ET is outside the regular session (09:30-16:00"),
            (_et(2026, 11, 10, 9, 29, 59), "09:29:59 ET is outside"),
            (_et(2026, 11, 10, 16, 0), "16:00:00 ET is outside"),
            (_et(2026, 11, 27, 13, 0), "(09:30-13:00 ET on 2026-11-27)"),
            (_et(2026, 11, 27, 14, 0), "(09:30-13:00 ET on 2026-11-27)"),
            (_et(2026, 11, 26, 11, 0), "is not on an NYSE trading day"),  # Thanksgiving
            (_et(2026, 11, 14, 11, 0), "is not on an NYSE trading day"),  # Saturday
            (dt.datetime(2026, 7, 1, 13, 29, tzinfo=dt.UTC), "09:29:00 ET is outside"),
        ],
    )
    def test_outside_it_a_start_is_refused_with_its_reason(self, now, says) -> None:
        reason = module.outside_market_hours(now)
        assert reason is not None and reason.startswith("outside market hours: ")
        assert says in reason

    def test_a_naive_time_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(ValueError, match="aware"):
            module.outside_market_hours(dt.datetime(2026, 11, 10, 14, 0))  # noqa: DTZ001
