"""The order router: off by default, paper only, capped, kill-switched, idempotent.

Every test drives a fake IB client that records each order call, so "sent
nothing" is an assertion on `order_calls`, not on the router's own say-so.
"""

from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace

import pytest
from crucible.keys import manifest_key
from crucible.release import TRADER_PIN_KEY
from crucible.serving import PredictionsFeed
from crucible.slots.strategy import BookUniverse
from crucible.store import LocalStore
from ib_fakes import T0, FakeClock, FakeIB, FakeSdk, ctx_for

from crucible_trader.broker_session import GatewayAddress, OrderRefusedError
from crucible_trader.construction import TargetBook
from crucible_trader.kill_switch import (
    KILL_SWITCH_KEY,
    IbBrokerControl,
    TradingHaltedError,
    halt_document,
)
from crucible_trader.order_router import (
    ORDER_REF_PREFIX,
    PAPER_GATEWAY_PORT,
    ROUTING_VAR,
    ExecutionRecord,
    ForeignPositionError,
    IbQuoteSource,
    NotPaperError,
    OrderRefError,
    PaperOrderRouter,
    PriceUnavailableError,
    RiskCapExceededError,
    RouterConfig,
    RouterSettingsError,
    RoutingDisabledError,
    RoutingRefusedError,
    ShadowRouter,
    build_router,
    open_paper_router,
    order_key,
    parse_order_ref,
    plan_orders,
)
from crucible_trader.session import session_cycle

DAY = "2026-09-08"
ACCOUNT = "DU1"
PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0}
PAPER = GatewayAddress("127.0.0.1", PAPER_GATEWAY_PORT, 7)


def _env(**extra: str) -> dict[str, str]:
    values = {
        ROUTING_VAR: "ib_paper",
        "CRUCIBLE_TRADER_IB_ACCOUNT": ACCOUNT,
        "CRUCIBLE_TRADER_IB_HOST": "127.0.0.1",
        "CRUCIBLE_TRADER_IB_PORT": str(PAPER_GATEWAY_PORT),
        "CRUCIBLE_TRADER_IB_CLIENT_ID": "7",
        "CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD": "50000",
        "CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD": "100000",
    }
    values.update(extra)
    return values


def _config(**overrides) -> RouterConfig:
    base = RouterConfig(
        enabled=True,
        account=ACCOUNT,
        address=PAPER,
        max_order_notional_usd=50_000.0,
        max_session_notional_usd=100_000.0,
    )
    return dataclasses.replace(base, **overrides)


def _book(weights: dict[str, float] | None = None, *, day: str = DAY) -> TargetBook:
    weights = weights if weights is not None else {"AAA": 0.3, "BBB": 0.2, "CCC": 0.0}
    tickers = (*weights, "SPY", "__CASH__")
    full = {**weights, "SPY": 0.0, "__CASH__": 1.0 - sum(weights.values())}
    return TargetBook(
        trading_day=day,
        s_champion="s:arm:def",
        m_champion="m:arm:abc",
        u_champion=None,
        universe=BookUniverse(
            tickers=tickers,
            sectors=tuple("x" for _ in tickers),
            benchmark_idx=len(tickers) - 2,
            cash_idx=len(tickers) - 1,
        ),
        weights=full,
        turnover_one_way_ratio=0.5,
        cost_bps=1.0,
        portfolio_notional_usd=100_000.0,
        evidence={},
        metric={"name": "portfolio_construction"},
        session_inputs_key="k",
    )


class RoutingIB(FakeIB):
    """FakeIB plus the quote, qualify, open/completed-order and execution reads."""

    def __init__(self, *, prices=None, slippage=0.01, stall=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prices = dict(PRICES if prices is None else prices)
        self.slippage = slippage
        self.stall = stall
        self.completed: list = []
        self.cancelled_mkt: list[str] = []
        self.unqualified: set[str] = set()
        self.on_place = None

    def qualifyContracts(self, contract):  # noqa: N802 - SDK spelling
        # `ib_async` 2.x keeps the slot and returns None for an unqualifiable contract.
        return [None] if contract.symbol in self.unqualified else [contract]

    def reqMarketDataType(self, kind):  # noqa: N802 - SDK spelling
        self.market_data_type = kind

    def reqMktData(self, contract, tick_list, snapshot, regulatory):  # noqa: N802
        price = self.prices.get(contract.symbol)
        return SimpleNamespace(last=math.nan if price is None else price, close=math.nan)

    def cancelMktData(self, contract):  # noqa: N802 - SDK spelling
        self.cancelled_mkt.append(contract.symbol)

    def reqAllOpenOrders(self):  # noqa: N802 - SDK spelling
        return self.openTrades()

    def reqCompletedOrders(self, api_only):  # noqa: N802 - SDK spelling
        assert api_only is True
        return list(self.completed)

    def placeOrder(self, contract, order):  # noqa: N802 - SDK spelling
        trade = super().placeOrder(contract, order)
        if self.on_place is not None:
            self.on_place()
        return trade

    def sleep(self, seconds):
        super().sleep(seconds)
        if self.stall:
            return
        for trade in self._trades:
            ref = trade.order.orderRef or ""
            if trade.orderStatus.status != "Submitted" or not ref.startswith(ORDER_REF_PREFIX):
                continue
            symbol, qty = trade.contract.symbol, int(trade.order.totalQuantity)
            buy = trade.order.action == "BUY"
            price = self.prices[symbol] * (1 + self.slippage if buy else 1 - self.slippage)
            self.book[symbol] = self.book.get(symbol, 0) + (qty if buy else -qty)
            if self.book[symbol] == 0:
                del self.book[symbol]
            trade.orderStatus.status = "Filled"
            self.broker_fills.append(
                SimpleNamespace(
                    contract=SimpleNamespace(symbol=symbol),
                    execution=SimpleNamespace(
                        orderRef=ref,
                        time=self.clock(),
                        side="BOT" if buy else "SLD",
                        shares=float(qty),
                        price=price,
                        avgPrice=price,
                    ),
                    commissionReport=SimpleNamespace(commission=1.0),
                )
            )


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


def _router(store, ib, *, config=None, clock=None, **kwargs) -> PaperOrderRouter:
    return PaperOrderRouter(
        config or _config(),
        client=ib,
        sdk=FakeSdk(ib),
        store=store,
        trading_day=DAY,
        clock=clock or ib.clock,
        **kwargs,
    )


def _route(router: PaperOrderRouter, book: TargetBook) -> None:
    for batch in router.batches(book):
        router.send(batch)


def _halt(store, mode: str, day: str = DAY) -> None:
    store.put_bytes(
        KILL_SWITCH_KEY,
        json.dumps(
            halt_document(
                engaged=True, mode=mode, cause="drill", trading_day=day, since="x", run_id="r"
            )
        ).encode(),
    )


# -- off by default -----------------------------------------------------------------


@pytest.mark.parametrize("env", [{}, {ROUTING_VAR: ""}, {ROUTING_VAR: "off"}])
def test_routing_is_off_unless_explicitly_switched_on(env) -> None:
    assert RouterConfig.from_env(env) == RouterConfig(enabled=False)


def test_the_process_environment_is_read_when_no_env_is_given(monkeypatch) -> None:
    monkeypatch.delenv(ROUTING_VAR, raising=False)
    assert RouterConfig.from_env().enabled is False


@pytest.mark.parametrize("value", ["IB_PAPER", "on", "true", "1", "live", "ib_live"])
def test_an_unrecognised_switch_value_is_refused_not_guessed(value) -> None:
    with pytest.raises(RouterSettingsError, match="not a routing mode"):
        RouterConfig.from_env(_env(**{ROUTING_VAR: value}))


def test_a_disabled_config_builds_a_shadow_router_without_touching_the_broker(store) -> None:
    def never():
        raise AssertionError("a shadow session must not load the broker SDK")

    router = build_router(RouterConfig.from_env({}), store=store, trading_day=DAY, sdk_loader=never)
    assert isinstance(router, ShadowRouter) and router.enabled is False
    assert list(router.batches(_book())) == [] and router.records == ()
    with pytest.raises(RoutingDisabledError):
        router.send(["anything"])
    document = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=T0)
    assert document["outcome"] == "no_orders" and "shadow session" in document["outcome_reason"]
    assert router.close() is None


def test_an_order_capable_router_refuses_a_disabled_config(store) -> None:
    ib = RoutingIB()
    with pytest.raises(RoutingDisabledError):
        _router(store, ib, config=RouterConfig(enabled=False))
    with pytest.raises(RoutingDisabledError):
        open_paper_router(RouterConfig(enabled=False), store=store, trading_day=DAY)
    assert ib.order_calls == []


# -- paper only ----------------------------------------------------------------------


def test_a_complete_paper_environment_enables_routing() -> None:
    config = RouterConfig.from_env(_env())
    assert config.enabled and config.account == ACCOUNT and config.address == PAPER
    assert config.max_order_notional_usd == 50_000.0
    assert config.max_session_notional_usd == 100_000.0


@pytest.mark.parametrize("account", ["U1234567", "F123", "DF1", "du1"])
def test_a_non_paper_account_is_refused(account) -> None:
    with pytest.raises(NotPaperError, match="not an IB paper account"):
        RouterConfig.from_env(_env(CRUCIBLE_TRADER_IB_ACCOUNT=account))


@pytest.mark.parametrize("port", ["4001", "7496", "7497"])
def test_a_port_other_than_the_paper_gateway_port_is_refused(port) -> None:
    with pytest.raises(NotPaperError, match="not the IB Gateway paper port"):
        RouterConfig.from_env(_env(CRUCIBLE_TRADER_IB_PORT=port))


def test_the_paper_guard_holds_on_direct_construction_too() -> None:
    with pytest.raises(NotPaperError):
        _config(account="U1")
    with pytest.raises(NotPaperError):
        _config(account=None)
    with pytest.raises(NotPaperError):
        _config(address=GatewayAddress("127.0.0.1", 4001, 7))
    with pytest.raises(RouterSettingsError, match="no gateway address"):
        _config(address=None)


def test_routing_on_with_no_account_is_refused() -> None:
    with pytest.raises(RouterSettingsError, match="CRUCIBLE_TRADER_IB_ACCOUNT"):
        RouterConfig.from_env(_env(CRUCIBLE_TRADER_IB_ACCOUNT=" "))


@pytest.mark.parametrize(
    ("var", "value", "match"),
    [
        ("CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD", "", "not a number"),
        ("CRUCIBLE_TRADER_MAX_ORDER_NOTIONAL_USD", "lots", "not a number"),
        ("CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD", "0", "positive"),
        ("CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD", "inf", "positive"),
        ("CRUCIBLE_TRADER_MAX_SESSION_NOTIONAL_USD", "-5", "positive"),
    ],
)
def test_caps_are_required_and_positive(var, value, match) -> None:
    with pytest.raises(RouterSettingsError, match=match):
        RouterConfig.from_env(_env(**{var: value}))


@pytest.mark.parametrize("field", ["max_order_notional_usd", "max_session_notional_usd"])
@pytest.mark.parametrize("value", [None, 0.0, math.nan])
def test_caps_are_required_on_direct_construction(field, value) -> None:
    with pytest.raises(RouterSettingsError, match=field):
        _config(**{field: value})


def test_a_gateway_managing_a_live_account_is_refused_before_any_order(store) -> None:
    ib = RoutingIB(accounts=("U9",))
    with pytest.raises(OrderRefusedError, match="non-paper"):
        open_paper_router(_config(), store=store, trading_day=DAY, sdk_loader=FakeSdk(ib))
    assert ib.order_calls == [] and ib.disconnected == 1


def test_a_gateway_managing_another_paper_account_is_refused(store) -> None:
    ib = RoutingIB(accounts=("DU2",))
    with pytest.raises(NotPaperError, match="DU2"):
        open_paper_router(_config(), store=store, trading_day=DAY, sdk_loader=FakeSdk(ib))
    assert ib.order_calls == [] and ib.disconnected == 1


def test_build_router_connects_an_order_capable_session_on_the_paper_port(store) -> None:
    ib = RoutingIB()
    router = build_router(
        RouterConfig.from_env(_env()), store=store, trading_day=DAY, sdk_loader=FakeSdk(ib)
    )
    assert isinstance(router, PaperOrderRouter) and router.enabled is True
    assert ib.connected_with["port"] == PAPER_GATEWAY_PORT
    assert ib.connected_with["readonly"] is False
    router.close()
    assert ib.disconnected == 1


# -- routing ---------------------------------------------------------------------------


def test_a_book_is_routed_sells_first_and_every_order_is_recorded(store) -> None:
    ib = RoutingIB(positions={"BBB": 500, "CCC": 100})
    router = _router(store, ib)
    batches = list(router.batches(_book()))
    assert [[o.decision.symbol for o in b] for b in batches] == [["BBB", "CCC"], ["AAA"]]
    for batch in batches:
        router.send(batch)

    placed = [(c.symbol, o.action, o.totalQuantity) for c, o in ib.order_calls]
    # AAA 0.3 * 100k / 100 = 300; BBB 0.2 * 100k / 50 = 400 (holds 500); CCC exits.
    assert placed == [("BBB", "SELL", 100), ("CCC", "SELL", 100), ("AAA", "BUY", 300)]
    for _, order in ib.order_calls:
        assert order.account == ACCOUNT and order.tif == "DAY"
        assert order.outsideRth is False and order.transmit is True
        assert order.orderRef.startswith(f"{ORDER_REF_PREFIX}|{DAY}|")
    assert ib.book == {"AAA": 300, "BBB": 400}
    assert ib.market_data_type == 3 and sorted(ib.cancelled_mkt) == ["AAA", "BBB", "CCC"]

    records = router.records
    assert [r.disposition for r in records] == ["placed"] * 3
    assert [r.broker_status for r in records] == ["Filled"] * 3
    aaa = records[-1].to_document()
    assert aaa["order_id"] == order_key(DAY, "AAA") and aaa["filled_quantity"] == 300.0
    assert aaa["decision_price"] == 100.0 and aaa["fill_price"] == pytest.approx(101.0)
    assert aaa["shortfall_bps"] == pytest.approx(100.0)

    document = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=T0)
    assert document["outcome"] == "computed" and document["summary"]["n_filled"] == 3


def test_a_book_already_held_sends_nothing_and_says_so(store) -> None:
    ib = RoutingIB(positions={"AAA": 300, "BBB": 400})
    router = _router(store, ib)
    assert list(router.batches(_book())) == []
    assert ib.order_calls == []
    document = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=T0)
    assert document["outcome"] == "no_orders"
    assert "no order was needed" in document["outcome_reason"]


def test_an_order_still_working_at_the_fill_bound_is_recorded_unfilled(store) -> None:
    ib = RoutingIB(stall=True)
    router = _router(store, ib, fill_timeout_s=2.0)
    _route(router, _book({"AAA": 0.1}))
    (record,) = router.records
    assert record.broker_status == "Submitted" and record.filled_quantity == 0.0
    assert record.fill_price is None and ib.clock() > T0
    document = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=T0)
    assert document["outcome"] == "computed" and document["summary"]["n_filled"] == 0


def test_a_contract_ib_cannot_qualify_is_refused_at_send(store) -> None:
    ib = RoutingIB()
    router = _router(store, ib, quote=lambda symbol: 100.0)
    (batch,) = router.batches(_book({"AAA": 0.1}))
    ib.unqualified.add("AAA")
    with pytest.raises(RoutingRefusedError, match="qualify AAA"):
        router.send(batch)
    assert ib.order_calls == []


def test_a_book_for_another_day_is_refused(store) -> None:
    with pytest.raises(RoutingRefusedError, match="opened for"):
        _router(store, RoutingIB()).batches(_book(day="2026-09-09"))


# -- kill switch ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["flatten", "freeze", "hold"])
def test_an_engaged_kill_switch_blocks_submission(store, mode) -> None:
    ib = RoutingIB()
    router = _router(store, ib)
    (batch,) = router.batches(_book({"AAA": 0.1}))
    _halt(store, mode)
    with pytest.raises(TradingHaltedError, match=mode):
        router.send(batch)
    assert ib.order_calls == [] and router.records == ()
    document = router.shortfall_document(champion="m:arm:abc", trading_day=DAY, now=T0)
    assert document["outcome"] == "no_orders" and "none routed" in document["outcome_reason"]


def test_a_switch_fired_mid_batch_stops_the_next_order(store) -> None:
    ib = RoutingIB()
    router = _router(store, ib)
    (batch,) = router.batches(_book({"AAA": 0.1, "BBB": 0.1}))
    ib.on_place = lambda: _halt(store, "freeze")
    with pytest.raises(TradingHaltedError):
        router.send(batch)
    assert [c.symbol for c, _ in ib.order_calls] == ["AAA"]
    (record,) = router.records  # the order already at the broker is still recorded
    assert record.order.decision.symbol == "AAA" and record.disposition == "placed"


def _pin_document(sha: str) -> dict:
    return {
        "sha": sha,
        "target": "trader",
        "pinned_at": "2026-09-05T22:00:00Z",
        "smoke_run_id": "01JG0000000000000000000000",
        "smoke_status": "ok",
        "smoke_manifest_key": manifest_key("trader.smoke", "2026-09-05", discriminator=sha[:12]),
    }


def _session(store, ib, router):
    store.put_bytes(TRADER_PIN_KEY, json.dumps(_pin_document("c" * 40)).encode())
    feed = PredictionsFeed(
        slot="m",
        trading_day=DAY,
        champion="m:arm:abc",
        feature_version="v1",
        source_key="arms/m/x.json",
        predicted_alpha={f"T{i}": 0.001 * i for i in range(10)},
    )
    return session_cycle(
        ctx_for(store, "trader.session"),
        broker=IbBrokerControl(ib, FakeSdk(ib)),
        router=router,
        installed_version=f"0.1.0+g{'c' * 12}",
        previous_weights=None,
        clock=ib.clock,
        contract_resolver=lambda s, day: SimpleNamespace(feed=feed, arm_id=feed.champion),
        construct=lambda s, **kwargs: _book({"AAA": 0.1}),
    )


def test_the_session_routes_through_the_router(store) -> None:
    ib = RoutingIB()
    result = _session(store, ib, _router(store, ib))
    assert result["batches_sent"] == 1 and len(ib.order_calls) == 1


def test_the_session_with_a_fired_kill_switch_sends_nothing(store) -> None:
    _halt(store, "flatten", day="2026-09-01")
    ib = RoutingIB()
    with pytest.raises(TradingHaltedError):
        _session(store, ib, _router(store, ib))
    assert ib.order_calls == []


def test_a_shadow_session_sends_nothing(store) -> None:
    ib = RoutingIB()
    result = _session(store, ib, ShadowRouter())
    assert result["batches_sent"] == 0 and ib.order_calls == []


# -- idempotency ---------------------------------------------------------------------


def test_a_retried_session_adopts_its_orders_and_sends_none_twice(store) -> None:
    ib = RoutingIB()
    _route(_router(store, ib), _book({"AAA": 0.1}))
    assert len(ib.order_calls) == 1
    first_ref = ib.order_calls[0][1].orderRef

    ib.prices["AAA"] = 110.0  # the retry would size at a different price
    retry = _router(store, ib)
    _route(retry, _book({"AAA": 0.1, "BBB": 0.1}))
    assert [c.symbol for c, _ in ib.order_calls] == ["AAA", "BBB"]  # BBB is new; AAA is not
    adopted = next(r for r in retry.records if r.disposition == "adopted")
    assert adopted.order.order_ref == first_ref and adopted.broker_status == "Adopted"
    assert adopted.order.decision.decision_price == 100.0  # the ORIGINAL decision
    assert adopted.submitted_at == adopted.order.decision.decided_at
    assert adopted.filled_quantity == 100.0


def test_orders_known_only_as_completed_or_executed_are_adopted(store) -> None:
    decision_ref = f"{order_key(DAY, 'AAA')}|buy|100|100.0|2026-09-08T13:00:00.000000Z"
    ib = RoutingIB()
    ib.completed.append(SimpleNamespace(order=SimpleNamespace(orderRef=decision_ref)))
    ib.broker_fills.append(
        SimpleNamespace(execution=SimpleNamespace(orderRef=None, shares=1.0, price=1.0))
    )
    router = _router(store, ib)
    _route(router, _book({"AAA": 0.1}))
    assert ib.order_calls == [] and router.records[0].disposition == "adopted"


def test_another_days_order_does_not_block_today(store) -> None:
    ib = RoutingIB()
    ib.completed.append(
        SimpleNamespace(
            order=SimpleNamespace(
                orderRef=f"{order_key('2026-09-04', 'AAA')}|buy|5|99.0|2026-09-04T13:00:00.000000Z"
            )
        )
    )
    _route(_router(store, ib), _book({"AAA": 0.1}))
    assert len(ib.order_calls) == 1


@pytest.mark.parametrize(
    "ref",
    [
        f"{ORDER_REF_PREFIX}|{DAY}|AAA",
        f"{ORDER_REF_PREFIX}|{DAY}|AAA|hold|1|1.0|2026-09-08T13:00:00.000000Z",
        f"{ORDER_REF_PREFIX}|{DAY}|AAA|buy|1|1.0|yesterday",
    ],
)
def test_an_unreadable_ref_under_this_prefix_refuses_the_batch(store, ref) -> None:
    ib = RoutingIB()
    ib.completed.append(SimpleNamespace(order=SimpleNamespace(orderRef=ref)))
    router = _router(store, ib)
    (batch,) = router.batches(_book({"AAA": 0.1}))
    with pytest.raises(OrderRefError, match="cannot be read"):
        router.send(batch)
    assert ib.order_calls == []


def test_a_ref_with_another_prefix_is_not_this_routers() -> None:
    with pytest.raises(OrderRefError, match="prefix"):
        parse_order_ref(f"other|{DAY}|AAA|buy|1|1.0|2026-09-08T13:00:00.000000Z")


def test_the_order_ref_round_trips_the_sizing_decision() -> None:
    ref = f"{order_key(DAY, 'AAA')}|sell|7|12.5|2026-09-08T13:00:00.000000Z"
    order = parse_order_ref(ref)
    assert order.order_ref == ref and order.action == "SELL" and order.key == order_key(DAY, "AAA")


# -- caps and refusals (the plan, before any order) -------------------------------------


def _plan(book, *, positions=None, quote=None, order_cap=50_000.0, session_cap=100_000.0):
    return plan_orders(
        book,
        positions=positions or {},
        quote=quote or PRICES.get,
        clock=FakeClock(),
        max_order_notional_usd=order_cap,
        max_session_notional_usd=session_cap,
    )


def test_an_order_over_the_per_order_cap_refuses_the_whole_plan() -> None:
    with pytest.raises(RiskCapExceededError, match="per-order cap"):
        _plan(_book(), order_cap=20_000.0)


def test_a_plan_over_the_per_session_cap_is_refused() -> None:
    with pytest.raises(RiskCapExceededError, match="per-session cap"):
        _plan(_book(), session_cap=40_000.0)


def test_a_cap_breach_through_the_router_sends_nothing(store) -> None:
    ib = RoutingIB()
    with pytest.raises(RiskCapExceededError):
        _route(_router(store, ib, config=_config(max_session_notional_usd=1_000.0)), _book())
    assert ib.order_calls == []


def test_a_position_the_book_does_not_know_is_never_traded() -> None:
    with pytest.raises(ForeignPositionError, match="ZZZ"):
        _plan(_book(), positions={"ZZZ": 10})
    with pytest.raises(ForeignPositionError, match="SPY"):  # a sentinel is not tradable
        _plan(_book(), positions={"SPY": 10})


def test_a_short_target_is_refused() -> None:
    with pytest.raises(RiskCapExceededError, match="long only"):
        _plan(_book({"AAA": -0.1, "BBB": 0.2}))


def test_a_book_over_its_notional_is_refused() -> None:
    with pytest.raises(RiskCapExceededError, match="gross exceeds"):
        _plan(_book({"AAA": 0.7, "BBB": 0.5}))


@pytest.mark.parametrize("price", [None, math.nan, 0.0, -1.0])
def test_a_missing_decision_price_refuses_the_whole_plan(price) -> None:
    with pytest.raises(PriceUnavailableError, match="AAA"):
        _plan(_book(), quote=lambda symbol: price if symbol == "AAA" else 50.0)


def test_the_decision_is_frozen_at_the_quote() -> None:
    (sells, buys) = _plan(_book({"AAA": 0.3}))
    (order,) = buys
    assert sells == () and order.decision.decision_price == 100.0
    assert order.decision.decided_at == T0 and order.decision.quantity == 300


# -- quotes (v1's `get_current_price`, ported) -------------------------------------------


def test_the_quote_falls_back_to_the_close_and_releases_the_line() -> None:
    ib = RoutingIB()
    ib.reqMktData = lambda *a: SimpleNamespace(last=None, close=42.0)
    assert IbQuoteSource(ib, FakeSdk(ib))("AAA") == 42.0
    assert ib.cancelled_mkt == ["AAA"]


def test_a_name_with_no_tick_times_out_to_none() -> None:
    ib = RoutingIB(prices={})
    assert IbQuoteSource(ib, FakeSdk(ib), max_wait=1.0)("AAA") is None
    assert ib.cancelled_mkt == ["AAA"] and (ib.clock() - T0).total_seconds() == 1.0


def test_an_unqualifiable_name_has_no_quote() -> None:
    ib = RoutingIB()
    ib.unqualified.add("AAA")
    assert IbQuoteSource(ib, FakeSdk(ib))("AAA") is None and ib.cancelled_mkt == []
    ib.qualifyContracts = lambda contract: []  # an empty result is refused the same way
    assert IbQuoteSource(ib, FakeSdk(ib))("AAA") is None


def test_an_execution_record_names_its_order_and_fill() -> None:
    order = parse_order_ref(f"{order_key(DAY, 'AAA')}|buy|10|10.0|2026-09-08T13:00:00.000000Z")
    record = ExecutionRecord(
        order=order,
        disposition="placed",
        broker_status="Filled",
        submitted_at=T0,
        filled_quantity=10.0,
        fill_price=10.1,
    )
    document = record.to_document()
    assert document["order_ref"] == order.order_ref and document["broker_status"] == "Filled"
    assert document["shortfall_bps"] == pytest.approx(100.0)
