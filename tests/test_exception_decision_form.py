"""The reviewer's decision form (alpha-engine-config-I13121, back office 1d).

The load-bearing properties: the page lists exactly the proposals that still
wait for a person, for the approver's tenants only; a deferral is offered with
the decision it must supersede; a conflict gets no form; the reviewer is never
a form field; and nothing from a record reaches the page unescaped.
"""

from __future__ import annotations

from crucible_trader.exception_decision_form import (
    ACTION_TEXT,
    NATURE_TEXT,
    content_security_policy,
    queue_items,
    render_queue_page,
)
from crucible_trader.exception_proposal import ACTIONS, NATURES

TENANT = "nous-ergon-paper"


def proposal(
    n: int, *, tenant: str = TENANT, day: str = "2026-11-03", **extra: object
) -> dict[str, object]:
    return {
        "proposal_id": f"p{n}",
        "tenant": tenant,
        "trading_day": day,
        "class": "stale_anchor",
        "nature": "detector",
        "action": "refresh_anchor",
        "rationale": "the position anchor is stale; refresh it",
        **extra,
    }


def decision(
    n: int, proposal_id: str, verdict: str, supersedes: str | None = None
) -> dict[str, object]:
    return {
        "decision_id": f"d{n}",
        "proposal_id": proposal_id,
        "verdict": verdict,
        "supersedes": supersedes,
    }


class TestQueueItems:
    def test_waiting_items_oldest_first_for_held_tenants_only(self) -> None:
        proposals = [
            proposal(1, day="2026-11-03"),
            proposal(2, day="2026-11-02"),
            proposal(3, tenant="other"),
            proposal(4),
            proposal(5),
            proposal(6),
        ]
        decisions = [
            decision(1, "p4", "agree"),
            decision(2, "p5", "defer"),
            decision(3, "p6", "agree"),
            decision(4, "p6", "correct"),
        ]
        items = queue_items(proposals, decisions, tenants=[TENANT])
        assert [(i.proposal["proposal_id"], i.status, i.supersedes) for i in items] == [
            ("p2", "undecided", None),
            ("p1", "undecided", None),
            ("p5", "deferred", "d2"),
            ("p6", "conflicted", None),
        ]

    def test_a_superseded_deferral_that_was_then_agreed_leaves_the_queue(self) -> None:
        decisions = [decision(1, "p1", "defer"), decision(2, "p1", "agree", supersedes="d1")]
        assert queue_items([proposal(1)], decisions, tenants=[TENANT]) == []


class TestPage:
    def test_one_form_per_decidable_item_and_none_for_a_conflict(self) -> None:
        items = queue_items(
            [proposal(1), proposal(2), proposal(3)],
            [decision(1, "p2", "defer"), decision(2, "p3", "agree"), decision(3, "p3", "defer")],
            tenants=[TENANT],
        )
        page = render_queue_page(items, reviewer="brian", tenants=[TENANT], nonce="n0nce")
        assert page.count('<form class="decide">') == 2
        assert page.count('data-verdict="agree"') == 2 and page.count('data-verdict="defer"') == 2
        assert '<input type="hidden" name="supersedes" value="d1">' in page
        assert "Two decisions stand side by side" in page
        assert '<script nonce="n0nce">' in page and "Signed in as <strong>brian</strong>" in page
        assert 'name="decided_by"' not in page and 'name="reviewer"' not in page
        for verdict in ("agree", "correct", "defer"):
            assert f'data-verdict="{verdict}"' in page

    def test_record_values_are_escaped(self) -> None:
        hostile = proposal(1, rationale='<script>alert("x")</script>', **{"class": "a&b"})
        hostile["proposal_id"] = '"><img src=x>'
        page = render_queue_page(
            queue_items([hostile], [], tenants=[TENANT]),
            reviewer="<b>",
            tenants=[TENANT],
            nonce="n",
        )
        assert "<script>alert" not in page and "&lt;script&gt;" in page
        assert '"><img' not in page and "a&amp;b" in page and "&lt;b&gt;" in page

    def test_an_empty_queue_and_no_tenants_say_so(self) -> None:
        page = render_queue_page([], reviewer="brian", tenants=[], nonce="n")
        assert "Nothing waits for a decision." in page and "deciding for: none" in page
        assert "wait for you" not in page

    def test_a_missing_field_renders_blank(self) -> None:
        bare = {"proposal_id": "p9", "tenant": TENANT}
        page = render_queue_page(
            queue_items([bare], [], tenants=[TENANT]), reviewer="r", tenants=[TENANT], nonce="n"
        )
        assert "<dt>Why</dt><dd></dd>" in page


def test_the_policy_admits_only_the_nonced_script_and_requests_to_itself() -> None:
    policy = content_security_policy("abc")
    assert "script-src 'nonce-abc'" in policy and "connect-src 'self'" in policy
    assert "default-src 'none'" in policy and "frame-ancestors 'none'" in policy


class TestDecidableFromTheScreen:
    """Brian 2026-11-12: "I need to be able to understand what we are doing here.
    From the UX alone." Each card leads with plain words, both sides and the
    proposed fix; codes and ids sit behind "Technical details"."""

    CASH = {
        "exception_id": "e1",
        "class": "cash_delta",
        "amount": 107.69,
        "unit": "usd",
        "detail": "broker cash differs from anchor + fills + declared flows + dividends",
        "expected": {"record": "trader/reconciliation/2026-11-04.json", "value": 496715.37},
        "actual": {"record": "trader/broker_statements/2026-11-04.json", "value": 496823.06},
        "detected_by": {"manifest_key": "runs/trader.reconcile/2026-11-04/run.json"},
    }

    def page(self, record: dict[str, object] | None, **extra: object) -> str:
        p = proposal(1, exception_id="e1", **extra)
        items = queue_items([p], [], tenants=[TENANT])
        exceptions = {"e1": record} if record else None
        return render_queue_page(
            items, reviewer="brian", tenants=[TENANT], nonce="n", exceptions=exceptions
        )

    def test_a_cash_break_shows_both_sides_the_difference_and_the_fix_in_words(self) -> None:
        page = self.page(
            self.CASH, **{"class": "cash_delta", "nature": "defect", "action": "investigate_cash"}
        )
        assert "IB shows $107.69 more cash than our books expect" in page
        assert "$496,715.37" in page and "$496,823.06" in page and "+$107.69" in page
        assert "Find the missing or wrong cash entry before anything is booked." in page
        assert "a mistake on our side" in page
        assert "trader/broker_statements/2026-11-04.json" in page
        assert "1 difference between our books and IB wait" in page
        assert page.index("Technical details") > page.index("Proposed fix")

    def test_less_cash_reads_less(self) -> None:
        record = dict(self.CASH, amount=-3.5)
        assert "IB shows $3.50 less cash" in self.page(record, **{"class": "cash_delta"})

    def test_without_the_exception_record_the_card_still_renders(self) -> None:
        page = self.page(None)
        assert "The daily check skipped a day" in page
        assert "Restart the daily comparison from today" in page

    def test_every_action_and_nature_has_plain_words(self) -> None:
        assert set(ACTION_TEXT) == set(ACTIONS)
        assert set(NATURE_TEXT) == set(NATURES)

    def test_record_text_on_a_card_is_escaped(self) -> None:
        record = dict(self.CASH, detail="<img src=x onerror=1>", instrument="<b>")
        page = self.page(record, **{"class": "cash_delta"})
        assert "<img src=x" not in page and "&lt;img src=x" in page

    def test_a_share_count_corporate_action_reads_plainly(self) -> None:
        record = {
            "exception_id": "e1",
            "class": "corporate_action",
            "instrument": "MU",
            "amount": -40,
            "unit": "shares",
            "expected": {"value": 120},
            "actual": {"value": 80},
        }
        page = self.page(record, **{"class": "corporate_action"})
        assert "A corporate action on MU is not yet booked at IB" in page
        assert "<b>120</b>" in page and "<b>80</b>" in page and "<b>-40</b>" in page

    def test_a_record_with_no_figures_or_sources_shows_neither_box(self) -> None:
        page = self.page({"exception_id": "e1", "detail": "something odd"})
        assert 'class="sides"' not in page and "<summary>Evidence</summary>" not in page
        assert "something odd" in page
