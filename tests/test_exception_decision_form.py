"""The reviewer's decision form (alpha-engine-config-I13121, back office 1d).

The load-bearing properties: the page lists exactly the proposals that still
wait for a person, for the approver's tenants only; a deferral is offered with
the decision it must supersede; a conflict gets no form; the reviewer is never
a form field; and nothing from a record reaches the page unescaped.
"""

from __future__ import annotations

from crucible_trader.exception_decision_form import (
    content_security_policy,
    queue_items,
    render_queue_page,
)

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
        assert '<input type="hidden" name="supersedes" value="d1">' in page
        assert "Two decisions stand side by side" in page
        assert '<script nonce="n0nce">' in page and "Signed in as <strong>brian</strong>" in page
        assert 'name="decided_by"' not in page and 'name="reviewer"' not in page
        for verdict in ("agree", "correct", "defer"):
            assert f'<option value="{verdict}">' in page

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
