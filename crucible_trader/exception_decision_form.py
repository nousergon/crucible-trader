"""The reviewer's decision form: the page the decision writer serves on ``GET``.

`alpha-engine-config-I13121` (back office 1d), the last step of option (b) of
Brian's ruling of 2026-11-03: "the console form posts to it". The console is
read-only and its page vocabulary renders no HTML, so the form is served by the
writer itself, on the same Access-gated route its ``POST`` answers. A queue row
on the console's ``/page/back-office`` and a row here are the same proposal.

What the page shows: every proposal of the tenants the verified approver may
decide (narrowed by ``?tenant=``) that still waits for a person -- undecided, or
deferred (its form then supersedes the standing deferral) -- oldest first, each
with one small form. A proposal with two standing decisions (``conflicted``) is
listed without a form: which of the two a new call replaces is a choice the page
does not make for the reviewer.

The form never names the reviewer: the writer takes that from the verified
identity on the ``POST``. The script sends JSON with ``fetch`` to the page's own
URL, so the request carries the console's origin and the JSON content type the
writer requires, and it runs only under this response's nonce
(``Content-Security-Policy``). Everything rendered from a record is escaped.

Pure: no store, clock or network here; the writer passes the records in.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from crucible_trader.exception_decision import VERDICTS, effective_decisions
from crucible_trader.exception_proposal import ACTIONS, NATURES

__all__ = ["QueueItem", "content_security_policy", "queue_items", "render_queue_page"]


@dataclass(frozen=True)
class QueueItem:
    """One proposal waiting for a person, as the page lists it."""

    proposal: Mapping[str, object]
    #: ``undecided``, ``deferred`` or ``conflicted``.
    status: str
    #: The standing decision a new call must supersede (a deferral), if any.
    supersedes: str | None


def queue_items(
    proposals: Iterable[Mapping[str, object]],
    decisions: Iterable[Mapping[str, object]],
    *,
    tenants: Iterable[str],
) -> list[QueueItem]:
    """The proposals of ``tenants`` that still wait for a person, oldest first."""
    allowed = set(tenants)
    standing = effective_decisions(decisions)
    out: list[QueueItem] = []
    for proposal in proposals:
        if proposal.get("tenant") not in allowed:
            continue
        state, decision = standing.get(
            cast("str", proposal.get("proposal_id")), ("undecided", None)
        )
        if state == "conflicted":
            out.append(QueueItem(proposal, "conflicted", None))
        elif decision is None:
            out.append(QueueItem(proposal, "undecided", None))
        elif decision.get("verdict") == "defer":
            out.append(QueueItem(proposal, "deferred", cast("str", decision["decision_id"])))
    out.sort(
        key=lambda item: (
            str(item.proposal.get("trading_day")),
            str(item.proposal.get("proposal_id")),
        )
    )
    return out


def content_security_policy(nonce: str) -> str:
    """The page's policy: its own nonce'd script, inline style, and requests to itself only."""
    return (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; "
        "style-src 'unsafe-inline'; "
        "connect-src 'self'; "
        "form-action 'none'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'"
    )


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _options(values: Iterable[str], *, blank: str | None = None) -> str:
    head = f'<option value="">{_e(blank)}</option>' if blank is not None else ""
    return head + "".join(f'<option value="{_e(v)}">{_e(v)}</option>' for v in values)


_FIELDS = (
    ("trading_day", "Day"),
    ("tenant", "Tenant"),
    ("class", "Class"),
    ("nature", "Proposed nature"),
    ("action", "Proposed action"),
    ("rationale", "Why"),
)


def _row(item: QueueItem) -> str:
    p = item.proposal
    facts = "".join(f"<dt>{_e(label)}</dt><dd>{_e(p.get(field))}</dd>" for field, label in _FIELDS)
    title = f"{_e(p.get('class'))} on {_e(p.get('trading_day'))}"
    head = f"<h2>{title} <small>{_e(item.status)}</small></h2>"
    if item.status == "conflicted":
        body = (
            "<p>Two decisions stand side by side for this proposal. Record the next one with "
            "the command line, naming the decision it supersedes.</p>"
        )
    else:
        hidden = "".join(
            f'<input type="hidden" name="{name}" value="{_e(value)}">'
            for name, value in (
                ("proposal_id", p.get("proposal_id")),
                ("tenant", p.get("tenant")),
                ("supersedes", item.supersedes or ""),
            )
        )
        body = (
            f'<form class="decide">{hidden}'
            f'<label>Verdict <select name="verdict" required>{_options(VERDICTS)}</select></label> '
            '<label>Nature (on correct) <select name="nature">'
            f"{_options(NATURES, blank='-')}</select></label> "
            '<label>Action (on correct) <select name="action">'
            f"{_options(ACTIONS, blank='-')}</select></label> "
            '<label>Note <input name="note" maxlength="2000"></label> '
            '<button type="submit">Record</button> <output></output></form>'
        )
    return f'<section id="p-{_e(p.get("proposal_id"))}">{head}<dl>{facts}</dl>{body}</section>'


_SCRIPT = """
for (const form of document.querySelectorAll("form.decide")) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const out = form.querySelector("output");
    const body = {};
    for (const [name, value] of new FormData(form)) { if (value !== "") body[name] = value; }
    form.querySelector("button").disabled = true;
    out.textContent = "Recording...";
    try {
      const response = await fetch(window.location.pathname, {
        method: "POST", credentials: "same-origin",
        headers: {"content-type": "application/json"}, body: JSON.stringify(body)});
      const reply = await response.json();
      if (response.status === 201) {
        out.textContent = "Recorded: " + reply.verdict + " by " + reply.decided_by
          + " (" + reply.decision_id + ")";
        for (const field of form.elements) field.disabled = true;
      } else {
        out.textContent = "Refused (" + response.status + "): " + reply.error;
        form.querySelector("button").disabled = false;
      }
    } catch (error) {
      out.textContent = "Not sent: " + error;
      form.querySelector("button").disabled = false;
    }
  });
}
"""


def render_queue_page(
    items: Iterable[QueueItem], *, reviewer: str, tenants: Iterable[str], nonce: str
) -> str:
    """The whole HTML page for ``reviewer``; ``nonce`` must match the CSP header."""
    rows = "".join(_row(item) for item in items) or "<p>Nothing waits for a decision.</p>"
    scope = ", ".join(sorted(tenants)) or "none"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Back office decisions</title>"
        "<style>body{font:15px/1.45 system-ui,sans-serif;max-width:60rem;"
        "margin:1rem auto;padding:0 1rem}section{border-top:1px solid #8884;padding:.5rem 0}"
        "dl{display:grid;grid-template-columns:9rem 1fr;margin:.25rem 0}dt{opacity:.7}"
        "dd{margin:0}label{display:inline-block;margin:.25rem .5rem .25rem 0}"
        "small{font-weight:normal;opacity:.7}</style></head><body>"
        "<h1>Back office: decide the queue</h1>"
        f"<p>Signed in as <strong>{_e(reviewer)}</strong>, deciding for: {_e(scope)}. "
        "Each call is written once and never edited; a later call supersedes it "
        "and both stay on record. "
        "<em>Correct</em> needs a nature, an action and a note.</p>"
        f'{rows}<script nonce="{_e(nonce)}">{_SCRIPT}</script></body></html>'
    )
