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

from crucible_trader.exception_decision import effective_decisions
from crucible_trader.exception_plain import (
    ACTION_TEXT,
    NATURE_TEXT,
    headline,
    side_figures,
    sub,
)
from crucible_trader.exception_proposal import ACTIONS, NATURES

__all__ = [
    "ACTION_TEXT",
    "NATURE_TEXT",
    "QueueItem",
    "content_security_policy",
    "headline",
    "queue_items",
    "render_queue_page",
]


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


#: The nature choices as the reviewer reads them in "Change the fix".
_NATURE_CHOICE: dict[str, str] = {
    "event": "A real event at the broker",
    "defect": "A mistake on our side",
    "detector": "Our checker misfired or lacked data",
}


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _options(
    values: Iterable[str], *, blank: str | None = None, words: Mapping[str, str] | None = None
) -> str:
    """``<option>`` tags whose value is the code the writer takes and whose text
    is the plain sentence the reviewer reads (a code is the fallback)."""
    head = f'<option value="">{_e(blank)}</option>' if blank is not None else ""
    said = words or {}
    return head + "".join(f'<option value="{_e(v)}">{_e(said.get(v, v))}</option>' for v in values)


def _sides(record: Mapping[str, object] | None) -> str:
    cells = side_figures(record)
    if not cells:
        return ""
    body = "".join(f'<div class="cell"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in cells)
    return f'<div class="sides">{body}</div>'


def _evidence(record: Mapping[str, object] | None) -> str:
    if not record:
        return ""
    rows: list[tuple[str, object]] = []
    for side, label in (
        ("expected", "Our figure comes from"),
        ("actual", "IB's figure comes from"),
    ):
        key = sub(record, side).get("record")
        if key:
            rows.append((label, key))
    manifest = sub(record, "detected_by").get("manifest_key")
    if manifest:
        rows.append(("The check's run record", manifest))
    if not rows:
        return ""
    items = "".join(f"<li>{_e(label)}: <code>{_e(key)}</code></li>" for label, key in rows)
    return f"<details><summary>Evidence</summary><ul>{items}</ul></details>"


def _row(item: QueueItem, record: Mapping[str, object] | None = None) -> str:
    p = item.proposal
    action = cast("str", p.get("action") or "")
    nature = cast("str", p.get("nature") or "")
    status = {"undecided": "", "deferred": " · you chose Not now earlier", "conflicted": ""}.get(
        item.status, f" · {item.status}"
    )
    head = (
        f"<h2>{_e(headline(p, record))}</h2>"
        f'<p class="meta">Found on {_e(p.get("trading_day"))}{_e(status)}</p>'
    )
    what = (record or {}).get("detail")
    what_html = f"<p><b>What the check saw:</b> {_e(what)}</p>" if what else ""
    likely = NATURE_TEXT.get(nature)
    likely_html = f"<p><b>Most likely:</b> {_e(likely)}.</p>" if likely else ""
    fix = ACTION_TEXT.get(action, action)
    fix_html = (
        f'<p class="fix"><b>Proposed fix:</b> {_e(fix)}'
        f'<br><span class="why">Why: {_e(p.get("rationale"))}</span></p>'
    )
    technical = "".join(
        f"<dt>{_e(label)}</dt><dd>{_e(p.get(field))}</dd>" for field, label in _FIELDS
    )
    tech_html = f"<details><summary>Technical details</summary><dl>{technical}</dl></details>"
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
                ("verdict", ""),
            )
        )
        body = (
            f'<form class="decide">{hidden}'
            '<label class="note">Note (optional; required for Change) '
            '<input name="note" maxlength="2000"></label>'
            '<div class="buttons">'
            '<button type="submit" class="approve" data-verdict="agree">Approve the fix</button>'
            '<button type="submit" data-verdict="defer">Not now</button></div>'
            "<details><summary>Change the fix</summary>"
            '<label>What it really is <select name="nature">'
            f"{_options(NATURES, blank='(keep)', words=_NATURE_CHOICE)}</select></label> "
            '<label>What to do instead <select name="action">'
            f"{_options(ACTIONS, blank='(keep)', words=ACTION_TEXT)}</select></label> "
            '<button type="submit" data-verdict="correct">Save my change</button></details>'
            "<output></output></form>"
        )
    sides = _sides(record)
    return (
        f'<section class="card" id="p-{_e(p.get("proposal_id"))}">{head}{sides}{what_html}'
        f"{likely_html}{fix_html}{_evidence(record)}{body}{tech_html}</section>"
    )


_FIELDS = (
    ("trading_day", "Day"),
    ("tenant", "Tenant"),
    ("class", "Class"),
    ("nature", "Proposed nature"),
    ("action", "Proposed action"),
    ("rationale", "Why"),
    ("proposal_id", "Proposal id"),
    ("exception_id", "Exception id"),
)


_SCRIPT = """
for (const form of document.querySelectorAll("form.decide")) {
  for (const button of form.querySelectorAll("button[data-verdict]")) {
    button.addEventListener("click", () => {
      form.querySelector("input[name=verdict]").value = button.dataset.verdict;
    });
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const out = form.querySelector("output");
    const body = {};
    for (const [name, value] of new FormData(form)) { if (value !== "") body[name] = value; }
    for (const b of form.querySelectorAll("button")) b.disabled = true;
    out.textContent = "Recording...";
    try {
      const response = await fetch(window.location.pathname, {
        method: "POST", credentials: "same-origin",
        headers: {"content-type": "application/json"}, body: JSON.stringify(body)});
      const reply = await response.json();
      if (response.status === 201) {
        const said = {agree: "Approved", defer: "Set aside for now", correct: "Changed"};
        out.textContent = (said[reply.verdict] || reply.verdict) + " by " + reply.decided_by
          + ". Saved permanently.";
        form.closest("section").classList.add("done");
        for (const field of form.elements) field.disabled = true;
      } else {
        out.textContent = "Not saved: " + reply.error;
        for (const b of form.querySelectorAll("button")) b.disabled = false;
      }
    } catch (error) {
      out.textContent = "Not sent: " + error;
      for (const b of form.querySelectorAll("button")) b.disabled = false;
    }
  });
}
"""


def render_queue_page(
    items: Iterable[QueueItem],
    *,
    reviewer: str,
    tenants: Iterable[str],
    nonce: str,
    exceptions: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    """The whole HTML page for ``reviewer``; ``nonce`` must match the CSP header.

    ``exceptions`` (exception id -> ``exception.v1`` record,
    :func:`~crucible_trader.exception_decision.load_exception_records`) supplies
    each card's two sides, amount, what the check saw and the evidence; without
    it a card shows the proposal alone."""
    found = exceptions or {}
    listed = list(items)
    rows = (
        "".join(
            _row(item, found.get(cast("str", item.proposal.get("exception_id") or "")))
            for item in listed
        )
        or "<p>Nothing waits for a decision. The books agree with IB.</p>"
    )
    waiting = len(listed)
    summary = (
        f'<p class="summary"><b>{waiting} difference{"s" if waiting != 1 else ""} '
        "between our books "
        "and IB wait for you.</b> For each one, read what is different and the proposed fix, then "
        "approve it, change it, or set it aside. Nothing is booked until you decide.</p>"
        if waiting
        else ""
    )
    books_line = (
        f'<p class="books"><b>Are the books right today?</b> Not yet: {waiting} '
        f"difference{'s' if waiting != 1 else ''} between our books and IB "
        "still need a decision.</p>"
        if waiting
        else '<p class="books"><b>Are the books right today?</b> '
        "Yes: nothing waits for a decision.</p>"
    )
    scope = ", ".join(sorted(tenants)) or "none"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Back office decisions</title>"
        "<style>:root{color-scheme:light dark}body{font:16px/1.5 system-ui,sans-serif;"
        "max-width:44rem;margin:1rem auto;padding:0 1rem}"
        ".card{border:1px solid #8886;border-radius:12px;padding:1rem;margin:1rem 0}"
        ".card.done{opacity:.55}.card h2{font-size:1.15rem;margin:0 0 .25rem}"
        ".meta{margin:0 0 .75rem;opacity:.7;font-size:.9rem}"
        ".sides{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0}"
        ".cell{flex:1 1 8rem;border:1px solid #8884;border-radius:8px;padding:.5rem}"
        ".cell span{display:block;font-size:.8rem;opacity:.7}.cell b{font-size:1.1rem}"
        ".fix{background:#8881;border-radius:8px;padding:.5rem .75rem}"
        ".why{opacity:.75;font-size:.9rem}.books{font-size:1.05rem}"
        ".buttons{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0}"
        "button{font:inherit;padding:.5rem 1rem;border-radius:8px;"
        "border:1px solid #8888;cursor:pointer}"
        "button.approve{background:#1f7a3a;color:#fff;border-color:#1f7a3a}"
        "label{display:inline-block;margin:.25rem .5rem .25rem 0}"
        ".note input{width:100%;max-width:30rem}"
        "details{margin:.5rem 0}summary{cursor:pointer;opacity:.8}"
        "dl{display:grid;grid-template-columns:9rem 1fr;margin:.25rem 0;font-size:.85rem}"
        "dt{opacity:.7}dd{margin:0;overflow-wrap:anywhere}output{display:block;font-weight:600}"
        "</style></head><body>"
        "<h1>Back office: differences to decide</h1>"
        f"{books_line}{summary}"
        f'<p class="meta">Signed in as <strong>{_e(reviewer)}</strong>, deciding for: {_e(scope)}. '
        "Each decision is saved permanently; a later one replaces it and both stay on record.</p>"
        f'{rows}<script nonce="{_e(nonce)}">{_SCRIPT}</script></body></html>'
    )
