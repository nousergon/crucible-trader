# Security policy

## Reporting a vulnerability

Report privately to **cipher813@gmail.com**. Do not open a public issue, and do
not describe the finding in a pull request body.

Expect an acknowledgement within three business days and a disposition — fixed,
accepted with a named residual, or disputed with a reason — within fourteen.

## What this repository is, in threat-model terms

`crucible-trader` is on the **money path**. It holds broker connection handling,
the kill switch, the account guard, reconciliation tolerances and the sizing
path. A defect here does not produce a wrong verdict; it produces a wrong order.

That is why the repository is private, and why the following are never committed
to it under any circumstance:

- credentials, API keys, tokens, broker passwords, session files
- account numbers, account allowlists with real values, ARNs, instance ids
- tuned risk, exit or sizing constants

Config files with real values use the `.example` pattern with the real file
gitignored. Real values live in SSM or the strategy tree, loaded at runtime.

## The controls that are not documentation

- **Refusal, not degradation.** An unusable or unattested champion stops the
  trader with a non-zero exit and a page. There is no fallback path to serve.
- **Read-only identity.** The trader's AWS identity may read `champions/*`,
  `predictions/*` and `runs/*` and may write exactly one prefix — its own
  evidence. It cannot move a champion pointer, publish a feed, or write a
  release. A trader that could promote what it trades is not a trader.
- **Zero LLM calls.** No LLM dependency, no call site, and a registry row
  declaring so.
- **Supply chain.** `uv.lock` is the dependency graph; CI runs `uv lock --check`
  before anything installs and `pip-audit` over the exported lockfile as a
  blocking gate.

## Credential exposure

If a credential reaches a log, an output or this repository, follow
`security-posture-policy` rather than rotating reflexively — but note that the
policy's calibration **inverts** for money-path assets, which is what everything
in this repository is.
