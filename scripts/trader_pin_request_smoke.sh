#!/usr/bin/env bash
# Smoke the QUEUED trader pin, box-side (alpha-engine-config-I11545).
#
#   scripts/trader_pin_request_smoke.sh
#
# Brian's ruling: anyone may queue a sha at any time (crucible's
# `trader-pin.yml` dispatch writes `trader/pin_request.json`); the next clean
# post-close evening pins it (the same workflow's schedule), and that pin is
# gated on THIS trader's passing paper smoke for the sha. The harness may not
# reach into the trader, so the box smokes the request itself, on the executor
# box's alpha-engine-trader-pin-smoke timer.
#
# 1. Read the request and the live pin (`crucible_trader.pin_request`, through
#    this checkout's locked environment -- the same way `trader_pinned.sh`
#    reads the pin, with the same CRUCIBLE_TRADER_STORE_URI).
# 2. `none` / `noop` / `stale`: print the one-line reason and exit 0. Nothing
#    is owed: nothing was queued, the pin already names it, or the pin moved
#    after the request and the harness will refuse to apply it.
# 3. `fresh`: exec `trader_paper_smoke.sh <sha>`, which files the
#    `runs/trader.smoke/...` manifest the pin is gated on. Its exit status is
#    this script's.
#
# A request that cannot be READ -- a 403 because the trader identity lacks the
# grant, a malformed document -- exits non-zero with the reason. It is never
# read as "no request", which would be a quiet day forever.
#
# Required environment (no defaults, by design): CRUCIBLE_TRADER_STORE_URI, and
# for a fresh request everything `trader_paper_smoke.sh` requires.
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0 (takes no arguments; the sha comes from the queued request)" >&2
  exit 2
fi
if [[ -z "${CRUCIBLE_TRADER_STORE_URI:-}" ]]; then
  echo "CRUCIBLE_TRADER_STORE_URI is unset; the trader has no default store" >&2
  exit 2
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

decision="$(uv run --frozen --project "$repo" python -c \
  'import sys; from crucible_trader.pin_request import main; sys.exit(main(sys.argv[1:]))')"
read -r state sha reason <<<"$decision"

case "$state" in
  none|noop|stale)
    echo "trader-pin-request: ${state}: ${reason}"
    exit 0
    ;;
  fresh)
    if [[ ! "$sha" =~ ^[0-9a-f]{40}$ ]]; then
      echo "trader-pin-request: a fresh request named '${sha}', not a 40-hex sha" >&2
      exit 1
    fi
    echo "trader-pin-request: fresh: ${reason}; smoking ${sha}"
    exec "$repo/scripts/trader_paper_smoke.sh" "$sha"
    ;;
  *)
    echo "trader-pin-request: unrecognised decision '${decision}'" >&2
    exit 1
    ;;
esac
