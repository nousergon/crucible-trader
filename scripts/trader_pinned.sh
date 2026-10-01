#!/usr/bin/env bash
# Run one of the trader's daily entry points on the PINNED release, box-side
# (alpha-engine-config-I11545: the v2 trader runs on the executor box).
#
#   scripts/trader_pinned.sh daily_session [--run-mode live|replay]
#   scripts/trader_pinned.sh shadow_books_daily
#
# Same self-contained shape as `trader_reconcile.sh` and
# `trader_paper_smoke.sh`: a FRESH virtualenv per invocation, built from the
# release wheel `trader/release_pin` names (`crucible.release.TRADER_PIN_KEY`)
# plus this repository's locked `ib` extra, never this checkout's git pin.
#
# 1. Refuse any entry point not in the allowlist below: this script runs a
#    named module's `main`, never an arbitrary one.
# 2. Read `trader/release_pin`; refuse if it is unset (nothing is promoted).
# 3. Build the fresh venv and run `crucible_trader.<module>.main` inside it.
#
# `daily_session` connects to IB Gateway PAPER and runs the day's session.
# It is SHADOW (no order leaves the trader) unless
# CRUCIBLE_TRADER_ORDER_ROUTING=ib_paper is set, and nothing in this
# repository or its box units sets it (`crucible_trader.order_router`).
# `shadow_books_daily` never connects to the broker.
#
# Required environment (no defaults, by design): CRUCIBLE_TRADER_STORE_URI,
# and for daily_session also CRUCIBLE_TRADER_IB_HOST, CRUCIBLE_TRADER_IB_PORT,
# CRUCIBLE_TRADER_IB_CLIENT_ID. Exit status is the entry point's own.
set -euo pipefail

module="${1:-}"
case "$module" in
  daily_session) required=(CRUCIBLE_TRADER_STORE_URI CRUCIBLE_TRADER_IB_HOST CRUCIBLE_TRADER_IB_PORT CRUCIBLE_TRADER_IB_CLIENT_ID) ;;
  shadow_books_daily) required=(CRUCIBLE_TRADER_STORE_URI) ;;
  *)
    echo "usage: $0 daily_session|shadow_books_daily [args...] (got '${module}')" >&2
    exit 2
    ;;
esac
shift
for var in "${required[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "$var is unset; the trader has no default store or gateway" >&2
    exit 2
  fi
done

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

wheel="$(uv run --frozen --project "$repo" python - "$work" <<'PY'
import hashlib, os, sys
from crucible.release import TRADER_PIN_KEY, resolve_published_wheel
from crucible.store import open_store

from crucible_trader.pin_request import read_trader_pin

out = sys.argv[1]
store = open_store(os.environ["CRUCIBLE_TRADER_STORE_URI"])
sha = read_trader_pin(store)
if sha is None:
    raise SystemExit(f"{TRADER_PIN_KEY} is unset; no release is pinned to the trader")
published = resolve_published_wheel(store, sha)
payload = store.get_bytes(published.wheel_key)
digest = hashlib.sha256(payload).hexdigest()
if digest != published.record.wheel_sha256:
    raise SystemExit(f"wheel sha256 {digest} != release.json {published.record.wheel_sha256}")
name = published.wheel_key.rsplit("/", 1)[1]
with open(os.path.join(out, name), "wb") as handle:
    handle.write(payload)
print(name, sha)
PY
)"
pinned_sha="${wheel#* }"
wheel="${wheel%% *}"

uv export --frozen --project "$repo" --extra ib --no-emit-project --no-hashes --format requirements-txt \
  | grep -v '^crucible @' | grep -v '^ *#' > "$work/requirements.txt"
uv venv --quiet --python 3.12 "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" -r "$work/requirements.txt" "$work/$wheel"
uv pip install --quiet --python "$work/venv/bin/python" --no-deps "$repo"

# `crucible.runner.run_job` stamps `code_sha` from $CRUCIBLE_CODE_SHA, and
# falls back to `git rev-parse HEAD` inside the installed wheel's directory,
# which is no checkout. So without this the job refuses before it runs
# (measured on the box 2026-10-01: `CodeShaError: $CRUCIBLE_CODE_SHA is
# unset`). The code that runs IS the release wheel, so its sha is the
# answer. crucible's own job boxes export the same value
# (nous-ergon-ops crucible-v2.yaml, alpha-engine-config-I10454).
export CRUCIBLE_CODE_SHA="$pinned_sha"
"$work/venv/bin/python" -c \
  'import importlib, sys; sys.exit(importlib.import_module("crucible_trader." + sys.argv[1]).main(sys.argv[2:]))' \
  "$module" "$@"
