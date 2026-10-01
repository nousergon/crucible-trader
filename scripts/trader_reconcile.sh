#!/usr/bin/env bash
# The daily broker-reconciliation run, box-side (alpha-engine-config-I11071).
#
#   scripts/trader_reconcile.sh [--genesis]
#
# Same self-contained shape as `trader_paper_smoke.sh` (I10649 deliverable 2):
# a FRESH virtualenv per invocation, the pinned release wheel plus this
# repository's locked `ib` extra, never this checkout's git pin. Unlike the
# smoke, the release under test is not an argument -- `trader/release_pin`
# (`crucible.release.TRADER_PIN_KEY`) IS what the trader runs, so this script
# reads it itself rather than taking it on the command line.
#
# 1. Read `trader/release_pin`; refuse if it is unset (nothing is promoted).
# 2. Build the fresh venv exactly as the smoke does.
# 3. Run `crucible_trader.commands reconcile run [--genesis]` inside it: it
#    connects to IB Gateway PAPER **read-only**, reads the book and today's
#    fills, and files `runs/trader.reconcile/{day}/run.json` --
#    `crucible.manifest.MONEY_PATH_PREDICATES` reads that manifest's outputs.
#
# `--genesis` is for the ONE declared-genesis run only (no prior broker
# statement exists yet) -- never pass it from the scheduled timer, which
# calls this script with no arguments.
#
# Required environment (no defaults, by design): CRUCIBLE_TRADER_STORE_URI,
# CRUCIBLE_TRADER_IB_HOST, CRUCIBLE_TRADER_IB_PORT, CRUCIBLE_TRADER_IB_CLIENT_ID.
# Exit status is the reconcile's: non-zero whenever the manifest is not `ok`
# (a discrepancy, a void control arm, or an unreadable book) -- the page
# condition, never swallowed here.
set -euo pipefail

genesis_flag=()
if [[ "${1:-}" == "--genesis" ]]; then
  genesis_flag=(--genesis)
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--genesis] (got '${1}')" >&2
  exit 2
fi
for var in CRUCIBLE_TRADER_STORE_URI CRUCIBLE_TRADER_IB_HOST CRUCIBLE_TRADER_IB_PORT CRUCIBLE_TRADER_IB_CLIENT_ID; do
  if [[ -z "${!var:-}" ]]; then
    echo "$var is unset; reconciliation has no default store or gateway" >&2
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
  'import sys; from crucible_trader.commands import main; sys.exit(main(sys.argv[1:]))' \
  reconcile run "${genesis_flag[@]}"
