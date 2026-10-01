#!/usr/bin/env bash
# The trader's paper smoke, box-side (alpha-engine-config-I10649 deliverable 2).
#
#   scripts/trader_paper_smoke.sh <40-char release sha>
#
# 1. Resolve and download the release wheel for <sha> through crucible's own
#    reader (release.json names the wheel), and verify its sha256 against the
#    record.
# 2. Build a FRESH virtualenv and install that wheel plus this repository (no
#    deps) plus the locked `ib` extra. The crucible under test is the published
#    wheel, never this checkout's git pin.
# 3. Run `crucible_trader.paper_smoke.main --release <sha>` inside it: it asserts
#    the installed crucible is that wheel, connects to IB Gateway PAPER read-only,
#    reads positions and cash, places no order, and files
#    runs/trader.smoke/{day}/{sha12}/run.json -- the manifest
#    `crucible release.pin --target trader <sha>` is gated on.
#
# Required environment (no defaults, by design): CRUCIBLE_TRADER_STORE_URI,
# CRUCIBLE_TRADER_IB_HOST, CRUCIBLE_TRADER_IB_PORT, CRUCIBLE_TRADER_IB_CLIENT_ID.
# Exit status is the smoke's: non-zero whenever the manifest is not `ok`.
set -euo pipefail

sha="${1:-}"
if [[ ! "$sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <40-character lowercase release sha> (got '${sha}')" >&2
  exit 2
fi
for var in CRUCIBLE_TRADER_STORE_URI CRUCIBLE_TRADER_IB_HOST CRUCIBLE_TRADER_IB_PORT CRUCIBLE_TRADER_IB_CLIENT_ID; do
  if [[ -z "${!var:-}" ]]; then
    echo "$var is unset; the smoke has no default store or gateway" >&2
    exit 2
  fi
done

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

wheel="$(uv run --frozen --project "$repo" python - "$sha" "$work" <<'PY'
import hashlib, os, sys
from crucible.release import resolve_published_wheel
from crucible.store import open_store

sha, out = sys.argv[1], sys.argv[2]
store = open_store(os.environ["CRUCIBLE_TRADER_STORE_URI"])
published = resolve_published_wheel(store, sha)
payload = store.get_bytes(published.wheel_key)
digest = hashlib.sha256(payload).hexdigest()
if digest != published.record.wheel_sha256:
    raise SystemExit(f"wheel sha256 {digest} != release.json {published.record.wheel_sha256}")
name = published.wheel_key.rsplit("/", 1)[1]
with open(os.path.join(out, name), "wb") as handle:
    handle.write(payload)
print(name)
PY
)"

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
export CRUCIBLE_CODE_SHA="$sha"
"$work/venv/bin/python" -c 'import sys; from crucible_trader.paper_smoke import main; sys.exit(main(sys.argv[1:]))' --release "$sha"
