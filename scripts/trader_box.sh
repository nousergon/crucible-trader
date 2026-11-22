#!/usr/bin/env bash
# Run one of the v2 box's alert / watchdog modules from this checkout
# (alpha-engine-config-I14129, part of alpha-engine-config-I14079):
#
#   scripts/trader_box.sh trade_alert|market_alert|release_alert|session_watchdog|day_end [args...]
#
# Runs `python -m crucible_trader.box.<module>` in this repository's locked
# environment. The module name is checked against a literal allow-list first:
# this script runs a named module, never an arbitrary one. Nothing on the box
# calls it yet; the units that will are a later part of the same arc.
set -euo pipefail

module="${1:-}"
case "$module" in
  trade_alert | market_alert | release_alert | session_watchdog | day_end) ;;
  *)
    echo "usage: $0 trade_alert|market_alert|release_alert|session_watchdog|day_end [args...] (got '${module}')" >&2
    exit 2
    ;;
esac
shift
exec uv run --frozen --project "$HOME/crucible-trader" python -m "crucible_trader.box.${module}" "$@"
