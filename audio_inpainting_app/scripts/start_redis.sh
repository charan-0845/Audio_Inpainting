#!/usr/bin/env bash
# Run on the gateway. Usage: scripts/start_redis.sh [password]
set -euo pipefail
PASS="${1:-}"
ARGS=(--port 6379 --bind 127.0.0.1 --protected-mode yes --save "" --appendonly no)
if [[ -n "${PASS}" ]]; then
  ARGS+=(--requirepass "${PASS}")
fi
exec redis-server "${ARGS[@]}"
