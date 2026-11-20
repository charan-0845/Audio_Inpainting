#!/usr/bin/env bash
# Usage: scripts/run_worker.sh <gateway-host> [name]
# Linux/Bash. Windows workers should run Celery with --pool=solo.
set -euo pipefail
GATEWAY="${1:?gateway host}"; NAME="${2:-w1}"
export DPAI_REDIS="${DPAI_REDIS:-redis://${GATEWAY}:6379/0}"
export DPAI_DEVICE="${DPAI_DEVICE:-cpu}"
export DPAI_TORCH_THREADS="${DPAI_TORCH_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
exec celery -A src.cluster.worker worker --concurrency=1 --loglevel=INFO -n "${NAME}@%h"
