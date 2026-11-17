#!/usr/bin/env bash
# Builds a synthetic fixture folder and runs the full pipeline on it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS="${1:-$ROOT/demo_workspace}"
SRC="${WS}_images"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
CLI="${IMAGE_PIPELINE:-$ROOT/.venv/bin/image-pipeline}"

if [ ! -d "$SRC" ]; then
  "$PY" "$ROOT/scripts/make_fixtures.py" "$SRC"
fi
"$CLI" run "$SRC" --workspace "$WS" --rights-status user_attested --license "synthetic test data"
"$CLI" status --workspace "$WS"
echo "Dataset: $WS/processed_dataset"
