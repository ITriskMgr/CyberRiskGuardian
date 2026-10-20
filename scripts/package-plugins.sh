#!/usr/bin/env bash
# Build Cowork-installable .plugin files (zips of each plugin folder) into dist/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/dist"
for p in "$ROOT"/plugins/*/; do
  name="$(basename "$p")"
  rm -f "$ROOT/dist/$name.plugin"
  (cd "$p" && zip -qr "$ROOT/dist/$name.plugin" . -x 'evals/results/*' '*/__pycache__/*' '*.pyc')
  echo "dist/$name.plugin"
done
