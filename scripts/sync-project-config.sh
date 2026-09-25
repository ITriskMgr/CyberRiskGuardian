#!/usr/bin/env bash
# Refresh the project-level .claude/ components from the plugin (single source of truth).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/plugins/cyberriskguardian"
DST="$ROOT/.claude"
for d in skills commands agents output-styles; do
  rm -rf "$DST/$d"
  cp -R "$SRC/$d" "$DST/$d"
done
cp "$SRC/hooks/session-context.md" "$DST/hooks/session-context.md"
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} +
echo "Synced skills, commands, agents, output-styles and session context from $SRC to $DST"
