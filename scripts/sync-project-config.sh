#!/usr/bin/env bash
# Refresh the project-level .claude/ components from the plugins (single source of truth).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORE="$ROOT/plugins/cyberriskguardian"
BOOK="$ROOT/plugins/crg-governance-textbook"
DST="$ROOT/.claude"
for d in skills commands agents output-styles; do
  rm -rf "$DST/$d"
  cp -R "$CORE/$d" "$DST/$d"
done
cp -R "$BOOK/skills/cybersecurity-governance" "$DST/skills/cybersecurity-governance"
cp "$CORE/hooks/session-context.md" "$DST/hooks/session-context.md"
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} +
echo "Synced Core skills, commands, agents, output styles, the governance textbook skill and the session context into $DST"
