#!/usr/bin/env bash
# Install discopt Claude Code skills (slash commands).
#
# Creates .claude/commands/ in the repo root and symlinks each skill file
# so that Claude Code picks them up as /command-name slash commands.
#
# Usage:
#   bash claude-skills/install.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_ROOT/claude-skills/commands"
DEST_DIR="$REPO_ROOT/.claude/commands"

mkdir -p "$DEST_DIR"

count=0
for src in "$SRC_DIR"/*.md; do
    name="$(basename "$src")"
    dest="$DEST_DIR/$name"
    if [ -L "$dest" ] || [ -e "$dest" ]; then
        echo "  skip  $name (already exists)"
    else
        ln -s "$src" "$dest"
        echo "  link  $name"
        count=$((count + 1))
    fi
done

echo ""
echo "Installed $count skill(s) into .claude/commands/"
echo "Available as slash commands: $(ls "$SRC_DIR"/*.md | xargs -I{} basename {} .md | paste -sd', ' -)"
