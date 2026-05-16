#!/bin/bash
# Fetch Mesa upstream commit info without full clone.
# Uses git ls-remote + archive for path verification.
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

MESA_URL="https://gitlab.freedesktop.org/mesa/mesa.git"
OUT_DIR="$(dirname "$0")/.."

echo "[mesa-fetch] Querying upstream HEAD..."
HEAD_SHA=$(git ls-remote "$MESA_URL" HEAD | cut -f1)
echo "[mesa-fetch] HEAD: $HEAD_SHA"

# Get latest tag for version
LATEST_TAG=$(git ls-remote --tags --sort=-v:refname "$MESA_URL" 'mesa-*' | head -1 | sed 's|.*refs/tags/||')
echo "[mesa-fetch] Latest tag: $LATEST_TAG"

echo "{
  \"mesa_upstream\": {
    \"url\": \"$MESA_URL\",
    \"tracking\": \"main\",
    \"pinned_commit\": \"$HEAD_SHA\",
    \"pinned_version\": \"$LATEST_TAG\",
    \"pin_date\": \"$(date -u +%Y-%m-%d)\",
    \"required_paths_verified\": false
  }
}" > "$OUT_DIR/upstream.lock.json"

echo "[mesa-fetch] upstream.lock.json updated"
