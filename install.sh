#!/usr/bin/env bash
# Copy only the plugin source tree (not training/) into AstrBot's data dir.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/astrbot_plugin_img_curator/astrbot_plugin_img_curator"
DATA_DIR="${ASTRBOT_DATA_DIR:-$HOME/astrbot/data}"
DEST="$DATA_DIR/plugins/astrbot_plugin_img_curator"

if [[ ! -f "$SRC/main.py" ]]; then
  echo "plugin source not found: $SRC" >&2
  exit 1
fi

ARGS=(-a --no-owner --no-group --chmod=D755,F644 --delete
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache')
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  ARGS+=(--dry-run --itemize-changes)
fi

RSYNC=(rsync)
SUDO_CHOWN=0
mkdir -p "$DEST" 2>/dev/null || true
if [[ ! -w "$DEST" ]]; then
  RSYNC=(sudo rsync)
  SUDO_CHOWN=1
fi

echo "source: $SRC"
echo "target: $DEST"
"${RSYNC[@]}" "${ARGS[@]}" "$SRC/" "$DEST/"
if [[ "$SUDO_CHOWN" == "1" && "$DRY_RUN" == "0" ]]; then
  sudo chown -R root:root "$DEST"
fi

echo "copied plugin only; training/ and tests/ were not included."
echo "restart AstrBot (or reload the plugin) to apply changes."
