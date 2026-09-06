#!/usr/bin/env bash
# sync_from_upstream.sh — Sync standalone bpd-substrate to LlamaTov's vendored copy
#
# Runs through the validation gate: only syncs validated changes.
# Triggered manually or on census milestones.
#
# Usage: ./sync_from_upstream.sh [--dry-run]

set -euo pipefail

UPSTREAM_REPO="https://github.com/heath-hunnicutt-ruach-tov/bpd-substrate.git"
UPSTREAM_DIR="/tmp/bpd-substrate-sync"
VENDOR_DIR="bpd-substrate"
DRY_RUN="${1:-}"

echo "=== LlamaTov ← bpd-substrate sync ==="
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Step 1: Fetch upstream
if [ -d "$UPSTREAM_DIR" ]; then
    cd "$UPSTREAM_DIR" && git pull -q && cd -
else
    git clone --depth 5 "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi

# Step 2: Detect changes
UPSTREAM_HEAD=$(cd "$UPSTREAM_DIR" && git log -1 --format='%h %s')
VENDOR_MAKEFILE_HASH=$(md5sum "$VENDOR_DIR/Makefile" 2>/dev/null | cut -d' ' -f1)
UPSTREAM_MAKEFILE_HASH=$(md5sum "$UPSTREAM_DIR/Makefile" 2>/dev/null | cut -d' ' -f1)

if [ "$VENDOR_MAKEFILE_HASH" = "$UPSTREAM_MAKEFILE_HASH" ]; then
    echo "  No Makefile changes. Checking other files..."
fi

# Step 3: Diff summary
CHANGED=$(diff -rq "$VENDOR_DIR/" "$UPSTREAM_DIR/" \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='build' --exclude='*.so' --exclude='*.o' \
    2>/dev/null | grep -c 'differ\|Only' || true)

echo "  Upstream HEAD: $UPSTREAM_HEAD"
echo "  Files differing: $CHANGED"

if [ "$CHANGED" -eq 0 ]; then
    echo "  Already in sync. Nothing to do."
    exit 0
fi

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "  DRY RUN — would sync $CHANGED files"
    diff -rq "$VENDOR_DIR/" "$UPSTREAM_DIR/" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='build' --exclude='*.so' --exclude='*.o' \
        2>/dev/null | head -20
    exit 0
fi

# Step 4: Sync (rsync, preserving structure)
rsync -a --delete \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='build' --exclude='*.so' --exclude='*.o' \
    "$UPSTREAM_DIR/" "$VENDOR_DIR/"

echo "  Synced $CHANGED files from upstream"

# Step 5: Stage + report
git add "$VENDOR_DIR/"
echo ""
echo "=== Staged. Review with: git diff --cached --stat ==="
echo "=== Commit with proper attribution when ready ==="
