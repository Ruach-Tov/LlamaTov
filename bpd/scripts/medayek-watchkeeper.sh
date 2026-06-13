#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# medayek-watchkeeper.sh — Situational awareness pulse for medayek's watch
# Gathers signals from the environment and composes a news tick.
# Called by shamash scheduler every 10 minutes.

set +e  # Don't exit on non-zero — many checks legitimately fail

REPO="/home/medayek/Ruach-Tov"
LAST_TICK_FILE="/tmp/medayek_last_tick"
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Track last tick time
if [ -f "$LAST_TICK_FILE" ]; then
    SINCE=$(cat "$LAST_TICK_FILE")
else
    SINCE=$(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%SZ)
fi
echo "$NOW" > "$LAST_TICK_FILE"

REPORT=""

# ─── GIT ACTIVITY ───
cd "$REPO" 2>/dev/null && {
    # New commits since last tick
    NEW_COMMITS=$(git log --oneline --since="$SINCE" 2>/dev/null | head -5)
    if [ -n "$NEW_COMMITS" ]; then
        COUNT=$(echo "$NEW_COMMITS" | wc -l)
        AUTHORS=$(git log --format='%an' --since="$SINCE" 2>/dev/null | sort -u | tr '\n' ', ' | sed 's/, $//')
        REPORT="${REPORT}📦 GIT: ${COUNT} new commit(s) by ${AUTHORS}\n"
        echo "$NEW_COMMITS" | while read -r line; do
            REPORT="${REPORT}  ${line}\n"
        done
    fi

    # Untracked files
    UNTRACKED=$(git status --porcelain 2>/dev/null | grep '^??' | head -5)
    if [ -n "$UNTRACKED" ]; then
        UCOUNT=$(echo "$UNTRACKED" | wc -l)
        REPORT="${REPORT}📂 UNTRACKED: ${UCOUNT} new file(s)\n"
        echo "$UNTRACKED" | head -3 | while read -r line; do
            REPORT="${REPORT}  ${line}\n"
        done
    fi
} || true

# ─── MEETING ACTIVITY ───
MEETING_COUNT=$(redis-cli XLEN ruach:stream:meeting 2>/dev/null || echo "0")
# Check for recent meeting messages (last 10 min)
RECENT_MEETING=$(redis-cli XREVRANGE ruach:stream:meeting + - COUNT 3 2>/dev/null | grep -c "sender" || echo "0")
if [ "$RECENT_MEETING" -gt 0 ] 2>/dev/null; then
    REPORT="${REPORT}💬 MEETING: ${RECENT_MEETING} recent message(s) in stream\n"
fi

# ─── AGENTS ONLINE ───
AGENTS=$(redis-cli KEYS 'ruach:stream:*' 2>/dev/null | sed 's/ruach:stream://' | grep -v meeting | sort -u | tr '\n' ', ' | sed 's/, $//')
if [ -n "$AGENTS" ] && [ "$AGENTS" != "," ]; then
    REPORT="${REPORT}👥 AGENTS WITH STREAMS: ${AGENTS}\n"
fi

# ─── /tmp ARTIFACTS (code-write) ───
NEW_TMP=$(find /tmp -maxdepth 3 -newer "$LAST_TICK_FILE" -type f \( -name "*.py" -o -name "*.pl" -o -name "*.ll" -o -name "*.c" -o -name "*.json" -o -name "*.npz" \) 2>/dev/null | head -5)
if [ -n "$NEW_TMP" ]; then
    TCOUNT=$(echo "$NEW_TMP" | wc -l)
    REPORT="${REPORT}🔧 /tmp ARTIFACTS: ${TCOUNT} new file(s)\n"
    echo "$NEW_TMP" | head -3 | while read -r line; do
        REPORT="${REPORT}  ${line}\n"
    done
fi

# ─── WEATHER (Tel Aviv / Heath's location) ───
# Uses wttr.in one-liner format, no API key needed
WEATHER=$(curl -s --max-time 3 "wttr.in/?format=%c+%t+%w+%h" 2>/dev/null || echo "")
if [ -n "$WEATHER" ]; then
    REPORT="${REPORT}🌤️ WEATHER: ${WEATHER}\n"
fi

FORECAST=$(curl -s --max-time 3 "wttr.in/?format=%c+%t+tomorrow" 2>/dev/null || echo "")

# ─── HEATH'S ONE-LINERS (~1/8 chance) ───
ONELINERS=(
    '"Consistency is structural, not disciplinary." — Declarative specifications in the same form are applied as inputs to the same template, resulting in the same output phrasings, always.'
    '"The substrate refuses to pass invalid claims." — ULP verification, locator validation, compute_graph_invariants: same discipline, different surfaces.'
    '"Do NOT perform for the audience. Keep the work TRUE."'
    '"Build the consumer before the producer." — The harness ships before the kernel.'
    '"Operation ORDER matters: (x+3)/6 ≠ x/6+0.5 for floating-point precision."'
    '"Source in git, artifacts in output directories, nothing in /tmp/."'
)
# Show a one-liner with ~1/8 probability
SHOW_ONELINER=$((RANDOM % 8))
if [ "$SHOW_ONELINER" -eq 0 ]; then
    RANDOM_IDX=$((RANDOM % ${#ONELINERS[@]}))
    ONELINER="${ONELINERS[$RANDOM_IDX]}"
else
    ONELINER=""
fi

# ─── COMPOSE THE TICK ───
if [ -z "$REPORT" ]; then
    REPORT="All quiet on the watch. No new activity since last tick."
fi

if [ -n "$ONELINER" ]; then
    echo -e "🕯️ WATCHKEEPER REPORT (${NOW})\n${REPORT}\n💡 ${ONELINER}"
else
    echo -e "🕯️ WATCHKEEPER REPORT (${NOW})\n${REPORT}"
fi
