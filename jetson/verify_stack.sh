#!/usr/bin/env bash
# verify_stack.sh — Pre-flight check for Qualia stack health
# Run before leaving an overnight session unattended.

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    local label=$1 expected=$2 actual=$3
    if [ "$actual" = "$expected" ]; then
        echo -e "  ${GREEN}✓${NC} $label: $actual (expected $expected)"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} $label: $actual (expected $expected)"
        ((FAIL++))
    fi
}

echo '=== Qualia Stack Verification ==='
echo ''

# 1. Process counts
echo '── Process Counts ──'
for svc in autonomous_explorer qualia_detect qualia_audio lore_store session_recorder; do
    count=$(ps aux | grep -c "[p]ython3.*${svc}" 2>/dev/null || echo 0)
    check "$svc" 1 "$count"
done

# qualia-watch (Rust binary)
watch_count=$(ps aux | grep -c "[q]ualia-watch" 2>/dev/null || echo 0)
check "qualia-watch" 1 "$watch_count"
echo ''

# 2. Serial port
echo '── Serial Port ──'
if [ -c /dev/ttyTHS1 ]; then
    serial_user=$(fuser /dev/ttyTHS1 2>/dev/null | xargs -I{} ps -p {} -o comm= 2>/dev/null | head -1)
    if echo "$serial_user" | grep -q python3; then
        echo -e "  ${GREEN}✓${NC} /dev/ttyTHS1 held by: $serial_user"
        ((PASS++))
    else
        echo -e "  ${YELLOW}!${NC} /dev/ttyTHS1 status: ${serial_user:-not held}"
    fi
else
    echo -e "  ${RED}✗${NC} /dev/ttyTHS1 not found"
    ((FAIL++))
fi

# 3. Conflicting services
echo ''
echo '── Conflicting Services ──'
for svc in jetson-read-serial jetson-capture-images; do
    if systemctl is-active --quiet $svc 2>/dev/null; then
        echo -e "  ${RED}✗${NC} $svc is RUNNING (conflicts with stack)"
        ((FAIL++))
    else
        echo -e "  ${GREEN}✓${NC} $svc stopped"
        ((PASS++))
    fi
done
echo ''

# 4. Motor state
echo '── Motor State ──'
if [ -f /tmp/qualia_motor_state.json ]; then
    motor_json=$(cat /tmp/qualia_motor_state.json)
    motor_ts=$(echo "$motor_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ts',0))" 2>/dev/null || echo 0)
    now=$(python3 -c 'import time; print(time.time())')
    age=$(python3 -c "print(int($now - $motor_ts))" 2>/dev/null || echo 999)
    echo "  Motor state: $motor_json"
    if [ "$age" -lt 30 ]; then
        echo -e "  ${GREEN}✓${NC} Updated ${age}s ago"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} Stale — ${age}s since last update"
        ((FAIL++))
    fi
else
    echo -e "  ${RED}✗${NC} /tmp/qualia_motor_state.json not found"
    ((FAIL++))
fi
echo ''

# 5. Learning health
echo '── Learning Health ──'
SESSION_DIR=$(ls -td ~/training-data/sessions/2026* 2>/dev/null | head -1)
if [ -n "$SESSION_DIR" ] && [ -f "$SESSION_DIR/qualia_beliefs.csv" ]; then
    last_line=$(tail -1 "$SESSION_DIR/qualia_beliefs.csv")
    l0_vfe=$(echo "$last_line" | cut -d, -f2)
    l1_vfe=$(echo "$last_line" | cut -d, -f6)
    l1_compression=$(echo "$last_line" | cut -d, -f7)
    l2_vfe=$(echo "$last_line" | cut -d, -f10)

    echo "  Session: $(basename $SESSION_DIR)"
    echo "  L0 VFE: $l0_vfe  |  L1 VFE: $l1_vfe (comp=$l1_compression)  |  L2 VFE: $l2_vfe"

    # Check if all VFE are zero (frozen state)
    all_zero=$(echo "$last_line" | cut -d, -f2,6,10 | python3 -c "
import sys
vals = sys.stdin.read().strip().split(',')
print('yes' if all(float(v) == 0.0 for v in vals) else 'no')
" 2>/dev/null || echo 'unknown')

    if [ "$all_zero" = "yes" ]; then
        echo -e "  ${RED}✗${NC} ALL VFE = 0 — layers are NOT learning (frozen state!)"
        ((FAIL++))
    else
        echo -e "  ${GREEN}✓${NC} VFE non-zero — layers are active"
        ((PASS++))
    fi

    row_count=$(wc -l < "$SESSION_DIR/qualia_beliefs.csv")
    echo "  Rows recorded: $row_count"
else
    echo -e "  ${RED}✗${NC} No active session found"
    ((FAIL++))
fi
echo ''

# 6. Disk space
echo '── Disk Space ──'
avail=$(df -h /home/jetson | tail -1 | awk '{print $4}')
use_pct=$(df /home/jetson | tail -1 | awk '{print $5}' | tr -d '%')
echo "  Available: $avail (${use_pct}% used)"
if [ "$use_pct" -gt 90 ]; then
    echo -e "  ${RED}✗${NC} Disk usage above 90%!"
    ((FAIL++))
else
    echo -e "  ${GREEN}✓${NC} Disk OK"
    ((PASS++))
fi
echo ''

# Summary
echo '══════════════════════════════'
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All $PASS checks passed. Safe for overnight run.${NC}"
else
    echo -e "${RED}$FAIL check(s) FAILED${NC}, $PASS passed."
    echo 'Fix issues before leaving unattended.'
fi
exit $FAIL
