#!/usr/bin/env bash
set -euo pipefail

# preflight.sh — validate environment before starting a pipeline batch
# Usage: bin/preflight.sh video1.mp4 video2.mp4 ...

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; ERRORS=$((ERRORS + 1)); }

ERRORS=0

echo "=== GoMoveShift Pre-flight Check ==="
echo ""

# ── Check video files ──
echo "Video files:"
for f in "$@"; do
    if [[ ! -f "$f" ]]; then
        fail "File not found: $f"
        continue
    fi

    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
    size_gb=$(echo "scale=1; $size / 1073741824" | bc)

    if (( size < 1073741824 )); then
        fail "$f — ${size_gb}GB (expected > 1GB for 4K video)"
    else
        pass "$f — ${size_gb}GB"
    fi

    # Check bitrate via ffprobe
    bitrate=$(ffprobe -v quiet -show_format -print_format json "$f" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
br = int(d.get('format', {}).get('bit_rate', 0)) // 1000
print(br)
" 2>/dev/null || echo "0")

    if [[ "$bitrate" == "0" ]]; then
        fail "  ffprobe returned bitrate=0 for $f"
    else
        pass "  Bitrate: ${bitrate} kbps"
    fi
done

echo ""

# ── Check GPU ──
echo "GPU:"
if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$gpu_name" ]]; then
        pass "GPU available: $gpu_name"
    else
        fail "nvidia-smi found but no GPU detected"
    fi
else
    fail "nvidia-smi not found — NVENC encoding unavailable"
fi

echo ""

# ── Check disk space ──
echo "Disk space:"

# Processing (staging) dir: GMS_STAGING_DIR, else <data home>/input
data_home="${GMS_DATA_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/gomoveshift}"
staging_dir="${GMS_STAGING_DIR:-$data_home/input}"
if [[ -d "$staging_dir" ]]; then
    avail_gb=$(df -BG "$staging_dir" 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4); print $4}')
    if (( ${avail_gb:-0} < 50 )); then
        fail "Staging filesystem: ${avail_gb:-unknown}GB free (need 50GB+ for processing)"
    else
        pass "Staging filesystem: ${avail_gb}GB free"
    fi
else
    warn "Staging dir not found: $staging_dir (set GMS_STAGING_DIR)"
fi

# Output dir for deliverables: GMS_OUTPUT_DIR, else <data home>/output
output_dir="${GMS_OUTPUT_DIR:-$data_home/output}"
if [[ -d "$output_dir" ]]; then
    avail_out=$(df -BG "$output_dir" 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4); print $4}')
    if (( avail_out < 100 )); then
        warn "Output filesystem: ${avail_out}GB free (want 100GB+ for deliverables)"
    else
        pass "Output filesystem: ${avail_out}GB free"
    fi
else
    warn "Output dir not found: $output_dir (set GMS_OUTPUT_DIR)"
fi

echo ""

# ── Check network (warn-only) ──
echo "Network:"
if ping -c 1 -W 2 nominatim.openstreetmap.org &>/dev/null; then
    pass "Nominatim reachable (chapter geocoding)"
else
    warn "Nominatim unreachable — chapter generation may fail"
fi

echo ""

# ── Summary ──
if (( ERRORS > 0 )); then
    echo -e "${RED}Pre-flight FAILED: $ERRORS error(s)${NC}"
    exit 1
else
    echo -e "${GREEN}Pre-flight PASSED — ready to run${NC}"
    exit 0
fi
