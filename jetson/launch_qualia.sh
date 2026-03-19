#!/usr/bin/env bash
set -euo pipefail

# Launch the full Qualia stack on Jetson.
#
# Usage:
#   ./launch_qualia.sh                    # Run stack only
#   ./launch_qualia.sh --record           # Run + record training data
#   ./launch_qualia.sh --drive            # Run + autonomous exploration
#   ./launch_qualia.sh --manual           # Run + manual keyboard drive
#   ./launch_qualia.sh --record --drive   # Full pipeline: stack + record + drive
#   ./launch_qualia.sh --record --manual  # Manual drive + record training data
#   ./launch_qualia.sh --gemini-key KEY   # Provide Gemini API key
#   ./launch_qualia.sh --camera /dev/video1  # Override camera device

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RECORD=false
DRIVE=false
MANUAL=false
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --record) RECORD=true; shift ;;
        --drive) DRIVE=true; shift ;;
        --manual) MANUAL=true; shift ;;
        --gemini-key) export GEMINI_API_KEY="$2"; shift 2 ;;
        --camera) CAMERA_DEVICE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "$DRIVE" = true ] && [ "$MANUAL" = true ]; then
    echo "ERROR: --drive and --manual are mutually exclusive"
    exit 1
fi

export CAMERA_DEVICE

# ── 1. CUDA environment ──────────────────────────────────────────
CUDA_DIR="/usr/local/cuda-12.9"
if [ -d "$CUDA_DIR" ]; then
    export CUDA_HOME="$CUDA_DIR"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ── 1b. CUDA MPS (shares GPU context across layer processes) ────
if [ -x /usr/bin/nvidia-cuda-mps-control ]; then
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true  # kill stale
    sleep 0.5
    nvidia-cuda-mps-control -d 2>/dev/null && echo "CUDA MPS daemon started" || echo "WARNING: CUDA MPS failed to start (non-fatal)"
fi

# ── 1d. Stop conflicting services ─────────────────────────────
if systemctl is-active --quiet jetson-capture-images 2>/dev/null; then
    echo "Stopping jetson-capture-images (conflicts with qualia-camera)..."
    sudo systemctl stop jetson-capture-images
fi
if systemctl is-active --quiet jetson-read-serial 2>/dev/null; then
    echo "Stopping jetson-read-serial (conflicts with explorer serial port)..."
    sudo systemctl stop jetson-read-serial
fi

# Kill orphaned processes from previous runs
pkill -f "session_recorder" 2>/dev/null || true
pkill -f "qualia-watch" 2>/dev/null || true
sleep 2
# Free up the camera device
fuser -k /dev/video0 2>/dev/null || true
sleep 1

# ── 1c. Checkpoint directory for weight persistence ─────────────
export QUALIA_CHECKPOINT_DIR="${QUALIA_CHECKPOINT_DIR:-$HOME/training-data/checkpoints}"
mkdir -p "$QUALIA_CHECKPOINT_DIR"

# ── 2. Rust / Cargo ──────────────────────────────────────────────
source "$HOME/.cargo/env" 2>/dev/null || true

# ── 3. Gemini API key ────────────────────────────────────────────
if [ -z "${GEMINI_API_KEY:-}" ]; then
    KEY_FILE="$HOME/.secrets/gemini_api_key"
    if [ -f "$KEY_FILE" ]; then
        export GEMINI_API_KEY="$(cat "$KEY_FILE")"
        echo "Loaded GEMINI_API_KEY from $KEY_FILE"
    else
        echo "WARNING: No GEMINI_API_KEY set — vision runner will use offline mode"
    fi
fi

# ── 4. Pids to clean up ──────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    echo "All processes stopped."
}

trap cleanup SIGINT SIGTERM

# ── 5. Launch qualia-watch (Rust TUI supervisor) ─────────────────
cd "$REPO_DIR/qualia"
export QUALIA_HEADLESS="${QUALIA_HEADLESS:-1}"
echo "Starting qualia-watch (7-layer stack + camera + vision + agent)..."
echo "  Headless: $QUALIA_HEADLESS (set QUALIA_HEADLESS=0 for TUI)"
cargo run --release -p qualia-watch &
WATCH_PID=$!
PIDS+=($WATCH_PID)

# Wait for SHM to be created
echo "Waiting for SHM initialization..."
sleep 3

# ── 6. Launch lore_store (persists LORE to SQLite) ───────────────
if [ -f "$SCRIPT_DIR/lore_store.py" ]; then
    echo "Starting lore_store..."
    python3 "$SCRIPT_DIR/lore_store.py" &
    PIDS+=($!)
fi

# ── 7. Launch session recorder (if --record) ─────────────────────
if [ "$RECORD" = true ]; then
    # Wait for camera runner to create the snapshot file
    echo "Waiting for camera snapshot..."
    for i in $(seq 1 10); do
        [ -f /tmp/qualia-camera-latest.jpg ] && break
        sleep 1
    done
    echo "Starting session recorder..."
    python3 "$SCRIPT_DIR/session_recorder.py" --camera "$CAMERA_DEVICE" &
    PIDS+=($!)
fi

# ── 8. Launch local object detection (Phase 2.2) ─────────────────
if [ -f "$SCRIPT_DIR/qualia_detect.py" ]; then
    echo "Starting local YOLO detection..."
    QUALIA_DETECT_HZ="${QUALIA_DETECT_HZ:-2}" python3 "$SCRIPT_DIR/qualia_detect.py" &
    PIDS+=($!)
    sleep 1
fi

# ── 9. Launch local embeddings (Phase 2.3) ───────────────────────
if [ -f "$SCRIPT_DIR/qualia_embed.py" ]; then
    echo "Starting local embeddings..."
    python3 "$SCRIPT_DIR/qualia_embed.py" --interval 2.0 &
    PIDS+=($!)
fi

# ── 10. Launch audio features (Phase 4.1) ────────────────────────
if [ -f "$SCRIPT_DIR/qualia_audio.py" ] && [ -d /proc/asound ]; then
    echo "Starting audio features..."
    python3 "$SCRIPT_DIR/qualia_audio.py" --bands 32 &
    PIDS+=($!)
fi

# ── 11. Launch autonomous explorer (if --drive) ──────────────────
if [ "$DRIVE" = true ]; then
    if [ -c /dev/ttyACM0 ]; then
        echo "Starting EFE autonomous explorer..."
        python3 "$SCRIPT_DIR/autonomous_explorer.py" &
        PIDS+=($!)
    else
        echo "WARNING: No UGV serial device at /dev/ttyACM0 — skipping drive mode"
    fi
fi

# ── 11b. Launch manual drive (if --manual) ───────────────────────
if [ "$MANUAL" = true ]; then
    if [ -c /dev/ttyACM0 ]; then
        echo "Starting manual keyboard drive..."
        echo "  Use WASD to drive, Space to stop, Q to quit"
        # Run in foreground so keyboard input works (replaces wait loop below)
        python3 "$SCRIPT_DIR/manual_drive.py"
        # When manual_drive exits, clean up everything
        cleanup
        exit 0
    else
        echo "WARNING: No UGV serial device at /dev/ttyACM0 — skipping manual mode"
    fi
fi

# ── 12. Launch watchdog (monitors stack health) ───────────────
pkill -f "qualia_watchdog" 2>/dev/null || true
if [ -f "$SCRIPT_DIR/qualia_watchdog.sh" ]; then
    echo "Starting watchdog (auto-restart on freeze)..."
    nohup bash "$SCRIPT_DIR/qualia_watchdog.sh" --restart-on-failure >> "$HOME/training-data/watchdog.log" 2>&1 &
    WATCHDOG_PID=$!
    PIDS+=($WATCHDOG_PID)
fi

echo ""
echo "=== Qualia Stack Running ==="
echo "  Camera:       $CAMERA_DEVICE"
echo "  Dashboard:    http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo localhost):8080"
echo "  Recording:    $RECORD"
echo "  Driving:      $([ "$DRIVE" = true ] && echo 'autonomous' || ([ "$MANUAL" = true ] && echo 'manual' || echo 'off'))"
echo "  Checkpoints:  $QUALIA_CHECKPOINT_DIR"
echo "  Watchdog:     ${WATCHDOG_PID:-disabled}"
echo "  Gemini:       ${GEMINI_API_KEY:+enabled}${GEMINI_API_KEY:-offline}"
echo "  Local detect: $(command -v python3 >/dev/null && python3 -c 'import super_gradients' 2>/dev/null && echo 'YOLO' || echo 'unavailable')"
echo "  Local embed:  $(command -v python3 >/dev/null && python3 -c 'import onnxruntime' 2>/dev/null && echo 'ONNX' || echo 'unavailable')"
echo ""
echo "Press Ctrl-C to stop all processes."

# Wait on the main Rust process
wait $WATCH_PID
