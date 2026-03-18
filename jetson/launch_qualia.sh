#!/usr/bin/env bash
set -euo pipefail

# Launch the full Qualia stack on Jetson.
#
# Usage:
#   ./launch_qualia.sh                    # Run stack only
#   ./launch_qualia.sh --record           # Run + record training data
#   ./launch_qualia.sh --drive            # Run + autonomous exploration
#   ./launch_qualia.sh --record --drive   # Full pipeline: stack + record + drive
#   ./launch_qualia.sh --gemini-key KEY   # Provide Gemini API key
#   ./launch_qualia.sh --camera /dev/video1  # Override camera device

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RECORD=false
DRIVE=false
CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --record) RECORD=true; shift ;;
        --drive) DRIVE=true; shift ;;
        --gemini-key) export GEMINI_API_KEY="$2"; shift 2 ;;
        --camera) CAMERA_DEVICE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

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

# ── 1d. Stop conflicting camera services ────────────────────────
if systemctl is-active --quiet jetson-capture-images 2>/dev/null; then
    echo "Stopping jetson-capture-images (conflicts with qualia-camera)..."
    sudo systemctl stop jetson-capture-images
fi

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
    echo "Starting session recorder..."
    python3 "$SCRIPT_DIR/session_recorder.py" --camera "$CAMERA_DEVICE" &
    PIDS+=($!)
fi

# ── 8. Launch autonomous explorer (if --drive) ───────────────────
if [ "$DRIVE" = true ]; then
    if [ -c /dev/ttyACM0 ]; then
        echo "Starting autonomous explorer..."
        python3 "$SCRIPT_DIR/autonomous_explorer.py" &
        PIDS+=($!)
    else
        echo "WARNING: No UGV serial device at /dev/ttyACM0 — skipping drive mode"
    fi
fi

echo ""
echo "=== Qualia Stack Running ==="
echo "  Camera:       $CAMERA_DEVICE"
echo "  Dashboard:    http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo localhost):8080"
echo "  Recording:    $RECORD"
echo "  Driving:      $DRIVE"
echo "  Checkpoints:  $QUALIA_CHECKPOINT_DIR"
echo "  Gemini:       ${GEMINI_API_KEY:+enabled}${GEMINI_API_KEY:-offline}"
echo ""
echo "Press Ctrl-C to stop all processes."

# Wait on the main Rust process
wait $WATCH_PID
