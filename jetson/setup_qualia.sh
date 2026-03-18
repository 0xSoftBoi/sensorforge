#!/usr/bin/env bash
set -euo pipefail

# Setup script for building qualia-cuda on Jetson Orin Nano
# Assumes: JetPack 6.1+ with CUDA 12.9 installed (cuda-toolkit-12-9)

REPO_URL="https://github.com/0xSoftBoi/sensorforge.git"
REPO_DIR="$HOME/sensorforge"
SWAP_FILE="/swapfile"
SWAP_SIZE_GB=4

echo "=== Jetson Qualia Setup ==="

# ── 1. CUDA PATH ─────────────────────────────────────────────────
CUDA_DIR="/usr/local/cuda-12.9"
if [ ! -d "$CUDA_DIR" ]; then
    echo "ERROR: CUDA 12.9 not found at $CUDA_DIR"
    echo "Install with: sudo apt install cuda-toolkit-12-9"
    exit 1
fi

if ! grep -q "cuda-12.9" "$HOME/.bashrc" 2>/dev/null; then
    echo "Adding CUDA 12.9 to PATH..."
    cat >> "$HOME/.bashrc" << 'ENVEOF'

# CUDA 12.9 (added by setup_qualia.sh)
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ENVEOF
fi

export CUDA_HOME="$CUDA_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "nvcc: $(nvcc --version | tail -1)"

# ── 2. Rust ───────────────────────────────────────────────────────
if ! command -v rustc &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust already installed: $(rustc --version)"
fi

source "$HOME/.cargo/env" 2>/dev/null || true

# ── 3. Swap (4GB file for compilation) ────────────────────────────
if [ ! -f "$SWAP_FILE" ]; then
    echo "Creating ${SWAP_SIZE_GB}GB swapfile for compilation..."
    sudo fallocate -l "${SWAP_SIZE_GB}G" "$SWAP_FILE"
    sudo chmod 600 "$SWAP_FILE"
    sudo mkswap "$SWAP_FILE"
    sudo swapon "$SWAP_FILE"
    echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
    echo "Swap active: $(free -h | grep Swap)"
else
    if ! swapon --show | grep -q "$SWAP_FILE"; then
        sudo swapon "$SWAP_FILE"
    fi
    echo "Swap already configured: $(free -h | grep Swap)"
fi

# ── 4. Build dependencies ────────────────────────────────────────
echo "Installing build dependencies..."
sudo apt-get update -qq 2>/dev/null || echo "apt-get update had warnings (stale repos?) — continuing"
sudo apt-get install -y -qq pkg-config libssl-dev build-essential git

# ── 5. Clone or update repo ──────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    echo "Updating existing repo..."
    cd "$REPO_DIR"
    git pull --no-rebase
else
    echo "Cloning sensorforge..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ── 6. Build ─────────────────────────────────────────────────────
cd "$REPO_DIR/qualia"

echo "Building qualia-cuda backend..."
cargo build -j2 --release -p qualia-cuda

echo "Building layer runners with CUDA backend..."
for layer in l0-superposition l1-belief l2-belief l3-belief l4-behavior l5-behavior; do
    echo "  qualia-$layer..."
    cargo build -j2 --release -p "qualia-$layer" --no-default-features --features cuda
done

echo "Building non-layer runners..."
for runner in qualia-camera qualia-health qualia-vision qualia-agent qualia-watch; do
    echo "  $runner..."
    cargo build -j2 --release -p "$runner"
done

echo ""
echo "=== Build complete ==="
echo "Run: ./jetson/launch_qualia.sh"
echo "     ./jetson/launch_qualia.sh --record  (with training data)"
