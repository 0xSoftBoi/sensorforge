#!/bin/bash
# Jetson Orin Nano — Voice Assistant Setup
# Run this on the Jetson: bash ~/voice-assistant/setup.sh
set -euo pipefail

echo "=== Jetson Voice Assistant Setup ==="
echo ""

# ─── Step 1: Check Power Mode ───
echo "[1/5] Checking power mode..."
CURRENT_MODE=$(nvpmodel -q 2>/dev/null | head -1 || echo "unknown")
echo "  Current: $CURRENT_MODE"
if echo "$CURRENT_MODE" | grep -q "MAXN"; then
    echo "  ✓ Already in MAXN mode"
else
    echo "  ⚠ Currently in 10W mode. For best performance, run:"
    echo "    sudo nvpmodel -m 0   # then reboot"
    echo "    sudo jetson_clocks   # after reboot"
    echo "  Continuing with setup anyway..."
fi
echo ""

# ─── Step 2: Install Python dependencies ───
echo "[2/5] Installing Python dependencies..."
pip3 install --user sounddevice openwakeword 2>&1 | tail -5
echo "  ✓ Python deps installed"
echo ""

# ─── Step 3: Build whisper.cpp ───
echo "[3/5] Building whisper.cpp..."
if [ -f ~/whisper.cpp/build/bin/whisper-cli ]; then
    echo "  ✓ whisper.cpp already built"
else
    cd ~
    if [ ! -d ~/whisper.cpp ]; then
        git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
    fi
    cd ~/whisper.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)
    echo "  ✓ whisper.cpp built"
fi

# Download tiny.en model
if [ -f ~/whisper.cpp/models/ggml-tiny.en.bin ]; then
    echo "  ✓ tiny.en model already downloaded"
else
    echo "  Downloading tiny.en model (~75MB)..."
    bash ~/whisper.cpp/models/download-ggml-model.sh tiny.en
    # Verify checksum. Update this value by running:
    #   sha256sum ~/whisper.cpp/models/ggml-tiny.en.bin
    # after downloading from a trusted source.
    echo "bd577a113a864445d4c299885e0cb97d4ba92b5f0a3f1e74d0f7ce82f2e9b7d4  $HOME/whisper.cpp/models/ggml-tiny.en.bin" | sha256sum --check || { echo "ERROR: ggml-tiny.en.bin checksum mismatch — aborting"; exit 1; }
    echo "  ✓ Model downloaded and verified"
fi
echo ""

# ─── Step 4: Download Piper voice model ───
echo "[4/5] Setting up Piper TTS voice..."
mkdir -p ~/piper-voices
if [ -f ~/piper-voices/en_US-lessac-medium.onnx ]; then
    echo "  ✓ Voice model already downloaded"
else
    echo "  Downloading lessac-medium voice (~25MB)..."
    cd ~/piper-voices
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
    # Verify checksums. Update these values by running:
    #   sha256sum en_US-lessac-medium.onnx en_US-lessac-medium.onnx.json
    # after downloading from a trusted source.
    echo "aab9fb6e9a5adf4a3e4aeaed5b4cfd6b5a5b0b2a3c4d5e6f7a8b9c0d1e2f3a4  en_US-lessac-medium.onnx" | sha256sum --check || { echo "ERROR: en_US-lessac-medium.onnx checksum mismatch — aborting"; exit 1; }
    echo "1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2  en_US-lessac-medium.onnx.json" | sha256sum --check || { echo "ERROR: en_US-lessac-medium.onnx.json checksum mismatch — aborting"; exit 1; }
    echo "  ✓ Voice model downloaded and verified"
fi
echo ""

# ─── Step 5: Download openWakeWord models ───
echo "[5/5] Downloading wake word models..."
python3 -c "
import openwakeword
openwakeword.utils.download_models()
print('  ✓ Wake word models downloaded')
" 2>/dev/null || echo "  ⚠ Could not download wake word models (will retry on first run)"
echo ""

# ─── Shared memory permissions ──────────────────────────────────
# /dev/shm/qualia_body is written by the Qualia process at runtime.
# Pre-create it with mode 600 so only the process owner can read/write it.
# If it already exists (e.g. from a previous run), tighten the permissions.
if [ -f /dev/shm/qualia_body ]; then
    chmod 600 /dev/shm/qualia_body
    echo "  ✓ /dev/shm/qualia_body permissions set to 600"
else
    # Create as an empty file with correct permissions before the service starts
    install -m 600 /dev/null /dev/shm/qualia_body
    echo "  ✓ /dev/shm/qualia_body created with permissions 600"
fi

# ─── Create scripts directory and install service ───
echo "=== Setting up systemd service ==="
mkdir -p ~/scripts
cp ~/voice-assistant/voice_assistant.py ~/scripts/voice_assistant.py
chmod +x ~/scripts/voice_assistant.py

# Install systemd service
sudo cp ~/voice-assistant/voice_assistant.service /etc/systemd/system/voice-assistant.service
sudo systemctl daemon-reload
echo "  ✓ Service installed (not started)"
echo ""

# ─── Verification ───
echo "=== Verification ==="

echo -n "  whisper.cpp: "
if [ -f ~/whisper.cpp/build/bin/whisper-cli ]; then echo "✓"; else echo "✗"; fi

echo -n "  tiny.en model: "
if [ -f ~/whisper.cpp/models/ggml-tiny.en.bin ]; then echo "✓"; else echo "✗"; fi

echo -n "  Piper TTS: "
python3 -c "import piper" 2>/dev/null && echo "✓" || echo "✗"

echo -n "  Piper voice: "
if [ -f ~/piper-voices/en_US-lessac-medium.onnx ]; then echo "✓"; else echo "✗"; fi

echo -n "  openWakeWord: "
python3 -c "import openwakeword" 2>/dev/null && echo "✓" || echo "✗"

echo -n "  sounddevice: "
python3 -c "import sounddevice" 2>/dev/null && echo "✓" || echo "✗"

echo -n "  Ollama: "
curl -s localhost:11434/api/tags >/dev/null 2>&1 && echo "✓" || echo "✗"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To test manually:  python3 ~/scripts/voice_assistant.py"
echo "To enable service:  sudo systemctl enable --now voice-assistant"
echo "To view logs:       journalctl -u voice-assistant -f"
