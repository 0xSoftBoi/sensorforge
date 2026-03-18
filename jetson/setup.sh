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
    echo "  ✓ Model downloaded"
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
    echo "  ✓ Voice model downloaded"
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
