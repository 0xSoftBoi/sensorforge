# Jetson Voice Assistant

Fully offline voice assistant for the Jetson Orin Nano with 24 tools for system monitoring, robot control, and Qualia brain interaction.

## Requirements

- **NVIDIA Jetson Orin Nano** (8GB recommended)
- **JetPack 6.0+** (Ubuntu 22.04 based)
- **USB microphone** (e.g., ReSpeaker, any ALSA-compatible)
- **USB or 3.5mm speaker**
- **Python 3.10+**

## Setup

Run the automated setup script:

```bash
bash setup.sh
```

This installs and verifies:

1. **Python dependencies**: `sounddevice`, `openwakeword`
2. **whisper.cpp**: Built from source with the `tiny.en` model (~75MB)
3. **Piper TTS**: `lessac-medium` voice (~25MB)
4. **openWakeWord**: Pre-trained wake word models
5. **systemd service**: Installed but not started

### Additional manual setup

```bash
# Install Ollama (LLM inference)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:1b

# For optimal performance
sudo nvpmodel -m 0       # Set MAXN power mode
sudo reboot
sudo jetson_clocks        # Max out clocks after reboot
```

## Usage

```bash
# Run interactively
python3 voice_assistant.py

# Test all 24 tools without voice
python3 voice_assistant.py --test-tools

# Run as a systemd service
sudo systemctl enable --now voice-assistant
journalctl -u voice-assistant -f    # View logs
```

## Voice Pipeline

```
  "Hey Jarvis"          Record           Transcribe          LLM              Speak
 ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐    ┌──────────┐
 │ openWakeWord│───►│ sounddevice│───►│ whisper.cpp│───►│  Ollama  │───►│ Piper TTS│
 │ threshold=  │    │  16kHz/1ch │    │  tiny.en   │    │ gemma3:1b│    │  lessac  │
 │  0.5        │    │ 80ms chunks│    │  ~75MB     │    │ 24 tools │    │  medium  │
 └────────────┘    └────────────┘    └────────────┘    └──────────┘    └──────────┘
                    Silence detect:
                    500 RMS / 1.5s
```

All processing runs locally on the Jetson — no cloud dependency.

## Tool Registry

### System Tools

| Tool | Description |
|------|-------------|
| `get_cpu_temperature` | Read CPU thermal zone temperature |
| `get_gpu_temperature` | Read GPU thermal zone temperature |
| `get_all_temperatures` | Read all thermal sensors at once |
| `get_disk_usage` | Disk space — total, used, available |
| `get_memory_usage` | RAM — total, used, available |
| `get_uptime` | Time since last boot |
| `get_cpu_load` | CPU load averages |
| `get_power_mode` | Current Jetson power mode (10W / MAXN) |
| `get_power_consumption` | Current power draw |
| `get_date_time` | Current date and time |
| `get_network_info` | Network interfaces and IP addresses |
| `get_service_status` | Check if a systemd service is running |
| `restart_service` | Restart a systemd service |
| `list_services` | List all Jetson services and their status |
| `get_system_telemetry` | Combined system health snapshot |

### Robot Control

| Tool | Description | Parameters |
|------|-------------|------------|
| `move_forward` | Drive forward | `duration` (max 5s), `speed` (0-128) |
| `move_backward` | Drive backward | `duration` (max 5s), `speed` (0-128) |
| `turn_left` | Turn left | `duration` (default 0.5s) |
| `turn_right` | Turn right | `duration` (default 0.5s) |
| `stop_robot` | Emergency stop | — |
| `robot_status` | Battery voltage and IMU data | — |
| `capture_and_describe` | Take photo and describe scene | — |

### Qualia Brain

| Tool | Description |
|------|-------------|
| `qualia_beliefs` | Current belief states across all 7 layers |
| `qualia_surprise` | Which layers have high prediction error |
| `qualia_lore` | Recent questions and answers from the brain |
| `qualia_directive` | Set the brain's top-level goal (e.g., "explore the room") |

### Conversation

| Tool | Description |
|------|-------------|
| `recall_conversation` | Semantic search over conversation history (SQLite) |

## Qualia Integration

The voice assistant reads Qualia's shared memory directly via `mmap` + `struct.unpack`, matching the `#[repr(C)]` layout from `qualia/crates/types/src/lib.rs`.

The bridge module (`qualia_bridge.py`) provides:

- `read_all_layers()` — Belief state for all 7 layers
- `read_world_model()` — Scene, objects, embedding, API counters
- `read_thought_buffer()` — Recent thought stream entries
- `read_lore_buffer()` — Question/answer pairs from the LORE system

## Other Modules

| File | Description |
|------|-------------|
| `wifi_bridge.py` | TCP server receiving sensor data from iPhone |
| `qualia_bridge.py` | Python reader for Qualia shared memory |
| `ugv_driver.py` | Motor control via GPIO/UART |
| `gemini_vision.py` | Gemini API client for scene understanding |
| `qualia_detect.py` | Scene detection integration |
| `qualia_embed.py` | Embedding generation |
| `qualia_audio.py` | Audio processing for Qualia |
| `lore_store.py` | LORE question/answer storage |
| `autonomous_explorer.py` | Autonomous navigation and exploration |
| `session_recorder.py` | Data recording for training |
| `debug_wakeword.py` | Wake word detection debugging utility |

## Configuration

Key constants in `voice_assistant.py` (hardcoded):

| Constant | Value | Description |
|----------|-------|-------------|
| Wake word | `hey_jarvis` | openWakeWord model name |
| Wake threshold | `0.5` | Activation sensitivity |
| Audio sample rate | `16000` Hz | Recording sample rate |
| Audio channels | `1` | Mono |
| Chunk duration | `80` ms | Audio buffer size |
| Silence threshold | `500` RMS | Below this = silence |
| Silence duration | `1.5` s | Silence before end of speech |
| Whisper model | `tiny.en` | ~75MB, English only |
| LLM model | `gemma3:1b` | Via Ollama |
| TTS voice | `lessac-medium` | Piper en_US voice |
