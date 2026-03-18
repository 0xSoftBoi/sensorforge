#!/usr/bin/env python3
"""
Jetson Orin Nano — Offline Voice Assistant

Fully offline, Jarvis-style voice assistant with tool calling and streaming TTS.
Wake word → STT → LLM (with tools) → Streaming TTS → Speaker

Components:
  - openWakeWord: wake word detection ("hey jarvis")
  - whisper.cpp:  speech-to-text (tiny.en, ~75MB)
  - Ollama:       LLM inference (gemma3:1b) with dispatch-based tool calling
  - Piper TTS:    text-to-speech (lessac-medium) with sentence streaming

Tool Calling:
  The LLM can call local functions to check system stats, control services,
  read sensors, and more. Define tools in TOOL_REGISTRY.

Streaming TTS:
  Instead of waiting for the full LLM response, text is streamed token-by-token,
  split into sentences, and each sentence is spoken immediately via a background
  TTS worker thread. This cuts perceived latency by ~50%.
"""

import base64
import glob as globmod
import json
import logging
import os
import queue
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np

# ─── Configuration ───────────────────────────────────────────────

WHISPER_CLI = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-tiny.en.bin")
PIPER_CLI = os.path.expanduser("~/.local/bin/piper")
PIPER_VOICE = os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx")
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:1b"

# Audio devices (ALSA)
MIC_DEVICE = "plughw:0,0"  # USB Camera (actual microphone)
SPEAKER_DEVICE = "plughw:1,0"  # USB PnP Audio Device (speaker)

# Telemetry / Camera / Database
TELEMETRY_JSONL = "/home/jetson/training-data/telemetry/system.jsonl"
CAMERA_DEVICE = "/dev/video0"
IMAGE_DIR = "/home/jetson/training-data/images"
CONVERSATIONS_DB = "/home/jetson/training-data/conversations.db"

# Model roster (only 1 loaded at a time)
REASONING_MODEL = "qwen3:0.6b"

# Gemini API for vision (no local VLM fits in 3.5GB)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_BYTES = 2560  # 1280 samples * 2 bytes = 80ms at 16kHz
SILENCE_THRESHOLD = 500  # RMS amplitude below this = silence (noise floor ~300)
SILENCE_DURATION = 1.5  # seconds of silence to stop recording
MAX_RECORD_SECONDS = 15  # max recording length
MIN_RECORD_SECONDS = 0.5  # min recording to process

# Wake word
WAKE_WORD_MODEL = "hey_jarvis"  # Pre-trained model
WAKE_THRESHOLD = 0.5  # Detection confidence threshold

# LLM system prompt
SYSTEM_PROMPT = (
    "You are a helpful voice assistant running locally on a Jetson Orin Nano "
    "mounted on a Waveshare UGV robot. "
    "Keep responses concise — 1-3 sentences max. Be direct and conversational. "
    "You have limited internet access for vision only. For other real-time data, say you're offline. "
    "You have tools to check system stats, service status, temperatures, "
    "disk and memory usage, network info, and more. Use them when the user "
    "asks about the system. You can see through a camera and describe what "
    "you see. You can control the robot — move forward, backward, turn, and stop. "
    "You can explore autonomously using your camera and navigation intelligence. "
    "You can also remember previous conversations."
)

# UGV serial port
UGV_PORT = "/dev/ttyACM0"
UGV_BAUD = 115200
MAX_AUTONOMOUS_DURATION = 60  # Max seconds for autonomous exploration

# ─── Logging ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice-assistant")

# ─── UGV Driver (lazy-loaded) ──────────────────────────────────

_ugv_driver = None


def _get_ugv():
    """Lazy-load UGV driver. Returns driver or None if not available."""
    global _ugv_driver
    if _ugv_driver is not None:
        return _ugv_driver
    try:
        from ugv_driver import UGVDriver
        _ugv_driver = UGVDriver(port=UGV_PORT, baud=UGV_BAUD)
        return _ugv_driver
    except Exception as e:
        log.warning(f"UGV not available: {e}")
        return None


# ─── Tool Functions ─────────────────────────────────────────────
# Each tool returns a string result. The LLM decides when to call them.


def tool_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp_mc = int(f.read().strip())
        return f"{temp_mc / 1000:.1f} degrees Celsius"
    except Exception as e:
        return f"Error reading CPU temperature: {e}"


def tool_gpu_temperature():
    try:
        # Jetson Orin: GPU temp is usually thermal_zone1
        with open("/sys/class/thermal/thermal_zone1/temp") as f:
            temp_mc = int(f.read().strip())
        return f"{temp_mc / 1000:.1f} degrees Celsius"
    except Exception as e:
        return f"Error reading GPU temperature: {e}"


def tool_disk_usage():
    try:
        result = subprocess.run(
            ["df", "-h", "/"], capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            return f"Total: {parts[1]}, Used: {parts[2]}, Available: {parts[3]}, Usage: {parts[4]}"
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def tool_memory_usage():
    try:
        result = subprocess.run(
            ["free", "-h"], capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            return f"Total: {parts[1]}, Used: {parts[2]}, Available: {parts[6]}"
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


def tool_uptime():
    try:
        result = subprocess.run(
            ["uptime", "-p"], capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


ALLOWED_STATUS_SERVICES = frozenset({
    "voice-assistant", "ollama", "jetson-capture-images",
    "jetson-capture-audio", "jetson-read-serial", "jetson-telemetry",
})

ALLOWED_RESTART_SERVICES = frozenset({
    "jetson-capture-images", "jetson-capture-audio",
    "jetson-read-serial", "jetson-telemetry",
})


def tool_service_status(service_name):
    if service_name not in ALLOWED_STATUS_SERVICES:
        return f"Unknown service. Available: {', '.join(sorted(ALLOWED_STATUS_SERVICES))}"
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True, text=True, timeout=5,
        )
        return f"{service_name} is {result.stdout.strip()}"
    except Exception as e:
        return f"Error: {e}"


def tool_restart_service(service_name):
    if service_name not in ALLOWED_RESTART_SERVICES:
        return (
            f"Cannot restart '{service_name}' for safety. "
            f"Allowed: {', '.join(sorted(ALLOWED_RESTART_SERVICES))}"
        )
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", service_name],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return f"{service_name} restarted successfully"
        return f"Failed: {result.stderr.strip()}"
    except Exception as e:
        return f"Error: {e}"


def tool_network_info():
    try:
        result = subprocess.run(
            ["ip", "-brief", "addr"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def tool_power_mode():
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"


def tool_date_time():
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def tool_list_services():
    results = []
    for svc in sorted(ALLOWED_STATUS_SERVICES):
        try:
            r = subprocess.run(
                ["systemctl", "is-active", svc],
                capture_output=True, text=True, timeout=5,
            )
            results.append(f"{svc}: {r.stdout.strip()}")
        except Exception:
            results.append(f"{svc}: unknown")
    return "\n".join(results)


# ─── Telemetry Tools (from JSONL) ────────────────────────────────

def _read_last_telemetry():
    """Read the last line of the telemetry JSONL file."""
    try:
        if not os.path.exists(TELEMETRY_JSONL):
            return None
        with open(TELEMETRY_JSONL, "rb") as f:
            # Seek near end for efficiency
            f.seek(0, 2)
            size = f.tell()
            pos = max(0, size - 4096)
            f.seek(pos)
            lines = f.read().decode("utf-8", errors="replace").strip().split("\n")
            for line in reversed(lines):
                line = line.strip()
                if line:
                    return json.loads(line)
        return None
    except Exception as e:
        log.warning(f"Telemetry read error: {e}")
        return None


def _parse_power_field(power_raw):
    """Parse the power field which may be a Python repr string or dict."""
    if isinstance(power_raw, dict):
        return power_raw
    if isinstance(power_raw, str):
        try:
            import ast
            return ast.literal_eval(power_raw)
        except Exception:
            return {}
    return {}


def tool_power_consumption():
    data = _read_last_telemetry()
    if not data:
        return "Telemetry data not available"
    power = _parse_power_field(data.get("power", {}))
    # Total power from 'tot' field (in milliwatts)
    tot = power.get("tot", {})
    if isinstance(tot, dict) and "power" in tot:
        watts = tot["power"] / 1000.0
        name = tot.get("name", "VDD_IN")
        parts = [f"{name}: {watts:.1f} watts"]
        # Add per-rail breakdown
        for rail_name, rail in power.get("rail", {}).items():
            if isinstance(rail, dict) and "power" in rail:
                parts.append(f"{rail_name}: {rail['power']/1000:.1f}W")
        return ", ".join(parts)
    return "Power data not available in telemetry"


def tool_cpu_load():
    data = _read_last_telemetry()
    if not data:
        return "Telemetry data not available"
    cpu = data.get("cpu", [])
    if isinstance(cpu, list):
        cores = []
        for i, core in enumerate(cpu):
            if isinstance(core, dict) and core.get("online"):
                user = core.get("user", 0)
                system = core.get("system", 0)
                total = user + system
                cores.append(f"Core {i}: {total:.0f}%")
        if cores:
            return ", ".join(cores)
    return "CPU load data not available"


def tool_all_temperatures():
    data = _read_last_telemetry()
    if not data:
        return "Telemetry data not available"
    temp = data.get("temp", {})
    if isinstance(temp, dict) and temp:
        temps = []
        for zone, info in sorted(temp.items()):
            if isinstance(info, dict) and info.get("online") and info.get("temp", -256) > -200:
                temps.append(f"{zone}: {info['temp']:.1f}°C")
        if temps:
            return ", ".join(temps)
    return "Temperature data not available"


def tool_system_telemetry():
    data = _read_last_telemetry()
    if not data:
        return "Telemetry data not available"
    parts = []
    ts = data.get("ts", "unknown")
    if isinstance(ts, (int, float)):
        parts.append(f"Timestamp: {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}")
    # CPU — average load across cores
    cpu = data.get("cpu", [])
    if isinstance(cpu, list):
        loads = []
        for core in cpu:
            if isinstance(core, dict) and core.get("online"):
                loads.append(core.get("user", 0) + core.get("system", 0))
        if loads:
            parts.append(f"CPU avg: {sum(loads)/len(loads):.0f}% across {len(loads)} cores")
    # GPU
    gpu = data.get("gpu", {})
    if isinstance(gpu, dict):
        inner = gpu.get("gpu", {})
        if isinstance(inner, dict):
            freq = inner.get("freq", {})
            if isinstance(freq, dict):
                cur = freq.get("cur")
                if cur:
                    parts.append(f"GPU freq: {cur/1000:.0f} MHz")
            load = inner.get("status", {}).get("load", None)
            if load is not None:
                parts.append(f"GPU load: {load}%")
    # Thermal — junction temp
    temp = data.get("temp", {})
    if isinstance(temp, dict):
        tj = temp.get("tj", {})
        if isinstance(tj, dict) and tj.get("online"):
            parts.append(f"Junction temp: {tj['temp']:.1f}°C")
    # Power
    power = _parse_power_field(data.get("power", {}))
    tot = power.get("tot", {})
    if isinstance(tot, dict) and "power" in tot:
        parts.append(f"Power: {tot['power']/1000:.1f}W")
    # RAM
    ram = data.get("ram", {})
    if isinstance(ram, dict):
        used = ram.get("used")
        total = ram.get("total")
        if used and total:
            parts.append(f"RAM: {used}/{total} MB")
    return ". ".join(parts)


# ─── UGV Tool Functions ─────────────────────────────────────────


def tool_ugv_forward(duration="1.0", speed="100"):
    """Move the robot forward."""
    ugv = _get_ugv()
    if not ugv:
        return "Robot not connected"
    ugv.forward(speed=int(speed), duration=float(duration))
    return f"Moved forward for {duration}s"


def tool_ugv_backward(duration="1.0", speed="100"):
    """Move the robot backward."""
    ugv = _get_ugv()
    if not ugv:
        return "Robot not connected"
    ugv.backward(speed=int(speed), duration=float(duration))
    return f"Moved backward for {duration}s"


def tool_ugv_turn_left(duration="0.5", speed="80"):
    """Turn the robot left."""
    ugv = _get_ugv()
    if not ugv:
        return "Robot not connected"
    ugv.turn_left(speed=int(speed), duration=float(duration))
    return "Turned left"


def tool_ugv_turn_right(duration="0.5", speed="80"):
    """Turn the robot right."""
    ugv = _get_ugv()
    if not ugv:
        return "Robot not connected"
    ugv.turn_right(speed=int(speed), duration=float(duration))
    return "Turned right"


def tool_ugv_stop():
    """Emergency stop the robot."""
    ugv = _get_ugv()
    if not ugv:
        return "Robot not connected"
    ugv.stop()
    return "Robot stopped"


def tool_ugv_status():
    """Get robot battery and sensor status."""
    ugv = _get_ugv()
    if not ugv:
        return "Robot not connected"
    return ugv.get_status()


# ─── Tool Registry ──────────────────────────────────────────────
# Maps name → (callable, [param_names])

TOOL_REGISTRY = {
    "get_cpu_temperature": (tool_cpu_temperature, []),
    "get_gpu_temperature": (tool_gpu_temperature, []),
    "get_disk_usage": (tool_disk_usage, []),
    "get_memory_usage": (tool_memory_usage, []),
    "get_uptime": (tool_uptime, []),
    "get_service_status": (tool_service_status, ["service_name"]),
    "restart_service": (tool_restart_service, ["service_name"]),
    "get_network_info": (tool_network_info, []),
    "get_power_mode": (tool_power_mode, []),
    "get_date_time": (tool_date_time, []),
    "list_services": (tool_list_services, []),
    "get_power_consumption": (tool_power_consumption, []),
    "get_cpu_load": (tool_cpu_load, []),
    "get_all_temperatures": (tool_all_temperatures, []),
    "get_system_telemetry": (tool_system_telemetry, []),
    # UGV robot control
    "move_forward": (tool_ugv_forward, ["duration", "speed"]),
    "move_backward": (tool_ugv_backward, ["duration", "speed"]),
    "turn_left": (tool_ugv_turn_left, ["duration", "speed"]),
    "turn_right": (tool_ugv_turn_right, ["duration", "speed"]),
    "stop_robot": (tool_ugv_stop, []),
    "robot_status": (tool_ugv_status, []),
}


def execute_tool(name, arguments):
    if name not in TOOL_REGISTRY:
        return f"Unknown tool: {name}"
    func, param_names = TOOL_REGISTRY[name]
    try:
        if param_names:
            kwargs = {k: arguments.get(k, "") for k in param_names}
            return func(**kwargs)
        return func()
    except Exception as e:
        return f"Tool error: {e}"


# ─── Conversation Memory (SQLite) ────────────────────────────────

class ConversationStore:
    """Persistent conversation history using SQLite with WAL mode."""

    def __init__(self, db_path=CONVERSATIONS_DB):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self.session_id = self._start_session()
        self._lock = threading.Lock()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_name TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_messages_content
                ON messages(content);
        """)
        self.conn.commit()

    def _start_session(self):
        cur = self.conn.execute(
            "INSERT INTO sessions (started_at) VALUES (datetime('now'))"
        )
        self.conn.commit()
        return cur.lastrowid

    def save_message(self, role, content, tool_name=None):
        with self._lock:
            self.conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_name) "
                "VALUES (?, ?, ?, ?)",
                (self.session_id, role, content, tool_name),
            )
            self.conn.commit()

    def get_recent_messages(self, limit=12):
        """Get recent messages for LLM context (current session first, then prior)."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
        # Return in chronological order
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def search_history(self, query, limit=5):
        """Search past conversations by keyword."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        if not rows:
            return f"No conversations found matching '{query}'."
        results = []
        for role, content, ts in rows:
            snippet = content[:100] + ("..." if len(content) > 100 else "")
            results.append(f"[{ts}] {role}: {snippet}")
        return "\n".join(results)

    def end_session(self):
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?",
                (self.session_id,),
            )
            self.conn.commit()

    def close(self):
        self.end_session()
        self.conn.close()


# Global conversation store — initialized in VoiceAssistant or test functions
conversation_store = None


def tool_recall_conversation(query):
    """Search conversation history."""
    if conversation_store is None:
        return "Conversation memory not initialized."
    return conversation_store.search_history(query)


# Add recall to registry
TOOL_REGISTRY["recall_conversation"] = (tool_recall_conversation, ["query"])


# Ollama tool definitions (JSON Schema for the LLM)
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_cpu_temperature",
            "description": "Read the CPU temperature of this Jetson device",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gpu_temperature",
            "description": "Read the GPU temperature of this Jetson device",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_disk_usage",
            "description": "Get disk space usage — total, used, and available",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_usage",
            "description": "Get RAM memory usage — total, used, and available",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_uptime",
            "description": "Get how long the system has been running since last boot",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_status",
            "description": (
                "Check if a systemd service is running. Available services: "
                "voice-assistant, ollama, jetson-capture-images, "
                "jetson-capture-audio, jetson-read-serial, jetson-telemetry"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the systemd service",
                    },
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": (
                "Restart a systemd service. Allowed: jetson-capture-images, "
                "jetson-capture-audio, jetson-read-serial, jetson-telemetry"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the service to restart",
                    },
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_network_info",
            "description": "Get network interface IP addresses and connection status",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_power_mode",
            "description": "Get the current Jetson power mode (10W or MAXN 15W)",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_services",
            "description": "List all available Jetson services and whether each is running or stopped",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    # UGV robot control tools
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Move the robot forward for a given duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string", "description": "Duration in seconds (default 1.0, max 5.0)"},
                    "speed": {"type": "string", "description": "Speed 0-128 (default 100)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_backward",
            "description": "Move the robot backward for a given duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string", "description": "Duration in seconds (default 1.0, max 5.0)"},
                    "speed": {"type": "string", "description": "Speed 0-128 (default 100)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_left",
            "description": "Turn the robot left for a given duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string", "description": "Duration in seconds (default 0.5)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_right",
            "description": "Turn the robot right for a given duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string", "description": "Duration in seconds (default 0.5)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_robot",
            "description": "Stop the robot immediately",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_status",
            "description": "Get robot battery voltage and IMU sensor data",
            "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        },
    },
]

# Gemini function calling tools for complex robot commands
GEMINI_UGV_TOOLS = {
    "function_declarations": [
        {
            "name": "move_forward",
            "description": "Move the robot forward for N seconds",
            "parameters": {
                "type": "OBJECT",
                "properties": {"duration": {"type": "NUMBER", "description": "Seconds to move (max 5)"}},
            },
        },
        {
            "name": "move_backward",
            "description": "Move the robot backward for N seconds",
            "parameters": {
                "type": "OBJECT",
                "properties": {"duration": {"type": "NUMBER", "description": "Seconds to move (max 5)"}},
            },
        },
        {
            "name": "turn_left",
            "description": "Turn the robot left",
            "parameters": {
                "type": "OBJECT",
                "properties": {"duration": {"type": "NUMBER", "description": "Seconds to turn (default 0.5)"}},
            },
        },
        {
            "name": "turn_right",
            "description": "Turn the robot right",
            "parameters": {
                "type": "OBJECT",
                "properties": {"duration": {"type": "NUMBER", "description": "Seconds to turn (default 0.5)"}},
            },
        },
        {
            "name": "stop_robot",
            "description": "Stop the robot immediately",
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "capture_and_describe",
            "description": "Take a photo with the robot's camera and describe what is seen",
            "parameters": {"type": "OBJECT", "properties": {}},
        },
    ],
}


# ─── Audio Helpers ───────────────────────────────────────────────


def rms(audio_bytes):
    """Calculate RMS amplitude of int16 audio data."""
    if len(audio_bytes) < 2:
        return 0
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(samples ** 2))


def save_wav(audio_data, filepath):
    """Save raw int16 audio data to WAV file."""
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)


def play_audio(filepath):
    """Play a WAV file through the USB speaker."""
    try:
        subprocess.run(
            ["aplay", "-D", SPEAKER_DEVICE, filepath],
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        log.warning("Audio playback timed out")
    except FileNotFoundError:
        log.error("aplay not found")


def generate_beep(filepath, freq=880, duration=0.15):
    """Generate a short beep WAV file as acknowledgment tone."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    envelope = np.ones_like(t)
    fade = int(SAMPLE_RATE * 0.01)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    tone = (np.sin(2 * np.pi * freq * t) * envelope * 16000).astype(np.int16)
    save_wav(tone.tobytes(), filepath)


def start_mic_stream():
    """Start arecord subprocess that streams raw PCM from the mic."""
    return subprocess.Popen(
        [
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
            "-t", "raw",
            "--buffer-size", "8192",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


# ─── STT: Whisper.cpp ───────────────────────────────────────────


def transcribe(wav_path):
    """Run whisper.cpp on a WAV file, return transcribed text."""
    if not os.path.exists(WHISPER_CLI):
        log.error(f"whisper-cli not found at {WHISPER_CLI}")
        return ""
    if not os.path.exists(WHISPER_MODEL):
        log.error(f"Whisper model not found at {WHISPER_MODEL}")
        return ""

    try:
        result = subprocess.run(
            [
                WHISPER_CLI,
                "-m", WHISPER_MODEL,
                "-f", wav_path,
                "--no-timestamps",
                "--no-prints",
                "-t", "4",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = result.stdout.strip()
        if not text or "[BLANK" in text or "(inaudible)" in text.lower():
            return ""
        text = text.strip("[] ")
        log.info(f'STT: "{text}"')
        return text
    except subprocess.TimeoutExpired:
        log.error("Whisper transcription timed out")
        return ""
    except Exception as e:
        log.error(f"Whisper error: {e}")
        return ""


# ─── LLM: Ollama (Streaming + Tool Calling) ─────────────────────

conversation_history = []
MAX_HISTORY = 6

# Set from CLI args
_tools_enabled = False  # Dispatch handles all tools; avoids 400 error + retry overhead

# Model swap lock — only one model in VRAM at a time
_model_lock = threading.Lock()


def _stream_ollama(messages, sentence_cb=None, use_tools=False, model=None, options=None):
    """Stream an Ollama chat response, emitting complete sentences for TTS.

    Returns (full_text, tool_calls_or_None).
    Reads NDJSON line-by-line from the streaming HTTP response.
    model: override the default OLLAMA_MODEL (used for vision/reasoning).
    options: override default Ollama options (num_predict, temperature, etc.).
    """
    active_model = model or OLLAMA_MODEL
    default_options = {"num_predict": 256, "temperature": 0.7}
    if options:
        default_options.update(options)
    payload = {
        "model": active_model,
        "messages": messages,
        "stream": True,
        "options": default_options,
    }
    if use_tools and _tools_enabled:
        payload["tools"] = OLLAMA_TOOLS

    try:
        req = Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp_handle = urlopen(req, timeout=120)
        except URLError as e:
            # Model doesn't support tools → retry without them
            if "tools" in payload and "400" in str(e):
                log.warning(f"{active_model} does not support tools, falling back to plain chat")
                payload.pop("tools")
                req = Request(
                    f"{OLLAMA_URL}/api/chat",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                resp_handle = urlopen(req, timeout=120)
            else:
                raise

        full_text = ""
        sentence_buf = ""
        tool_calls = None

        with resp_handle as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message", {})
                content = msg.get("content", "")

                if content:
                    full_text += content
                    if sentence_cb:
                        sentence_buf += content
                        # Split on sentence boundaries (. ! ?) followed by whitespace
                        parts = re.split(r"(?<=[.!?])\s+", sentence_buf)
                        if len(parts) > 1:
                            for s in parts[:-1]:
                                s = s.strip()
                                if s:
                                    sentence_cb(s)
                            sentence_buf = parts[-1]

                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]

                if chunk.get("done"):
                    break

        # Emit any remaining buffered text as the final sentence
        if sentence_cb and sentence_buf.strip():
            sentence_cb(sentence_buf.strip())

        return full_text, tool_calls

    except URLError as e:
        log.error(f"Ollama connection error: {e}")
        err = "Sorry, I can't reach the language model right now."
        if sentence_cb:
            sentence_cb(err)
        return err, None
    except Exception as e:
        log.error(f"Ollama error: {e}")
        err = "Sorry, something went wrong."
        if sentence_cb:
            sentence_cb(err)
        return err, None


def query_ollama(prompt, sentence_cb=None):
    """Send prompt to Ollama with tool calling and streaming TTS.

    sentence_cb: called with each complete sentence for the TTS worker.
    Returns the full response text.

    Flow:
      1. Stream LLM response with tools enabled
      2. If LLM returns tool_calls → execute tools → stream follow-up
      3. Each complete sentence is emitted via sentence_cb for TTS
    """
    global conversation_history

    # Persist to SQLite if available
    if conversation_store:
        conversation_store.save_message("user", prompt)

    conversation_history.append({"role": "user", "content": prompt})
    if len(conversation_history) > MAX_HISTORY * 2:
        conversation_history = conversation_history[-MAX_HISTORY * 2:]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
    ]

    full_text, tool_calls = _stream_ollama(
        messages, sentence_cb=sentence_cb, use_tools=True,
    )

    # Handle tool calls (max 3 rounds to prevent loops)
    rounds = 0
    while tool_calls and rounds < 3:
        rounds += 1
        messages.append({
            "role": "assistant",
            "content": full_text or "",
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments", {})
            # Ollama may return arguments as a JSON string
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except json.JSONDecodeError:
                    fn_args = {}
            log.info(f"Tool: {fn_name}({fn_args})")
            result = execute_tool(fn_name, fn_args)
            log.info(f"Result: {result[:120]}")
            messages.append({"role": "tool", "content": str(result)})

        # Stream the follow-up (no more tools — just the spoken answer)
        full_text, tool_calls = _stream_ollama(
            messages, sentence_cb=sentence_cb, use_tools=False,
        )

    if full_text:
        conversation_history.append({"role": "assistant", "content": full_text})
        if conversation_store:
            conversation_store.save_message("assistant", full_text)

    log.info(f'LLM: "{full_text[:80]}{"..." if len(full_text) > 80 else ""}"')
    return full_text


# ─── Camera Vision (P4) ─────────────────────────────────────────

def capture_camera_frame():
    """Capture a frame from USB camera, return base64 JPEG string."""
    try:
        import cv2
    except ImportError:
        log.error("OpenCV not available for camera capture")
        return None

    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        log.warning(f"Cannot open camera {CAMERA_DEVICE}, trying image dir fallback")
        return _latest_image_fallback()

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            log.warning("Failed to read camera frame")
            return _latest_image_fallback()

        # Resize to 640px wide for faster inference
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    finally:
        cap.release()


def _latest_image_fallback():
    """Get the most recent image from the training-data images dir."""
    if not os.path.isdir(IMAGE_DIR):
        return None
    images = sorted(
        globmod.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        + globmod.glob(os.path.join(IMAGE_DIR, "*.png")),
        key=os.path.getmtime,
    )
    if not images:
        return None
    latest = images[-1]
    log.info(f"Using fallback image: {latest}")
    with open(latest, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _query_gemini_vision(img_b64, prompt):
    """Send image to Gemini API for vision analysis. Returns text or None on failure."""
    if not GEMINI_API_KEY:
        return None
    payload = json.dumps({
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {"maxOutputTokens": 300, "thinkingConfig": {"thinkingBudget": 0}},
    })
    try:
        req = Request(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        return (data.get("candidates", [{}])[0]
                .get("content", {}).get("parts", [{}])[0].get("text"))
    except Exception as e:
        log.warning(f"Gemini API failed: {e}")
        return None


def query_vision(prompt, sentence_cb=None):
    """Capture camera image and analyze it via Gemini API.

    Falls back to camera metadata + local LLM if Gemini is unavailable.
    No local model swap needed either way.
    """
    if sentence_cb:
        sentence_cb("Let me take a look...")

    img_b64 = capture_camera_frame()
    user_prompt = prompt or "Describe what you see concisely."

    # Try Gemini API vision if we have an image and API key
    if img_b64 and GEMINI_API_KEY:
        log.info(f"Vision query via Gemini ({GEMINI_MODEL})...")
        gemini_prompt = (
            f"{user_prompt} Keep your response concise — 2-3 sentences max, "
            "suitable for text-to-speech."
        )
        result = _query_gemini_vision(img_b64, gemini_prompt)
        if result:
            log.info(f"Gemini vision: {result[:80]}...")
            if sentence_cb:
                sentence_cb(result)
            if conversation_store:
                conversation_store.save_message("user", f"[vision] {user_prompt}")
                conversation_store.save_message("assistant", result)
            return result
        log.warning("Gemini vision failed, falling back to metadata")

    # Fallback: camera metadata + local LLM
    camera_available = os.path.exists(CAMERA_DEVICE)
    image_count = 0
    latest_info = ""

    if os.path.isdir(IMAGE_DIR):
        images = sorted(
            globmod.glob(os.path.join(IMAGE_DIR, "*.jpg"))
            + globmod.glob(os.path.join(IMAGE_DIR, "*.png")),
            key=os.path.getmtime,
        )
        image_count = len(images)
        if images:
            latest_ts = datetime.fromtimestamp(
                os.path.getmtime(images[-1])
            ).strftime("%Y-%m-%d %H:%M:%S")
            latest_info = f"Latest capture: {os.path.basename(images[-1])} at {latest_ts}"

    context = (
        f"Camera ({CAMERA_DEVICE}): {'accessible' if camera_available else 'unavailable'}, "
        f"{image_count} images captured. {latest_info}"
    )
    log.info(f"Vision fallback (metadata): {context}")

    messages = [
        {"role": "system", "content": (
            "You have a USB camera but cannot see image contents right now. "
            f"Camera info: {context}\n"
            "Answer based on this metadata. Be honest about your limitations."
        )},
        {"role": "user", "content": user_prompt},
    ]
    full_text, _ = _stream_ollama(messages, sentence_cb=sentence_cb)

    if conversation_store and full_text:
        conversation_store.save_message("user", f"[vision] {user_prompt}")
        conversation_store.save_message("assistant", full_text)

    return full_text


# ─── Reasoning Mode (P5) ────────────────────────────────────────

def query_reasoning(prompt, sentence_cb=None):
    """Route complex questions to qwen3:0.6b for chain-of-thought.

    Uses /think prefix for qwen3's native thinking mode.
    Strips <think>...</think> tags before sending to TTS.
    """
    if sentence_cb:
        sentence_cb("Let me think about that...")

    messages = [
        {"role": "system", "content": (
            "/think\n"
            "You are a reasoning assistant. Think briefly, then give a "
            "clear, concise answer in 2-4 sentences. Keep your thinking short."
        )},
        {"role": "user", "content": prompt},
    ]

    # Collect full text first (don't stream think tags to TTS)
    # Use higher token limit since reasoning needs room for <think> + answer
    with _model_lock:
        log.info(f"Swapping to {REASONING_MODEL} for reasoning...")
        full_text, _ = _stream_ollama(
            messages, sentence_cb=None, model=REASONING_MODEL,
            options={"num_predict": 384, "temperature": 0.6},
        )
        # Swap back
        log.info(f"Swapping back to {OLLAMA_MODEL}...")
        try:
            req = Request(
                f"{OLLAMA_URL}/api/chat",
                data=json.dumps({
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"num_predict": 1},
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urlopen(req, timeout=30)
        except Exception:
            log.warning("Failed to preload default model back")

    # Strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
    if not cleaned:
        # Everything was in think tags — extract last meaningful paragraph from inside
        think_match = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            # Take the last paragraph as the most likely conclusion
            paragraphs = [p.strip() for p in think_content.split("\n\n") if p.strip()]
            cleaned = paragraphs[-1] if paragraphs else think_content[-300:]
        else:
            # No closing </think> tag — model ran out of tokens mid-thought
            # Strip the opening tag and take what we can
            cleaned = re.sub(r"<think>", "", full_text).strip()
            if cleaned:
                # Add a note that reasoning was truncated
                cleaned = cleaned.split("\n")[-1]  # last line is usually most useful
        if not cleaned:
            cleaned = "I ran out of thinking time, but let me try to answer with what I know."

    # Now stream cleaned text to TTS sentence by sentence
    if sentence_cb and cleaned:
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        for s in sentences:
            s = s.strip()
            if s:
                sentence_cb(s)

    if conversation_store and cleaned:
        conversation_store.save_message("user", f"[reasoning] {prompt}")
        conversation_store.save_message("assistant", cleaned)

    log.info(f'Reasoning: "{cleaned[:80]}{"..." if len(cleaned) > 80 else ""}"')
    return cleaned


# ─── Prompt-Based Dispatch (P1) ──────────────────────────────────
# Intercepts tool-able questions via regex BEFORE hitting the LLM.
# Falls through to LLM for anything that doesn't match.

def _extract_service_name(text):
    """Extract a service name from user text."""
    for svc in ALLOWED_STATUS_SERVICES:
        if svc.replace("-", " ") in text.lower() or svc in text.lower():
            return {"service_name": svc}
    return {"service_name": "unknown"}


def _extract_recall_query(text):
    """Extract search query for conversation recall."""
    # Remove trigger words and use the rest as query
    cleaned = re.sub(
        r"\b(remember|recall|earlier|last time|history|what did I|"
        r"did I ask|did we talk)\b", "", text, flags=re.I,
    ).strip()
    return {"query": cleaned if cleaned else ""}


DISPATCH_PATTERNS = [
    # Emergency stop (P0 — highest priority, checked first)
    (re.compile(r"\b(stop|halt|freeze|don't move)\b", re.I),
     "stop_robot", None, "{result}"),

    # Telemetry (P2) — rich data from JSONL
    (re.compile(r"\b(power|watt|consumption|draw)\b", re.I),
     "get_power_consumption", None, "Power consumption: {result}"),
    (re.compile(r"\b(cpu|processor)\b.*\b(load|usage|busy|utilization)\b", re.I),
     "get_cpu_load", None, "CPU load: {result}"),
    (re.compile(r"\ball\b.*\btemp", re.I),
     "get_all_temperatures", None, "Temperatures: {result}"),
    (re.compile(r"\b(system status|full telemetry|system telemetry|overview)\b", re.I),
     "get_system_telemetry", None, "System telemetry: {result}"),

    # Basic tools (use \btemp without trailing \b so "temperature" matches)
    (re.compile(r"\b(cpu|processor)\b.*\b(temp|hot|thermal|heat)", re.I),
     "get_cpu_temperature", None, "The CPU temperature is {result}."),
    (re.compile(r"\bgpu\b.*\b(temp|hot|thermal|heat)", re.I),
     "get_gpu_temperature", None, "The GPU temperature is {result}."),
    (re.compile(r"\b(disk|storage|space)\b", re.I),
     "get_disk_usage", None, "Disk usage: {result}."),
    (re.compile(r"\b(memory|ram)\b.*\b(usage|free|available|used)\b", re.I),
     "get_memory_usage", None, "Memory: {result}."),
    (re.compile(r"\b(uptime|up time|running since|how long)\b", re.I),
     "get_uptime", None, "System uptime: {result}."),
    (re.compile(r"\b(network|ip|address|wifi|ethernet)\b", re.I),
     "get_network_info", None, "Network info:\n{result}"),
    (re.compile(r"\b(power mode|nvpmodel|performance mode)\b", re.I),
     "get_power_mode", None, "Power mode: {result}"),
    (re.compile(r"\b(time|date|day|what day)\b", re.I),
     "get_date_time", None, "It's {result}."),
    (re.compile(r"\b(list|all)\b.*\bservice", re.I),
     "list_services", None, "Services:\n{result}"),
    (re.compile(r"\b(status|check|running)\b.*\bservice\b|\bservice\b.*\b(status|running|check)\b", re.I),
     "get_service_status", _extract_service_name,
     "Service status: {result}"),

    # Conversation recall (P3)
    (re.compile(r"\b(remember|recall|earlier|last time|history|what did I|did I ask|did we talk)\b", re.I),
     "recall_conversation", _extract_recall_query,
     "Here's what I found: {result}"),

    # UGV robot movement
    (re.compile(r"\b(go|move|drive)\b.*\b(forward|ahead|straight)\b", re.I),
     "move_forward", None, "{result}"),
    (re.compile(r"\b(go|move|drive)\b.*\b(back|backward|reverse)\b", re.I),
     "move_backward", None, "{result}"),
    (re.compile(r"\b(turn|go|rotate)\b.*\bleft\b", re.I),
     "turn_left", None, "{result}"),
    (re.compile(r"\b(turn|go|rotate)\b.*\bright\b", re.I),
     "turn_right", None, "{result}"),
    (re.compile(r"\b(battery|charge|robot status|imu)\b", re.I),
     "robot_status", None, "Robot status: {result}"),
]

# Vision patterns (P4) — handled separately since they don't use tool registry
VISION_PATTERN = re.compile(
    r"\b(what do you see|what can you see|look|camera|take a (picture|photo|look)|"
    r"describe what|show me|what's in front|what is that)\b", re.I,
)

# Exploration pattern (P3.5) — autonomous vision-guided navigation
EXPLORE_PATTERN = re.compile(
    r"\b(explore|navigate|look around|scout|patrol|wander)\b", re.I,
)

# Navigate-to pattern — "go to the door", "find the red chair"
NAVIGATE_TO_PATTERN = re.compile(
    r"\b(go to|find|navigate to|drive to|head to|move to)\b\s+(?:the\s+)?(.+)",
    re.I,
)

# Reasoning patterns (P5)
REASONING_PATTERN = re.compile(
    r"\b(explain why|why does|how does|think about|step by step|reason|analyze|"
    r"figure out|work through|break down)\b", re.I,
)


# ─── Autonomous Exploration (Vision + UGV) ─────────────────────

# Flag to request exploration abort from any thread
_explore_abort = threading.Event()


def _execute_gemini_tool_call(func_name, func_args):
    """Execute a Gemini function call and return the result string."""
    if func_name == "move_forward":
        return tool_ugv_forward(duration=str(func_args.get("duration", 1.0)))
    elif func_name == "move_backward":
        return tool_ugv_backward(duration=str(func_args.get("duration", 1.0)))
    elif func_name == "turn_left":
        return tool_ugv_turn_left(duration=str(func_args.get("duration", 0.5)))
    elif func_name == "turn_right":
        return tool_ugv_turn_right(duration=str(func_args.get("duration", 0.5)))
    elif func_name == "stop_robot":
        return tool_ugv_stop()
    elif func_name == "capture_and_describe":
        img_b64 = capture_camera_frame()
        if img_b64 and GEMINI_API_KEY:
            result = _query_gemini_vision(img_b64, "Describe what you see concisely in 1-2 sentences.")
            return result or "Could not analyze image"
        return "Camera not available"
    return f"Unknown function: {func_name}"


def _query_gemini_with_tools(prompt, img_b64=None):
    """Send a prompt (optionally with image) to Gemini with UGV function calling.

    Returns list of (function_name, args) tuples, or a text response.
    """
    if not GEMINI_API_KEY:
        return None

    parts = []
    if img_b64:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": img_b64}})
    parts.append({"text": prompt})

    payload = json.dumps({
        "contents": [{"role": "user", "parts": parts}],
        "tools": [GEMINI_UGV_TOOLS],
        "generationConfig": {"maxOutputTokens": 300, "thinkingConfig": {"thinkingBudget": 0}},
    })

    try:
        req = Request(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            data=payload.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())

        candidate = data.get("candidates", [{}])[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Check for function calls
        calls = []
        text_parts = []
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append((fc["name"], fc.get("args", {})))
            elif "text" in part:
                text_parts.append(part["text"])

        if calls:
            return calls
        return " ".join(text_parts) if text_parts else None
    except Exception as e:
        log.warning(f"Gemini tool call failed: {e}")
        return None


def explore_autonomous(duration=30, sentence_cb=None):
    """Vision-guided autonomous exploration.

    Loop: capture frame → Gemini decides direction → move → narrate.
    """
    ugv = _get_ugv()
    if not ugv:
        msg = "Cannot explore — robot not connected."
        if sentence_cb:
            sentence_cb(msg)
        return msg

    if not GEMINI_API_KEY:
        msg = "Cannot explore — Gemini API key not set."
        if sentence_cb:
            sentence_cb(msg)
        return msg

    duration = min(float(duration), MAX_AUTONOMOUS_DURATION)
    _explore_abort.clear()
    log.info(f"Starting autonomous exploration for {duration}s")
    if sentence_cb:
        sentence_cb("Starting exploration. I'll narrate what I see.")

    start = time.time()
    narration_parts = []

    while time.time() - start < duration:
        if _explore_abort.is_set():
            ugv.stop()
            msg = "Exploration stopped."
            if sentence_cb:
                sentence_cb(msg)
            narration_parts.append(msg)
            break

        # 1. Capture frame
        img_b64 = capture_camera_frame()
        if not img_b64:
            log.warning("Explore: no camera frame, waiting...")
            time.sleep(2)
            continue

        # 2. Ask Gemini what to do
        prompt = (
            "You are controlling a robot with a camera. Describe what you see in "
            "one short sentence, then call exactly ONE function to move the robot. "
            "If there's an obstacle ahead (wall, furniture, object closer than 1 meter), "
            "turn left or right. If the path is clear, move forward. "
            "If you see something interesting, stop and describe it."
        )
        result = _query_gemini_with_tools(prompt, img_b64=img_b64)

        if result is None:
            log.warning("Explore: Gemini failed, stopping")
            ugv.stop()
            break

        # 3. Execute the response
        if isinstance(result, list):
            # Function calls from Gemini
            for func_name, func_args in result:
                log.info(f"Explore → {func_name}({func_args})")
                action_result = _execute_gemini_tool_call(func_name, func_args)
                if sentence_cb and func_name == "capture_and_describe":
                    sentence_cb(action_result)
                    narration_parts.append(action_result)
        elif isinstance(result, str):
            # Text response (Gemini described but didn't call a function)
            log.info(f"Explore narration: {result[:80]}")
            if sentence_cb:
                sentence_cb(result)
            narration_parts.append(result)
            # Default: move forward briefly since no function was called
            ugv.forward(speed=80, duration=0.8)

        time.sleep(0.5)  # Brief pause between iterations

    ugv.stop()
    summary = "Exploration complete. " + " ".join(narration_parts[-3:]) if narration_parts else "Exploration complete."
    if conversation_store:
        conversation_store.save_message("user", "[explore] autonomous exploration")
        conversation_store.save_message("assistant", summary)
    return summary


def navigate_to(target, sentence_cb=None):
    """Navigate toward a specific target using vision.

    Repeatedly capture → ask Gemini if target is visible → move toward it.
    """
    ugv = _get_ugv()
    if not ugv:
        msg = "Cannot navigate — robot not connected."
        if sentence_cb:
            sentence_cb(msg)
        return msg

    if not GEMINI_API_KEY:
        msg = "Cannot navigate — Gemini API key not set."
        if sentence_cb:
            sentence_cb(msg)
        return msg

    _explore_abort.clear()
    log.info(f"Navigating to: {target}")
    if sentence_cb:
        sentence_cb(f"Looking for {target}.")

    for attempt in range(15):  # Max 15 attempts (~30s)
        if _explore_abort.is_set():
            ugv.stop()
            break

        img_b64 = capture_camera_frame()
        if not img_b64:
            time.sleep(1)
            continue

        prompt = (
            f"You are controlling a robot looking for: {target}. "
            "Look at the camera image. If you can see the target, call move_forward "
            "to approach it (short duration). If the target is to the left, call turn_left. "
            "If to the right, call turn_right. If you've reached it (it's very close/centered), "
            "call stop_robot. If you can't see it at all, call turn_right to search. "
            "Also briefly describe what you see."
        )
        result = _query_gemini_with_tools(prompt, img_b64=img_b64)

        if result is None:
            ugv.stop()
            break

        stopped = False
        if isinstance(result, list):
            for func_name, func_args in result:
                log.info(f"Navigate → {func_name}({func_args})")
                _execute_gemini_tool_call(func_name, func_args)
                if func_name == "stop_robot":
                    stopped = True
        elif isinstance(result, str):
            if sentence_cb:
                sentence_cb(result)
            ugv.forward(speed=60, duration=0.5)

        if stopped:
            msg = f"I think I found {target}!"
            if sentence_cb:
                sentence_cb(msg)
            if conversation_store:
                conversation_store.save_message("user", f"[navigate] find {target}")
                conversation_store.save_message("assistant", msg)
            return msg

        time.sleep(0.5)

    ugv.stop()
    msg = f"Could not find {target} after searching."
    if sentence_cb:
        sentence_cb(msg)
    return msg


def gemini_complex_command(text, sentence_cb=None):
    """Handle complex multi-step robot commands via Gemini function calling.

    e.g., "Go forward until you see something interesting, then describe it"
    """
    ugv = _get_ugv()
    if not ugv:
        msg = "Robot not connected."
        if sentence_cb:
            sentence_cb(msg)
        return msg

    if not GEMINI_API_KEY:
        return query_ollama(text, sentence_cb=sentence_cb)

    log.info(f"Gemini complex command: {text[:60]}")

    # Multi-turn loop: let Gemini orchestrate tool calls
    messages = []
    img_b64 = capture_camera_frame()

    prompt = (
        f"You are controlling a robot. The user said: \"{text}\"\n"
        "You have these tools: move_forward, move_backward, turn_left, turn_right, "
        "stop_robot, capture_and_describe. Execute the user's request by calling "
        "the appropriate tools. Call one or more tools now."
    )

    for step in range(5):  # Max 5 tool-call rounds
        result = _query_gemini_with_tools(prompt, img_b64=img_b64 if step == 0 else None)

        if result is None:
            break

        if isinstance(result, list):
            for func_name, func_args in result:
                log.info(f"Complex → {func_name}({func_args})")
                action_result = _execute_gemini_tool_call(func_name, func_args)
                if sentence_cb and action_result:
                    sentence_cb(action_result)
                if func_name == "stop_robot":
                    return action_result
        elif isinstance(result, str):
            if sentence_cb:
                sentence_cb(result)
            return result

        # Capture new frame for next round
        img_b64 = capture_camera_frame()
        prompt = "Continue executing the user's request. What should the robot do next?"

    ugv.stop()
    msg = "Done."
    if sentence_cb:
        sentence_cb(msg)
    return msg


def dispatch(text, sentence_cb=None):
    """Route user text to the best handler: tools → explore → vision → reasoning → LLM.

    Priority: stop (P0) → tools (P1-P3) → explore (P3.5) → vision (P4) → reasoning (P5) → LLM.
    """
    # P0-P3: Regex tool dispatch — instant, no LLM needed
    for pattern, tool_name, arg_extractor, template in DISPATCH_PATTERNS:
        if pattern.search(text):
            # "stop" during exploration aborts it
            if tool_name == "stop_robot":
                _explore_abort.set()
            args = arg_extractor(text) if arg_extractor else {}
            log.info(f"Dispatch → tool: {tool_name}({args})")
            result = execute_tool(tool_name, args)
            response = template.format(result=result)
            if sentence_cb:
                sentence_cb(response)
            if conversation_store:
                conversation_store.save_message("user", text, tool_name=tool_name)
                conversation_store.save_message("assistant", response, tool_name=tool_name)
            return response

    # P3.5: Autonomous exploration — "explore the room", "look around"
    if EXPLORE_PATTERN.search(text):
        log.info(f"Dispatch → explore: {text[:60]}")
        return explore_autonomous(duration=30, sentence_cb=sentence_cb)

    # P3.5: Navigate to target — "go to the door", "find the red chair"
    nav_match = NAVIGATE_TO_PATTERN.search(text)
    if nav_match:
        target = nav_match.group(2).strip()
        if target and len(target) > 2:
            log.info(f"Dispatch → navigate_to: {target}")
            return navigate_to(target, sentence_cb=sentence_cb)

    # P4: Vision — "what do you see?"
    if VISION_PATTERN.search(text):
        log.info(f"Dispatch → vision: {text[:60]}")
        return query_vision(text, sentence_cb=sentence_cb)

    # P5: Reasoning — "explain why...", "step by step"
    if REASONING_PATTERN.search(text):
        log.info(f"Dispatch → reasoning: {text[:60]}")
        return query_reasoning(text, sentence_cb=sentence_cb)

    # P6: Complex robot commands via Gemini (if UGV is connected)
    if _ugv_driver is not None and GEMINI_API_KEY:
        # Check if the command seems robot-related
        robot_keywords = re.compile(
            r"\b(robot|drive|move|go|turn|spin|circle|forward|backward|"
            r"approach|avoid|follow|around|room|door|wall|chair|table)\b", re.I,
        )
        if robot_keywords.search(text):
            log.info(f"Dispatch → Gemini complex command: {text[:60]}")
            return gemini_complex_command(text, sentence_cb=sentence_cb)

    # Fallthrough → LLM
    log.info(f"Dispatch → LLM: {text[:60]}")
    return query_ollama(text, sentence_cb=sentence_cb)


# ─── TTS: Piper ─────────────────────────────────────────────────


def speak(text, output_path):
    """Convert text to speech using Piper TTS, save to WAV."""
    if not text:
        return False
    try:
        result = subprocess.run(
            [
                PIPER_CLI,
                "--model", PIPER_VOICE,
                "--output_file", output_path,
            ],
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.error(f"Piper error: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        log.error(f"Piper not found at {PIPER_CLI}")
        return False
    except subprocess.TimeoutExpired:
        log.error("Piper TTS timed out")
        return False


# ─── Voice Assistant ─────────────────────────────────────────────


class VoiceAssistant:
    """Main voice assistant loop using arecord for audio capture."""

    def __init__(self, skip_wakeword=False):
        global conversation_store
        self.running = True
        self.skip_wakeword = skip_wakeword
        self.tmp_dir = tempfile.mkdtemp(prefix="voice_assist_")
        self.beep_path = os.path.join(self.tmp_dir, "beep.wav")
        self.beep_done_path = os.path.join(self.tmp_dir, "beep_done.wav")
        generate_beep(self.beep_path, freq=880, duration=0.15)
        generate_beep(self.beep_done_path, freq=660, duration=0.1)
        self._tts_counter = 0

        # Initialize persistent conversation memory
        try:
            conversation_store = ConversationStore()
            log.info(f"Conversation memory: {CONVERSATIONS_DB}")
        except Exception as e:
            log.warning(f"Conversation memory unavailable: {e}")
            conversation_store = None

        if not skip_wakeword:
            log.info("Loading wake word model...")
            from openwakeword.model import Model
            self.wakeword_model = Model(
                wakeword_models=[WAKE_WORD_MODEL],
                inference_framework="onnx",
            )
            log.info(f"Wake word model loaded: {WAKE_WORD_MODEL}")
        else:
            self.wakeword_model = None
            log.info("Wake word disabled — press Enter to talk")

    def listen_for_wake_word(self):
        """Stream mic audio via arecord, feed to wake word model."""
        log.info(f'Waiting for wake word: "{WAKE_WORD_MODEL}"...')
        proc = start_mic_stream()

        try:
            while self.running:
                # Read 80ms chunk (1280 samples * 2 bytes)
                raw = proc.stdout.read(CHUNK_BYTES)
                if not raw or len(raw) < CHUNK_BYTES:
                    break

                # Feed raw int16 audio to openwakeword
                audio_int16 = np.frombuffer(raw, dtype=np.int16)
                self.wakeword_model.predict(audio_int16)

                # Check predictions
                for name, scores in self.wakeword_model.prediction_buffer.items():
                    if len(scores) > 0 and scores[-1] > WAKE_THRESHOLD:
                        log.info(f"Wake word detected! ({name}: {scores[-1]:.2f})")
                        return True
        except Exception as e:
            log.error(f"Wake word listener error: {e}")
        finally:
            proc.terminate()
            proc.wait()

        return False

    def record_speech(self):
        """Record speech via arecord until silence detected."""
        log.info("Recording speech...")
        proc = start_mic_stream()
        chunks = []
        silence_start = None
        recording_start = time.time()

        try:
            while self.running:
                raw = proc.stdout.read(CHUNK_BYTES)
                if not raw or len(raw) < CHUNK_BYTES:
                    break

                chunks.append(raw)
                elapsed = time.time() - recording_start

                if elapsed > MAX_RECORD_SECONDS:
                    log.info(f"Max recording time ({MAX_RECORD_SECONDS}s) reached")
                    break

                # Silence detection
                amplitude = rms(raw)
                if amplitude < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        if elapsed > MIN_RECORD_SECONDS:
                            log.info(f"Silence detected, recorded {elapsed:.1f}s")
                            break
                else:
                    silence_start = None
        except Exception as e:
            log.error(f"Recording error: {e}")
        finally:
            proc.terminate()
            proc.wait()

        if not chunks:
            return None

        audio_data = b"".join(chunks)
        duration = len(audio_data) / (SAMPLE_RATE * 2)  # 2 bytes per sample
        if duration < MIN_RECORD_SECONDS:
            log.info(f"Recording too short ({duration:.1f}s), ignoring")
            return None

        wav_path = os.path.join(self.tmp_dir, "recording.wav")
        save_wav(audio_data, wav_path)
        log.info(f"Recorded {duration:.1f}s")
        return wav_path

    def _tts_worker(self, tts_queue):
        """Background thread: speak sentences as they arrive from the LLM stream."""
        while True:
            sentence = tts_queue.get()
            if sentence is None:  # poison pill — done
                break
            wav_path = os.path.join(self.tmp_dir, f"stream_{self._tts_counter}.wav")
            self._tts_counter += 1
            if speak(sentence, wav_path):
                play_audio(wav_path)
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def process_conversation(self):
        """Record → Transcribe → LLM (with tools) → Streaming TTS.

        The LLM response is streamed token-by-token. Complete sentences are
        pushed to a TTS worker thread that speaks them immediately, so the
        user hears the first sentence while the LLM is still generating.
        """
        play_audio(self.beep_path)

        wav_path = self.record_speech()
        if not wav_path:
            log.info("No speech detected")
            play_audio(self.beep_done_path)
            return

        log.info("Transcribing...")
        text = transcribe(wav_path)
        if not text:
            log.info("Empty transcription")
            play_audio(self.beep_done_path)
            return

        log.info("Thinking...")

        # Start background TTS worker
        tts_queue = queue.Queue()
        tts_thread = threading.Thread(
            target=self._tts_worker, args=(tts_queue,), daemon=True,
        )
        tts_thread.start()

        def on_sentence(sentence):
            log.info(f'TTS: "{sentence[:60]}"')
            tts_queue.put(sentence)

        response = dispatch(text, sentence_cb=on_sentence)

        # Signal worker to finish and wait for all sentences to play
        tts_queue.put(None)
        tts_thread.join(timeout=60)

        if not response:
            log.info("Empty LLM response")
            play_audio(self.beep_done_path)

    def run(self):
        """Main loop."""
        log.info("=" * 50)
        log.info("Jetson Voice Assistant started")
        log.info(f"  Model:     {OLLAMA_MODEL}")
        log.info(f"  Wake word: {WAKE_WORD_MODEL}")
        log.info(f"  Mic:       {MIC_DEVICE}")
        log.info(f"  Speaker:   {SPEAKER_DEVICE}")
        log.info(f"  STT:       whisper.cpp tiny.en")
        log.info(f"  TTS:       Piper lessac-medium (streaming)")
        log.info(f"  Tools:     {'enabled' if _tools_enabled else 'disabled'}"
                 f" ({len(TOOL_REGISTRY)} functions)")
        log.info(f"  Dispatch:  {len(DISPATCH_PATTERNS)} patterns + vision + explore + reasoning")
        log.info(f"  Vision:    {'Gemini API (' + GEMINI_MODEL + ')' if GEMINI_API_KEY else 'metadata-only (no API key)'}")
        log.info(f"  Reasoning: {REASONING_MODEL}")
        log.info(f"  UGV:       {UGV_PORT}@{UGV_BAUD} ({'connected' if _ugv_driver else 'not connected'})")
        log.info(f"  Memory:    {'SQLite' if conversation_store else 'in-memory only'}")
        log.info("=" * 50)

        self._self_test()

        while self.running:
            try:
                if self.skip_wakeword:
                    try:
                        input("\nPress Enter to talk (Ctrl+C to quit)...")
                    except EOFError:
                        break
                    self.process_conversation()
                else:
                    if self.listen_for_wake_word():
                        self.process_conversation()
                        self.wakeword_model.reset()
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error(f"Loop error: {e}")
                time.sleep(1)

        self.shutdown()

    def _self_test(self):
        """Verify all components are working."""
        issues = []

        if not os.path.exists(WHISPER_CLI):
            issues.append(f"whisper-cli not found: {WHISPER_CLI}")
        if not os.path.exists(WHISPER_MODEL):
            issues.append(f"Whisper model not found: {WHISPER_MODEL}")
        if not os.path.exists(PIPER_VOICE):
            issues.append(f"Piper voice not found: {PIPER_VOICE}")

        # Test mic
        try:
            proc = subprocess.run(
                ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                 "-c", "1", "-t", "raw", "-d", "1"],
                capture_output=True, timeout=5,
            )
            if proc.returncode != 0:
                issues.append(f"Mic {MIC_DEVICE} not accessible")
        except Exception:
            issues.append(f"Mic test failed for {MIC_DEVICE}")

        # Test Ollama
        try:
            req = Request(f"{OLLAMA_URL}/api/tags")
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if not any(OLLAMA_MODEL in m for m in models):
                    issues.append(f"Ollama model {OLLAMA_MODEL} not found")
        except Exception:
            issues.append("Ollama not reachable")

        if issues:
            log.warning("Self-test issues:")
            for issue in issues:
                log.warning(f"  - {issue}")
        else:
            log.info("Self-test passed — all components ready")

    def shutdown(self):
        """Clean shutdown."""
        self.running = False
        _explore_abort.set()
        log.info("Shutting down...")
        # Stop robot motors before anything else
        if _ugv_driver:
            try:
                _ugv_driver.close()
            except Exception:
                pass
        if conversation_store:
            try:
                conversation_store.close()
            except Exception:
                pass
        import shutil
        try:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        except Exception:
            pass


def main():
    global _tools_enabled

    import argparse
    parser = argparse.ArgumentParser(description="Jetson Voice Assistant")
    parser.add_argument("--no-wakeword", action="store_true",
                        help="Skip wake word, press Enter to talk")
    parser.add_argument("--no-tools", action="store_true",
                        help="Disable tool calling (plain chat only)")
    parser.add_argument("--mic", type=str, default=None,
                        help="Override mic device (e.g., plughw:0,0)")
    parser.add_argument("--test-mic", action="store_true",
                        help="Record 5s and transcribe, then exit")
    parser.add_argument("--test-tools", action="store_true",
                        help="Run each tool and print results, then exit")
    parser.add_argument("--test-dispatch", action="store_true",
                        help="Test dispatch patterns with sample phrases")
    parser.add_argument("--test-telemetry", action="store_true",
                        help="Test telemetry tools (reads JSONL)")
    parser.add_argument("--test-memory", action="store_true",
                        help="Test conversation memory (SQLite)")
    parser.add_argument("--test-vision", action="store_true",
                        help="Test camera capture + Gemini vision (or metadata fallback)")
    parser.add_argument("--test-reasoning", action="store_true",
                        help="Test reasoning mode (qwen3:0.6b)")
    parser.add_argument("--test-serial", action="store_true",
                        help="Probe /dev/ttyACM0, print device info, test stop command")
    parser.add_argument("--test-drive", action="store_true",
                        help="Move forward 1s, backward 1s, left, right, stop")
    parser.add_argument("--test-explore", action="store_true",
                        help="Run 15s autonomous exploration with vision narration")
    args = parser.parse_args()

    if args.no_tools:
        _tools_enabled = False
        log.info("Tool calling disabled")

    if args.mic:
        global MIC_DEVICE
        MIC_DEVICE = args.mic
        log.info(f"Mic override: {MIC_DEVICE}")

    if args.test_mic:
        log.info(f"Testing mic: {MIC_DEVICE}")
        wav = "/tmp/mic_test.wav"
        subprocess.run(
            ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
             "-c", "1", "-d", "5", wav],
            timeout=10,
        )
        text = transcribe(wav)
        log.info(f"Heard: '{text}'" if text else "Nothing detected")
        return

    if args.test_tools:
        log.info("Testing all tools...")
        for name, (func, params) in TOOL_REGISTRY.items():
            if params:
                log.info(f"  {name}: (skipped — requires arguments)")
            else:
                result = func()
                log.info(f"  {name}: {result}")
        return

    if args.test_dispatch:
        log.info("Testing dispatch patterns...")
        test_phrases = [
            "What's the CPU temperature?",
            "How much power is the system drawing?",
            "What's the CPU load?",
            "Show me all temperatures",
            "Give me system status",
            "How much disk space do I have?",
            "How much memory is available?",
            "What's the uptime?",
            "What's the network info?",
            "What time is it?",
            "List all services",
            "What do you see?",
            "Explain why the sky is blue",
            "Tell me a joke",  # should fallthrough to LLM
            # UGV patterns
            "Stop!",
            "Go forward",
            "Move backward",
            "Turn left",
            "Turn right",
            "What's the battery level?",
            "Explore the room",
            "Go to the door",
            "Find the red chair",
        ]
        for phrase in test_phrases:
            matched = "LLM (fallthrough)"
            # Tools checked first (same order as dispatch())
            for pattern, tool_name, _, _ in DISPATCH_PATTERNS:
                if pattern.search(phrase):
                    matched = f"TOOL:{tool_name}"
                    break
            else:
                if EXPLORE_PATTERN.search(phrase):
                    matched = "EXPLORE"
                elif NAVIGATE_TO_PATTERN.search(phrase):
                    target = NAVIGATE_TO_PATTERN.search(phrase).group(2).strip()
                    matched = f"NAVIGATE:{target}"
                elif VISION_PATTERN.search(phrase):
                    matched = "VISION"
                elif REASONING_PATTERN.search(phrase):
                    matched = "REASONING"
            log.info(f'  "{phrase}" → {matched}')
        return

    if args.test_telemetry:
        log.info("Testing telemetry tools...")
        log.info(f"  JSONL path: {TELEMETRY_JSONL}")
        data = _read_last_telemetry()
        if data:
            log.info(f"  Raw keys: {list(data.keys())}")
            log.info(f"  Power: {tool_power_consumption()}")
            log.info(f"  CPU load: {tool_cpu_load()}")
            log.info(f"  All temps: {tool_all_temperatures()}")
            log.info(f"  Full telemetry: {tool_system_telemetry()}")
        else:
            log.warning(f"  No telemetry data found at {TELEMETRY_JSONL}")
        return

    if args.test_memory:
        global conversation_store
        log.info("Testing conversation memory...")
        test_db = "/tmp/test_conversations.db"
        try:
            os.remove(test_db)
        except FileNotFoundError:
            pass
        conversation_store = ConversationStore(db_path=test_db)
        log.info(f"  DB: {test_db}, session: {conversation_store.session_id}")
        conversation_store.save_message("user", "What's the CPU temperature?")
        conversation_store.save_message("assistant", "The CPU temperature is 42.5°C.")
        conversation_store.save_message("user", "Tell me a joke about robots")
        conversation_store.save_message("assistant", "Why did the robot go to therapy? It had too many bytes of emotional baggage.")
        recent = conversation_store.get_recent_messages(limit=4)
        log.info(f"  Recent messages ({len(recent)}):")
        for msg in recent:
            log.info(f"    {msg['role']}: {msg['content'][:60]}")
        search = conversation_store.search_history("temperature")
        log.info(f"  Search 'temperature': {search}")
        recall = tool_recall_conversation("joke")
        log.info(f"  Recall 'joke': {recall}")
        conversation_store.close()
        os.remove(test_db)
        log.info("  Memory test passed!")
        return

    if args.test_vision:
        log.info("Testing vision mode...")
        log.info(f"  Camera device: {CAMERA_DEVICE}")
        log.info(f"  Gemini API: {'configured (' + GEMINI_MODEL + ')' if GEMINI_API_KEY else 'NOT configured (set GEMINI_API_KEY)'}")
        log.info("  Querying vision...")
        result = query_vision(
            "Describe what you see in this image.",
            sentence_cb=lambda s: log.info(f'  Vision says: "{s}"'),
        )
        log.info(f"  Full response: {result[:200]}")
        return

    if args.test_reasoning:
        log.info("Testing reasoning mode...")
        log.info("  Querying reasoning model (this will take 10-20s)...")
        result = query_reasoning(
            "Why is the sky blue? Explain step by step.",
            sentence_cb=lambda s: log.info(f'  Reasoning says: "{s}"'),
        )
        log.info(f"  Full response: {result[:300]}")
        return

    if args.test_serial:
        log.info("Probing serial device...")
        from ugv_driver import UGVDriver
        for baud in [115200, 9600, 57600]:
            log.info(f"  Trying {UGV_PORT}@{baud}...")
            ok, info = UGVDriver.probe(port=UGV_PORT, baud=baud)
            log.info(f"  {'OK' if ok else 'FAIL'}: {info}")
            if ok:
                break
        return

    if args.test_drive:
        log.info("Testing UGV drive...")
        ugv = _get_ugv()
        if not ugv:
            log.error("  UGV not available — check serial connection")
            return
        log.info("  Forward 1s...")
        ugv.forward(speed=80, duration=1.0)
        time.sleep(0.5)
        log.info("  Backward 1s...")
        ugv.backward(speed=80, duration=1.0)
        time.sleep(0.5)
        log.info("  Turn left...")
        ugv.turn_left(speed=80, duration=0.5)
        time.sleep(0.5)
        log.info("  Turn right...")
        ugv.turn_right(speed=80, duration=0.5)
        time.sleep(0.5)
        log.info("  Stop.")
        ugv.stop()
        ugv.close()
        log.info("  Drive test complete!")
        return

    if args.test_explore:
        log.info("Testing autonomous exploration (15s)...")
        log.info(f"  Gemini API: {'configured' if GEMINI_API_KEY else 'NOT configured'}")
        result = explore_autonomous(
            duration=15,
            sentence_cb=lambda s: log.info(f'  Robot: "{s}"'),
        )
        log.info(f"  Result: {result[:200]}")
        if _ugv_driver:
            _ugv_driver.close()
        return

    assistant = VoiceAssistant(skip_wakeword=args.no_wakeword)

    def signal_handler(sig, frame):
        log.info(f"Received signal {sig}")
        _explore_abort.set()  # Abort any exploration in progress
        assistant.running = False

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    assistant.run()


if __name__ == "__main__":
    main()
