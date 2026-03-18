# Qualia Engine

A predictive coding architecture running 7 hierarchical layers on Apple Silicon Metal, with Google Gemini as the semantic embedding model that teaches the lower layers what to expect. Layers that can't resolve their own prediction errors ask questions to the outside world — the answers become **LORE**, accumulated world-knowledge that shapes future predictions.

## Prerequisites

### macOS (Apple Silicon required for Metal GPU)

```bash
# 1. Install Xcode Command Line Tools (for C compiler + Metal SDK)
xcode-select --install

# 2. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 3. Install ffmpeg (for webcam capture)
brew install ffmpeg

# 4. Get a Google Gemini API key
#    https://aistudio.google.com/apikey
#    Free tier works fine — Flash is cheap
export GEMINI_API_KEY="your-key-here"
```

### Jetson Orin Nano (CUDA)

```bash
# 1. Ensure JetPack 6.0+ is installed (includes CUDA toolkit)
nvcc --version   # should show CUDA 12.x

# 2. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 3. Install ffmpeg
sudo apt install ffmpeg

# 4. Get a Google Gemini API key (optional — runs offline without it)
export GEMINI_API_KEY="your-key-here"
```

### Verify your setup

```bash
rustc --version     # needs 1.75+
cargo --version
ffmpeg -version     # needs avfoundation support (macOS default)
# Metal is built-in on all Apple Silicon Macs — no driver needed
```

## Quick Start

```bash
# From the SensorForge repo root
cd qualia
cargo build --release

# Run — the TUI is the supervisor, spawns everything
export GEMINI_API_KEY="your-key-here"
cargo run --release --bin qualia-watch
```

That's it. The TUI creates shared memory, spawns all 10 runner processes, starts the webcam, and connects to Gemini. Press `q` to quit and cleanly shut everything down.

Without `GEMINI_API_KEY` it runs in offline mode with synthetic sensor data and hash-based embeddings — still useful for development.

## Architecture

```mermaid
graph TB
    subgraph "Gemini — The Semantic Layer"
        GV[Gemini Vision<br/>gemini-2.0-flash]
        GE[Gemini Embedding<br/>text-embedding-004]
        GV -->|scene + lore answers| GE
        GE -->|64-dim embedding| WM[World Model<br/>shared memory]
    end

    subgraph "Predictive Hierarchy"
        L5[L5 deep patterns<br/>0.1 Hz] -->|prediction| L4
        L4[L4 short-term behavior<br/>1 Hz] -->|prediction| L3
        L3[L3 visual patterns<br/>100 Hz] -->|prediction| L2
        L2[L2 local structure<br/>100 Hz] -->|prediction| L1
        L1[L1 motor patterns<br/>100 Hz] -->|prediction| L0
        L0[L0 superposition<br/>1000 Hz] -->|prediction error| L1
        L1 -->|prediction error| L2
        L2 -->|prediction error| L3
        L3 -->|prediction error| L4
        L4 -->|prediction error| L5
    end

    subgraph "Sensor"
        CAM[Mac Camera<br/>ffmpeg 30fps] --> L6[L6 sensor<br/>8x8 grayscale]
        L6 -->|raw input| L0
    end

    subgraph "LORE"
        LB[(Lore Buffer<br/>128 entries)]
        L5 -.->|question: stuck| LB
        L4 -.->|question: plateau| LB
        L3 -.->|question: novel| LB
        LB -.->|harvested| GV
    end

    WM -.->|injection 30%| L5
    WM -.->|injection 15%| L4
    WM -.->|injection 5%| L3
    WM -.->|injection 2%| L2

    CAM -->|320x240 JPEG| GV

    style GV fill:#4285f4,color:#fff
    style GE fill:#34a853,color:#fff
    style WM fill:#fbbc04,color:#000
    style LB fill:#9c27b0,color:#fff
    style L6 fill:#1a1a25,color:#00d4ff
```

## The LORE System

When a layer can't resolve its prediction errors on its own, it poses a question to the outside world. The vision runner harvests these questions and batches them into the next Gemini call. The answers become LORE — accumulated world-knowledge that persists in shared memory.

```mermaid
sequenceDiagram
    participant L5 as L5 deep patterns
    participant L4 as L4 behavior
    participant Q as Question Slots<br/>(shared memory)
    participant Vis as Vision Runner
    participant Gem as Gemini
    participant Lore as Lore Buffer
    participant Dash as Dashboard

    Note over L5: VFE stuck high for 50+ cycles
    L5->>Q: "What deep pattern exists here<br/>that I'm missing?"
    Note over L4: Compression plateaued
    L4->>Q: "What category describes<br/>this behavior?"

    loop Every 30s
        Vis->>Q: Harvest pending questions
        Vis->>Gem: Image + scene prompt + layer questions
        Gem->>Vis: {scene, objects, lore_answers[]}
        Vis->>Lore: Write answered lore entries
        Lore-->>Dash: Stream via WebSocket
    end

    Note over L5,L4: Questions are SPARSE:<br/>L5 every 60s, L4 every 2min,<br/>L3 every 5min, lower every 10min
```

### Question Triggers

| Reason | Code | Trigger | Example |
|--------|------|---------|---------|
| STUCK | 0 | VFE > 5x threshold for 50+ cycles | "My predictions keep failing — what pattern am I missing?" |
| PLATEAU | 1 | Compression stalled, still challenged | "I learned something but can't simplify — what abstraction am I missing?" |
| NOVEL | 2 | Sudden VFE spike > 8x threshold | "Something new appeared — what changed?" |

## Process Architecture

```mermaid
graph LR
    subgraph "qualia-watch — TUI + Supervisor"
        INIT[Create SHM 64 MiB]
        TUI[Ratatui TUI<br/>7 views + controls]
        INIT --> TUI
    end

    TUI -->|spawns| C[qualia-camera]
    TUI -->|spawns| L0B[qualia-l0]
    TUI -->|spawns| L1B[qualia-l1]
    TUI -->|spawns| L2B[qualia-l2]
    TUI -->|spawns| L3B[qualia-l3]
    TUI -->|spawns| L4B[qualia-l4]
    TUI -->|spawns| L5B[qualia-l5]
    TUI -->|spawns| V[qualia-vision]
    TUI -->|spawns| H[qualia-health]
    TUI -->|spawns| A[qualia-agent :8080]

    SHM[(POSIX Shared Memory<br/>64 MiB)]
    C <--> SHM
    L0B <--> SHM
    L1B <--> SHM
    L2B <--> SHM
    L3B <--> SHM
    L4B <--> SHM
    L5B <--> SHM
    V <--> SHM
    H --> SHM
    A --> SHM
    TUI --> SHM

    style SHM fill:#2a2a3a,color:#c8c8d8
    style TUI fill:#0a0a0f,color:#00d4ff,stroke:#00d4ff
```

## Shared Memory Layout

```mermaid
block-beta
    columns 1
    block:header["Header (4096 bytes)"]
        magic["QUAL1AEN"] version["v1"] num_layers["7 layers"]
    end
    block:layers["Layer Slots (7 x ~40KB each)"]
        L0S["L0-L5: 2x BeliefSlot + 64x64 weights + bias + QuestionSlot"]
        L6S["L6: sensor slot + QuestionSlot"]
    end
    block:ledger["Ledger Ring Buffer (16 MiB)"]
        events["Challenge | Confirm | Habit | Escalate events"]
    end
    block:world["World Model"]
        objects["16 objects"] scene["scene + activity"] embedding["64-dim Gemini embedding"] tokens["API counters"]
    end
    block:thoughts["Thought Buffer (512 entries)"]
        thought_ring["observe | predict | surprise | learn | resolve | escalate"]
    end
    block:lore["Lore Buffer (128 entries)"]
        lore_ring["question | answer | layer | reason | embedding_delta"]
    end
```

## Belief Update — Metal GPU Kernel

Each layer runs 64 GPU threads (one per dimension) on Apple Silicon Metal:

```mermaid
flowchart LR
    A[Read below.mean] --> B["Predict: W @ mean + bias"]
    B --> C["Residual = below - prediction"]
    C --> D["VFE = sum of precision * residual^2"]
    D --> E{VFE > threshold?}
    E -->|Challenge| F["Update beliefs<br/>mean += lr * precision * residual"]
    E -->|Confirm| G["Increment streak<br/>Increase compression"]
    F --> H["Learn weights<br/>W += lr * precision * residual * mean_T"]
    H --> I["Adapt precision"]
    I --> J{Stuck too long?}
    J -->|yes| K["Post QUESTION<br/>to shared memory"]
    J -->|no| L[Next cycle]
    K --> L
```

## How Gemini Teaches the Hierarchy

```mermaid
sequenceDiagram
    participant Cam as Camera
    participant Vis as Vision Runner
    participant GV as Gemini Vision
    participant GE as Gemini Embedding
    participant WM as World Model
    participant L5 as L5 deep patterns
    participant L4 as L4 behavior

    loop Every 30s (configurable)
        Vis->>Vis: Harvest layer questions
        Cam->>Vis: 320x240 JPEG frame
        Vis->>GV: Image + directive + questions
        GV->>Vis: scene, activity, objects, lore_answers
        Vis->>GE: scene text
        GE->>Vis: 64-dim embedding
        Vis->>WM: Write embedding + scene + objects + lore
        WM->>L5: Inject embedding alpha=0.30
        WM->>L4: Inject embedding alpha=0.15
        Note over L5,L4: New embedding creates prediction errors<br/>that cascade down through weight learning
    end

    loop Between calls (200ms)
        Cam->>WM: Blend sensor into embedding alpha=0.05
    end
```

## TUI Controls

| Key | Action |
|-----|--------|
| `1`-`7` | Switch view mode |
| `j`/`k` or arrows | Select layer |
| `Tab` | Cycle views |
| `r` | Restart all runners |
| `q` / `Esc` / `Ctrl+C` | Quit and shutdown |

### View Modes

| # | View | What you see |
|---|------|--------------|
| 1 | Overview | Layer table: VFE, compression, streaks, challenge/confirm |
| 2 | Detail | Single layer deep dive — all 64 dimensions |
| 3 | Hex | Raw belief memory dump |
| 4 | Sparklines | VFE history for all layers |
| 5 | Residuals | Residual heatmap across dimensions |
| 6 | Weights | 64x64 weight matrix heatmap |
| 7 | World | Scene, objects, directive, Gemini embedding |

## Web Dashboard

Auto-launched at `http://localhost:8080` with real-time WebSocket streaming:

- Layer hierarchy with VFE coloring
- 16x16 weight matrix heatmap
- Belief vector heatmaps (mean, precision, prediction, residual)
- Scrolling volatility charts (VFE, challenge rate, residual energy)
- Thought stream
- **Lore Codex** — scrolling question/answer pairs from the layers
- Gemini API cost tracking with token breakdown and price graph
- Scene embedding 8x8 heatmap

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | _(none)_ | Google Gemini API key. Without it, runs offline with synthetic data |
| `QUALIA_SHM_NAME` | `/qualia_body` | POSIX shared memory region name |
| `QUALIA_SOCK_PATH` | `/tmp/qualia_body.sock` | Unix domain socket for IPC |
| `QUALIA_WEB_PORT` | `8080` | Web dashboard port |
| `QUALIA_LLM_INTERVAL` | `30` | Seconds between Gemini API calls |
| `QUALIA_LLM_MAX_CALLS` | `50` | Max Gemini calls per session (budget) |
| `RUST_LOG` | `info` | Log level |

## Layer Reference

| Layer | Name | Hz | Role | Gemini Injection | Questions |
|-------|------|----|------|-----------------|-----------|
| L0 | Superposition | 1000 | Raw sensation | 0% | Every 10min |
| L1 | Motor | 100 | How things move | 0% | Every 10min |
| L2 | Local | 100 | Nearby structure | 2% | Every 10min |
| L3 | Visual | 100 | What things look like | 5% | Every 5min |
| L4 | Behavior | 1 | What's happening now | 15% | Every 2min |
| L5 | Deep | 0.1 | Persistent regularities | 30% | Every 1min |
| L6 | Sensor | 30 | Raw camera input | 0% | _(none)_ |

## Project Structure

```
qualia/
  crates/
    types/       repr(C) shared data structures
    shm/         POSIX shared memory with double-buffering
    ipc/         Unix domain socket control plane
    metal/       Apple Silicon Metal GPU compute
    cuda/        (stub) future NVIDIA support
  runners/
    watch/       TUI supervisor — entry point, spawns everything
    camera/      Webcam via ffmpeg -> L6
    vision/      Gemini Vision + Embedding + LORE answering
    agent/       Web dashboard (Axum + WebSocket)
    health/      10 Hz monitoring
    l0-superposition/  through  l5-behavior/
  kernels/
    belief_update.metal   GPU kernel
```

## License

MIT
