# Contributing to SensorForge

## Project Structure

| Directory | Language | Build |
|-----------|----------|-------|
| `ios/` | Swift | Xcode |
| `jetson/` | Python | pip |
| `qualia/` | Rust | Cargo workspace |
| `protocol/` | JSON + Python | — |
| `scripts/` | Python | — |

## Development Setup

### iOS

Open `ios/SensorForge.xcodeproj` in Xcode. No external dependencies — the app uses only Apple frameworks. Requires a physical iPhone (simulators lack ARKit, LiDAR, and most sensors).

### Jetson

```bash
cd jetson
bash setup.sh              # Installs all dependencies
python3 voice_assistant.py --test-tools   # Verify everything works
```

### Qualia

```bash
cd qualia
cargo build --release      # Metal (macOS) by default
cargo build --release --features cuda   # CUDA (Jetson)
cargo test                 # Run tests
```

## Code Style

- **Swift**: Follow standard Swift conventions. Use `MARK` comments to organize sections.
- **Python**: Standard Python style. Keep imports organized (stdlib, third-party, local).
- **Rust**: Run `cargo fmt` before committing. Run `cargo clippy` for lint warnings.

## Testing

- **Qualia**: `cargo test` in the `qualia/` directory
- **Voice Assistant**: `python3 voice_assistant.py --test-tools` exercises all 24 tools
- **Protocol**: Test manually by running both the WiFi bridge and the iOS app on the same network

## Making Changes

1. Create a feature branch from `main`
2. Make your changes, keeping commits focused
3. Ensure the relevant component builds and tests pass
4. Open a pull request with a clear description of what and why

### Cross-component changes

When modifying the WiFi bridge protocol:

1. Update `protocol/schema.json` first (source of truth)
2. Update `protocol/messages.py` to match
3. Update `ios/SensorForge/WiFiBridge.swift` to match
4. Test with both sides connected

When modifying shared memory layout:

1. Update `qualia/crates/types/src/lib.rs` first (source of truth)
2. Update `jetson/qualia_bridge.py` struct offsets to match
3. Increment the version in the SHM header
