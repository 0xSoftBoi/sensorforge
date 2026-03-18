#!/usr/bin/env python3
"""
Qualia Audio Features — Mel-spectrogram sensor input.

Phase 4.1: Feed 64-band mel-spectrogram features from USB mic into
the Qualia sensor layer. Features are multiplexed with visual data
(32 dims visual + 32 dims audio) or can use a dedicated sensor slot.

Runs at 10Hz, generating mel-spectrogram features from 100ms audio windows.
Writes features into a shared file for qualia-camera to multiplex,
or directly into L6 sensor slot (configurable).

Usage:
    python3 qualia_audio.py [--device default] [--bands 32]
"""

import argparse
import json
import logging
import os
import signal
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s audio: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AUDIO_FEATURES_FILE = "/tmp/qualia_audio_features.json"


def acquire_singleton(name):
    """Ensure only one instance runs. Kill stale instance if found."""
    pidfile = f"/tmp/qualia_{name}.pid"
    try:
        with open(pidfile) as f:
            old_pid = int(f.read().strip())
        os.kill(old_pid, 0)
        log.warning(f"Killing stale {name} process (PID {old_pid})")
        os.kill(old_pid, signal.SIGTERM)
        time.sleep(1)
        try:
            os.kill(old_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        pass
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
    log.info(f"Singleton lock acquired: {pidfile} (PID {os.getpid()})")


SAMPLE_RATE = 16000
WINDOW_MS = 100  # 100ms windows at 10Hz
N_FFT = 512
N_MELS = 64


def compute_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Compute mel filterbank matrix (simplified, no librosa dependency)."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                fbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fbank[i, j] = (right - j) / (right - center)

    return fbank


def audio_to_mel_features(audio: np.ndarray, mel_bank: np.ndarray, n_bands: int) -> np.ndarray:
    """Convert audio samples to mel-spectrogram features."""
    # Apply Hanning window
    windowed = audio * np.hanning(len(audio))

    # FFT
    spectrum = np.abs(np.fft.rfft(windowed, n=N_FFT)) ** 2

    # Apply mel filterbank
    mel_spec = mel_bank @ spectrum

    # Log scale (with floor to avoid log(0))
    mel_log = np.log10(np.maximum(mel_spec, 1e-10))

    # Normalize to [0, 1] range
    mel_min, mel_max = mel_log.min(), mel_log.max()
    if mel_max > mel_min:
        mel_norm = (mel_log - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_log)

    # Pool to requested number of bands
    if len(mel_norm) > n_bands:
        bin_size = len(mel_norm) / n_bands
        pooled = np.array([
            mel_norm[int(i * bin_size):int((i + 1) * bin_size)].mean()
            for i in range(n_bands)
        ])
        return pooled
    return mel_norm[:n_bands]


def main():
    parser = argparse.ArgumentParser(description="Audio features for Qualia")
    parser.add_argument("--device", default="default", help="ALSA device name")
    parser.add_argument("--bands", type=int, default=32, help="Number of mel bands (32 or 64)")
    parser.add_argument("--hz", type=float, default=10.0, help="Update frequency")
    args = parser.parse_args()

    acquire_singleton("audio")

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Precompute mel filterbank
    mel_bank = compute_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
    interval = 1.0 / max(args.hz, 0.1)
    samples_per_window = int(SAMPLE_RATE * WINDOW_MS / 1000)

    log.info(f"Audio features: {args.bands} bands at {args.hz:.0f}Hz, window={WINDOW_MS}ms")

    # Try to open audio device
    audio_stream = None
    try:
        import sounddevice as sd
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=samples_per_window,
            device=None if args.device == "default" else args.device,
        )
        audio_stream.start()
        log.info("Audio stream started (sounddevice)")
    except (ImportError, Exception) as e:
        log.warning(f"Cannot open audio device: {e}")
        log.warning("Running in synthetic mode (generates test patterns)")
        audio_stream = None

    update_count = 0

    try:
        while running:
            loop_start = time.monotonic()

            if audio_stream is not None:
                try:
                    audio_data, overflowed = audio_stream.read(samples_per_window)
                    if overflowed:
                        log.warning("Audio buffer overflow")
                    audio = audio_data[:, 0]  # mono
                except Exception as e:
                    log.error(f"Audio read error: {e}")
                    audio = np.zeros(samples_per_window, dtype=np.float32)
            else:
                # Synthetic audio: sine wave with varying frequency
                t = np.linspace(0, WINDOW_MS / 1000, samples_per_window)
                freq = 200 + 100 * np.sin(update_count * 0.1)
                audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5

            # Compute mel features
            features = audio_to_mel_features(audio, mel_bank, args.bands)

            # Write to shared file for camera runner to multiplex
            data = {
                "ts": time.time(),
                "bands": args.bands,
                "features": features.tolist(),
                "rms": float(np.sqrt(np.mean(audio ** 2))),
            }
            tmp = AUDIO_FEATURES_FILE + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(data, f)
                os.replace(tmp, AUDIO_FEATURES_FILE)
            except OSError:
                pass

            update_count += 1
            if update_count % 100 == 0:
                rms = float(np.sqrt(np.mean(audio ** 2)))
                log.info(f"#{update_count}: rms={rms:.4f}, bands={args.bands}")

            elapsed = time.monotonic() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    finally:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        log.info(f"Audio shutdown after {update_count} frames")


if __name__ == "__main__":
    main()
