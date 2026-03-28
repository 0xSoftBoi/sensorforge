#!/usr/bin/env python3
"""
Qualia Audio Features — Mel-spectrogram + MFCC sensor input.

Phase 4.1: Feed mel-spectrogram features from USB mic into Qualia sensor layer.
Phase 6.5: Upgraded with SOTA audio features:
  - MFCCs via DCT (Davis & Mermelstein 1980) for decorrelated spectral features
  - Delta and delta-delta coefficients for temporal dynamics
  - Voice Activity Detection (VAD) to skip silent frames
  - Spectral contrast and zero-crossing rate for richer features
  - Overlapping windows (50%) for transient capture

Runs at 10Hz, generating features from 100ms audio windows with 50% overlap.

Usage:
    python3 qualia_audio.py [--device default] [--bands 64]
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import deque

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
WINDOW_MS = 100  # 100ms windows
HOP_MS = 50      # 50% overlap (Phase 6.5)
N_FFT = 512
N_MELS = 64
N_MFCC = 20     # number of MFCCs to compute

# VAD parameters (Phase 6.5)
VAD_ENERGY_THRESHOLD = 0.005   # RMS below this = silence
VAD_ZCR_THRESHOLD = 0.3       # zero-crossing rate above this = likely unvoiced/noise
VAD_HOLD_FRAMES = 5           # keep "active" for N frames after last speech


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


def compute_dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    """Compute DCT-II matrix for MFCC extraction (Type II, ortho normalized).
    This is the standard DCT used in speech processing."""
    dct = np.zeros((n_mfcc, n_mels))
    for k in range(n_mfcc):
        for n in range(n_mels):
            dct[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    # Orthonormal normalization
    dct[0, :] *= np.sqrt(1.0 / n_mels)
    dct[1:, :] *= np.sqrt(2.0 / n_mels)
    return dct


def zero_crossing_rate(audio: np.ndarray) -> float:
    """Compute zero-crossing rate of audio signal."""
    if len(audio) < 2:
        return 0.0
    signs = np.sign(audio)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings) / (len(audio) - 1)


def spectral_contrast(spectrum: np.ndarray, n_bands: int = 6) -> np.ndarray:
    """Compute spectral contrast: difference between peaks and valleys in each
    sub-band. Useful for distinguishing harmonic (speech/music) from noise."""
    band_size = len(spectrum) // n_bands
    contrast = np.zeros(n_bands)
    for i in range(n_bands):
        band = spectrum[i * band_size:(i + 1) * band_size]
        if len(band) == 0:
            continue
        sorted_band = np.sort(band)
        n_top = max(1, len(band) // 4)
        peak = sorted_band[-n_top:].mean()
        valley = sorted_band[:n_top].mean()
        contrast[i] = np.log10(max(peak, 1e-10)) - np.log10(max(valley, 1e-10))
    return contrast


class AudioFeatureExtractor:
    """Full audio feature pipeline: mel + MFCC + delta + VAD + spectral contrast."""

    def __init__(self, n_bands: int = 64):
        self.n_bands = n_bands
        self.mel_bank = compute_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
        self.dct_matrix = compute_dct_matrix(N_MFCC, N_MELS)

        # History buffer for delta computation (Phase 6.5)
        self.mfcc_history: deque = deque(maxlen=5)

        # VAD state
        self.vad_active = False
        self.vad_hold_counter = 0

        # Running normalization stats (EMA-based, no per-frame normalization)
        self.mel_ema_mean = None
        self.mel_ema_var = None
        self.norm_alpha = 0.02

    def extract(self, audio: np.ndarray) -> dict:
        """Extract all features from a single audio window.
        Returns dict with mel, mfcc, delta, delta2, vad, zcr, contrast, rms."""

        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Apply Hanning window
        windowed = audio * np.hanning(len(audio))

        # Power spectrum
        spectrum = np.abs(np.fft.rfft(windowed, n=N_FFT)) ** 2

        # Mel-spectrogram (log scale)
        mel_spec = self.mel_bank @ spectrum
        mel_log = np.log10(np.maximum(mel_spec, 1e-10))

        # Running normalization (Phase 6.5: global stats instead of per-frame)
        if self.mel_ema_mean is None:
            self.mel_ema_mean = mel_log.copy()
            self.mel_ema_var = np.ones_like(mel_log)
        else:
            self.mel_ema_mean += self.norm_alpha * (mel_log - self.mel_ema_mean)
            dev = mel_log - self.mel_ema_mean
            self.mel_ema_var += self.norm_alpha * (dev * dev - self.mel_ema_var)

        mel_norm = (mel_log - self.mel_ema_mean) / np.sqrt(np.maximum(self.mel_ema_var, 1e-6))
        mel_norm = np.clip(mel_norm, -3.0, 3.0) / 3.0  # normalize to [-1, 1]

        # MFCCs via DCT (Phase 6.5)
        mfcc = self.dct_matrix @ mel_log

        # Store for delta computation
        self.mfcc_history.append(mfcc.copy())

        # Delta features: first-order temporal derivative
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
        if len(self.mfcc_history) >= 3:
            # Simple regression-based delta (Furui 1986)
            h = list(self.mfcc_history)
            delta = (h[-1] - h[-3]) / 2.0
        if len(self.mfcc_history) >= 5:
            h = list(self.mfcc_history)
            delta2 = (h[-1] - 2 * h[-3] + h[-5]) / 4.0

        # Zero-crossing rate
        zcr = zero_crossing_rate(audio)

        # Spectral contrast
        contrast = spectral_contrast(spectrum, n_bands=6)

        # Voice Activity Detection (Phase 6.5)
        is_speech = rms > VAD_ENERGY_THRESHOLD and zcr < VAD_ZCR_THRESHOLD
        if is_speech:
            self.vad_active = True
            self.vad_hold_counter = VAD_HOLD_FRAMES
        elif self.vad_hold_counter > 0:
            self.vad_hold_counter -= 1
        else:
            self.vad_active = False

        # Pool mel to requested bands
        if len(mel_norm) > self.n_bands:
            bin_size = len(mel_norm) / self.n_bands
            mel_pooled = np.array([
                mel_norm[int(i * bin_size):int((i + 1) * bin_size)].mean()
                for i in range(self.n_bands)
            ])
        else:
            mel_pooled = mel_norm[:self.n_bands]

        return {
            "mel": mel_pooled,
            "mfcc": mfcc,
            "delta": delta,
            "delta2": delta2,
            "zcr": zcr,
            "contrast": contrast,
            "rms": rms,
            "vad": self.vad_active,
        }

    def to_feature_vector(self, features: dict) -> np.ndarray:
        """Flatten all features into a single vector for Qualia ingestion.
        Layout: [mel(n_bands) | mfcc(20) | delta(20) | delta2(20) | contrast(6) | zcr(1) | rms(1) | vad(1)]
        Total with 64 mel bands: 64 + 20 + 20 + 20 + 6 + 1 + 1 + 1 = 133 dims
        We pool/truncate to n_bands for the output file (backward compat) but write full vector too."""
        parts = [
            features["mel"],
            features["mfcc"],
            features["delta"],
            features["delta2"],
            features["contrast"],
            np.array([features["zcr"]]),
            np.array([features["rms"]]),
            np.array([1.0 if features["vad"] else 0.0]),
        ]
        return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser(description="Audio features for Qualia")
    parser.add_argument("--device", default="default", help="ALSA device name")
    parser.add_argument("--bands", type=int, default=64, help="Number of mel bands (32 or 64)")
    parser.add_argument("--hz", type=float, default=10.0, help="Update frequency")
    args = parser.parse_args()

    acquire_singleton("audio")

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    extractor = AudioFeatureExtractor(n_bands=args.bands)
    interval = 1.0 / max(args.hz, 0.1)
    hop_samples = int(SAMPLE_RATE * HOP_MS / 1000)
    window_samples = int(SAMPLE_RATE * WINDOW_MS / 1000)

    log.info(f"Audio features: {args.bands} mel + {N_MFCC} MFCC + deltas at {args.hz:.0f}Hz, "
             f"window={WINDOW_MS}ms, hop={HOP_MS}ms (50% overlap)")

    # Try to open audio device
    audio_stream = None
    try:
        import sounddevice as sd
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=hop_samples,
            device=None if args.device == "default" else args.device,
        )
        audio_stream.start()
        log.info("Audio stream started (sounddevice)")
    except (ImportError, Exception) as e:
        log.warning(f"Cannot open audio device: {e}")
        log.warning("Running in synthetic mode (pink noise + speech-like modulation)")
        audio_stream = None

    # Overlap buffer: keep previous hop for 50% overlap
    prev_hop = np.zeros(hop_samples, dtype=np.float32)
    update_count = 0

    try:
        while running:
            loop_start = time.monotonic()

            if audio_stream is not None:
                try:
                    audio_data, overflowed = audio_stream.read(hop_samples)
                    if overflowed:
                        log.warning("Audio buffer overflow")
                    current_hop = audio_data[:, 0]
                except Exception as e:
                    log.error(f"Audio read error: {e}")
                    current_hop = np.zeros(hop_samples, dtype=np.float32)
            else:
                # Synthetic: pink noise with speech-like amplitude modulation
                t = np.linspace(0, HOP_MS / 1000, hop_samples)
                # Pink noise approximation: 1/f spectrum
                white = np.random.randn(hop_samples).astype(np.float32)
                # Simple 1/f filter via cumulative sum + high-pass
                pink = np.cumsum(white) * 0.002
                pink -= np.mean(pink)
                # Speech-like AM: slow modulation (3-8 Hz syllable rate)
                am = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 5.0 * t + update_count * 0.3))
                current_hop = (pink * am).astype(np.float32)

            # Assemble overlapping window
            audio = np.concatenate([prev_hop, current_hop])[:window_samples]
            prev_hop = current_hop.copy()

            # Extract features
            features = extractor.extract(audio)

            # Write to shared file
            data = {
                "ts": time.time(),
                "bands": args.bands,
                "features": features["mel"].tolist(),
                "mfcc": features["mfcc"].tolist(),
                "delta": features["delta"].tolist(),
                "delta2": features["delta2"].tolist(),
                "contrast": features["contrast"].tolist(),
                "zcr": features["zcr"],
                "rms": features["rms"],
                "vad": features["vad"],
                "full_vector": extractor.to_feature_vector(features).tolist(),
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
                vad_str = "SPEECH" if features["vad"] else "silent"
                log.info(f"#{update_count}: rms={features['rms']:.4f}, "
                         f"zcr={features['zcr']:.3f}, {vad_str}, "
                         f"bands={args.bands}+{N_MFCC}mfcc")

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
