#!/usr/bin/env python3
"""Debug wake word detection — shows live prediction scores."""
import numpy as np
import subprocess
import time
from openwakeword.model import Model

print("Loading wake word model...")
model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
print("Model loaded. Listening for 20 seconds...")
print('Say "Hey Jarvis" now!')
print()

proc = subprocess.Popen(
    ["arecord", "-D", "plughw:0,0", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
)

CHUNK = 2560  # 1280 samples * 2 bytes
start = time.time()
max_score = 0.0
last_print = 0

try:
    while time.time() - start < 20:
        raw = proc.stdout.read(CHUNK)
        if not raw or len(raw) < CHUNK:
            print("ERROR: No audio data from mic")
            break

        audio_int16 = np.frombuffer(raw, dtype=np.int16)
        rms_val = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))

        model.predict(audio_int16)

        elapsed = time.time() - start
        for name, scores in model.prediction_buffer.items():
            if len(scores) > 0:
                score = scores[-1]
                if score > max_score:
                    max_score = score

                # Print every second regardless, or immediately if score > 0.01
                if score > 0.01 or (elapsed - last_print > 1.0):
                    triggered = " *** TRIGGERED ***" if score > 0.5 else ""
                    print(f"  [{elapsed:5.1f}s] RMS={rms_val:6.0f}  {name}={score:.4f}{triggered}")
                    last_print = elapsed

                    if score > 0.5:
                        print("\nWake word detected! It works!")

except KeyboardInterrupt:
    pass
finally:
    proc.terminate()
    proc.wait()

print(f"\nMax score seen: {max_score:.4f} (threshold: 0.5)")
if max_score < 0.05:
    print("Very low scores - mic may not be picking up voice well")
    print("Try: speak louder, closer to mic, or check if mic is correct device")
elif max_score < 0.5:
    print("Some detection but below threshold - try speaking louder/closer")
    print("Or try lowering WAKE_THRESHOLD in voice_assistant.py")
