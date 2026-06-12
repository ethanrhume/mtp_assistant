"""Generate a minimal WAV test file (3s of silence) without external dependencies."""
import struct
import wave
from pathlib import Path

out = Path("test_audio.wav")
sr, dur = 16000, 3
samples = [0] * (sr * dur)  # silence — Whisper will return empty/minimal text, that's fine

with wave.open(str(out), "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    f.writeframes(struct.pack(f"<{len(samples)}h", *samples))

print(f"Created {out} ({out.stat().st_size} bytes)")
