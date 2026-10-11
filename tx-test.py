# check_wav.py  —  run: python check_wav.py out.wav
import sys, wave
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "out.wav"
with wave.open(path, "rb") as w:
    fs   = w.getframerate()
    ch   = w.getnchannels()
    sw   = w.getsampwidth()
    nfr  = w.getnframes()
    raw  = w.readframes(nfr)

print(f"header: fs={fs} Hz  ch={ch}  width={sw*8} bit  frames={nfr}  "
      f"duration={nfr/fs:.3f} s")

dtype = {2: "<i2", 4: "<i4"}.get(sw)
x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
if ch > 1:
    x = x.reshape(-1, ch).mean(axis=1)

N = 32768
# skip the first 0.5 s (may be a silence gap); take a mid-file block
start = min(len(x) - N, fs // 2)
seg   = x[start : start + N] * np.hanning(N)
X     = np.abs(np.fft.rfft(seg))
f     = np.fft.rfftfreq(N, 1 / fs)

peak_bin = int(np.argmax(X))
peak_hz  = f[peak_bin]
print(f"peak spectral line: {peak_hz:8.1f} Hz")

# energy in the near-ultrasonic band vs. below 15 kHz
band_hi = (f >= 18_000) & (f <= 24_000)
band_lo = (f < 15_000)
e_hi = np.sum(X[band_hi] ** 2)
e_lo = np.sum(X[band_lo] ** 2)
print(f"energy 18–24 kHz: {e_hi:.3e}")
print(f"energy  <15 kHz: {e_lo:.3e}")
print(f"ratio hi/lo    : {e_hi / max(e_lo, 1e-30):.3e}")