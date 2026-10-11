#!/usr/bin/env python3
"""
Transmit a text string as OOK-modulated sound at a given carrier frequency.
Default carrier: 21.5 kHz (near/above the upper limit of human hearing and
of many laptop/phone speakers).

Requires: numpy, sounddevice  ->  pip install numpy sounddevice
"""

import numpy as np
import sounddevice as sd


# ---------- encoding ----------

def text_to_bits(text: str):
    """UTF-8 encode text and return a list of bits, MSB first."""
    return [(byte >> i) & 1
            for byte in text.encode("utf-8")
            for i in range(7, -1, -1)]


# ---------- modulation ----------

def ook_modulate(bits, sr, carrier, bit_dur, amp, fade_ms=1.0):
    """
    Modulate a bit sequence with OOK.
    A continuous-phase carrier is generated for the whole waveform and
    multiplied by an envelope that is 1 for a '1' bit and 0 for a '0' bit.
    Short raised-cosine-ish fades are applied at each bit edge to avoid
    clicks / spectral splatter.
    """
    spb       = int(round(sr * bit_dur))          # samples per bit
    fade_len  = max(1, int(sr * fade_ms / 1000))  # samples for fade
    total     = spb * len(bits)

    t            = np.arange(total) / sr
    carrier_wave = np.sin(2 * np.pi * carrier * t)

    # per-bit envelope with fades on both ends
    fade = np.ones(spb)
    fade[:fade_len]  = np.linspace(0.0, 1.0, fade_len)
    fade[-fade_len:] = np.linspace(1.0, 0.0, fade_len)

    env = (np.asarray(bits, dtype=np.float64).reshape(-1, 1) * fade).ravel()
    return amp * carrier_wave * env


# ---------- optional verification (requires a working mic) ----------

def check_recording(sr, expected_freq, duration=1.0, tolerance=100.0):
    """
    Record from the default input and check for energy near expected_freq.
    NOTE: the microphone must also be able to capture 21.5 kHz — many can't.
    """
    print(f"Recording {duration}s from default input...")
    rec = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    rec = rec.ravel()

    # Hann window -> FFT -> magnitude
    win  = np.hanning(len(rec))
    spec = np.abs(np.fft.rfft(rec * win))
    freqs = np.fft.rfftfreq(len(rec), 1.0 / sr)

    band  = (freqs > expected_freq - tolerance) & (freqs < expected_freq + tolerance)
    peak  = spec[band].max()
    floor = np.median(spec) + 1e-12

    snr_db = 20 * np.log10(peak / floor)
    print(f"Peak magnitude in band {expected_freq - tolerance:.0f}-"
          f"{expected_freq + tolerance:.0f} Hz: {peak:.3e}")
    print(f"Median spectrum floor:                        {floor:.3e}")
    print(f"Approx. SNR in that band: {snr_db:6.1f} dB")
    return snr_db


# ---------- main ----------

def main():
    import argparse
    p = argparse.ArgumentParser(description="OOK acoustic transmitter")
    p.add_argument("text", nargs="?", default="Hello 21.5 kHz",
                   help="text to transmit")
    p.add_argument("--sr",       type=int,   default=48000, help="sample rate (Hz)")
    p.add_argument("--freq",     type=float, default=21500, help="carrier (Hz)")
    p.add_argument("--bit-dur",  type=float, default=0.010, help="seconds per bit")
    p.add_argument("--amp",      type=float, default=0.5,   help="amplitude 0..1")
    p.add_argument("--verify",   action="store_true",
                   help="after transmitting, record & check the spectrum")
    p.add_argument("--list",     action="store_true", help="list audio devices and exit")
    args = p.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    # Warn if the Nyquist limit is uncomfortably close to the carrier.
    if args.sr < 2.2 * args.freq:
        print(f"[warn] sample rate {args.sr} Hz is low for a {args.freq} Hz carrier. "
              f"Prefer >= {int(2.5*args.freq)} Hz.")

    # Preamble: alternating 1/0 for receiver sync, plus a leading 1 so the
    # tone actually starts on a known phase.
    preamble = [1, 0] * 16
    bits = preamble + text_to_bits(args.text)

    wave = ook_modulate(bits, args.sr, args.freq, args.bit_dur, args.amp)
    pad  = np.zeros(int(args.sr * 0.2))
    wave = np.concatenate([pad, wave, pad]).astype(np.float32)

    dur = len(bits) * args.bit_dur
    print(f"Transmitting {len(bits)} bits ({dur:.2f}s) @ {args.freq:.0f} Hz "
          f"carrier, sr={args.sr} Hz, bit={args.bit_dur*1000:.1f} ms")

    sd.play(wave, args.sr)
    sd.wait()

    if args.verify:
        # Let the room settle, then listen back.
        import time; time.sleep(0.3)
        check_recording(args.sr, args.freq, duration=max(dur, 1.0))


if __name__ == "__main__":
    main()