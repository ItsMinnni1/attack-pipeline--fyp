#!/usr/bin/env python3
"""
txrx.py — Acoustic FSK test rig for high-frequency speaker/mic testing.

Features
--------
* Chirp preamble (18->22 kHz) for start-of-message sync via matched filter.
* Continuous-phase 2-FSK payload (default 21.0 / 21.5 kHz).
* Frame: sync word + length byte + payload + CRC32.
* Goertzel-based non-coherent FSK decoder + bit confidences.
* Loopback mode (DSP-only, no hardware needed) with BER measurement.
* Live send, receive, and simultaneous send+receive.
* Frequency sweep mode with auto-generated spectrogram PNG.
* Auto-PNG report on every decode.

Install:  pip install numpy sounddevice matplotlib
Usage:    python txrx.py --help
"""

import argparse
import binascii
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import sounddevice as sd
except ImportError:
    sd = None


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
SR_DEFAULT       = 96000
F0_DEFAULT       = 21000.0    # FSK '0'
F1_DEFAULT       = 21500.0    # FSK '1'
BIT_DUR_DEFAULT  = 0.010      # 100 bps
AMP_DEFAULT      = 0.5

CHIRP_F0   = 18000.0
CHIRP_F1   = 22000.0
CHIRP_DUR  = 0.100
GAP_DUR    = 0.020
TAIL_DUR   = 0.250

SYNC_WORD  = bytes([0xB5, 0x5A])


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def apply_fade(w, sr, fade_ms=2.0):
    """Raised-cosine fades at both ends to kill clicks."""
    n = max(1, int(sr * fade_ms / 1000))
    if 2 * n >= len(w):
        return w
    w = w.copy()
    f = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n)))
    w[:n]  *= f
    w[-n:] *= f[::-1]
    return w


def chirp(sr, f0, f1, dur, amp=1.0, fade_ms=2.0):
    """Linear chirp from f0 to f1."""
    n = int(round(sr * dur))
    t = np.arange(n) / sr
    k = (f1 - f0) / dur
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    return apply_fade(amp * np.sin(phase), sr, fade_ms)


def bytes_to_bits(b):
    return [(byte >> i) & 1 for byte in b for i in range(7, -1, -1)]


def bits_to_bytes(bits):
    n = len(bits) // 8 * 8
    out = bytearray()
    for i in range(0, n, 8):
        v = 0
        for j in range(8):
            v = (v << 1) | bits[i + j]
        out.append(v)
    return bytes(out)


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #
def build_frame(payload_bytes: bytes) -> bytes:
    if len(payload_bytes) > 255:
        raise ValueError("payload too long (max 255 bytes)")
    header = SYNC_WORD + bytes([len(payload_bytes)])
    crc = binascii.crc32(header + payload_bytes) & 0xFFFFFFFF
    return header + payload_bytes + crc.to_bytes(4, "big")


# --------------------------------------------------------------------------- #
# modulation
# --------------------------------------------------------------------------- #
def fsk_modulate(bits, sr, f0, f1, bit_dur, amp):
    """Continuous-phase 2-FSK — one sine wave whose instantaneous frequency
    is f0 for a '0' bit and f1 for a '1' bit."""
    spb = int(round(sr * bit_dur))
    freqs = np.concatenate([np.full(spb, f1 if b else f0) for b in bits])
    phase = 2 * np.pi * np.cumsum(freqs) / sr
    return amp * np.sin(phase)


def build_waveform(text, sr, f0, f1, bit_dur, amp):
    """Returns (full_waveform, tx_bits, framed_bytes)."""
    frame = build_frame(text.encode("utf-8"))
    bits  = bytes_to_bits(frame)

    up   = chirp(sr, CHIRP_F0, CHIRP_F1, CHIRP_DUR, amp)
    down = chirp(sr, CHIRP_F1, CHIRP_F0, CHIRP_DUR, amp)
    gap  = np.zeros(int(sr * GAP_DUR))

    data = apply_fade(fsk_modulate(bits, sr, f0, f1, bit_dur, amp), sr, 1.0)

    return np.concatenate([up, gap, data, gap, down]), bits, frame


# --------------------------------------------------------------------------- #
# demodulation
# --------------------------------------------------------------------------- #
def find_chirp(rec, ref):
    """Normalized matched filter. Returns (peak_index, peak_value, curve)."""
    corr = np.correlate(rec, ref, mode="valid")
    win_energy = np.convolve(rec * rec, np.ones(len(ref)), mode="valid")
    ref_energy = float(np.sum(ref * ref))
    denom = np.sqrt(np.maximum(win_energy, 1e-20) * ref_energy)
    norm = np.abs(corr) / denom
    peak = int(np.argmax(norm))
    return peak, float(norm[peak]), norm


def demod_fsk(rec, sr, f0, f1, bit_dur, start_sample, n_bits):
    """Goertzel (single-bin DFT) per bit. Returns (bits, confidences)."""
    spb = int(round(sr * bit_dur))
    n   = np.arange(spb)
    win = np.hanning(spb)
    lo  = np.exp(-2j * np.pi * f0 * n / sr)
    hi  = np.exp(-2j * np.pi * f1 * n / sr)

    bits, confs = [], []
    for i in range(n_bits):
        a = start_sample + i * spb
        b = a + spb
        if b > len(rec):
            break
        seg = rec[a:b] * win
        e0 = abs(np.sum(seg * lo))
        e1 = abs(np.sum(seg * hi))
        bits.append(1 if e1 > e0 else 0)
        confs.append((e1 - e0) / (e1 + e0 + 1e-12))
    return bits, np.asarray(confs)


def decode(rec, sr, f0, f1, bit_dur):
    """Full decode pipeline. Returns a dict with an 'ok' flag and diagnostics."""
    up_ref = chirp(sr, CHIRP_F0, CHIRP_F1, CHIRP_DUR, 1.0)
    peak, peak_val, _ = find_chirp(rec, up_ref)

    start = peak + len(up_ref) + int(sr * GAP_DUR)

    header_bits, _ = demod_fsk(rec, sr, f0, f1, bit_dur, start, 56)
    if len(header_bits) < 56:
        return {"ok": False, "reason": "short header",
                "chirp_peak": peak, "chirp_val": peak_val}

    header_bytes = bits_to_bytes(header_bits)
    if header_bytes[:2] != SYNC_WORD:
        return {"ok": False,
                "reason": f"sync mismatch (got {header_bytes[:2].hex()})",
                "chirp_peak": peak, "chirp_val": peak_val}

    length = header_bytes[2]
    total_bits = 56 + length * 8
    all_bits, confs = demod_fsk(rec, sr, f0, f1, bit_dur, start, total_bits)

    data = bits_to_bytes(all_bits)
    need = 7 + length + 4
    if len(data) < need:
        return {"ok": False, "reason": "short frame",
                "chirp_peak": peak, "chirp_val": peak_val}

    frame_bytes = data[:7 + length]
    crc_rx = int.from_bytes(data[7 + length: 11 + length], "big")
    crc_calc = binascii.crc32(frame_bytes) & 0xFFFFFFFF

    payload = frame_bytes[3:3 + length].decode("utf-8", errors="replace")

    return {
        "ok":        crc_rx == crc_calc,
        "payload":   payload,
        "crc_rx":    crc_rx,
        "crc_calc":  crc_calc,
        "chirp_peak": peak,
        "chirp_val": peak_val,
        "mean_conf": float(confs.mean()) if len(confs) else 0.0,
        "bits":      all_bits,
        "confs":     confs,
        "length":    length,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_result(r):
    if r.get("ok"):
        print(f"  ✓ decoded: {r['payload']!r}")
    else:
        print(f"  ✗ decode failed: {r.get('reason', 'CRC mismatch')}")
    if "chirp_peak" in r:
        print(f"  chirp peak: sample {r['chirp_peak']}  "
              f"(corr={r.get('chirp_val', 0):.3f})")
    if "mean_conf" in r:
        print(f"  mean bit confidence: {r['mean_conf']:+.3f}")
    if "crc_rx" in r:
        print(f"  CRC rx={r['crc_rx']:08x}  calc={r['crc_calc']:08x}")


def save_report(rec, sr, result, path, f0, f1):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    ax = axes[0]
    ax.specgram(rec, NFFT=1024, Fs=sr, noverlap=896, cmap="viridis")
    ax.set_ylim(15000, min(sr / 2, 25000))
    ax.axhline(f0, color="r", ls="--", lw=0.7, alpha=0.7, label=f"f0={f0:.0f}")
    ax.axhline(f1, color="orange", ls="--", lw=0.7, alpha=0.7, label=f"f1={f1:.0f}")
    ax.set_ylabel("Hz")
    ax.set_title("Spectrogram (15 kHz – Nyquist)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    if "confs" in result and len(result["confs"]):
        ax.plot(result["confs"], lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Bit confidence (e1−e0)/(e1+e0)")
    ax.set_xlabel("bit index")
    ax.set_ylabel("confidence")

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  report: {path}")


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def _warn_sr(sr, f1):
    if sr < 2.2 * f1:
        print(f"[warn] sample rate {sr} Hz is low for a {f1:.0f} Hz carrier. "
              f"Use >= {int(2.5 * f1)} Hz.", file=sys.stderr)


def cmd_loopback(args):
    """DSP-only test — no hardware touched."""
    wave, bits_tx, _ = build_waveform(args.text, args.sr, args.f0, args.f1,
                                      args.bit_dur, args.amp)
    result = decode(wave, args.sr, args.f0, args.f1, args.bit_dur)

    print(f"Loopback: {len(bits_tx)} data bits, "
          f"{len(wave)/args.sr:.2f}s waveform")
    print_result(result)

    if "bits" in result:
        n = min(len(bits_tx), len(result["bits"]))
        errs = sum(a != b for a, b in zip(bits_tx[:n], result["bits"][:n]))
        ber = errs / max(n, 1)
        print(f"  BER: {ber:.2e}  ({errs}/{n} bits)")

    if args.report:
        save_report(wave, args.sr, result, args.report, args.f0, args.f1)

    return 0 if result.get("ok") else 2


def cmd_send(args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    _warn_sr(args.sr, args.f1)
    wave, bits, _ = build_waveform(args.text, args.sr, args.f0, args.f1,
                                   args.bit_dur, args.amp)
    dur = len(wave) / args.sr
    print(f"Transmitting {len(bits)} bits ({dur:.2f}s) "
          f"@ {args.f0:.0f}/{args.f1:.0f} Hz, sr={args.sr}")
    sd.play(wave.astype(np.float32), args.sr, device=args.device)
    sd.wait()
    return 0


def _record_and_decode(args, rec):
    result = decode(rec, args.sr, args.f0, args.f1, args.bit_dur)
    print_result(result)
    if args.report:
        save_report(rec, args.sr, result, args.report, args.f0, args.f1)
    return 0 if result.get("ok") else 2


def cmd_recv(args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    _warn_sr(args.sr, args.f1)
    n = int(args.sr * args.duration)
    print(f"Recording {args.duration:.2f}s …")
    rec = sd.rec(n, samplerate=args.sr, channels=1,
                 dtype="float32", device=args.device)
    sd.wait()
    return _record_and_decode(args, rec.ravel())


def cmd_sendrecv(args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    _warn_sr(args.sr, args.f1)

    wave, bits, _ = build_waveform(args.text, args.sr, args.f0, args.f1,
                                   args.bit_dur, args.amp)
    wave = wave.astype(np.float32)
    dur = len(wave) / args.sr
    print(f"Send+recv: {len(bits)} bits ({dur:.2f}s) @ "
          f"{args.f0:.0f}/{args.f1:.0f} Hz, sr={args.sr}")

    in_dev  = args.in_device  if args.in_device  is not None else args.device
    out_dev = args.out_device if args.out_device is not None else args.device

    chunks = []
    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    stream = sd.InputStream(samplerate=args.sr, channels=1,
                            dtype="float32", device=in_dev, callback=cb)
    with stream:
        sd.play(wave, args.sr, device=out_dev)
        sd.wait()
        time.sleep(args.tail)

    if not chunks:
        print("error: no audio captured", file=sys.stderr)
        return 1
    rec = np.concatenate(chunks).ravel()
    return _record_and_decode(args, rec)


def cmd_sweep(args):
    """Log-free linear sweep used to characterise the speaker's HF rolloff."""
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    _warn_sr(args.sr, args.sweep_f1)

    sweep = apply_fade(chirp(args.sr, args.sweep_f0, args.sweep_f1,
                             args.duration, args.amp, fade_ms=10.0),
                       args.sr, 10.0)

    print(f"Sweeping {args.sweep_f0/1000:.1f} → {args.sweep_f1/1000:.1f} kHz "
          f"over {args.duration:.2f}s")

    chunks = []
    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    in_dev  = args.in_device  if args.in_device  is not None else args.device
    out_dev = args.out_device if args.out_device is not None else args.device

    stream = sd.InputStream(samplerate=args.sr, channels=1,
                            dtype="float32", device=in_dev, callback=cb)
    with stream:
        sd.play(sweep, args.sr, device=out_dev)
        sd.wait()
        time.sleep(args.tail)

    if not chunks:
        print("error: no audio captured", file=sys.stderr)
        return 1
    rec = np.concatenate(chunks).ravel()

    # offline matched-filter spectrogram via FFT, then report peak-vs-freq
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.specgram(rec, NFFT=2048, Fs=args.sr, noverlap=1536, cmap="magma")
    ax.set_ylim(args.sweep_f0 - 1000, min(args.sr / 2, args.sweep_f1 + 1000))
    ax.set_title(f"Sweep response  {args.sweep_f0/1000:.1f}–"
                 f"{args.sweep_f1/1000:.1f} kHz")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    plt.tight_layout()

    # extract envelope along the known sweep path
    t_rec = np.arange(len(rec)) / args.sr
    k = (args.sweep_f1 - args.sweep_f0) / args.duration
    # find where the sweep starts by looking for max energy
    win = int(args.sr * 0.005)
    energy = np.convolve(rec * rec, np.ones(win), mode="same")
    start = int(np.argmax(energy)) - int(args.sr * 0.02)
    start = max(0, start)

    freqs, levels = [], []
    step = int(args.sr * 0.005)
    n = np.arange(step)
    for i in range(0, len(rec) - step, step):
        t_global = (i - start) / args.sr
        if not (0 <= t_global <= args.duration):
            continue
        f_now = args.sweep_f0 + k * t_global
        ref = np.exp(-2j * np.pi * f_now * n / args.sr)
        seg = rec[i:i + step] * np.hanning(step)
        freqs.append(f_now)
        levels.append(20 * np.log10(abs(np.sum(seg * ref)) + 1e-12))
    if freqs:
        ax2 = ax.twinx()
        ax2.plot([start / args.sr + (f - args.sweep_f0) / k for f in freqs],
                 levels, color="cyan", lw=1.0, alpha=0.8)
        ax2.set_ylabel("level (dB, arb.)", color="cyan")

    plt.savefig(args.report, dpi=140)
    plt.close(fig)
    print(f"  report: {args.report}")
    return 0


def cmd_list(_args):
    if sd is None:
        print("error: sounddevice not installed", file=sys.stderr)
        return 1
    print(sd.query_devices())
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_common(p):
    p.add_argument("--sr",       type=int,   default=SR_DEFAULT)
    p.add_argument("--f0",       type=float, default=F0_DEFAULT,
                   help="FSK frequency for bit 0 (Hz)")
    p.add_argument("--f1",       type=float, default=F1_DEFAULT,
                   help="FSK frequency for bit 1 (Hz)")
    p.add_argument("--bit-dur",  type=float, default=BIT_DUR_DEFAULT,
                   help="bit duration (s)")
    p.add_argument("--amp",      type=float, default=AMP_DEFAULT)
    p.add_argument("--device",   type=int, default=None)
    p.add_argument("--in-device",  type=int, default=None)
    p.add_argument("--out-device", type=int, default=None)
    p.add_argument("--tail",     type=float, default=TAIL_DUR,
                   help="extra recording time after playback (s)")
    p.add_argument("--report",   type=Path, default=None,
                   help="save a PNG report here")


def main():
    ap = argparse.ArgumentParser(prog="txrx",
                                 description="Acoustic FSK test rig")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("loopback", help="DSP-only test, no hardware")
    p.add_argument("text", nargs="?", default="hello 21.5 kHz")
    add_common(p); p.set_defaults(func=cmd_loopback)

    p = sub.add_parser("send", help="transmit a string")
    p.add_argument("text", nargs="?", default="hello 21.5 kHz")
    add_common(p); p.set_defaults(func=cmd_send)

    p = sub.add_parser("recv", help="record and try to decode")
    p.add_argument("--duration", type=float, default=4.0)
    add_common(p); p.set_defaults(func=cmd_recv)

    p = sub.add_parser("sendrecv", help="send + record simultaneously")
    p.add_argument("text", nargs="?", default="hello 21.5 kHz")
    add_common(p); p.set_defaults(func=cmd_sendrecv)

    p = sub.add_parser("sweep", help="frequency sweep + report")
    p.add_argument("--sweep-f0", type=float, default=15000.0)
    p.add_argument("--sweep-f1", type=float, default=23000.0)
    p.add_argument("--duration", type=float, default=3.0)
    add_common(p)
    p.set_defaults(report=Path("sweep.png"))
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("list", help="list audio devices")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)