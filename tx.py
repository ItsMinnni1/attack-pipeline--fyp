#!/usr/bin/env python3
"""
usound_secrets.py — Transmit RS-protected, XOR-encrypted .txt files over
near-ultrasonic BPSK/QPSK audio at 21.5 kHz, and decode them back.

Framing (matches the original script, plus a 4-byte frame index):
    every 201-byte chunk  ->  [idx:4 LE][nonce:16][payload:201][crc16:2]
                          ->  Reed-Solomon(255, 223)
                          ->  255-byte on-air frame

On-air (per 255-byte frame):
    [64-bit preamble][2040 bits]            BPSK  (2040 syms @ 1000 baud)
    [64-bit preamble][1020 QPSK symbols]    QPSK  (1020 syms @ 1000 baud)
    + 20 ms silence gap, concatenated.

Usage:
    tx  SRC_DIR  OUT.wav  [--mode qpsk] [--baud 1000]
    rx  IN.wav   OUT.txt  [--mode qpsk] [--baud 1000]

The rx side ignores any frame that fails RS decode or CRC.
"""

from __future__ import annotations

import argparse
import binascii
import sys
import wave
from pathlib import Path

import numpy as np
from reedsolo import RSCodec

# --------------------------------------------------------------------- config
CARRIER_HZ      = 21_500.0          # Hz
DEFAULT_FS      = 48_000            # Hz
DEFAULT_BAUD    = 1_000.0           # symbols / s
ROLLOFF         = 0.35              # RRC excess bandwidth
FILTER_SPAN     = 8                 # RRC span (symbols)
GAP_MS          = 20.0              # silence between packets
XOR_KEY         = b"SecretKey123!"  # same key as your script
DATA_CHUNK_SIZE = 201               # 223 − 4 (idx) − 16 (nonce) − 2 (CRC)
FRAME_BYTES     = 255               # RS codeword length
RS_DATA_BYTES   = 223               # RS payload length

# 64-bit sync word with a sharp autocorrelation peak
PREAMBLE_BITS = np.array([
    1,1,0,1,0,0,1,1, 0,0,1,0,1,1,0,1,
    1,0,1,0,0,0,1,0, 0,1,1,1,0,1,0,0,
    0,1,0,1,1,0,0,1, 1,0,0,1,0,1,0,1,
    1,1,0,0,1,1,0,0, 0,1,1,0,1,0,0,0,
], dtype=np.uint8)

rs = RSCodec(32)


# =====================================================================
#  DSP building blocks
# =====================================================================
def rrc_filter(beta: float, sps: int, span: int) -> np.ndarray:
    N = span * sps
    t = np.arange(-N // 2, N // 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                - (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = (np.sin(np.pi * ti * (1.0 - beta))
                   + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta)))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h * h))
    return h


def shape(syms: np.ndarray, sps: int, rrc: np.ndarray) -> np.ndarray:
    up = np.zeros(len(syms) * sps, dtype=np.complex128)
    up[::sps] = syms
    return np.convolve(up, rrc, mode="full")


def xcorr(x: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    L = len(tpl)
    n = len(x) + L - 1
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.fft(x, nfft)
    T = np.fft.fft(np.conj(tpl[::-1]), nfft)
    return np.fft.ifft(X * T)[L - 1 : L - 1 + len(x) - L + 1]


def find_peaks(arr: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    above = arr > threshold
    peaks = []
    i, n = 0, len(arr)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            peaks.append(i + int(np.argmax(arr[i:j])))
            i = j
        else:
            i += 1
    if not peaks:
        return np.array([], dtype=int)
    peaks = np.array(peaks)
    keep = [peaks[0]]
    for p in peaks[1:]:
        if p - keep[-1] >= min_distance:
            keep.append(p)
        elif arr[p] > arr[keep[-1]]:
            keep[-1] = p
    return np.array(keep)


# =====================================================================
#  Bits <-> symbols
# =====================================================================
def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    n = len(bits) - (len(bits) % 8)
    return np.packbits(bits[:n]).tobytes()


def bits_to_symbols(bits: np.ndarray, mode: str) -> np.ndarray:
    bits = bits.astype(np.uint8)
    if mode == "bpsk":
        return (1.0 - 2.0 * bits.astype(np.float64)).astype(np.complex128)
    if len(bits) % 2:
        bits = np.append(bits, 0)
    b = bits.reshape(-1, 2)
    I = 1.0 - 2.0 * b[:, 0].astype(np.float64)
    Q = 1.0 - 2.0 * b[:, 1].astype(np.float64)
    return ((I + 1j * Q) / np.sqrt(2.0)).astype(np.complex128)


def symbols_to_bits(syms: np.ndarray, mode: str) -> np.ndarray:
    if mode == "bpsk":
        return (np.real(syms) < 0).astype(np.uint8)
    b0 = (np.real(syms) < 0).astype(np.uint8)
    b1 = (np.imag(syms) < 0).astype(np.uint8)
    out = np.empty(2 * len(syms), dtype=np.uint8)
    out[0::2] = b0
    out[1::2] = b1
    return out


# =====================================================================
#  WAV I/O
# =====================================================================
def write_wav(path: str, x: np.ndarray, fs: int) -> None:
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


def read_wav(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, "rb") as w:
        fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw*8} bits")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return fs, data


def apply_fade(x: np.ndarray, fs: int, ms: float = 5.0) -> np.ndarray:
    n = min(int(fs * ms / 1000.0), len(x) // 4)
    if n < 1:
        return x
    win = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    y = x.copy()
    y[:n] *= win
    y[-n:] *= win[::-1]
    return y


# =====================================================================
#  Framing (your script's logic, + 4-byte frame index)
# =====================================================================
def xor_encrypt_decrypt(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def calculate_crc16(data: bytes) -> bytes:
    return binascii.crc_hqx(data, 0).to_bytes(2, byteorder="big")


def build_frame(idx: int, chunk: bytes) -> bytes:
    """Build one 255-byte RS codeword."""
    if len(chunk) != DATA_CHUNK_SIZE:
        raise ValueError(f"chunk must be {DATA_CHUNK_SIZE} bytes")
    nonce = np.random.bytes(16)
    payload_enc = xor_encrypt_decrypt(chunk, XOR_KEY)
    inner = idx.to_bytes(4, "little") + nonce + payload_enc         # 221 B
    crc = calculate_crc16(inner)                                     # 2 B
    rs_data = inner + crc                                            # 223 B
    return bytes(rs.encode(rs_data))                                 # 255 B


def unframe_one(frame: bytes) -> tuple[int, bytes]:
    """RS-decode + CRC-check + decrypt one 255-byte frame."""
    if len(frame) != FRAME_BYTES:
        raise ValueError("bad frame length")
    rs_data = bytes(rs.decode(frame)[0])                             # 223 B
    inner = rs_data[:221]
    crc_stored = int.from_bytes(rs_data[221:223], "big")
    if calculate_crc16(inner) != crc_stored.to_bytes(2, "big"):
        raise ValueError("CRC-16 mismatch")
    idx = int.from_bytes(inner[:4], "little")
    payload_enc = inner[20:221]                                      # skip nonce
    return idx, xor_encrypt_decrypt(payload_enc, XOR_KEY)


# =====================================================================
#  Modulator
# =====================================================================
def modulate(framed: bytes, mode: str, fs: int, baud: float,
             amplitude: float = 0.5) -> np.ndarray:
    if len(framed) % FRAME_BYTES:
        raise ValueError("framed stream length must be a multiple of 255")

    upper_edge = CARRIER_HZ + baud * (1.0 + ROLLOFF) / 2.0
    if upper_edge >= fs / 2.0:
        sys.exit(f"fs={fs} Hz too low: signal reaches {upper_edge:.0f} Hz "
                 f"(Nyquist {fs/2:.0f} Hz).")

    sps = int(round(fs / baud))
    if abs(sps * baud - fs) > 1e-6:
        print(f"[mod] warning: fs/baud not integer; sps = {sps}", file=sys.stderr)

    rrc = rrc_filter(ROLLOFF, sps, FILTER_SPAN)
    pre_syms = bits_to_symbols(PREAMBLE_BITS, mode)
    gap = np.zeros(int(fs * GAP_MS / 1000.0))

    packets = []
    for off in range(0, len(framed), FRAME_BYTES):
        frame = framed[off : off + FRAME_BYTES]
        data_syms = bits_to_symbols(bytes_to_bits(frame), mode)
        syms = np.concatenate([pre_syms, data_syms])
        bb = shape(syms, sps, rrc)
        n = np.arange(len(bb))
        tx = np.real(bb * np.exp(1j * 2 * np.pi * CARRIER_HZ * n / fs))
        packets.append(tx)
        packets.append(gap)

    out = np.concatenate(packets)
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out / peak * amplitude
    return apply_fade(out, fs)


# =====================================================================
#  Demodulator
# =====================================================================
def demodulate(rx: np.ndarray, fs: int, mode: str, baud: float) -> list[bytes]:
    sps = int(round(fs / baud))
    rrc = rrc_filter(ROLLOFF, sps, FILTER_SPAN)

    n = np.arange(len(rx))
    bb = rx * np.exp(-1j * 2 * np.pi * CARRIER_HZ * n / fs) * 2.0
    mf = np.convolve(bb, rrc, mode="full")

    pre_syms = bits_to_symbols(PREAMBLE_BITS, mode)
    tpl = np.convolve(shape(pre_syms, sps, rrc), rrc, mode="full")

    corr = xcorr(mf, tpl)
    mag = np.abs(corr)
    if not np.any(mag > 0):
        return []
    peaks = find_peaks(mag, 0.5 * mag.max(), min_distance=int(sps * 50))

    n_pre   = len(pre_syms)
    n_data  = (FRAME_BYTES * 8) // (2 if mode == "qpsk" else 1)
    n_total = n_pre + n_data

    frames: list[bytes] = []
    for pk in peaks:
        start = int(pk) + FILTER_SPAN * sps
        idx = start + np.arange(n_total) * sps
        if idx[-1] >= len(mf):
            continue
        syms = mf[idx]

        # resolve QPSK 4-fold (or BPSK 2-fold) phase ambiguity
        best_rot, best_err = 1 + 0j, None
        for rot in (1 + 0j, -1 + 0j, 1j, -1j):
            b = symbols_to_bits(syms[:n_pre] * rot, mode)
            e = int(np.sum(b != PREAMBLE_BITS))
            if best_err is None or e < best_err:
                best_err, best_rot = e, rot
        if best_err is None or best_err > n_pre // 4:
            continue
        syms = syms * best_rot

        data_bits = symbols_to_bits(syms[n_pre:], mode)[: FRAME_BYTES * 8]
        frame = bits_to_bytes(data_bits)
        if len(frame) == FRAME_BYTES:
            frames.append(frame)
    return frames


# =====================================================================
#  CLI
# =====================================================================
def cmd_tx(args: argparse.Namespace) -> None:
    src = Path(args.source_dir)
    candidates = [
        p for p in src.rglob("*.txt")
        if not p.name.endswith("_framed.txt")
    ]
    if not candidates:
        sys.exit(f"[tx] no .txt file found under {src}")
    target = candidates[0]
    raw = target.read_bytes()
    print(f"[tx] source: {target} ({len(raw)} B)", file=sys.stderr)

    framed = bytearray()
    n = 0
    for off in range(0, len(raw), DATA_CHUNK_SIZE):
        chunk = raw[off : off + DATA_CHUNK_SIZE]
        if len(chunk) < DATA_CHUNK_SIZE:
            chunk = chunk.ljust(DATA_CHUNK_SIZE, b"\x00")
        framed += build_frame(n, chunk)
        n += 1
    if n == 0:
        sys.exit("[tx] source file is empty")
    print(f"[tx] framed {n} RS(255,223) codewords", file=sys.stderr)

    tx = modulate(bytes(framed), args.mode, args.sample_rate, args.baud,
                  amplitude=args.amplitude)
    write_wav(args.output, tx, args.sample_rate)
    print(f"[tx] mode={args.mode} baud={args.baud:g} "
          f"duration={len(tx)/args.sample_rate:.2f}s -> {args.output}",
          file=sys.stderr)


def cmd_rx(args: argparse.Namespace) -> None:
    fs, rx = read_wav(args.input)
    frames = demodulate(rx, fs, args.mode, args.baud)
    print(f"[rx] {len(frames)} packet(s) detected in {args.input}", file=sys.stderr)

    chunks: dict[int, bytes] = {}
    for f in frames:
        try:
            idx, payload = unframe_one(f)
        except Exception as e:
            print(f"[rx] dropping frame: {e}", file=sys.stderr)
            continue
        chunks[idx] = payload

    if not chunks:
        sys.exit("[rx] no valid frames decoded")

    indices = sorted(chunks)
    missing = sorted(set(range(indices[-1] + 1)) - set(indices))
    if missing:
        print(f"[rx] warning: missing frames {missing}", file=sys.stderr)

    out = b"".join(chunks[i] for i in indices).rstrip(b"\x00")
    Path(args.output).write_bytes(out)
    print(f"[rx] recovered {len(out)} B -> {args.output}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Near-ultrasonic BPSK/QPSK transport for RS-framed .txt files."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    tx = sub.add_parser("tx", help="transmit: SOURCE_DIR -> .wav")
    tx.add_argument("source_dir", help="folder to search for a .txt file")
    tx.add_argument("output", help="output .wav")
    tx.add_argument("--mode", choices=("bpsk", "qpsk"), default="qpsk")
    tx.add_argument("--baud", type=float, default=DEFAULT_BAUD,
                    help="symbol rate in symbols/s (default 1000)")
    tx.add_argument("--sample-rate", type=int, default=DEFAULT_FS)
    tx.add_argument("--amplitude", type=float, default=0.5)
    tx.set_defaults(func=cmd_tx)

    rx = sub.add_parser("rx", help="receive: .wav -> recovered .txt")
    rx.add_argument("input", help="input .wav")
    rx.add_argument("output", help="recovered .txt")
    rx.add_argument("--mode", choices=("bpsk", "qpsk"), default="qpsk")
    rx.add_argument("--baud", type=float, default=DEFAULT_BAUD)
    rx.set_defaults(func=cmd_rx)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    import os
    raise SystemExit(main())