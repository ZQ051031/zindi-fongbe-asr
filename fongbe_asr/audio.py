from functools import lru_cache
from math import gcd

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

RATE = 16000
WINDOW = 640
HOP = 160
FFT = 1024
MELS = 64
FEATURES = MELS + 3


def read_audio(path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(samples):
        raise ValueError(f"Empty audio: {path}")
    duration = len(samples) / rate
    mono = samples.mean(axis=1)
    if rate != RATE:
        divisor = gcd(rate, RATE)
        mono = resample_poly(mono, RATE // divisor, rate // divisor)
    return mono, duration


@lru_cache(maxsize=1)
def mel_bank():
    scale = lambda hz: 2595 * np.log10(1 + hz / 700)
    edges = 700 * (10 ** (np.linspace(scale(50), scale(7600), MELS + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(FFT, 1 / RATE)
    return np.maximum(0, np.minimum(
        (bins - edges[:-2, None]) / np.diff(edges)[:-1, None],
        (edges[2:, None] - bins) / np.diff(edges)[1:, None],
    )).astype(np.float32)


def features(samples):
    padded = np.pad(samples, (WINDOW // 2, WINDOW // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, WINDOW)[::HOP]
    windowed = frames * np.hanning(WINDOW).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed, FFT)).astype(np.float32) ** 2
    mel = np.log(np.maximum(spectrum @ mel_bank().T, 1e-6))
    mel = np.clip((mel - mel.mean(axis=0)) / np.maximum(mel.std(axis=0), 0.5), -8, 8)

    # Local autocorrelation peaks and utterance-relative log F0.
    corr = np.fft.irfft(spectrum, FFT)
    corr /= np.maximum(corr[:, :1], 1e-9)
    lo, hi = RATE // 420, RATE // 65
    band = corr[:, lo:hi + 1]
    peaks = (band > corr[:, lo - 1:hi]) & (band >= corr[:, lo + 1:hi + 2])
    best = np.max(np.where(peaks, band, 0), axis=1)
    first = (peaks & (band >= best[:, None] * 0.9)).argmax(axis=1)
    voiced = best >= 0.35
    pitch = np.log2(RATE / (lo + first))
    center = np.median(pitch[voiced]) if voiced.any() else np.log2(160)
    return np.column_stack((mel, np.where(voiced, pitch - center, 0),
                            np.where(voiced, pitch - np.log2(160), 0), best)).astype(np.float32)
