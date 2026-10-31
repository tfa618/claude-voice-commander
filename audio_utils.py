"""Audio helpers that depend only on numpy: resampling, RMS, utterance segmentation, beeps."""
import math
import numbers

import numpy as np

SAMPLING_RATE = 16000


def _lowpass_for_downsample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Hann-windowed-sinc low-pass to suppress aliasing before downsampling.
    Reflect pad + 'valid' convolution keeps the input length unchanged so
    downstream duration math stays correct. Buffers shorter than the kernel
    half-width are returned untouched (every sample would be edge artifact).
    """
    cutoff = 0.9 * (SAMPLING_RATE / 2.0) / orig_sr
    half = max(8, int(round(3.0 / cutoff)))
    if audio.size <= half:
        return audio
    n = 2 * half + 1
    t = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.sinc(2.0 * cutoff * t) * np.hanning(n)
    kernel = kernel / kernel.sum()
    padded = np.pad(audio, (half, half), mode="reflect")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Downsample to 16 kHz with anti-aliasing; untouched if already 16 kHz."""
    # numbers.Real also accepts numpy ints/floats; bool is excluded explicitly.
    # isfinite rejects NaN and +/-inf (inf > 0 is True, so a plain "> 0" let
    # +inf through and it crashed with ZeroDivisionError in the low-pass).
    if (isinstance(orig_sr, bool) or not isinstance(orig_sr, numbers.Real)
            or not math.isfinite(orig_sr) or orig_sr <= 0):
        raise ValueError(f"orig_sr must be a positive number, got {orig_sr!r}")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if orig_sr == SAMPLING_RATE or audio.size < 8:
        return audio
    if orig_sr > SAMPLING_RATE:
        audio = _lowpass_for_downsample(audio, orig_sr)
    duration = audio.shape[0] / float(orig_sr)
    target_len = int(round(duration * SAMPLING_RATE))
    if target_len <= 0:
        return np.empty(0, dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))


class UtteranceSegmenter:
    """
    Groups fixed-size audio chunks into utterances using an energy gate.

    An utterance starts at the first loud chunk (plus a little pre-roll, so
    the first syllable isn't clipped) and ends after `pause_seconds` of quiet.
    Utterances with less than `min_speech_seconds` of loud audio are dropped
    as clicks/coughs. Utterances are force-split at `max_seconds` so a noisy
    room can't grow a buffer forever.
    """

    def __init__(self, sample_rate: int, chunk_seconds: float, silence_threshold: float,
                 pause_seconds: float, min_speech_seconds: float,
                 max_seconds: float = 30.0, preroll_chunks: int = 3):
        self.sr = sample_rate
        self.chunk_seconds = chunk_seconds
        self.threshold = silence_threshold
        # ceil like min_speech_chunks below: a pause must last AT LEAST the
        # configured time (round() is banker's rounding: 0.45 s -> 4 chunks).
        self.pause_chunks = max(1, math.ceil(pause_seconds / chunk_seconds - 1e-9))
        # ceil, not round: 0.25 s / 0.1 s chunks must require 3 loud chunks
        # (0.3 s), never 2 (0.2 s, shorter than configured). The small epsilon
        # stops float noise (0.3 / 0.1 = 3.0000000000000004) adding a chunk.
        self.min_speech_chunks = max(1, math.ceil(min_speech_seconds / chunk_seconds - 1e-9))
        self.max_chunks = max(1, int(round(max_seconds / chunk_seconds)))
        self.preroll_chunks = preroll_chunks
        self.reset()

    def reset(self):
        self._preroll: list[np.ndarray] = []
        self._buf: list[np.ndarray] = []
        self._speech_chunks = 0
        self._silent_run = 0
        self._peak = 0.0

    @property
    def in_speech(self) -> bool:
        return bool(self._buf)

    def feed(self, chunk: np.ndarray):
        """Feed one chunk. Returns (audio, peak_rms) when an utterance completes, else None."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        level = rms(chunk)
        loud = level >= self.threshold
        if not self._buf:
            if loud:
                self._buf = self._preroll + [chunk]
                self._preroll = []
                self._speech_chunks = 1
                self._silent_run = 0
                self._peak = level
            else:
                self._preroll.append(chunk)
                if len(self._preroll) > self.preroll_chunks:
                    self._preroll.pop(0)
            return None

        self._buf.append(chunk)
        self._peak = max(self._peak, level)
        if loud:
            self._speech_chunks += 1
            self._silent_run = 0
        else:
            self._silent_run += 1
        if self._silent_run >= self.pause_chunks or len(self._buf) >= self.max_chunks:
            return self._finish()
        return None

    def _finish(self):
        enough = self._speech_chunks >= self.min_speech_chunks
        audio = np.concatenate(self._buf) if self._buf else np.empty(0, np.float32)
        peak = self._peak
        self.reset()
        return (audio, peak) if enough else None


def make_tone(freq: float, seconds: float, sr: int = 44100, volume: float = 0.2) -> np.ndarray:
    """Short sine beep with 10 ms fade in/out (avoids speaker clicks)."""
    n = max(1, int(sr * seconds))
    t = np.arange(n) / sr
    tone = volume * np.sin(2 * np.pi * freq * t)
    fade = min(n // 2, int(sr * 0.01))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    return tone.astype(np.float32)


def make_beep(kind: str, sr: int = 44100) -> np.ndarray:
    """'start' = rising two-note, 'sent' = single high, 'cancel' = falling low."""
    gap = np.zeros(int(sr * 0.04), np.float32)
    if kind == "start":
        return np.concatenate([make_tone(660, 0.08, sr), gap, make_tone(990, 0.1, sr)])
    if kind == "sent":
        return make_tone(1320, 0.09, sr)
    if kind == "cancel":
        return np.concatenate([make_tone(500, 0.09, sr), gap, make_tone(330, 0.12, sr)])
    return make_tone(800, 0.08, sr)
