"""Filler Word Engine — pre-generate and cache filler audio for instant playback.

v2.9.1: Guard against VoxCPM generation loops — cap filler audio to MAX_FILLER_SEC.
"""
import random
import logging
import numpy as np

log = logging.getLogger("voxlabs.filler")

FILLER_TEXTS = ["嗯，", "好的，", "是的，", "嗯嗯，", "好，"]

MAX_FILLER_SEC = 2.0


class FillerEngine:
    def __init__(self, tts_engine):
        self._tts = tts_engine
        self._cache: list[tuple[str, np.ndarray]] = []
        self._idx = 0

    def pregenerate(self):
        sr = getattr(self._tts, "SAMPLE_RATE", 44100)
        max_bytes = int(MAX_FILLER_SEC * sr * 2)
        log.info("Pre-generating %d filler words (max %.1fs / %d bytes each)...",
                 len(FILLER_TEXTS), MAX_FILLER_SEC, max_bytes)
        for text in FILLER_TEXTS:
            try:
                result = self._tts.synthesize(text)
            except Exception as e:
                log.warning("  Filler '%s' TTS failed: %s — skipped", text, e)
                continue
            audio = result["audio"]
            if audio.dtype != np.int16:
                audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            raw = audio.tobytes()
            if len(raw) > max_bytes:
                log.warning("  Filler '%s' too long: %d bytes (%.1fs) — truncating to %.1fs",
                            text, len(raw), len(raw) / sr / 2, MAX_FILLER_SEC)
                raw = raw[:max_bytes]
            self._cache.append((text, raw))
            log.info("  Filler '%s': %.1fms, %d bytes", text, result["ttfa_ms"], len(raw))
        if not self._cache:
            log.error("No filler words cached! Filler disabled.")
        else:
            log.info("Filler cache ready: %d items", len(self._cache))

    def get_filler(self) -> tuple[str, bytes]:
        """Get next filler audio (round-robin with shuffle)."""
        if not self._cache:
            return "", b""
        text, audio_bytes = self._cache[self._idx % len(self._cache)]
        self._idx += 1
        if self._idx >= len(self._cache):
            random.shuffle(self._cache)
            self._idx = 0
        return text, audio_bytes

    @property
    def sample_rate(self):
        return self._tts.SAMPLE_RATE
