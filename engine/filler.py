"""Filler Word Engine — pre-generate and cache filler audio for instant playback."""
import random
import logging
import numpy as np

log = logging.getLogger("voxlabs.filler")

FILLER_TEXTS = ["嗯，", "好的，", "是的，", "了解，", "嗯嗯，", "好，", "对，"]


class FillerEngine:
    def __init__(self, tts_engine):
        self._tts = tts_engine
        self._cache: list[tuple[str, np.ndarray]] = []
        self._idx = 0

    def pregenerate(self):
        log.info("Pre-generating %d filler words...", len(FILLER_TEXTS))
        for text in FILLER_TEXTS:
            result = self._tts.synthesize(text)
            audio = result["audio"]
            if audio.dtype != np.int16:
                audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            self._cache.append((text, audio.tobytes()))
            log.info("  Filler '%s': %.1fms, %d bytes", text, result["ttfa_ms"], len(audio.tobytes()))
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
