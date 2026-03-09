"""Moonshine Tiny ASR — ultra-lightweight 27M model for speculative/fast ASR.

CPU-only, ONNX Runtime, ~20-50ms for 1-3s audio.
Used as the speculative ASR engine: runs during silence before endpointing confirms.
"""
import os
import time
import logging
import numpy as np

log = logging.getLogger("voxlabs.asr_moonshine")


class MoonshineASR:
    def __init__(self, model_name: str = "moonshine/tiny"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None

    def load(self):
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        self._model = MoonshineOnnxModel(model_name=self._model_name)
        self._tokenizer = load_tokenizer()
        self._warmup()
        log.info("MoonshineASR loaded (model=%s)", self._model_name)
        return self

    def _warmup(self):
        dummy = np.random.randn(16000).astype(np.float32)
        self.transcribe(dummy)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """Transcribe audio. Returns {'text': str, 'latency_ms': float}."""
        t0 = time.perf_counter()

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        tokens = self._model.generate(audio.astype(np.float32))
        texts = self._tokenizer.decode_batch(tokens)
        text = texts[0] if texts else ""

        latency = (time.perf_counter() - t0) * 1000
        return {"text": text.strip(), "latency_ms": latency}
