"""SenseVoiceSmall ASR via FunASR framework."""
import time
import os
import numpy as np


class SenseVoiceASR:
    def __init__(self, model_dir: str, device: str = "cuda:2"):
        self._model_dir = model_dir
        self._device = device
        self._model = None

    def load(self):
        from funasr import AutoModel
        self._model = AutoModel(
            model=self._model_dir,
            trust_remote_code=True,
            device=self._device,
            disable_update=True,
        )
        self._warmup()
        return self

    def _warmup(self):
        dummy = np.random.randn(16000).astype(np.float32)
        self.transcribe(dummy)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """Transcribe audio array. Returns {'text': str, 'latency_ms': float}."""
        t0 = time.perf_counter()
        result = self._model.generate(
            input=audio,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )
        latency = (time.perf_counter() - t0) * 1000

        text = ""
        if result and len(result) > 0:
            text = result[0].get("text", "")

        return {"text": text, "latency_ms": latency}
