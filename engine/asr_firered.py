"""FireRedASR2-AED wrapper — SOTA Chinese ASR (CER 2.89%, 20+ dialects including Cantonese).

Requires model at models/FireRedASR2-AED/ and fireredasr2 source at engine/fireredasr2/.
"""
import os
import sys
import time
import tempfile
import logging
import numpy as np
import soundfile as sf

log = logging.getLogger("voxlabs.asr_firered")


class FireRedASR:
    def __init__(self, model_dir: str, device: str = "cuda:0", use_half: bool = True):
        self._model_dir = model_dir
        self._device = device
        self._use_half = use_half
        self._model = None
        self._tmp_dir = tempfile.mkdtemp(prefix="firered_asr_")

    def load(self):
        engine_dir = os.path.dirname(os.path.abspath(__file__))
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)

        import torch
        if "cuda" in self._device:
            gpu_idx = int(self._device.split(":")[-1]) if ":" in self._device else 0
            torch.cuda.set_device(gpu_idx)

        from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
        use_gpu = "cuda" in self._device
        config = FireRedAsr2Config(use_gpu=use_gpu, use_half=self._use_half)
        self._model = FireRedAsr2.from_pretrained("aed", self._model_dir, config)
        self._warmup()
        log.info("FireRedASR loaded (model_dir=%s, device=%s)", self._model_dir, self._device)
        return self

    def _warmup(self):
        dummy = np.random.randn(16000).astype(np.float32) * 0.01
        self.transcribe(dummy)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """Transcribe audio array. Returns {'text': str, 'latency_ms': float}."""
        t0 = time.perf_counter()

        tmp_path = os.path.join(self._tmp_dir, "chunk.wav")
        int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        sf.write(tmp_path, int16, sr)

        results = self._model.transcribe(["utt"], [tmp_path])
        latency = (time.perf_counter() - t0) * 1000

        text = ""
        if results and len(results) > 0:
            text = results[0].get("text", "")

        return {"text": text.strip(), "latency_ms": latency}
