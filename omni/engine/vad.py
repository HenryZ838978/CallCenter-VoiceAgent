"""Silero VAD wrapper — copied from voiceagent for isolation."""
import os
import torch
import numpy as np


class SileroVAD:
    def __init__(self, model_dir: str, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._model_dir = model_dir

    def load(self):
        jit_path = os.path.join(self._model_dir, "silero_vad.jit")
        if os.path.exists(jit_path):
            self._model = torch.jit.load(jit_path, map_location="cpu")
        else:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=False,
            )
        self._model.eval()
        self.reset()
        return self

    def reset(self):
        if self._model is not None:
            self._model.reset_states()
        self._speech_active = False
        self._silence_count = 0

    def process_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        """Process 512 samples at 16kHz (32ms).
        Returns dict with speech_prob, speech_start, speech_end.
        """
        tensor = torch.from_numpy(audio_chunk).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        prob = self._model(tensor, sr).item()
        event = {"speech_prob": prob, "speech_start": False, "speech_end": False}

        if prob >= self.threshold and not self._speech_active:
            self._speech_active = True
            self._silence_count = 0
            event["speech_start"] = True
        elif prob < self.threshold and self._speech_active:
            self._silence_count += 1
            if self._silence_count >= 15:
                self._speech_active = False
                self._silence_count = 0
                event["speech_end"] = True
        elif prob >= self.threshold and self._speech_active:
            self._silence_count = 0

        return event

    @property
    def is_speech_active(self):
        return self._speech_active
