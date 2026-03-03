"""VoxCPM 1.5 TTS via nanovllm-voxcpm with voice cloning support."""
import os
import time
import logging
import numpy as np

log = logging.getLogger("voxlabs.tts")


class VoxCPMTTS:
    SAMPLE_RATE = 44100

    def __init__(self, model_dir: str, device: str = "cuda:2",
                 gpu_memory_utilization: float = 0.55):
        self._model_dir = model_dir
        self._device = device
        self._gpu_memory_utilization = gpu_memory_utilization
        self._engine = None
        self._warmed_up = False
        self._default_prompt_id = None

    def load(self):
        from nanovllm_voxcpm import VoxCPM
        gpu_idx = int(self._device.split(":")[-1]) if ":" in self._device else 0
        self._engine = VoxCPM.from_pretrained(
            model=self._model_dir,
            gpu_memory_utilization=self._gpu_memory_utilization,
            devices=[gpu_idx],
        )
        return self

    def register_voice(self, wav_path: str, prompt_text: str) -> str:
        """Register a voice clone prompt from a wav file.
        Returns the prompt_id for future use.
        """
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        fmt = "wav"
        if wav_path.endswith(".mp3"):
            fmt = "mp3"
        elif wav_path.endswith(".flac"):
            fmt = "flac"

        prompt_id = self._engine.add_prompt(wav_bytes, fmt, prompt_text)
        log.info("Voice prompt registered: id=%s, file=%s", prompt_id, os.path.basename(wav_path))
        return prompt_id

    def set_default_voice(self, prompt_id: str):
        """Set the default voice for all subsequent synthesize calls."""
        self._default_prompt_id = prompt_id
        log.info("Default voice set to: %s", prompt_id)

    def warmup(self):
        if not self._warmed_up:
            self.synthesize("你好，欢迎致电。")
            self._warmed_up = True

    def synthesize(self, text: str, prompt_id: str = None) -> dict:
        """Synthesize speech from text using cloned voice.
        If prompt_id is not specified, uses the default registered voice.
        """
        t0 = time.perf_counter()
        chunks = []
        ttfa = None

        pid = prompt_id or self._default_prompt_id
        kwargs = {"target_text": text, "temperature": 0.7, "cfg_value": 3.0}
        if pid:
            kwargs["prompt_id"] = pid

        chunk_idx = 0
        for audio_chunk in self._engine.generate(**kwargs):
            if isinstance(audio_chunk, np.ndarray):
                c = audio_chunk
            elif hasattr(audio_chunk, 'numpy'):
                c = audio_chunk.numpy()
            else:
                c = np.array(audio_chunk, dtype=np.float32)

            chunk_idx += 1
            if chunk_idx == 1:
                ttfa = (time.perf_counter() - t0) * 1000
                # Skip first chunk entirely — VoxCPM VAE has startup transient
                continue

            if chunk_idx == 2:
                # Fade-in on the second chunk (now the first audible one)
                fade = int(self.SAMPLE_RATE * 0.01)
                if len(c) > fade:
                    c[:fade] *= np.linspace(0, 1, fade, dtype=c.dtype)

            chunks.append(c)

        total = (time.perf_counter() - t0) * 1000
        audio_out = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

        return {
            "audio": audio_out,
            "sr": self.SAMPLE_RATE,
            "ttfa_ms": ttfa or total,
            "total_ms": total,
            "chunks": len(chunks),
        }
