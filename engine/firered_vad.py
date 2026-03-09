"""FireRedVAD streaming wrapper — drop-in replacement for Silero VAD.

FireRedVAD achieves 97.57% F1 (vs Silero 95.95%), with false alarm rate
2.69% (vs Silero 9.41%). Supports 100+ languages.

Frame size: 400 samples (25ms) at 16kHz, shift 160 samples (10ms).
Our pipeline sends 512 samples (32ms) chunks — we internally resample to
FireRedVAD's frame cadence.
"""
import os
import logging
import numpy as np

log = logging.getLogger("voxlabs.firered_vad")

FIRERED_FRAME_LEN = 400    # 25ms at 16kHz
FIRERED_FRAME_SHIFT = 160  # 10ms at 16kHz


class FireRedVADWrapper:
    """Wraps FireRedStreamVad to match the SileroVAD interface used by ConversationManager."""

    def __init__(self, model_dir: str, threshold: float = 0.5):
        self.threshold = threshold
        self._model_dir = model_dir
        self._stream_vad = None
        self._speech_active = False
        self._silence_count = 0
        self._leftover = np.array([], dtype=np.int16)

    def load(self):
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

        stream_model_dir = os.path.join(self._model_dir, "Stream-VAD")
        if not os.path.isdir(stream_model_dir):
            stream_model_dir = self._model_dir

        config = FireRedStreamVadConfig(
            use_gpu=False,
            speech_threshold=self.threshold,
            smooth_window_size=5,
            pad_start_frame=5,
            min_speech_frame=8,
            min_silence_frame=15,
        )
        self._stream_vad = FireRedStreamVad.from_pretrained(stream_model_dir, config)
        log.info("FireRedVAD loaded (model_dir=%s, threshold=%.2f)", stream_model_dir, self.threshold)
        return self

    def reset(self):
        if self._stream_vad is not None:
            self._stream_vad.reset()
        self._speech_active = False
        self._silence_count = 0
        self._leftover = np.array([], dtype=np.int16)

    def process_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        """Process a chunk of audio (512 samples = 32ms at 16kHz).
        Internally feeds FireRedVAD frame-by-frame (10ms shift) and returns
        aggregated result compatible with SileroVAD interface.
        """
        int16 = (np.clip(audio_chunk, -1, 1) * 32767).astype(np.int16) if audio_chunk.dtype != np.int16 else audio_chunk

        samples = np.concatenate([self._leftover, int16])

        max_prob = 0.0
        any_speech_start = False
        any_speech_end = False

        any_is_speech = False
        pos = 0
        while pos + FIRERED_FRAME_LEN <= len(samples):
            frame = samples[pos:pos + FIRERED_FRAME_LEN]
            result = self._stream_vad.detect_frame(frame)

            max_prob = max(max_prob, result.smoothed_prob)
            if result.is_speech:
                any_is_speech = True
            if result.is_speech_start:
                any_speech_start = True
            if result.is_speech_end:
                any_speech_end = True

            pos += FIRERED_FRAME_SHIFT

        self._leftover = samples[pos:]

        if any_is_speech or any_speech_start:
            effective_prob = max(max_prob, 0.8)
        elif any_speech_end:
            effective_prob = 0.1
        else:
            effective_prob = min(max_prob, 0.3)

        event = {
            "speech_prob": effective_prob,
            "speech_start": any_speech_start,
            "speech_end": any_speech_end,
        }

        if any_speech_start and not self._speech_active:
            self._speech_active = True
            self._silence_count = 0
        if any_speech_end and self._speech_active:
            self._speech_active = False
            self._silence_count = 0

        if not any_speech_start and not any_speech_end:
            if not any_is_speech and self._speech_active:
                self._silence_count += 1
            elif any_is_speech:
                self._silence_count = 0

        return event

    @property
    def is_speech_active(self):
        return self._speech_active
