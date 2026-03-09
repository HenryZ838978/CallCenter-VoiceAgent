"""Semantic Turn Detector — combines Pipecat Smart Turn v3 (audio-native, 8M ONNX)
with optional Moonshine streaming partial ASR for richer context.

Smart Turn v3 analyzes raw audio waveforms for prosodic cues (intonation fall = done,
rise = not done). When combined with streaming partial ASR, the system also checks
semantic completeness.

Usage in ConversationManager:
    After VAD silence >= 160ms, call turn_detector.should_endpoint(audio_buffer)
    → True (user done) / False (user still thinking, wait longer)
"""
import logging
import time
import numpy as np

log = logging.getLogger("voxlabs.turn_detector")

SAMPLE_RATE = 16000


class SmartTurnDetector:
    """Audio-native turn detector using Pipecat Smart Turn v3 ONNX model."""

    def __init__(self):
        self._analyzer = None

    def load(self):
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        self._analyzer = LocalSmartTurnAnalyzerV3()
        self._analyzer.set_sample_rate(SAMPLE_RATE)
        log.info("SmartTurnDetector loaded (Pipecat Smart Turn v3, 8M ONNX)")
        return self

    def reset(self):
        if self._analyzer:
            self._analyzer.clear()

    def should_endpoint(self, audio_buffer: list, is_speech_active: bool) -> bool:
        """Feed buffered audio to Smart Turn and check if user is done speaking.

        Args:
            audio_buffer: list of float32 numpy chunks (512 samples each)
            is_speech_active: whether VAD currently detects speech

        Returns:
            True if the model thinks the user is done speaking
        """
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

        self._analyzer.clear()

        for chunk in audio_buffer:
            int16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            state = self._analyzer.append_audio(int16.tobytes(), is_speech=True)

        silence = np.zeros(512, dtype=np.int16)
        state = self._analyzer.append_audio(silence.tobytes(), is_speech=False)

        return state == EndOfTurnState.COMPLETE

    def get_probability(self, audio_buffer: list) -> float:
        """Return a soft probability of turn completion (0.0 = not done, 1.0 = done).
        Uses the Smart Turn model's internal scoring.
        """
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

        self._analyzer.clear()

        for chunk in audio_buffer:
            int16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            state = self._analyzer.append_audio(int16.tobytes(), is_speech=True)

        silence_chunks = 5
        complete_count = 0
        for _ in range(silence_chunks):
            silence = np.zeros(512, dtype=np.int16)
            state = self._analyzer.append_audio(silence.tobytes(), is_speech=False)
            if state == EndOfTurnState.COMPLETE:
                complete_count += 1

        return complete_count / silence_chunks
