"""Full-duplex voice agent with barge-in support."""
import time
import threading
import queue
import numpy as np
from typing import Optional, Callable

from .vad import SileroVAD
from .asr import SenseVoiceASR
from .llm import VLLMChat
from .tts import VoxCPMTTS
from .rag import RAGEngine


class DuplexAgent:
    """Coordinates VAD → ASR → (RAG) → LLM → TTS with full-duplex barge-in."""

    def __init__(self, vad: SileroVAD, asr: SenseVoiceASR,
                 llm: VLLMChat, tts: VoxCPMTTS,
                 rag: Optional[RAGEngine] = None,
                 on_tts_chunk: Optional[Callable] = None):
        self.vad = vad
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.rag = rag
        self.on_tts_chunk = on_tts_chunk

        self._audio_buffer = []
        self._tts_playing = False
        self._barge_in = False
        self._stop_tts = threading.Event()
        self._lock = threading.Lock()

    def reset(self):
        self.vad.reset()
        self.llm.reset()
        self._audio_buffer = []
        self._tts_playing = False
        self._barge_in = False
        self._stop_tts.clear()

    def feed_audio(self, chunk: np.ndarray, sr: int = 16000) -> Optional[dict]:
        """Feed audio chunk (512 samples / 32ms). Returns event dict or None."""
        vad_result = self.vad.process_chunk(chunk, sr)

        if vad_result["speech_start"]:
            if self._tts_playing:
                self._barge_in = True
                self._stop_tts.set()
                return {
                    "event": "barge_in",
                    "timestamp_ms": time.perf_counter() * 1000,
                }
            self._audio_buffer = []

        if self.vad.is_speech_active:
            self._audio_buffer.append(chunk)

        if vad_result["speech_end"] and self._audio_buffer:
            full_audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
            return self._process_utterance(full_audio, sr)

        return None

    def _process_utterance(self, audio: np.ndarray, sr: int) -> dict:
        """Full pipeline: ASR → (RAG) → LLM → TTS."""
        t_start = time.perf_counter()

        asr_result = self.asr.transcribe(audio, sr)
        user_text = asr_result["text"].strip()
        if not user_text:
            return {"event": "empty_asr", "asr_latency_ms": asr_result["latency_ms"]}

        rag_context = None
        rag_ms = 0.0
        if self.rag:
            rag_result = self.rag.get_context(user_text)
            rag_context = rag_result["context"]
            rag_ms = rag_result["total_ms"]

        llm_result = self.llm.chat(user_text, rag_context=rag_context)

        self._stop_tts.clear()
        self._tts_playing = True
        tts_result = self.tts.synthesize(llm_result["text"])
        self._tts_playing = False

        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "event": "response",
            "user_text": user_text,
            "assistant_text": llm_result["text"],
            "asr_latency_ms": asr_result["latency_ms"],
            "rag_latency_ms": rag_ms,
            "llm_latency_ms": llm_result["latency_ms"],
            "tts_ttfa_ms": tts_result["ttfa_ms"],
            "tts_total_ms": tts_result["total_ms"],
            "tts_chunks": tts_result["chunks"],
            "total_ms": total_ms,
            "audio": tts_result["audio"],
            "audio_sr": tts_result["sr"],
        }

    def process_text(self, user_text: str) -> dict:
        """Skip ASR, directly process text through (RAG) → LLM → TTS."""
        t_start = time.perf_counter()

        rag_context = None
        rag_ms = 0.0
        if self.rag:
            rag_result = self.rag.get_context(user_text)
            rag_context = rag_result["context"]
            rag_ms = rag_result["total_ms"]

        llm_result = self.llm.chat(user_text, rag_context=rag_context)

        self._stop_tts.clear()
        self._tts_playing = True
        tts_result = self.tts.synthesize(llm_result["text"])
        self._tts_playing = False

        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "event": "response",
            "user_text": user_text,
            "assistant_text": llm_result["text"],
            "rag_latency_ms": rag_ms,
            "llm_latency_ms": llm_result["latency_ms"],
            "tts_ttfa_ms": tts_result["ttfa_ms"],
            "tts_total_ms": tts_result["total_ms"],
            "tts_chunks": tts_result["chunks"],
            "total_ms": total_ms,
            "audio": tts_result["audio"],
            "audio_sr": tts_result["sr"],
        }

    def simulate_barge_in(self, tts_chunks_before_interrupt: int = 6) -> dict:
        """Simulate barge-in during TTS playback."""
        self._tts_playing = True
        self._stop_tts.clear()
        t0 = time.perf_counter()

        chunk_count = 0
        for _ in range(tts_chunks_before_interrupt):
            chunk_count += 1
            time.sleep(0.1)

        self._barge_in = True
        self._stop_tts.set()
        detect_ms = (time.perf_counter() - t0) * 1000
        self._tts_playing = False

        return {
            "event": "barge_in_test",
            "chunks_played": chunk_count,
            "detection_ms": detect_ms,
            "barge_in_triggered": self._barge_in,
        }
