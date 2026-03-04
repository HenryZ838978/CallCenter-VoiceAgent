"""Production Conversation Manager v2.1 — enhanced full-duplex with robust barge-in.

Changes from v2.0:
  - Barge-in: high VAD threshold (0.85) + RMS energy gate + 3-chunk confirmation window
  - THINKING state buffers user speech instead of discarding
  - Adaptive endpointing: short utterances → fast, long utterances → patient
  - TTS cancel granularity: 200ms → 50ms chunks
  - Frontend now sends audio always (echo suppression via server-side filtering)

States:
  IDLE       → waiting for user
  LISTENING  → user is speaking, buffering audio
  THINKING   → user done, running pipeline (buffers late speech)
  SPEAKING   → streaming TTS playback
  INTERRUPTED→ user spoke during AI speech, collecting new input
"""
import re
import time
import asyncio
import logging
import numpy as np
from enum import Enum
from typing import Optional, Callable, Awaitable

log = logging.getLogger("voxlabs.cm")

_ASR_TAG_RE = re.compile(r"<\|[^>]*\|>")


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512               # 32ms
CHUNK_MS = 32

INTERRUPT_SILENCE_CHUNKS = 10     # ~320ms before processing interrupted speech

BARGE_IN_VAD_THRESHOLD = 0.85
BARGE_IN_RMS_THRESHOLD = 0.015
BARGE_IN_CONFIRM_CHUNKS = 3       # ~96ms consecutive confirmation

ENDPOINT_FAST_CHUNKS = 6          # ~192ms for short utterances (<0.5s)
ENDPOINT_DEFAULT_CHUNKS = 10      # ~320ms for normal utterances
ENDPOINT_SLOW_CHUNKS = 20         # ~640ms for long utterances (>3s)

TTS_SEND_CHUNK_BYTES = 2205 * 2   # ~50ms at 44.1kHz 16-bit


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2)))


class ConversationManager:
    def __init__(self, asr, llm, tts, rag, vad, filler, send_fn: Callable[..., Awaitable]):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.rag = rag
        self.vad = vad
        self.filler = filler
        self._send = send_fn

        self.state = State.IDLE
        self._audio_buffer = []
        self._interrupted_texts = []
        self._silence_count = 0
        self._turn = 0
        self._speaking_task: Optional[asyncio.Task] = None
        self._cancel_speaking = asyncio.Event()

        self._barge_confirm_count = 0
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False

    def reset(self):
        self.state = State.IDLE
        self._audio_buffer = []
        self._interrupted_texts = []
        self._silence_count = 0
        self._turn = 0
        self._barge_confirm_count = 0
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False
        self.vad.reset()
        self.llm.reset()
        if self._speaking_task and not self._speaking_task.done():
            self._speaking_task.cancel()

    async def feed_audio(self, samples: np.ndarray):
        """Feed raw PCM samples (float32, 16kHz). Called per WebSocket message."""
        for i in range(0, len(samples), CHUNK_SAMPLES):
            chunk = samples[i:i + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            await self._process_chunk(chunk)

    async def _process_chunk(self, chunk: np.ndarray):
        vad_result = self.vad.process_chunk(chunk, SAMPLE_RATE)
        speech_prob = vad_result["speech_prob"]
        is_speech = speech_prob >= 0.5
        rms = _rms(chunk)

        if self.state == State.IDLE:
            if is_speech and rms > 0.01:
                self.state = State.LISTENING
                self._audio_buffer = [chunk]
                self._silence_count = 0
                await self._send({"type": "state", "state": "listening"})

        elif self.state == State.LISTENING:
            if is_speech:
                self._audio_buffer.append(chunk)
                self._silence_count = 0
            else:
                self._audio_buffer.append(chunk)
                self._silence_count += 1
                threshold = self._adaptive_endpoint_threshold()
                if self._silence_count >= threshold:
                    await self._on_endpointing()

        elif self.state == State.SPEAKING:
            is_real_speech = (
                speech_prob >= BARGE_IN_VAD_THRESHOLD
                and rms > BARGE_IN_RMS_THRESHOLD
            )
            if is_real_speech:
                self._barge_confirm_count += 1
                if self._barge_confirm_count >= BARGE_IN_CONFIRM_CHUNKS:
                    if not self._cancel_speaking.is_set():
                        self._cancel_speaking.set()
                        self.state = State.INTERRUPTED
                        self._audio_buffer = [chunk]
                        self._barge_confirm_count = 0
                        await self._send({"type": "barge_in", "turn": self._turn})
                        await self._send({"type": "state", "state": "interrupted"})
                    else:
                        self._audio_buffer.append(chunk)
            else:
                self._barge_confirm_count = 0

        elif self.state == State.INTERRUPTED:
            if is_speech:
                self._audio_buffer.append(chunk)
                self._silence_count = 0
            else:
                self._audio_buffer.append(chunk)
                self._silence_count += 1
                if self._silence_count >= INTERRUPT_SILENCE_CHUNKS:
                    await self._on_endpointing()

        elif self.state == State.THINKING:
            if is_speech and rms > 0.01:
                self._thinking_extra_buffer.append(chunk)
                self._thinking_has_speech = True
            elif self._thinking_has_speech:
                self._thinking_extra_buffer.append(chunk)

    def _adaptive_endpoint_threshold(self) -> int:
        """Shorter utterances get faster endpointing, longer ones get more patience."""
        audio_duration_ms = len(self._audio_buffer) * CHUNK_MS
        if audio_duration_ms < 500:
            return ENDPOINT_FAST_CHUNKS
        elif audio_duration_ms > 3000:
            return ENDPOINT_SLOW_CHUNKS
        else:
            return ENDPOINT_DEFAULT_CHUNKS

    async def _on_endpointing(self):
        """User finished speaking — start pipeline."""
        self.state = State.THINKING
        self._silence_count = 0
        self._barge_confirm_count = 0
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False
        self._turn += 1
        await self._send({"type": "state", "state": "thinking"})

        audio = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([], dtype=np.float32)
        self._audio_buffer = []

        asyncio.create_task(self._run_pipeline(audio, self._turn))

    async def _run_pipeline(self, audio: np.ndarray, turn: int):
        """Full pipeline: ASR → (merge late speech) → RAG → LLM(streaming) → TTS(per-sentence) → play."""
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        # 1. ASR
        asr_result = await loop.run_in_executor(None, self.asr.transcribe, audio)
        user_text = _ASR_TAG_RE.sub("", asr_result["text"]).strip()
        asr_ms = asr_result["latency_ms"]

        # 1b. Check if user spoke more during ASR processing
        if self._thinking_has_speech and self._thinking_extra_buffer:
            extra_audio = np.concatenate(self._thinking_extra_buffer)
            self._thinking_extra_buffer = []
            self._thinking_has_speech = False
            if len(extra_audio) > CHUNK_SAMPLES * 3:
                extra_result = await loop.run_in_executor(None, self.asr.transcribe, extra_audio)
                extra_text = _ASR_TAG_RE.sub("", extra_result["text"]).strip()
                if extra_text and len(extra_text) >= 2:
                    user_text = f"{user_text}，{extra_text}" if user_text else extra_text
                    asr_ms += extra_result["latency_ms"]
                    log.info("[Turn %d] Merged late speech: '%s'", turn, extra_text)

        await self._send({"type": "asr", "text": user_text, "latency_ms": round(asr_ms, 1), "turn": turn})

        if not user_text or len(user_text) < 2:
            self.state = State.IDLE
            await self._send({"type": "state", "state": "idle"})
            return

        # 2. Check for interrupted texts to aggregate
        if self._interrupted_texts:
            combined = "；".join(self._interrupted_texts + [user_text])
            user_text = f"用户依次说了：{combined}。请自然地综合回应。"
            self._interrupted_texts = []
        else:
            if asr_ms > 0:
                self._interrupted_texts = []

        # 3. RAG
        rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
        rag_context = rag_result["context"]
        rag_ms = rag_result["total_ms"]
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": turn,
        })

        # 4. LLM streaming → TTS per sentence → play
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = asyncio.create_task(
            self._stream_llm_tts(user_text, rag_context, turn, t_start, asr_ms, rag_ms)
        )

    async def _stream_llm_tts(self, user_text, rag_context, turn, t_start, asr_ms, rag_ms):
        """True streaming: LLM yields sentences via queue → TTS per sentence → play."""
        loop = asyncio.get_event_loop()
        sentence_queue = asyncio.Queue()
        full_response = ""
        sentence_count = 0

        def _produce_sentences():
            for s in self.llm.stream_sentences(user_text, rag_context=rag_context):
                if self._cancel_speaking.is_set():
                    break
                asyncio.run_coroutine_threadsafe(sentence_queue.put(s), loop)
            asyncio.run_coroutine_threadsafe(sentence_queue.put(None), loop)

        try:
            loop.run_in_executor(None, _produce_sentences)

            while True:
                if self._cancel_speaking.is_set():
                    log.info("[Turn %d] Cancelled before next sentence", turn)
                    break

                s = await sentence_queue.get()
                if s is None:
                    break

                if self._cancel_speaking.is_set():
                    break

                sentence = s["sentence"]
                full_response += sentence
                sentence_count += 1

                await self._send({
                    "type": "llm_sentence", "text": sentence,
                    "latency_ms": round(s["ttfs_ms"], 1) if sentence_count == 1 else 0,
                    "sentence_idx": sentence_count - 1, "turn": turn,
                })

                if self._cancel_speaking.is_set():
                    break

                tts_result = await loop.run_in_executor(
                    None, lambda sent=sentence: self.tts.synthesize(sent)
                )

                if sentence_count == 1:
                    first_response_ms = asr_ms + rag_ms + s["ttfs_ms"] + tts_result["ttfa_ms"]
                    await self._send({
                        "type": "metrics",
                        "asr_ms": round(asr_ms, 1),
                        "rag_ms": round(rag_ms, 1),
                        "llm_ms": round(s["ttfs_ms"], 1),
                        "tts_ttfa_ms": round(tts_result["ttfa_ms"], 1),
                        "first_response_ms": round(first_response_ms, 1),
                        "turn": turn,
                    })

                if self._cancel_speaking.is_set():
                    break

                audio_data = tts_result["audio"]
                if audio_data.dtype != np.int16:
                    audio_data = (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
                await self._send_audio(audio_data.tobytes())

            await self._send({
                "type": "llm", "text": full_response,
                "turn": turn, "sentences": sentence_count,
            })

        except asyncio.CancelledError:
            log.info("[Turn %d] Pipeline cancelled", turn)
        except Exception as e:
            log.error("[Turn %d] Pipeline error: %s", turn, e, exc_info=True)
            await self._send({"type": "error", "message": str(e), "turn": turn})
        finally:
            if self.state == State.SPEAKING:
                self.state = State.IDLE
                await self._send({"type": "state", "state": "idle"})

    async def _send_audio(self, audio_bytes: bytes):
        """Send audio in small chunks, checking cancel between each."""
        for i in range(0, len(audio_bytes), TTS_SEND_CHUNK_BYTES):
            if self._cancel_speaking.is_set():
                return
            await self._send({"_audio": audio_bytes[i:i + TTS_SEND_CHUNK_BYTES]})
            await asyncio.sleep(0.002)

    async def handle_text_input(self, text: str):
        """Handle text input (skip ASR)."""
        self._turn += 1
        self.state = State.THINKING
        await self._send({"type": "state", "state": "thinking"})
        await self._send({"type": "asr", "text": text, "latency_ms": 0, "turn": self._turn, "source": "text"})

        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        rag_result = await loop.run_in_executor(None, self.rag.get_context, text)
        await self._send({
            "type": "rag", "context": rag_result["context"],
            "latency_ms": round(rag_result["total_ms"], 1), "turn": self._turn,
        })

        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = asyncio.create_task(
            self._stream_llm_tts(
                text, rag_result["context"], self._turn,
                t_start, 0, rag_result["total_ms"]
            )
        )
