"""Production Conversation Manager v2.2 — speculative inference + paralinguistic awareness.

Changes from v2.1:
  - Speculative ASR: starts after 150ms silence, before endpointing confirmed
  - Paralinguistic captioner: optional audio scene description injected into LLM prompt
  - All v2.1 features preserved (dual-layer barge-in, adaptive endpointing, THINKING buffer)

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

BARGE_IN_VAD_THRESHOLD = 0.6
BARGE_IN_RMS_THRESHOLD = 0.008
BARGE_IN_CONFIRM_CHUNKS = 2       # ~64ms consecutive confirmation

ENDPOINT_FAST_CHUNKS = 6          # ~192ms for short utterances (<0.5s)
ENDPOINT_DEFAULT_CHUNKS = 10      # ~320ms for normal utterances
ENDPOINT_SLOW_CHUNKS = 20         # ~640ms for long utterances (>3s)

SPECULATIVE_ASR_CHUNKS = 5        # ~160ms silence → start speculative ASR

TTS_SEND_CHUNK_BYTES = 8820 * 2   # ~200ms at 44.1kHz 16-bit (larger chunks for network stability)


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2)))


class ConversationManager:
    def __init__(self, asr, llm, tts, rag, vad, filler, send_fn: Callable[..., Awaitable],
                 captioner=None, spec_asr=None, turn_detector=None, denoiser=None):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.rag = rag
        self.vad = vad
        self.filler = filler
        self.captioner = captioner
        self.spec_asr = spec_asr
        self.turn_detector = turn_detector
        self.denoiser = denoiser
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

        self._spec_asr_task: Optional[asyncio.Task] = None
        self._spec_asr_result: Optional[dict] = None
        self._spec_asr_audio_len = 0

    def reset(self):
        self.state = State.IDLE
        self._audio_buffer = []
        self._interrupted_texts = []
        self._silence_count = 0
        self._turn = 0
        self._barge_confirm_count = 0
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False
        self._cancel_speculative_asr()
        self.vad.reset()
        self.llm.reset()
        if self._speaking_task and not self._speaking_task.done():
            self._speaking_task.cancel()

    def _cancel_speculative_asr(self):
        if self._spec_asr_task and not self._spec_asr_task.done():
            self._spec_asr_task.cancel()
        self._spec_asr_task = None
        self._spec_asr_result = None
        self._spec_asr_audio_len = 0

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
                self._cancel_speculative_asr()
                await self._send({"type": "state", "state": "listening"})

        elif self.state == State.LISTENING:
            if is_speech:
                self._audio_buffer.append(chunk)
                self._silence_count = 0
                if self._spec_asr_task:
                    self._cancel_speculative_asr()
                if self.turn_detector:
                    self.turn_detector.reset()
            else:
                self._audio_buffer.append(chunk)
                self._silence_count += 1

                if (self._silence_count == SPECULATIVE_ASR_CHUNKS
                        and self._spec_asr_task is None
                        and len(self._audio_buffer) > CHUNK_SAMPLES):
                    self._launch_speculative_asr()

                if self.turn_detector and self._silence_count >= SPECULATIVE_ASR_CHUNKS:
                    if self._check_smart_turn():
                        await self._on_endpointing()
                    elif self._silence_count >= ENDPOINT_SLOW_CHUNKS:
                        await self._on_endpointing()
                else:
                    threshold = self._adaptive_endpoint_threshold()
                    if self._silence_count >= threshold:
                        await self._on_endpointing()

        elif self.state == State.SPEAKING:
            if not hasattr(self, '_speak_log_count'):
                self._speak_log_count = 0
            self._speak_log_count += 1
            if self._speak_log_count <= 5 or (speech_prob > 0.3 and self._speak_log_count % 10 == 0):
                log.info("[SPEAKING] chunk %d: vad=%.2f rms=%.4f threshold=%.2f/%.4f",
                         self._speak_log_count, speech_prob, rms,
                         BARGE_IN_VAD_THRESHOLD, BARGE_IN_RMS_THRESHOLD)

            has_speaker_vad = vad_result.get("is_target_speaker") is not None
            if has_speaker_vad:
                is_real_speech = (
                    vad_result.get("is_target_speaker", False)
                    and rms > BARGE_IN_RMS_THRESHOLD
                )
            else:
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

    def _launch_speculative_asr(self):
        """Fire-and-forget ASR on current buffer snapshot. Result cached for _run_pipeline.
        Uses Moonshine (27M, CPU, ~20ms) if available, otherwise falls back to main ASR.
        """
        audio_snapshot = np.concatenate(self._audio_buffer) if self._audio_buffer else None
        if audio_snapshot is None or len(audio_snapshot) < CHUNK_SAMPLES * 3:
            return
        self._spec_asr_audio_len = len(audio_snapshot)
        asr_engine = self.spec_asr if self.spec_asr else self.asr

        async def _run():
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, asr_engine.transcribe, audio_snapshot)
                self._spec_asr_result = result
            except Exception:
                self._spec_asr_result = None

        self._spec_asr_task = asyncio.create_task(_run())
        engine_name = "Moonshine" if self.spec_asr else "SenseVoice"
        log.debug("Speculative ASR (%s) launched on %d samples", engine_name, len(audio_snapshot))

    def _check_smart_turn(self) -> bool:
        """Ask the Smart Turn detector if the user is done speaking."""
        if not self.turn_detector or len(self._audio_buffer) < 3:
            return False
        try:
            return self.turn_detector.should_endpoint(self._audio_buffer, is_speech_active=False)
        except Exception:
            return False

    def _adaptive_endpoint_threshold(self) -> int:
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

        spec_result = self._spec_asr_result
        spec_audio_len = self._spec_asr_audio_len
        spec_task = self._spec_asr_task
        self._spec_asr_result = None
        self._spec_asr_task = None
        self._spec_asr_audio_len = 0

        asyncio.create_task(self._run_pipeline(audio, self._turn, spec_result, spec_audio_len, spec_task))

    async def _run_pipeline(self, audio: np.ndarray, turn: int,
                            spec_result: Optional[dict], spec_audio_len: int,
                            spec_task: Optional[asyncio.Task]):
        """Full pipeline: denoise → (speculative) ASR → paralinguistic → RAG → LLM(streaming) → TTS → play."""
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        # 0. Denoise (if available)
        if self.denoiser and len(audio) > CHUNK_SAMPLES * 3:
            audio = await loop.run_in_executor(None, self.denoiser.process, audio)
            log.info("[Turn %d] Denoised %d samples", turn, len(audio))

        # 1. ASR — use speculative result if audio hasn't grown much
        audio_grew = len(audio) > spec_audio_len * 1.15 if spec_audio_len > 0 else True

        if spec_task and not spec_task.done():
            try:
                await asyncio.wait_for(spec_task, timeout=0.2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                spec_result = None

        if spec_result and not audio_grew:
            user_text = _ASR_TAG_RE.sub("", spec_result["text"]).strip()
            asr_ms = spec_result["latency_ms"]
            log.info("[Turn %d] Used speculative ASR (saved ~%dms)", turn, round(asr_ms))
            asr_ms = 0.0
        else:
            asr_result = await loop.run_in_executor(None, self.asr.transcribe, audio)
            user_text = _ASR_TAG_RE.sub("", asr_result["text"]).strip()
            asr_ms = asr_result["latency_ms"]

        # 1b. Auto-enroll speaker on first turn (for SpeakerAwareVAD)
        if turn == 1 and hasattr(self.vad, 'enroll_speaker') and not self.vad.is_enrolled:
            if len(audio) >= 8000:
                self.vad.enroll_speaker(audio)
                log.info("[Turn %d] Speaker auto-enrolled from first utterance", turn)

        # 1c. Merge late speech from THINKING state
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

        # 2. Interrupted text aggregation
        if self._interrupted_texts:
            combined = "；".join(self._interrupted_texts + [user_text])
            user_text = f"用户依次说了：{combined}。请自然地综合回应。"
            self._interrupted_texts = []
        else:
            if asr_ms > 0:
                self._interrupted_texts = []

        # 3. Paralinguistic captioner (optional, async, non-blocking with timeout)
        caption = ""
        caption_ms = 0.0
        if self.captioner and len(audio) >= SAMPLE_RATE:
            try:
                t_cap = time.perf_counter()
                caption = await asyncio.wait_for(
                    loop.run_in_executor(None, self.captioner.describe, audio),
                    timeout=0.5
                )
                caption_ms = (time.perf_counter() - t_cap) * 1000
                log.info("[Turn %d] Captioner: '%s' (%.0fms)", turn, caption, caption_ms)
                await self._send({
                    "type": "caption", "text": caption,
                    "latency_ms": round(caption_ms, 1), "turn": turn,
                })
            except (asyncio.TimeoutError, Exception) as e:
                log.warning("[Turn %d] Captioner failed/timeout: %s", turn, e)
                caption = ""

        # 4. RAG
        rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
        rag_context = rag_result["context"]
        rag_ms = rag_result["total_ms"]
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": turn,
        })

        # 5. Inject caption into user text for LLM
        llm_user_text = user_text
        if caption:
            llm_user_text = f"[语音观察：{caption}]\n用户说：{user_text}"

        # 6. LLM streaming → TTS per sentence → play
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = asyncio.create_task(
            self._stream_llm_tts(llm_user_text, rag_context, turn, t_start, asr_ms, rag_ms)
        )

    async def _stream_llm_tts(self, user_text, rag_context, turn, t_start, asr_ms, rag_ms):
        """Streaming: LLM sentences → TTS per-chunk streaming → send each chunk with cancel check."""
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

        def _tts_stream_sentence(text):
            """Generator: yields TTS chunks one at a time."""
            chunks = []
            for chunk_data in self.tts.synthesize_stream(text):
                chunks.append(chunk_data)
            return chunks

        try:
            loop.run_in_executor(None, _produce_sentences)
            first_ttfa = None

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

                tts_chunks = await loop.run_in_executor(
                    None, _tts_stream_sentence, sentence
                )

                for i, chunk_data in enumerate(tts_chunks):
                    if self._cancel_speaking.is_set():
                        log.info("[Turn %d] Cancelled mid-TTS chunk %d/%d", turn, i, len(tts_chunks))
                        break

                    if sentence_count == 1 and i == 0:
                        first_ttfa = chunk_data.get("ttfa_ms", 0)
                        await self._send({"type": "audio_start", "turn": turn})
                        first_response_ms = asr_ms + rag_ms + s["ttfs_ms"] + first_ttfa
                        await self._send({
                            "type": "metrics",
                            "asr_ms": round(asr_ms, 1),
                            "rag_ms": round(rag_ms, 1),
                            "llm_ms": round(s["ttfs_ms"], 1),
                            "tts_ttfa_ms": round(first_ttfa, 1),
                            "first_response_ms": round(first_response_ms, 1),
                            "turn": turn,
                        })

                    audio = chunk_data["audio"]
                    if audio.dtype != np.int16:
                        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                    await self._send({"_audio": audio.tobytes()})
                    await asyncio.sleep(0.01)

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
        """Send audio in chunks, checking cancel between each.
        Uses larger chunks (200ms) for network stability over Cloudflare tunnels,
        with periodic yield to keep the event loop responsive for keepalive.
        """
        for i in range(0, len(audio_bytes), TTS_SEND_CHUNK_BYTES):
            if self._cancel_speaking.is_set():
                return
            await self._send({"_audio": audio_bytes[i:i + TTS_SEND_CHUNK_BYTES]})
            await asyncio.sleep(0)

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
