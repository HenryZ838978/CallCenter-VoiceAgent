"""Production Conversation Manager v3.0 — iPad demo hardening.

v3.0 (over v2.9):
  - LLM: MiniCPM4.1-8B-GPTQ → Qwen3-14B-AWQ (no thinking leakage)
  - Voice clone: lipsync_real_clone_sample_5s.wav (higher fidelity)
  - VAD: IDLE/barge-in thresholds raised (0.6/0.02) for iPad AEC environment
  - Endpointing: short audio (<0.5s) uses SLOW_CHUNKS to avoid premature cuts
  - ASR empty: silent return to LISTENING instead of "没听清" (first attempts)
  - Filler gating: only send filler for utterances >1s
  - Filler engine: MAX_FILLER_SEC cap, removed problematic words
  - TTS safety: MAX_TTS_AUDIO_BYTES (30s) hard cap on all outgoing audio
  - Manual barge-in: client-triggered via WebSocket

v2.9 features preserved (state-race fix, crash-safe, metrics, D1-D4 experience).
v2.2 features preserved (speculative ASR, captioner, dual-layer barge-in,
adaptive endpointing, THINKING buffer).

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

ENDPOINT_FAST_CHUNKS = 8          # ~256ms for short utterances (<0.5s)
ENDPOINT_DEFAULT_CHUNKS = 13      # ~416ms for normal utterances
ENDPOINT_SLOW_CHUNKS = 20         # ~640ms for long utterances (>3s)

SPECULATIVE_ASR_CHUNKS = 5        # ~160ms silence → start speculative ASR

TTS_SEND_CHUNK_BYTES = 8820 * 2   # ~200ms at 44.1kHz 16-bit (larger chunks for network stability)

IDLE_CHECK_INTERVAL_CHUNKS = 31   # ~1s — poll idle timeout at this cadence


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2)))


class _SessionMetrics:
    """Lightweight in-memory metrics for one session."""
    __slots__ = ("turns", "errors", "asr_ms_list", "llm_ms_list", "tts_ms_list", "fr_ms_list")

    def __init__(self):
        self.turns = 0
        self.errors = 0
        self.asr_ms_list: list[float] = []
        self.llm_ms_list: list[float] = []
        self.tts_ms_list: list[float] = []
        self.fr_ms_list: list[float] = []

    def record_turn(self, asr_ms=0.0, llm_ms=0.0, tts_ms=0.0, fr_ms=0.0):
        self.turns += 1
        if asr_ms: self.asr_ms_list.append(asr_ms)
        if llm_ms: self.llm_ms_list.append(llm_ms)
        if tts_ms: self.tts_ms_list.append(tts_ms)
        if fr_ms: self.fr_ms_list.append(fr_ms)

    def record_error(self):
        self.errors += 1

    def summary(self) -> dict:
        def _avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0
        return {
            "turns": self.turns,
            "errors": self.errors,
            "avg_asr_ms": _avg(self.asr_ms_list),
            "avg_llm_ms": _avg(self.llm_ms_list),
            "avg_tts_ms": _avg(self.tts_ms_list),
            "avg_first_response_ms": _avg(self.fr_ms_list),
        }


class ConversationManager:
    def __init__(self, asr, llm, tts, rag, vad, filler, send_fn: Callable[..., Awaitable],
                 captioner=None, spec_asr=None, turn_detector=None, denoiser=None,
                 experience_config: dict = None):
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
        self._dead = False

        self._barge_confirm_count = 0
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False

        self._spec_asr_task: Optional[asyncio.Task] = None
        self._spec_asr_result: Optional[dict] = None
        self._spec_asr_audio_len = 0

        self._tasks: set[asyncio.Task] = set()

        self.metrics = _SessionMetrics()

        self._error_apology = "抱歉系统开了个小差，您能再说一次吗？"

        # --- D1-D4 experience layer ---
        cfg = experience_config or {}
        self._greeting_text = cfg.get("greeting_text", "")
        self._idle_timeout_s = cfg.get("idle_timeout_s", 15.0)
        self._idle_goodbye_s = cfg.get("idle_goodbye_s", 30.0)
        self._idle_prompt_text = cfg.get("idle_prompt_text", "您还在吗？有什么需要帮助的吗？")
        self._idle_goodbye_text = cfg.get("idle_goodbye_text", "好的，如果后续有问题随时找我，再见！")
        self._asr_empty_text = cfg.get("asr_empty_retry_text", "抱歉没太听清，您能再说一次吗？")
        self._asr_noisy_text = cfg.get("asr_noisy_suggest_text", "您那边环境好像有点嘈杂，要不您试试打字发给我？")
        self._max_empty_asr = cfg.get("max_consecutive_empty_asr", 3)
        self._farewell_keywords = cfg.get("farewell_keywords", ["再见", "拜拜", "挂了"])

        self._idle_since: float = time.monotonic()
        self._idle_chunk_count = 0
        self._idle_prompted = False
        self._idle_goodbye_sent = False
        self._consecutive_empty_asr = 0
        self._session_ending = False

    def _track_task(self, coro, *, name: str = "") -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

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
        self._idle_since = time.monotonic()
        self._idle_chunk_count = 0
        self._idle_prompted = False
        self._idle_goodbye_sent = False
        self._consecutive_empty_asr = 0
        self._session_ending = False

    async def handle_barge_in(self):
        """Manual barge-in triggered by client (spacebar / button)."""
        if self.state == State.SPEAKING and not self._cancel_speaking.is_set():
            self._cancel_speaking.set()
            self.state = State.LISTENING
            self._audio_buffer = []
            self._barge_confirm_count = 0
            self._silence_count = 0
            log.info("[Turn %d] Manual barge-in from client", self._turn)
            await self._send({"type": "barge_in", "turn": self._turn, "source": "manual"})
            await self._send({"type": "state", "state": "listening"})
        elif self.state == State.THINKING:
            self._cancel_speaking.set()
            self.state = State.LISTENING
            log.info("[Turn %d] Manual barge-in (was THINKING)", self._turn)
            await self._send({"type": "barge_in", "turn": self._turn, "source": "manual"})
            await self._send({"type": "state", "state": "listening"})

    def shutdown(self):
        """Cancel all tracked tasks. Call on WebSocket disconnect."""
        self._dead = True
        self._cancel_speaking.set()
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._tasks.clear()
        self.reset()

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
        is_speech = speech_prob >= 0.6
        rms = _rms(chunk)

        if self.state == State.IDLE:
            if is_speech and rms > 0.02:
                self.state = State.LISTENING
                self._audio_buffer = [chunk]
                self._silence_count = 0
                self._idle_prompted = False
                self._idle_goodbye_sent = False
                self._consecutive_empty_asr = 0
                self._cancel_speculative_asr()
                await self._send({"type": "state", "state": "listening"})
            else:
                self._idle_chunk_count += 1
                if (self._idle_chunk_count >= IDLE_CHECK_INTERVAL_CHUNKS
                        and not self._idle_goodbye_sent
                        and not self._session_ending):
                    self._idle_chunk_count = 0
                    await self._check_idle_timeout()

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
            if is_speech and rms > 0.02:
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

        self._spec_asr_task = self._track_task(_run(), name=f"spec-asr-{self._turn}")
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
            return ENDPOINT_SLOW_CHUNKS
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

        audio_duration_ms = len(self._audio_buffer) * CHUNK_MS
        if self.filler and audio_duration_ms > 1000:
            await self._send_filler()

        audio = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([], dtype=np.float32)
        self._audio_buffer = []

        spec_result = self._spec_asr_result
        spec_audio_len = self._spec_asr_audio_len
        spec_task = self._spec_asr_task
        self._spec_asr_result = None
        self._spec_asr_task = None
        self._spec_asr_audio_len = 0

        self._track_task(
            self._run_pipeline(audio, self._turn, spec_result, spec_audio_len, spec_task),
            name=f"pipeline-{self._turn}",
        )

    async def _run_pipeline(self, audio: np.ndarray, turn: int,
                            spec_result: Optional[dict], spec_audio_len: int,
                            spec_task: Optional[asyncio.Task]):
        """Full pipeline with crash safety. Any exception → spoken apology + IDLE."""
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        try:
            await self._run_pipeline_inner(
                audio, turn, spec_result, spec_audio_len, spec_task,
                loop, t_start,
            )
        except asyncio.CancelledError:
            log.info("[Turn %d] Pipeline cancelled (disconnect?)", turn)
        except Exception as e:
            log.error("[Turn %d] Pipeline CRASHED: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            try:
                await self._send({"type": "error", "message": str(e), "turn": turn})
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._play_system_tts(self._error_apology, turn)
            except Exception:
                pass
        finally:
            if self.state == State.THINKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                try:
                    await self._send({"type": "state", "state": "idle"})
                except Exception:
                    pass

    async def _run_pipeline_inner(self, audio, turn, spec_result, spec_audio_len,
                                  spec_task, loop, t_start):
        """Inner pipeline logic — exceptions propagate to _run_pipeline wrapper."""
        # 0. Denoise
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
            try:
                asr_result = await loop.run_in_executor(None, self.asr.transcribe, audio)
                user_text = _ASR_TAG_RE.sub("", asr_result["text"]).strip()
                asr_ms = asr_result["latency_ms"]
            except Exception as e:
                log.error("[Turn %d] ASR failed: %s", turn, e)
                self.metrics.record_error()
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._send({"type": "system_message", "text": self._asr_empty_text, "turn": turn})
                await self._play_system_tts(self._asr_empty_text, turn)
                return

        # 1b. Auto-enroll speaker on first turn
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
                try:
                    extra_result = await loop.run_in_executor(None, self.asr.transcribe, extra_audio)
                    extra_text = _ASR_TAG_RE.sub("", extra_result["text"]).strip()
                    if extra_text and len(extra_text) >= 2:
                        user_text = f"{user_text}，{extra_text}" if user_text else extra_text
                        asr_ms += extra_result["latency_ms"]
                        log.info("[Turn %d] Merged late speech: '%s'", turn, extra_text)
                except Exception as e:
                    log.warning("[Turn %d] Late ASR merge failed: %s", turn, e)

        await self._send({"type": "asr", "text": user_text, "latency_ms": round(asr_ms, 1), "turn": turn})

        # D3: ASR empty → silent return to LISTENING (first attempts), then recovery
        if not user_text or len(user_text) < 2:
            self._consecutive_empty_asr += 1
            if self._consecutive_empty_asr >= self._max_empty_asr:
                recovery = self._asr_noisy_text
                log.info("[Turn %d] ASR empty (#%d), recovery: '%s'", turn, self._consecutive_empty_asr, recovery)
                self._cancel_speaking.clear()
                self._barge_confirm_count = 0
                self._speak_log_count = 0
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._send({"type": "system_message", "text": recovery, "turn": turn})
                await self._play_system_tts(recovery, turn)
            else:
                log.info("[Turn %d] ASR too short ('%s', #%d), silent return to LISTENING",
                         turn, user_text or "", self._consecutive_empty_asr)
                self.state = State.LISTENING
                self._audio_buffer = []
                self._silence_count = 0
                await self._send({"type": "state", "state": "listening"})
            return

        self._consecutive_empty_asr = 0

        # D4: Farewell detection
        if any(kw in user_text for kw in self._farewell_keywords):
            self._session_ending = True
            log.info("[Turn %d] Farewell detected in: '%s'", turn, user_text)

        # 2. Interrupted text aggregation
        if self._interrupted_texts:
            combined = "；".join(self._interrupted_texts + [user_text])
            user_text = f"用户依次说了：{combined}。请自然地综合回应。"
            self._interrupted_texts = []
        else:
            if asr_ms > 0:
                self._interrupted_texts = []

        # 3. Paralinguistic captioner
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

        # 4. RAG — degrade gracefully on failure
        rag_context = ""
        rag_ms = 0.0
        try:
            rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
            rag_context = rag_result["context"]
            rag_ms = rag_result["total_ms"]
        except Exception as e:
            log.warning("[Turn %d] RAG failed (degraded to no-context): %s", turn, e)
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": turn,
        })

        # 5. Inject caption
        llm_user_text = user_text
        if caption:
            llm_user_text = f"[语音观察：{caption}]\n用户说：{user_text}"

        # 6. LLM streaming → TTS
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = self._track_task(
            self._stream_llm_tts(llm_user_text, rag_context, turn, t_start, asr_ms, rag_ms),
            name=f"llm-tts-{turn}",
        )
        await self._speaking_task

    async def _stream_llm_tts(self, user_text, rag_context, turn, t_start, asr_ms, rag_ms):
        """Streaming: LLM sentences → TTS per-chunk → send. Crash-safe."""
        loop = asyncio.get_event_loop()
        sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        full_response = ""
        sentence_count = 0
        first_ttfa = None
        llm_first_ms = 0.0
        was_interrupted = False

        def _produce_sentences():
            try:
                for s in self.llm.stream_sentences(user_text, rag_context=rag_context):
                    if self._cancel_speaking.is_set():
                        break
                    fut = asyncio.run_coroutine_threadsafe(sentence_queue.put(s), loop)
                    fut.result(timeout=30)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    sentence_queue.put({"_error": str(e)}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(sentence_queue.put(None), loop)

        def _tts_stream_sentence(text):
            return list(self.tts.synthesize_stream(text))

        producer_future = loop.run_in_executor(None, _produce_sentences)

        try:
            while True:
                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    log.info("[Turn %d] Cancelled before next sentence", turn)
                    break

                try:
                    s = await asyncio.wait_for(sentence_queue.get(), timeout=35.0)
                except asyncio.TimeoutError:
                    log.error("[Turn %d] LLM sentence queue timeout (35s)", turn)
                    self.metrics.record_error()
                    break

                if s is None:
                    break
                if "_error" in s:
                    raise RuntimeError(f"LLM stream error: {s['_error']}")

                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    break

                sentence = s["sentence"]
                full_response += sentence
                sentence_count += 1
                if sentence_count == 1:
                    llm_first_ms = s["ttfs_ms"]

                await self._send({
                    "type": "llm_sentence", "text": sentence,
                    "latency_ms": round(s["ttfs_ms"], 1) if sentence_count == 1 else 0,
                    "sentence_idx": sentence_count - 1, "turn": turn,
                })

                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    break

                tts_chunks = await loop.run_in_executor(None, _tts_stream_sentence, sentence)

                for i, chunk_data in enumerate(tts_chunks):
                    if self._cancel_speaking.is_set():
                        was_interrupted = True
                        log.info("[Turn %d] Cancelled mid-TTS chunk %d/%d", turn, i, len(tts_chunks))
                        break

                    if sentence_count == 1 and i == 0:
                        first_ttfa = chunk_data.get("ttfa_ms", 0)
                        await self._send({"type": "audio_start", "turn": turn})
                        first_response_ms = asr_ms + rag_ms + llm_first_ms + first_ttfa
                        self.metrics.record_turn(
                            asr_ms=asr_ms, llm_ms=llm_first_ms,
                            tts_ms=first_ttfa, fr_ms=first_response_ms,
                        )
                        await self._send({
                            "type": "metrics",
                            "asr_ms": round(asr_ms, 1),
                            "rag_ms": round(rag_ms, 1),
                            "llm_ms": round(llm_first_ms, 1),
                            "tts_ttfa_ms": round(first_ttfa, 1),
                            "first_response_ms": round(first_response_ms, 1),
                            "turn": turn,
                        })

                    audio = chunk_data["audio"]
                    if audio.dtype != np.int16:
                        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                    raw = audio.tobytes()
                    if len(raw) > self.MAX_TTS_AUDIO_BYTES:
                        log.warning("[Turn %d] TTS chunk oversized (%d bytes) — truncating",
                                    turn, len(raw))
                        raw = raw[:self.MAX_TTS_AUDIO_BYTES]
                    await self._send({"_audio": raw})
                    await asyncio.sleep(0.01)

            await self._send({
                "type": "llm", "text": full_response,
                "turn": turn, "sentences": sentence_count,
            })

        except asyncio.CancelledError:
            was_interrupted = True
            log.info("[Turn %d] Pipeline cancelled", turn)
        except Exception as e:
            log.error("[Turn %d] LLM/TTS error: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            await self._send({"type": "error", "message": str(e), "turn": turn})
        finally:
            try:
                await asyncio.wait_for(asyncio.wrap_future(producer_future), timeout=2.0)
            except Exception:
                pass

            if was_interrupted and full_response:
                self._interrupted_texts.append(full_response)
                log.info("[Turn %d] Saved partial AI response (%d chars) for context",
                         turn, len(full_response))

            if self.state == State.SPEAKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                await self._send({"type": "state", "state": "idle"})
            if self._session_ending:
                self._session_ending = False
                await self._send({"type": "session_end", "reason": "farewell"})

    MAX_TTS_AUDIO_BYTES = 44100 * 2 * 30  # 30s at 44.1kHz 16-bit — hard safety cap

    async def _send_audio(self, audio_bytes: bytes):
        """Send audio in chunks, checking cancel between each."""
        if len(audio_bytes) > self.MAX_TTS_AUDIO_BYTES:
            log.warning("Audio too large (%d bytes, %.1fs) — truncating to 30s",
                        len(audio_bytes), len(audio_bytes) / 44100 / 2)
            audio_bytes = audio_bytes[:self.MAX_TTS_AUDIO_BYTES]
        for i in range(0, len(audio_bytes), TTS_SEND_CHUNK_BYTES):
            if self._cancel_speaking.is_set():
                return
            await self._send({"_audio": audio_bytes[i:i + TTS_SEND_CHUNK_BYTES]})
            await asyncio.sleep(0)

    async def _send_filler(self):
        """Send a pre-cached filler audio clip to fill LLM thinking time."""
        try:
            filler_text, filler_bytes = self.filler.get_filler()
            if filler_bytes:
                max_filler = 44100 * 2 * 2  # 2s hard cap for fillers
                if len(filler_bytes) > max_filler:
                    log.warning("[Turn %d] Filler '%s' oversized (%d bytes) — truncating",
                                self._turn, filler_text, len(filler_bytes))
                    filler_bytes = filler_bytes[:max_filler]
                await self._send({"type": "filler", "text": filler_text, "turn": self._turn})
                await self._send({"type": "audio_start", "turn": self._turn})
                await self._send({"_audio": filler_bytes})
                log.info("[Turn %d] Filler sent: '%s' (%d bytes)", self._turn, filler_text, len(filler_bytes))
        except Exception as e:
            log.warning("Filler send failed: %s", e)

    async def handle_text_input(self, text: str):
        """Handle text input (skip ASR)."""
        self._turn += 1
        self.state = State.THINKING
        await self._send({"type": "state", "state": "thinking"})
        await self._send({"type": "asr", "text": text, "latency_ms": 0, "turn": self._turn, "source": "text"})

        self._consecutive_empty_asr = 0

        # D4: Farewell detection for text input
        if any(kw in text for kw in self._farewell_keywords):
            self._session_ending = True
            log.info("[Turn %d] Farewell detected (text): '%s'", self._turn, text)

        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        rag_context = ""
        rag_ms = 0.0
        try:
            rag_result = await loop.run_in_executor(None, self.rag.get_context, text)
            rag_context = rag_result["context"]
            rag_ms = rag_result["total_ms"]
        except Exception as e:
            log.warning("[Turn %d] RAG failed in text_input (degraded): %s", self._turn, e)
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": self._turn,
        })

        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = self._track_task(
            self._stream_llm_tts(text, rag_context, self._turn, t_start, 0, rag_ms),
            name=f"text-tts-{self._turn}",
        )

    # ------------------------------------------------------------------
    # D1: Greeting
    # ------------------------------------------------------------------
    async def start_greeting(self):
        """Play greeting message via TTS. Barge-in aware (runs as a task)."""
        if not self._greeting_text:
            self._idle_since = time.monotonic()
            return
        log.info("Playing greeting: '%s'", self._greeting_text)
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self._speak_log_count = 0
        self.state = State.SPEAKING
        self._turn += 1
        await self._send({"type": "state", "state": "speaking"})
        await self._send({"type": "system_message", "text": self._greeting_text, "turn": self._turn})
        await self._play_system_tts(self._greeting_text, self._turn)

    # ------------------------------------------------------------------
    # D2: Idle timeout
    # ------------------------------------------------------------------
    async def _check_idle_timeout(self):
        elapsed = time.monotonic() - self._idle_since
        if not self._idle_prompted and elapsed >= self._idle_timeout_s:
            self._idle_prompted = True
            log.info("Idle timeout (%.0fs), prompting user", elapsed)
            await self._speak_system(self._idle_prompt_text)
        elif self._idle_prompted and not self._idle_goodbye_sent and elapsed >= self._idle_goodbye_s:
            self._idle_goodbye_sent = True
            self._session_ending = True
            log.info("Idle goodbye (%.0fs), ending session", elapsed)
            await self._speak_system(self._idle_goodbye_text, session_end_reason="idle_timeout")

    async def _speak_system(self, text: str, session_end_reason: str = None):
        """Schedule a system message as a background speaking task."""
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self._speak_log_count = 0
        self.state = State.SPEAKING
        self._turn += 1
        await self._send({"type": "state", "state": "speaking"})
        await self._send({"type": "system_message", "text": text, "turn": self._turn})
        self._speaking_task = self._track_task(
            self._play_system_tts(text, self._turn, session_end_reason),
            name=f"system-tts-{self._turn}",
        )

    # ------------------------------------------------------------------
    # Shared: TTS playback for system-generated messages (D1/D2/D3)
    # ------------------------------------------------------------------
    async def _play_system_tts(self, text: str, turn: int, session_end_reason: str = None):
        """Synthesize and stream TTS for a system message. Handles state transitions."""
        loop = asyncio.get_event_loop()
        try:
            def _gen():
                return list(self.tts.synthesize_stream(text))

            tts_chunks = await loop.run_in_executor(None, _gen)
            await self._send({"type": "audio_start", "turn": turn})

            for i, chunk_data in enumerate(tts_chunks):
                if self._cancel_speaking.is_set():
                    log.info("[Turn %d] System message interrupted at chunk %d", turn, i)
                    break
                audio = chunk_data["audio"]
                if audio.dtype != np.int16:
                    audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                await self._send({"_audio": audio.tobytes()})
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("[Turn %d] System TTS error: %s", turn, e)
        finally:
            if self.state == State.SPEAKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                await self._send({"type": "state", "state": "idle"})
            if session_end_reason and not self._cancel_speaking.is_set():
                await self._send({"type": "session_end", "reason": session_end_reason})
