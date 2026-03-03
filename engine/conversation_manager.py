"""Production Conversation Manager — state machine for natural full-duplex voice agent.

States:
  IDLE       → waiting for user
  LISTENING  → user is speaking, buffering audio
  THINKING   → user done, playing filler, running pipeline
  SPEAKING   → streaming TTS playback
  INTERRUPTED→ user spoke during AI speech, collecting new input

Flow:
  IDLE → LISTENING (user starts speaking)
  LISTENING → THINKING (endpointing: silence > threshold + ASR text looks complete)
  THINKING → SPEAKING (first TTS chunk ready)
  SPEAKING → INTERRUPTED (user starts speaking during AI output)
  SPEAKING → LISTENING (TTS finished, ready for next turn)
  INTERRUPTED → THINKING (user stops, aggregate intent)
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


ENDPOINTING_SILENCE_CHUNKS = 8    # ~256ms at 32ms/chunk
INTERRUPT_SILENCE_CHUNKS = 10     # ~320ms before processing interrupted speech
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512               # 32ms


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

    def reset(self):
        self.state = State.IDLE
        self._audio_buffer = []
        self._interrupted_texts = []
        self._silence_count = 0
        self._turn = 0
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
        is_speech = vad_result["speech_prob"] >= 0.5

        if self.state == State.IDLE:
            if is_speech:
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
                if self._silence_count >= ENDPOINTING_SILENCE_CHUNKS:
                    await self._on_endpointing()

        elif self.state == State.SPEAKING:
            if is_speech:
                self._silence_count = 0
                if not self._cancel_speaking.is_set():
                    self._cancel_speaking.set()
                    self.state = State.INTERRUPTED
                    self._audio_buffer = [chunk]
                    await self._send({"type": "barge_in", "turn": self._turn})
                    await self._send({"type": "state", "state": "interrupted"})
                else:
                    self._audio_buffer.append(chunk)
            # ignore silence during SPEAKING

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
            if is_speech:
                pass  # ignore speech during thinking (filler is playing)

    async def _on_endpointing(self):
        """User finished speaking — start pipeline."""
        self.state = State.THINKING
        self._silence_count = 0
        self._turn += 1
        await self._send({"type": "state", "state": "thinking"})

        audio = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([], dtype=np.float32)
        self._audio_buffer = []

        asyncio.create_task(self._run_pipeline(audio, self._turn))

    async def _run_pipeline(self, audio: np.ndarray, turn: int):
        """Full pipeline: filler → ASR → RAG → LLM(streaming) → TTS(per-sentence) → play."""
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        # 1. Filler disabled for now — short TTS fillers sound mechanical.
        # TODO: re-enable with real human-recorded filler audio
        # filler_text, filler_audio = self.filler.get_filler()
        # if filler_audio:
        #     await self._send({"type": "filler", "text": filler_text, "turn": turn})
        #     await self._send_audio(filler_audio)

        # 2. ASR
        asr_result = await loop.run_in_executor(None, self.asr.transcribe, audio)
        user_text = _ASR_TAG_RE.sub("", asr_result["text"]).strip()
        asr_ms = asr_result["latency_ms"]
        await self._send({"type": "asr", "text": user_text, "latency_ms": round(asr_ms, 1), "turn": turn})

        if not user_text or len(user_text) < 2:
            self.state = State.IDLE
            await self._send({"type": "state", "state": "idle"})
            return

        # 3. Check for interrupted texts to aggregate
        if self._interrupted_texts:
            combined = "；".join(self._interrupted_texts + [user_text])
            user_text = f"用户依次说了：{combined}。请自然地综合回应。"
            self._interrupted_texts = []
        else:
            if asr_ms > 0:
                self._interrupted_texts = []

        # 4. RAG
        rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
        rag_context = rag_result["context"]
        rag_ms = rag_result["total_ms"]
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": turn,
        })

        # 5. LLM streaming → TTS per sentence → play
        self._cancel_speaking.clear()
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = asyncio.create_task(
            self._stream_llm_tts(user_text, rag_context, turn, t_start, asr_ms, rag_ms)
        )

    async def _stream_llm_tts(self, user_text, rag_context, turn, t_start, asr_ms, rag_ms):
        """True streaming: LLM yields sentences via queue → TTS per sentence → play.
        Cancel is checked between every sentence and every audio chunk.
        """
        loop = asyncio.get_event_loop()
        sentence_queue = asyncio.Queue()
        full_response = ""
        sentence_count = 0

        def _produce_sentences():
            """Runs in thread: streams LLM and puts sentences into async queue."""
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
        chunk_size = 8820 * 2  # ~200ms at 44.1kHz 16-bit — small enough for responsive cancel
        for i in range(0, len(audio_bytes), chunk_size):
            if self._cancel_speaking.is_set():
                return
            await self._send({"_audio": audio_bytes[i:i + chunk_size]})
            await asyncio.sleep(0.005)

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
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = asyncio.create_task(
            self._stream_llm_tts(
                text, rag_result["context"], self._turn,
                t_start, 0, rag_result["total_ms"]
            )
        )
