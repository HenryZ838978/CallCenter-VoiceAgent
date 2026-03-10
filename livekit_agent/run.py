"""LiveKit Voice Agent — WebRTC transport for VoxLabs pipeline.

Usage:
    # 1. Start LiveKit Server:  ./livekit-server --config /tmp/livekit.yaml --dev
    # 2. Start vLLM:            CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server ...
    # 3. Start Agent Worker:    python livekit_agent/run.py dev

SSH tunnel from Mac:
    ssh -L 7880:localhost:7880 -L 7881:localhost:7881 user@10.158.0.7
    Then open playground.html, connect to ws://localhost:7880
"""
import os
import sys
import asyncio
import logging
import threading
import numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobProcess,
    stt,
    tts,
    llm,
    cli,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions

from config import (
    ASR_MODEL_DIR, TTS_MODEL_DIR, VAD_MODEL_DIR,
    VLLM_BASE_URL, VLLM_MODEL_NAME,
    EMBED_MODEL_DIR, KB_DATA_PATH, RAG_TOP_K,
    SYSTEM_PROMPT_RAG, VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("voxlabs.lk")

ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")

_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_XML_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")
_ASR_TAG_RE = re.compile(r"<\|[^>]*\|>")

_tts_lock = threading.Lock()
_tts_engine = None


def _get_tts():
    """Lazy-load TTS on first use. nanovllm spawns its own subprocess, so we
    defer loading until we're inside the worker process (after fork)."""
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    with _tts_lock:
        if _tts_engine is not None:
            return _tts_engine
        from engine.tts import VoxCPMTTS
        log.info("Lazy-loading TTS (VoxCPM nanovllm)...")
        tts_util = float(os.environ.get("TTS_GPU_UTIL", "0.55"))
        engine = VoxCPMTTS(TTS_MODEL_DIR, device=TTS_DEVICE, gpu_memory_utilization=tts_util)
        engine.load()
        if os.path.exists(VOICE_PROMPT_WAV):
            pid = engine.register_voice(VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT)
            engine.set_default_voice(pid)
        engine.warmup()
        _tts_engine = engine
        log.info("TTS ready")
        return _tts_engine


class VoxLabsSTT(stt.STT):
    def __init__(self, asr_engine):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._asr = asr_engine

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options=None):
        loop = asyncio.get_event_loop()
        audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_data) == 0:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="", language="zh")],
            )
        result = await loop.run_in_executor(None, self._asr.transcribe, audio_data)
        text = _ASR_TAG_RE.sub("", result["text"]).strip()
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language="zh")],
        )


class VoxLabsTTS(tts.TTS):
    def __init__(self):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=44100,
            num_channels=1,
        )

    def synthesize(self, text, *, conn_options=None):
        return VoxLabsChunkedStream(self, text, conn_options)


class VoxLabsChunkedStream(tts.ChunkedStream):
    def __init__(self, tts_instance, text, conn_options):
        super().__init__(tts=tts_instance, input_text=text,
                         conn_options=conn_options or APIConnectOptions())
        self._text = text

    async def _run(self):
        loop = asyncio.get_event_loop()
        engine = _get_tts()
        result = await loop.run_in_executor(None, engine.synthesize, self._text)
        audio = result["audio"]
        if audio.dtype != np.int16:
            audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

        chunk_samples = 44100 // 5
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=44100,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(request_id=self._request_id, frame=frame)
            )


class VoxLabsLLM(llm.LLM):
    def __init__(self, base_url, model, system_prompt, rag_engine=None):
        super().__init__()
        self._base_url = base_url
        self._model = model
        self._system_prompt = system_prompt
        self._rag = rag_engine

    def chat(self, *, chat_ctx, tools=None, conn_options=None, parallel_tool_calls=None,
             extra_body=None, temperature=None, response_format=None, tool_choice=None):
        return VoxLabsLLMStream(self, chat_ctx, self._base_url, self._model,
                                self._system_prompt, self._rag)


class VoxLabsLLMStream(llm.LLMStream):
    def __init__(self, llm_instance, chat_ctx, base_url, model, system_prompt, rag):
        super().__init__(llm=llm_instance, chat_ctx=chat_ctx,
                         tools=[], conn_options=APIConnectOptions())
        self._base_url = base_url
        self._model = model
        self._system_prompt = system_prompt
        self._rag = rag

    async def _run(self):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=self._base_url, api_key="dummy")

        user_text = ""
        for msg in self._chat_ctx.items:
            if hasattr(msg, 'role') and msg.role == "user":
                for c in msg.content:
                    if hasattr(c, 'text'):
                        user_text = c.text

        system = self._system_prompt
        if self._rag and user_text:
            loop = asyncio.get_event_loop()
            rag_result = await loop.run_in_executor(None, self._rag.get_context, user_text)
            context = rag_result.get("context", "")
            if "{context}" in system:
                system = system.replace("{context}", context)

        messages = [{"role": "system", "content": system}]
        for msg in self._chat_ctx.items:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                text_parts = [c.text for c in msg.content if hasattr(c, 'text')]
                if text_parts:
                    messages.append({"role": msg.role, "content": " ".join(text_parts)})

        extra = {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.2}
        stream = await client.chat.completions.create(
            model=self._model, messages=messages,
            max_tokens=150, temperature=0.7, stream=True, extra_body=extra,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                token = _TAG_RE.sub("", token)
                token = _XML_RE.sub("", token)
                if token:
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            request_id=self._request_id,
                            choices=[llm.Choice(
                                delta=llm.ChoiceDelta(role="assistant", content=token),
                                index=0,
                            )],
                        )
                    )


def prewarm(proc: JobProcess):
    """Load ASR and RAG in worker subprocess. TTS is lazy-loaded on first request
    to avoid nanovllm fork-in-fork CUDA issues."""
    from engine.asr import SenseVoiceASR
    from engine.rag import RAGEngine
    import json

    log.info("=== Prewarming VoxLabs engines (ASR + RAG) ===")

    asr = SenseVoiceASR(ASR_MODEL_DIR, device=ASR_DEVICE)
    asr.load()
    log.info("ASR loaded")

    rag = RAGEngine(EMBED_MODEL_DIR, device="cpu", top_k=RAG_TOP_K)
    rag.load()
    if os.path.exists(KB_DATA_PATH):
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        rag.build_index(docs)
    log.info("RAG loaded (%d docs)", len(rag._documents))

    proc.userdata["asr"] = asr
    proc.userdata["rag"] = rag
    log.info("=== Prewarm done (TTS will lazy-load on first request) ===")


server = AgentServer(
    setup_fnc=prewarm,
    ws_url=os.environ.get("LIVEKIT_URL", "ws://localhost:7880"),
    api_key=os.environ.get("LIVEKIT_API_KEY", "devkey"),
    api_secret=os.environ.get("LIVEKIT_API_SECRET", "secret"),
    initialize_process_timeout=120.0,
    num_idle_processes=1,
)


@server.rtc_session()
async def entrypoint(ctx):
    asr = ctx.proc.userdata["asr"]
    rag = ctx.proc.userdata["rag"]

    from livekit_agent.silero_vad_lk import SileroVADPlugin
    silero_vad = SileroVADPlugin(VAD_MODEL_DIR, threshold=0.5)

    session = AgentSession(
        stt=VoxLabsSTT(asr),
        tts=VoxLabsTTS(),
        llm=VoxLabsLLM(VLLM_BASE_URL, VLLM_MODEL_NAME, SYSTEM_PROMPT_RAG, rag),
        vad=silero_vad,
        turn_detection="vad",
        allow_interruptions=True,
        min_endpointing_delay=0.3,
        max_endpointing_delay=1.5,
    )

    agent = Agent(instructions=SYSTEM_PROMPT_RAG.split("\n")[0])
    await session.start(agent=agent, room=ctx.room)
    log.info("Session started for room: %s", ctx.room.name)


if __name__ == "__main__":
    cli.run_app(server)
