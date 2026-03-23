"""LiveKit Voice Agent — WebRTC transport with official Silero VAD.

Architecture:
  VPS (82.156.207.59): LiveKit Server (:7880) + Nginx SSL (:7443)
  4090 Server: Agent Worker (ASR GPU3 + TTS GPU2 + vLLM GPU1)
  No SSH tunnel needed — Agent Worker connects outbound to VPS.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 ASR_DEVICE=cuda:1 TTS_DEVICE=cuda:0 \
      LIVEKIT_URL=ws://82.156.207.59:7880 \
      python livekit_agent/run.py dev
"""
import os
import sys
import asyncio
import logging
import numpy as np
import re
import uuid

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
from livekit.plugins.silero import VAD as SileroVAD

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
        src_sr = getattr(buffer, 'sample_rate', 16000)
        if src_sr != 16000 and src_sr > 0:
            from scipy.signal import resample
            target_len = int(len(audio_data) * 16000 / src_sr)
            audio_data = resample(audio_data, target_len).astype(np.float32)
        result = await loop.run_in_executor(None, self._asr.transcribe, audio_data)
        text = _ASR_TAG_RE.sub("", result["text"]).strip()
        log.info("STT: '%s' (%.0fms)", text, result["latency_ms"])
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

    async def _run(self, output_emitter):
        """Call TTS HTTP server, push PCM audio via output_emitter."""
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{TTS_SERVER_URL}/synthesize",
                json={"text": self._text},
            )
            resp.raise_for_status()
            audio_bytes = resp.content
            latency = resp.headers.get("X-TTS-Latency-Ms", "?")

        req_id = str(uuid.uuid4())
        output_emitter.initialize(
            request_id=req_id,
            sample_rate=44100,
            num_channels=1,
            mime_type="audio/pcm",
        )

        chunk_size = 44100 // 5 * 2
        for i in range(0, len(audio_bytes), chunk_size):
            output_emitter.push(audio_bytes[i:i + chunk_size])

        log.info("TTS: '%s' (%sms via HTTP, %d bytes)", self._text[:30], latency, len(audio_bytes))


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

        extra = {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.15}
        stream = await client.chat.completions.create(
            model=self._model, messages=messages,
            max_tokens=120, temperature=0.85, top_p=0.9,
            stream=True, extra_body=extra,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                token = _TAG_RE.sub("", token)
                token = _XML_RE.sub("", token)
                if token:
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=chunk.id or str(uuid.uuid4()),
                            delta=llm.ChoiceDelta(role="assistant", content=token),
                        )
                    )


TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://localhost:8200")


def prewarm(proc: JobProcess):
    """Load ASR + RAG + VAD. TTS loaded on first session (nanovllm fork-in-fork issue)."""
    import torch

    from engine.asr_firered import FireRedASR
    from engine.rag import RAGEngine
    import json

    log.info("=== Prewarming (ASR %s, VAD Silero, TTS deferred) ===", ASR_DEVICE)

    gpu_idx = int(ASR_DEVICE.split(":")[-1]) if ":" in ASR_DEVICE else 0
    torch.cuda.set_device(gpu_idx)
    asr = FireRedASR(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "FireRedASR2-AED"),
        device=ASR_DEVICE
    )
    asr.load()
    log.info("ASR loaded (FireRedASR2-AED on %s)", ASR_DEVICE)

    rag = RAGEngine(EMBED_MODEL_DIR, device="cpu", top_k=RAG_TOP_K)
    rag.load()
    if os.path.exists(KB_DATA_PATH):
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        rag.build_index(docs)
    log.info("RAG loaded (%d docs)", len(rag._documents))

    silero_vad = SileroVAD.load(
        activation_threshold=0.35,
        min_speech_duration=0.05,
        min_silence_duration=0.4,
    )
    log.info("Silero VAD loaded (threshold=0.35, silence=0.4s)")

    proc.userdata["asr"] = asr
    proc.userdata["rag"] = rag
    proc.userdata["vad"] = silero_vad
    log.info("=== Prewarm done (TTS via HTTP %s) ===", TTS_SERVER_URL)


server = AgentServer(
    setup_fnc=prewarm,
    ws_url=os.environ.get("LIVEKIT_URL", "ws://82.156.207.59:7880"),
    api_key=os.environ.get("LIVEKIT_API_KEY", "hzai_key"),
    api_secret=os.environ.get("LIVEKIT_API_SECRET", "hzai_secret_long_enough_for_production_use_2026"),
    initialize_process_timeout=300.0,
    num_idle_processes=1,
)


@server.rtc_session()
async def entrypoint(ctx):
    asr = ctx.proc.userdata["asr"]
    rag = ctx.proc.userdata["rag"]
    vad = ctx.proc.userdata["vad"]

    session = AgentSession(
        stt=VoxLabsSTT(asr),
        tts=VoxLabsTTS(),
        llm=VoxLabsLLM(VLLM_BASE_URL, VLLM_MODEL_NAME, SYSTEM_PROMPT_RAG, rag),
        vad=vad,
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
