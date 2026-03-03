"""
VoxLabs Full-Duplex Voice Agent — WebSocket Server

Exposes:
  wss://:3000/ws/voice   — duplex voice channel (binary audio + JSON events)
  https://:3000/         — dashboard / voice call UI
  https://:3000/api/*    — REST management endpoints
"""
import os
import sys
import json
import time
import asyncio
import logging
import ssl
import struct
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASR_MODEL_DIR, TTS_MODEL_DIR, VAD_MODEL_DIR,
    VLLM_BASE_URL, VLLM_MODEL_NAME,
    EMBED_MODEL_DIR, KB_DATA_PATH, RAG_TOP_K,
    SYSTEM_PROMPT_RAG, SAMPLE_RATE, CHUNK_SAMPLES,
    VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("voxlabs")

ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")
RAG_DEVICE = os.environ.get("RAG_DEVICE", "cuda:0")

engine = {}
session_metrics = []


def load_engines():
    """Load all ML models. Called once at startup."""
    from engine.vad import SileroVAD
    from engine.asr import SenseVoiceASR
    from engine.llm import VLLMChat
    from engine.tts import VoxCPMTTS
    from engine.rag import RAGEngine

    log.info("Loading VAD...")
    engine["vad_template"] = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
    engine["vad_template"].load()

    log.info("Loading ASR...")
    engine["asr"] = SenseVoiceASR(ASR_MODEL_DIR, device=ASR_DEVICE)
    engine["asr"].load()

    log.info("Loading RAG (bge-small-zh-v1.5)...")
    rag = RAGEngine(EMBED_MODEL_DIR, device=RAG_DEVICE, top_k=RAG_TOP_K)
    rag.load()
    if os.path.exists(KB_DATA_PATH):
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        build_info = rag.build_index(docs)
        log.info(f"RAG index built: {build_info['num_docs']} docs, dim={build_info['dim']}")
    engine["rag"] = rag

    log.info("LLM client ready (vLLM @ %s)", VLLM_BASE_URL)
    engine["llm_factory"] = lambda: VLLMChat(
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL_NAME,
        system_prompt=SYSTEM_PROMPT_RAG,
    )

    log.info("Loading TTS (VoxCPM nanovllm)...")
    tts = VoxCPMTTS(TTS_MODEL_DIR, device=TTS_DEVICE, gpu_memory_utilization=0.55)
    tts.load()

    if os.path.exists(VOICE_PROMPT_WAV):
        log.info("Registering voice clone from %s ...", os.path.basename(VOICE_PROMPT_WAV))
        prompt_id = tts.register_voice(VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT)
        tts.set_default_voice(prompt_id)
        log.info("Voice clone registered: %s", prompt_id)

    log.info("Warming up TTS...")
    tts.warmup()
    engine["tts"] = tts

    log.info("All engines loaded.")


app = FastAPI(title="VoxLabs Voice Agent")

BASE_DIR = Path(__file__).parent


# ── REST API ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "voice_agent.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/info")
async def api_info():
    rag = engine.get("rag")
    return {
        "models": {
            "asr": {"name": "SenseVoiceSmall", "framework": "FunASR", "device": ASR_DEVICE},
            "llm": {"name": VLLM_MODEL_NAME, "endpoint": VLLM_BASE_URL, "thinking": False},
            "tts": {"name": "VoxCPM 1.5", "framework": "nanovllm-voxcpm", "device": TTS_DEVICE},
            "embedding": {"name": "bge-small-zh-v1.5", "device": RAG_DEVICE},
        },
        "rag": {
            "num_docs": len(rag._documents) if rag else 0,
            "top_k": RAG_TOP_K,
        },
    }


@app.get("/api/metrics")
async def api_metrics():
    recent = session_metrics[-50:]
    if not recent:
        return {"count": 0, "avg": {}}
    avg = {}
    keys = ["asr_ms", "rag_ms", "llm_ms", "tts_ttfa_ms", "first_response_ms"]
    for k in keys:
        vals = [m[k] for m in recent if k in m]
        avg[k] = round(sum(vals) / len(vals), 1) if vals else 0
    return {"count": len(recent), "avg": avg, "recent": recent[-10:]}


@app.get("/api/rag/docs")
async def rag_docs():
    rag = engine.get("rag")
    if not rag:
        return {"docs": []}
    return {"docs": [{"id": d.get("id"), "question": d.get("question"), "answer": d.get("answer", "")[:100]}
                      for d in rag._documents]}


@app.get("/api/rag/query")
async def rag_query(q: str):
    rag = engine.get("rag")
    if not rag:
        return {"error": "RAG not loaded"}
    result = rag.query(q)
    return result


@app.post("/api/rag/reload")
async def rag_reload():
    rag = engine.get("rag")
    if not rag or not os.path.exists(KB_DATA_PATH):
        return {"error": "No KB file"}
    with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    info = rag.build_index(docs)
    return {"status": "ok", **info}


# ── WebSocket duplex voice ────────────────────────────

@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    session_id = f"s-{int(time.time()*1000)}"
    log.info("[%s] WebSocket connected", session_id)

    from engine.vad import SileroVAD
    vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
    vad.load()

    llm = engine["llm_factory"]()
    asr = engine["asr"]
    tts = engine["tts"]
    rag = engine["rag"]

    audio_buffer = []
    speech_active = False
    tts_playing = False
    turn_count = 0

    async def send_json(data):
        try:
            await ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    await send_json({"type": "ready", "session_id": session_id})

    try:
        while True:
            data = await ws.receive()

            if "bytes" in data and data["bytes"]:
                raw = data["bytes"]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                for i in range(0, len(samples), CHUNK_SAMPLES):
                    chunk = samples[i:i + CHUNK_SAMPLES]
                    if len(chunk) < CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

                    vad_result = vad.process_chunk(chunk, SAMPLE_RATE)

                    if vad_result["speech_start"]:
                        speech_active = True
                        audio_buffer = []
                        if tts_playing:
                            tts_playing = False
                            await send_json({"type": "barge_in", "turn": turn_count})

                    if speech_active:
                        audio_buffer.append(chunk)

                    if vad_result["speech_end"] and audio_buffer:
                        speech_active = False
                        full_audio = np.concatenate(audio_buffer)
                        audio_buffer = []
                        turn_count += 1

                        try:
                            await process_utterance(
                                ws, full_audio, asr, rag, llm, tts,
                                turn_count, session_id, send_json
                            )
                        except Exception as e:
                            log.error("[%s] Pipeline error: %s", session_id, e)
                            await send_json({"type": "error", "message": str(e)})

            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if msg.get("type") == "text_input":
                    turn_count += 1
                    try:
                        await process_text_input(
                            ws, msg["text"], rag, llm, tts,
                            turn_count, session_id, send_json
                        )
                    except Exception as e:
                        log.error("[%s] Pipeline error: %s", session_id, e)
                        await send_json({"type": "error", "message": str(e)})
                elif msg.get("type") == "reset":
                    llm.reset()
                    vad.reset()
                    turn_count = 0
                    await send_json({"type": "reset_ack"})

    except WebSocketDisconnect:
        log.info("[%s] WebSocket disconnected (%d turns)", session_id, turn_count)
    except Exception as e:
        log.error("[%s] Error: %s", session_id, e, exc_info=True)


def _clean_asr(text: str) -> str:
    """Strip SenseVoice format tags like <|zh|><|NEUTRAL|><|Speech|><|withitn|>."""
    import re
    return re.sub(r"<\|[^>]*\|>", "", text).strip()


async def process_utterance(ws, audio, asr, rag, llm, tts, turn, sid, send_json):
    """ASR → RAG → LLM → TTS pipeline."""
    t_start = time.perf_counter()

    loop = asyncio.get_event_loop()

    asr_result = await loop.run_in_executor(None, asr.transcribe, audio)
    user_text = _clean_asr(asr_result["text"])
    await send_json({
        "type": "asr", "text": user_text,
        "latency_ms": round(asr_result["latency_ms"], 1), "turn": turn,
    })

    if not user_text or len(user_text) < 2:
        return

    await _do_rag_llm_tts(ws, user_text, rag, llm, tts, turn, sid, send_json,
                          asr_ms=asr_result["latency_ms"], t_start=t_start)


async def process_text_input(ws, user_text, rag, llm, tts, turn, sid, send_json):
    """Text → RAG → LLM → TTS (skip ASR)."""
    t_start = time.perf_counter()
    await send_json({"type": "asr", "text": user_text, "latency_ms": 0, "turn": turn, "source": "text"})
    await _do_rag_llm_tts(ws, user_text, rag, llm, tts, turn, sid, send_json,
                          asr_ms=0, t_start=t_start)


async def _do_rag_llm_tts(ws, user_text, rag, llm, tts, turn, sid, send_json,
                           asr_ms, t_start):
    loop = asyncio.get_event_loop()

    rag_result = await loop.run_in_executor(None, rag.get_context, user_text)
    rag_context = rag_result["context"]
    await send_json({
        "type": "rag", "context": rag_context,
        "num_results": rag_result["num_results"],
        "latency_ms": round(rag_result["total_ms"], 1), "turn": turn,
    })

    llm_result = await loop.run_in_executor(
        None, lambda: llm.chat(user_text, rag_context=rag_context)
    )
    assistant_text = llm_result["text"]
    await send_json({
        "type": "llm", "text": assistant_text,
        "latency_ms": round(llm_result["latency_ms"], 1), "turn": turn,
    })

    tts_result = await loop.run_in_executor(
        None, lambda: tts.synthesize(assistant_text)
    )
    first_response_ms = asr_ms + rag_result["total_ms"] + llm_result["latency_ms"] + tts_result["ttfa_ms"]

    await send_json({
        "type": "tts_start",
        "ttfa_ms": round(tts_result["ttfa_ms"], 1),
        "chunks": tts_result["chunks"], "turn": turn,
    })

    audio_data = tts_result["audio"]
    if audio_data.dtype != np.int16:
        audio_data = (np.clip(audio_data, -1, 1) * 32767).astype(np.int16)
    audio_bytes = audio_data.tobytes()

    chunk_size = 44100 * 2  # ~1s chunks at 44100Hz, 16-bit (larger chunks = fewer sends)
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        try:
            await ws.send_bytes(chunk)
        except Exception:
            break
        await asyncio.sleep(0.02)

    await send_json({
        "type": "tts_end",
        "total_ms": round(tts_result["total_ms"], 1), "turn": turn,
    })

    metrics = {
        "session_id": sid, "turn": turn, "user_text": user_text,
        "assistant_text": assistant_text[:80],
        "asr_ms": round(asr_ms, 1),
        "rag_ms": round(rag_result["total_ms"], 1),
        "llm_ms": round(llm_result["latency_ms"], 1),
        "tts_ttfa_ms": round(tts_result["ttfa_ms"], 1),
        "first_response_ms": round(first_response_ms, 1),
        "timestamp": time.time(),
    }
    session_metrics.append(metrics)
    await send_json({"type": "metrics", **metrics})

    log.info("[%s] Turn %d: ASR=%dms RAG=%dms LLM=%dms TTS=%dms FirstResp=%dms | %s → %s",
             sid, turn, asr_ms, rag_result["total_ms"],
             llm_result["latency_ms"], tts_result["ttfa_ms"],
             first_response_ms, user_text[:30], assistant_text[:30])


if __name__ == "__main__":
    load_engines()

    cert_dir = BASE_DIR / "certs"
    ssl_certfile = str(cert_dir / "cert.pem")
    ssl_keyfile = str(cert_dir / "key.pem")

    port = int(os.environ.get("PORT", "3000"))
    log.info("Starting VoxLabs server on https://0.0.0.0:%d", port)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        log_level="info",
    )
