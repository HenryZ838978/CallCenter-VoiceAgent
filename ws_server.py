"""
VoxLabs Voice Agent v2.9 — Production Full-Duplex WebSocket Server

v2.9: state-race fix, API key auth, auto speaker-VAD/denoise,
      structured logging, RAG confidence, LLM context expansion.
v2.8: crash-safe pipeline, session metrics, dead-socket detection,
      task lifecycle management, filler activation, barge-in consistency.
"""
import os
import sys
import json
import time
import hmac
import asyncio
import logging
import numpy as np
import uvicorn
import nest_asyncio
nest_asyncio.apply()
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.status import HTTP_403_FORBIDDEN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASR_MODEL_DIR, TTS_MODEL_DIR, VAD_MODEL_DIR,
    VLLM_BASE_URL, VLLM_MODEL_NAME,
    EMBED_MODEL_DIR, KB_DATA_PATH, RAG_TOP_K, RAG_SCORE_THRESHOLD,
    SYSTEM_PROMPT_RAG, VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT,
    GREETING_TEXT, IDLE_TIMEOUT_S, IDLE_GOODBYE_S,
    IDLE_PROMPT_TEXT, IDLE_GOODBYE_TEXT,
    ASR_EMPTY_RETRY_TEXT, ASR_NOISY_SUGGEST_TEXT, MAX_CONSECUTIVE_EMPTY_ASR,
    FAREWELL_KEYWORDS,
    LLM_MAX_HISTORY, API_KEY,
)
CAPTIONER_URL = os.environ.get("CAPTIONER_URL", "")
CAPTIONER_MODEL = os.environ.get("CAPTIONER_MODEL", "MiniCPM-o-4.5-awq")
USE_FIRERED_VAD = os.environ.get("USE_FIRERED_VAD", "0") == "1"
USE_MOONSHINE_ASR = os.environ.get("USE_MOONSHINE_ASR", "0") == "1"
USE_FIRERED_ASR = os.environ.get("USE_FIRERED_ASR", "0") == "1"
USE_SMART_TURN = os.environ.get("USE_SMART_TURN", "0") == "1"

_speaker_vad_env = os.environ.get("USE_SPEAKER_VAD", "auto")
_denoise_env = os.environ.get("USE_DENOISE", "auto")
DTLN_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "dtln")
SPEAKER_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "spkrec-ecapa-voxceleb")
FIRERED_VAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "FireRedVAD-stream")
FIRERED_ASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "FireRedASR2-AED")

def _model_dir_ready(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    entries = [e for e in os.listdir(d) if not e.startswith(".")]
    return len(entries) >= 2

USE_SPEAKER_VAD = (
    _speaker_vad_env == "1" or
    (_speaker_vad_env == "auto" and _model_dir_ready(SPEAKER_MODEL_DIR))
)
USE_DENOISE = (
    _denoise_env == "1" or
    (_denoise_env == "auto" and _model_dir_ready(DTLN_MODEL_DIR))
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger("voxlabs")

ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")
RAG_DEVICE = os.environ.get("RAG_DEVICE", "cuda:0")

engine = {}


def load_engines():
    from engine.vad import SileroVAD
    from engine.asr import SenseVoiceASR
    from engine.llm import VLLMChat
    from engine.tts import VoxCPMTTS
    from engine.rag import RAGEngine
    from engine.filler import FillerEngine

    log.info("=== Loading engines (v2.0) ===")

    if USE_FIRERED_VAD:
        from engine.firered_vad import FireRedVADWrapper
        log.info("Loading FireRedVAD (F1=97.57%%, false alarm 2.69%%)...")
        engine["vad_type"] = "firered"
        engine["vad_template"] = FireRedVADWrapper(FIRERED_VAD_DIR, threshold=0.5)
        engine["vad_template"].load()
    elif USE_SPEAKER_VAD:
        from engine.speaker_vad import SpeakerAwareVAD
        log.info("Loading SpeakerAwareVAD (Silero + ECAPA-TDNN)...")
        engine["vad_type"] = "speaker"
        svad = SpeakerAwareVAD(VAD_MODEL_DIR, SPEAKER_MODEL_DIR, threshold=0.5)
        svad.load(device="cpu")
        engine["vad_template"] = svad
    else:
        log.info("Loading VAD (Silero)...")
        engine["vad_type"] = "silero"
        engine["vad_template"] = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
        engine["vad_template"].load()

    if USE_MOONSHINE_ASR:
        from engine.asr_moonshine import MoonshineASR
        log.info("Loading Moonshine Tiny (speculative ASR, 27M, ONNX CPU)...")
        engine["spec_asr"] = MoonshineASR(model_name="moonshine/tiny")
        engine["spec_asr"].load()
    else:
        engine["spec_asr"] = None

    if USE_DENOISE:
        from engine.denoiser import DTLNDenoiser
        log.info("Loading DTLN Denoiser (4MB ONNX, ~8ms/block)...")
        engine["denoiser"] = DTLNDenoiser(DTLN_MODEL_DIR)
        engine["denoiser"].load()
    else:
        engine["denoiser"] = None

    if USE_SMART_TURN:
        from engine.turn_detector import SmartTurnDetector
        log.info("Loading Smart Turn Detector (Pipecat v3, 8M ONNX)...")
        engine["turn_detector"] = SmartTurnDetector()
        engine["turn_detector"].load()
    else:
        engine["turn_detector"] = None

    if USE_FIRERED_ASR:
        from engine.asr_firered import FireRedASR
        log.info("Loading ASR (FireRedASR2-AED, CER 2.89%%)...")
        engine["asr"] = FireRedASR(FIRERED_ASR_DIR, device=ASR_DEVICE)
        engine["asr"].load()
    else:
        log.info("Loading ASR (SenseVoice)...")
        engine["asr"] = SenseVoiceASR(ASR_MODEL_DIR, device=ASR_DEVICE)
        engine["asr"].load()

    log.info("Loading RAG (bge-small-zh-v1.5, threshold=%.2f)...", RAG_SCORE_THRESHOLD)
    rag = RAGEngine(EMBED_MODEL_DIR, device=RAG_DEVICE, top_k=RAG_TOP_K,
                    score_threshold=RAG_SCORE_THRESHOLD)
    rag.load()
    if os.path.exists(KB_DATA_PATH):
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        info = rag.build_index(docs)
        log.info("RAG index: %d docs, dim=%d", info["num_docs"], info["dim"])
    engine["rag"] = rag

    log.info("LLM client ready (vLLM @ %s, max_history=%d)", VLLM_BASE_URL, LLM_MAX_HISTORY)
    engine["llm_factory"] = lambda: VLLMChat(
        base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME,
        system_prompt=SYSTEM_PROMPT_RAG, max_history=LLM_MAX_HISTORY,
    )

    log.info("Loading TTS (VoxCPM nanovllm)...")
    tts_util = float(os.environ.get("TTS_GPU_UTIL", "0.45"))
    tts = VoxCPMTTS(TTS_MODEL_DIR, device=TTS_DEVICE, gpu_memory_utilization=tts_util)
    tts.load()

    if os.path.exists(VOICE_PROMPT_WAV):
        log.info("Registering voice clone...")
        pid = tts.register_voice(VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT)
        tts.set_default_voice(pid)
        log.info("Voice clone: %s", pid)

    log.info("Warming up TTS...")
    tts.warmup()
    engine["tts"] = tts

    log.info("Pre-generating filler words...")
    filler = FillerEngine(tts)
    filler.pregenerate()
    engine["filler"] = filler

    if CAPTIONER_URL:
        from engine.captioner import OmniCaptioner
        log.info("Loading Captioner (OmniCaptioner @ %s)...", CAPTIONER_URL)
        engine["captioner"] = OmniCaptioner(
            base_url=CAPTIONER_URL, model=CAPTIONER_MODEL,
        ).load()
    else:
        from engine.captioner import HeuristicCaptioner
        log.info("Using HeuristicCaptioner (no CAPTIONER_URL set)")
        engine["captioner"] = HeuristicCaptioner()

    log.info("=== All engines loaded (v2.9) ===")
    log.info("  Speaker-aware VAD: %s (env=%s)", USE_SPEAKER_VAD, _speaker_vad_env)
    log.info("  Denoise (DTLN): %s (env=%s)", USE_DENOISE, _denoise_env)
    log.info("  API key auth: %s", "enabled" if API_KEY else "disabled")


app = FastAPI(title="VoxLabs Voice Agent v2.9")
BASE_DIR = Path(__file__).parent

_active_sessions: dict[str, dict] = {}
_total_sessions = 0
_total_errors = 0


def _check_api_key(token: str | None) -> bool:
    if not API_KEY:
        return True
    if not token:
        return False
    return hmac.compare_digest(token, API_KEY)


async def verify_api_key(request: Request):
    """FastAPI dependency — checks Bearer token or ?token= query param."""
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not token:
        token = request.query_params.get("token", "")
    if not _check_api_key(token):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((BASE_DIR / "static" / "voice_agent.html").read_text(encoding="utf-8"))


@app.get("/api/info", dependencies=[Depends(verify_api_key)])
async def api_info():
    rag = engine.get("rag")
    asr_name = "FireRedASR2-AED" if USE_FIRERED_ASR else "SenseVoiceSmall"
    asr_fw = "FireRedASR" if USE_FIRERED_ASR else "FunASR"
    return {
        "version": "2.9",
        "models": {
            "asr": {"name": asr_name, "framework": asr_fw},
            "llm": {"name": VLLM_MODEL_NAME, "endpoint": VLLM_BASE_URL, "streaming": True},
            "tts": {"name": "VoxCPM 1.5", "framework": "nanovllm", "sample_rate": 44100},
            "embedding": {"name": "bge-small-zh-v1.5"},
        },
        "features": [
            "filler_words", "sentence_streaming", "barge_in",
            "utterance_aggregation", "experience_layer_d1_d4", "crash_safe",
            "api_key_auth", "rag_confidence", "context_summary",
        ],
        "rag": {"num_docs": len(rag._documents) if rag else 0, "top_k": RAG_TOP_K},
    }


@app.get("/api/metrics", dependencies=[Depends(verify_api_key)])
async def api_metrics():
    sessions = []
    for sid, info in _active_sessions.items():
        cm = info.get("cm")
        m = cm.metrics.summary() if cm else {}
        m["session_id"] = sid
        m["uptime_s"] = round(time.time() - info["started"], 1)
        m["state"] = cm.state.value if cm else "unknown"
        sessions.append(m)
    return {
        "total_sessions": _total_sessions,
        "active_sessions": len(_active_sessions),
        "total_errors": _total_errors,
        "sessions": sessions,
    }


@app.get("/api/rag/docs", dependencies=[Depends(verify_api_key)])
async def rag_docs():
    rag = engine.get("rag")
    if not rag:
        return {"docs": []}
    return {"docs": [{"id": d.get("id"), "question": d.get("question"),
                       "answer": d.get("answer", "")[:100]} for d in rag._documents]}


@app.get("/api/rag/query", dependencies=[Depends(verify_api_key)])
async def rag_query(q: str):
    rag = engine.get("rag")
    if not rag:
        return {"error": "RAG not loaded"}
    return rag.query(q)


@app.post("/api/rag/reload", dependencies=[Depends(verify_api_key)])
async def rag_reload():
    rag = engine.get("rag")
    if not rag or not os.path.exists(KB_DATA_PATH):
        return {"error": "No KB file"}
    with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {"status": "ok", **rag.build_index(docs)}


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket, token: str = Query(default="")):
    if not _check_api_key(token):
        await ws.close(code=4003, reason="Invalid API key")
        return
    await ws.accept()
    session_id = f"s-{int(time.time() * 1000)}"
    slog = logging.LoggerAdapter(log, {"session": session_id})
    slog.info("Connected")

    from engine.conversation_manager import ConversationManager

    vad_type = engine.get("vad_type", "silero")
    if vad_type == "firered":
        from engine.firered_vad import FireRedVADWrapper
        vad = FireRedVADWrapper(FIRERED_VAD_DIR, threshold=0.5)
        vad.load()
    elif vad_type == "speaker":
        from engine.speaker_vad import SpeakerAwareVAD
        vad = SpeakerAwareVAD(VAD_MODEL_DIR, SPEAKER_MODEL_DIR, threshold=0.5)
        vad.load(device="cpu")
    else:
        from engine.vad import SileroVAD
        vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
        vad.load()

    llm = engine["llm_factory"]()

    _ws_dead = False

    async def send_fn(data):
        nonlocal _ws_dead
        if _ws_dead:
            return
        try:
            if "_audio" in data:
                await ws.send_bytes(data["_audio"])
            else:
                await ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            if not _ws_dead:
                _ws_dead = True
                slog.warning("WebSocket send failed (marking dead): %s", e)

    exp_cfg = {
        "greeting_text": GREETING_TEXT,
        "idle_timeout_s": IDLE_TIMEOUT_S,
        "idle_goodbye_s": IDLE_GOODBYE_S,
        "idle_prompt_text": IDLE_PROMPT_TEXT,
        "idle_goodbye_text": IDLE_GOODBYE_TEXT,
        "asr_empty_retry_text": ASR_EMPTY_RETRY_TEXT,
        "asr_noisy_suggest_text": ASR_NOISY_SUGGEST_TEXT,
        "max_consecutive_empty_asr": MAX_CONSECUTIVE_EMPTY_ASR,
        "farewell_keywords": FAREWELL_KEYWORDS,
    }

    cm = ConversationManager(
        asr=engine["asr"], llm=llm, tts=engine["tts"],
        rag=engine["rag"], vad=vad, filler=engine["filler"],
        send_fn=send_fn,
        captioner=engine.get("captioner"),
        spec_asr=engine.get("spec_asr"),
        turn_detector=engine.get("turn_detector"),
        denoiser=engine.get("denoiser"),
        experience_config=exp_cfg,
    )

    global _total_sessions, _total_errors
    _total_sessions += 1
    _active_sessions[session_id] = {"started": time.time(), "cm": cm}

    await send_fn({"type": "ready", "session_id": session_id, "version": "2.9"})

    if GREETING_TEXT:
        cm._track_task(cm.start_greeting(), name="greeting")

    try:
        while True:
            if _ws_dead:
                slog.info("send_fn dead, exiting loop")
                break

            data = await ws.receive()

            if "bytes" in data and data["bytes"]:
                raw = data["bytes"]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                await cm.feed_audio(samples)

            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if msg.get("type") == "text_input":
                    await cm.handle_text_input(msg["text"])
                elif msg.get("type") == "manual_barge_in":
                    await cm.handle_barge_in()
                elif msg.get("type") == "reset":
                    cm.reset()
                    await send_fn({"type": "reset_ack"})

    except WebSocketDisconnect:
        slog.info("Disconnected (turn %d)", cm._turn)
    except Exception as e:
        slog.error("Error: %s", e, exc_info=True)
    finally:
        metrics = cm.metrics.summary()
        _total_errors += metrics.get("errors", 0)
        slog.info("Session metrics: %s", metrics)
        _active_sessions.pop(session_id, None)
        cm.shutdown()


if __name__ == "__main__":
    load_engines()

    port = int(os.environ.get("PORT", "3000"))
    use_ssl = os.environ.get("USE_SSL", "0") == "1"

    if use_ssl:
        cert_dir = BASE_DIR / "certs"
        log.info("Starting on https://0.0.0.0:%d (self-signed SSL)", port)
        uvicorn.run(app, host="0.0.0.0", port=port,
                    ssl_certfile=str(cert_dir / "cert.pem"),
                    ssl_keyfile=str(cert_dir / "key.pem"),
                    log_level="info")
    else:
        log.info("Starting on http://0.0.0.0:%d (SSL via Nginx reverse proxy)", port)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
