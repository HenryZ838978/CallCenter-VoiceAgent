"""
VoxLabs Voice Agent v2.0 — Production Full-Duplex WebSocket Server

Architecture:
  ConversationManager (state machine) orchestrates:
    Streaming ASR → RAG → LLM(sentence streaming) → TTS(per-sentence) → Audio playback
  With:
    Filler words for instant perceived response
    Barge-in with utterance aggregation
    Echo suppression via state signaling
"""
import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
import uvicorn
import nest_asyncio
nest_asyncio.apply()
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASR_MODEL_DIR, TTS_MODEL_DIR, VAD_MODEL_DIR,
    VLLM_BASE_URL, VLLM_MODEL_NAME,
    EMBED_MODEL_DIR, KB_DATA_PATH, RAG_TOP_K,
    SYSTEM_PROMPT_RAG, VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT,
)
CAPTIONER_URL = os.environ.get("CAPTIONER_URL", "")
CAPTIONER_MODEL = os.environ.get("CAPTIONER_MODEL", "MiniCPM-o-4.5-awq")
USE_SPEAKER_VAD = os.environ.get("USE_SPEAKER_VAD", "0") == "1"
USE_FIRERED_VAD = os.environ.get("USE_FIRERED_VAD", "0") == "1"
USE_MOONSHINE_ASR = os.environ.get("USE_MOONSHINE_ASR", "0") == "1"
USE_FIRERED_ASR = os.environ.get("USE_FIRERED_ASR", "0") == "1"
USE_SMART_TURN = os.environ.get("USE_SMART_TURN", "0") == "1"
SPEAKER_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "spkrec-ecapa-voxceleb")
FIRERED_VAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "FireRedVAD-stream")
FIRERED_ASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "FireRedASR2-AED")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

    log.info("Loading RAG (bge-small-zh-v1.5)...")
    rag = RAGEngine(EMBED_MODEL_DIR, device=RAG_DEVICE, top_k=RAG_TOP_K)
    rag.load()
    if os.path.exists(KB_DATA_PATH):
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        info = rag.build_index(docs)
        log.info("RAG index: %d docs, dim=%d", info["num_docs"], info["dim"])
    engine["rag"] = rag

    log.info("LLM client ready (vLLM @ %s)", VLLM_BASE_URL)
    engine["llm_factory"] = lambda: VLLMChat(
        base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME,
        system_prompt=SYSTEM_PROMPT_RAG,
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

    log.info("=== All engines loaded (v2.2) ===")


app = FastAPI(title="VoxLabs Voice Agent v2.0")
BASE_DIR = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((BASE_DIR / "static" / "voice_agent.html").read_text(encoding="utf-8"))


@app.get("/api/info")
async def api_info():
    rag = engine.get("rag")
    return {
        "version": "2.0",
        "models": {
            "asr": {"name": "SenseVoiceSmall", "framework": "FunASR"},
            "llm": {"name": VLLM_MODEL_NAME, "endpoint": VLLM_BASE_URL, "streaming": True},
            "tts": {"name": "VoxCPM 1.5", "framework": "nanovllm", "sample_rate": 44100},
            "embedding": {"name": "bge-small-zh-v1.5"},
        },
        "features": ["filler_words", "sentence_streaming", "barge_in", "utterance_aggregation"],
        "rag": {"num_docs": len(rag._documents) if rag else 0, "top_k": RAG_TOP_K},
    }


@app.get("/api/metrics")
async def api_metrics():
    return {"message": "Metrics available per-session via WebSocket"}


@app.get("/api/rag/docs")
async def rag_docs():
    rag = engine.get("rag")
    if not rag:
        return {"docs": []}
    return {"docs": [{"id": d.get("id"), "question": d.get("question"),
                       "answer": d.get("answer", "")[:100]} for d in rag._documents]}


@app.get("/api/rag/query")
async def rag_query(q: str):
    rag = engine.get("rag")
    if not rag:
        return {"error": "RAG not loaded"}
    return rag.query(q)


@app.post("/api/rag/reload")
async def rag_reload():
    rag = engine.get("rag")
    if not rag or not os.path.exists(KB_DATA_PATH):
        return {"error": "No KB file"}
    with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {"status": "ok", **rag.build_index(docs)}


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    session_id = f"s-{int(time.time() * 1000)}"
    log.info("[%s] Connected", session_id)

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

    async def send_fn(data):
        try:
            if "_audio" in data:
                await ws.send_bytes(data["_audio"])
            else:
                await ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    cm = ConversationManager(
        asr=engine["asr"], llm=llm, tts=engine["tts"],
        rag=engine["rag"], vad=vad, filler=engine["filler"],
        send_fn=send_fn,
        captioner=engine.get("captioner"),
        spec_asr=engine.get("spec_asr"),
        turn_detector=engine.get("turn_detector"),
    )

    await send_fn({"type": "ready", "session_id": session_id, "version": "2.0"})

    try:
        while True:
            data = await ws.receive()

            if "bytes" in data and data["bytes"]:
                raw = data["bytes"]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                await cm.feed_audio(samples)

            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if msg.get("type") == "text_input":
                    await cm.handle_text_input(msg["text"])
                elif msg.get("type") == "reset":
                    cm.reset()
                    await send_fn({"type": "reset_ack"})

    except WebSocketDisconnect:
        log.info("[%s] Disconnected (turn %d)", session_id, cm._turn)
    except Exception as e:
        log.error("[%s] Error: %s", session_id, e, exc_info=True)
    finally:
        cm.reset()


if __name__ == "__main__":
    load_engines()

    cert_dir = BASE_DIR / "certs"
    port = int(os.environ.get("PORT", "3000"))
    log.info("Starting VoxLabs v2.0 on https://0.0.0.0:%d", port)

    uvicorn.run(app, host="0.0.0.0", port=port,
                ssl_certfile=str(cert_dir / "cert.pem"),
                ssl_keyfile=str(cert_dir / "key.pem"),
                log_level="info")
