"""
Omni Voice Agent -- WebSocket server using MiniCPM-o-4.5

Single-model end-to-end: audio in (16kHz) -> audio out (24kHz)
No separate ASR / LLM / TTS.
KB is embedded in the system prompt for zero-latency retrieval.
"""
import os
import sys
import json
import time
import asyncio
import logging
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from config import (
    VAD_MODEL_DIR, OMNI_MODEL_DIR, OMNI_DEVICE, LOAD_IN_4BIT,
    VOICE_PROMPT_WAV, SYSTEM_PROMPT,
    SAMPLE_RATE_IN, SAMPLE_RATE_OUT, CHUNK_SAMPLES,
    HOST, PORT, VAD_THRESHOLD, MIN_SPEECH_CHUNKS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("omni_agent")

engine = {}
metrics_history = []
MAX_METRICS = 50

_omni_lock = threading.Lock()


def load_engines():
    from engine.vad import SileroVAD
    from engine.omni import OmniEngine

    log.info("=== Loading Omni Agent engines ===")

    log.info("Loading VAD (Silero) ...")
    vad = SileroVAD(VAD_MODEL_DIR, threshold=VAD_THRESHOLD)
    vad.load()
    engine["vad_template"] = vad

    log.info("Loading Omni (MiniCPM-o-4.5) on %s (4bit=%s) ...", OMNI_DEVICE, LOAD_IN_4BIT)
    omni = OmniEngine(OMNI_MODEL_DIR, device=OMNI_DEVICE, load_in_4bit=LOAD_IN_4BIT)
    omni.load()

    if os.path.exists(VOICE_PROMPT_WAV):
        log.info("Setting voice reference: %s", VOICE_PROMPT_WAV)
        omni.set_voice(VOICE_PROMPT_WAV)

    omni.warmup()
    engine["omni"] = omni

    log.info("=== All engines loaded ===")


app = FastAPI(title="Omni Voice Agent (MiniCPM-o-4.5)")


@app.on_event("startup")
async def startup():
    load_engines()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/info")
async def api_info():
    omni = engine.get("omni")
    return {
        "model": "MiniCPM-o-4.5",
        "model_dir": OMNI_MODEL_DIR,
        "device": OMNI_DEVICE,
        "load_in_4bit": LOAD_IN_4BIT,
        "sample_rate_in": SAMPLE_RATE_IN,
        "sample_rate_out": SAMPLE_RATE_OUT,
        "voice_ref": VOICE_PROMPT_WAV if omni and omni.ref_audio is not None else None,
        "architecture": "end-to-end omni (no separate ASR/LLM/TTS)",
    }


@app.get("/api/metrics")
async def api_metrics():
    return {"history": metrics_history[-MAX_METRICS:]}


STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    from engine.vad import SileroVAD
    vad = SileroVAD(VAD_MODEL_DIR, threshold=VAD_THRESHOLD)
    vad.load()

    state = STATE_IDLE
    audio_buffer = []
    speech_chunk_count = 0
    cancel_event = asyncio.Event()
    speaking_task = None

    async def send_json_safe(data):
        try:
            await ws.send_json(data)
        except Exception:
            pass

    async def set_state(s):
        nonlocal state
        state = s
        await send_json_safe({"type": "state", "state": s})

    async def run_omni(audio_np):
        nonlocal state
        import torch

        omni_eng = engine["omni"]
        t_start = time.perf_counter()
        ttfa = None
        text_acc = ""
        chunk_count = 0

        await set_state(STATE_THINKING)
        await send_json_safe({
            "type": "processing",
            "audio_duration_s": round(len(audio_np) / SAMPLE_RATE_IN, 2),
        })

        def _do_generate():
            with _omni_lock:
                return list(omni_eng.stream_chat(audio_np, SYSTEM_PROMPT))

        loop = asyncio.get_event_loop()
        try:
            gen_results = await loop.run_in_executor(None, _do_generate)
        except Exception as exc:
            log.error("Omni generation failed: %s", exc, exc_info=True)
            await set_state(STATE_IDLE)
            return

        if cancel_event.is_set():
            await set_state(STATE_IDLE)
            return

        await set_state(STATE_SPEAKING)
        await send_json_safe({"type": "audio_start", "sample_rate": SAMPLE_RATE_OUT})

        for wav_chunk, text_chunk in gen_results:
            if cancel_event.is_set():
                break

            if ttfa is None:
                ttfa = (time.perf_counter() - t_start) * 1000

            text_acc += text_chunk
            chunk_count += 1

            if wav_chunk is not None:
                try:
                    chunk_np = wav_chunk[0].cpu().float().numpy()
                    chunk_int16 = (chunk_np * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_bytes(chunk_int16.tobytes())
                except Exception:
                    break

        total_ms = (time.perf_counter() - t_start) * 1000

        m = {
            "ttfa_ms": round(ttfa or 0, 1),
            "total_ms": round(total_ms, 1),
            "text": text_acc,
            "chunks": chunk_count,
            "timestamp": time.time(),
        }
        metrics_history.append(m)

        await send_json_safe({
            "type": "response",
            "text": text_acc,
            "ttfa_ms": m["ttfa_ms"],
            "total_ms": m["total_ms"],
            "chunks": chunk_count,
        })
        await send_json_safe({"type": "audio_end", "total_ms": m["total_ms"]})

        if not cancel_event.is_set():
            await set_state(STATE_IDLE)

    try:
        await set_state(STATE_IDLE)

        while True:
            data = await ws.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                raw = data["bytes"]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                for i in range(0, len(samples), CHUNK_SAMPLES):
                    chunk = samples[i:i + CHUNK_SAMPLES]
                    if len(chunk) < CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

                    ev = vad.process_chunk(chunk)

                    if ev["speech_start"]:
                        if state == STATE_SPEAKING:
                            log.info("Barge-in detected")
                            cancel_event.set()
                            if speaking_task and not speaking_task.done():
                                speaking_task.cancel()
                            await send_json_safe({"type": "barge_in"})

                        audio_buffer.clear()
                        speech_chunk_count = 0
                        await set_state(STATE_LISTENING)

                    if state == STATE_LISTENING:
                        audio_buffer.append(chunk.copy())
                        speech_chunk_count += 1

                    if ev["speech_end"] and state == STATE_LISTENING:
                        if speech_chunk_count >= MIN_SPEECH_CHUNKS:
                            full_audio = np.concatenate(audio_buffer)
                            audio_buffer.clear()
                            speech_chunk_count = 0
                            cancel_event.clear()
                            speaking_task = asyncio.create_task(run_omni(full_audio))
                        else:
                            audio_buffer.clear()
                            speech_chunk_count = 0
                            await set_state(STATE_IDLE)

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "reset":
                    vad.reset()
                    audio_buffer.clear()
                    speech_chunk_count = 0
                    cancel_event.set()
                    await set_state(STATE_IDLE)
                    log.info("Session reset")

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.error("WebSocket error: %s", e, exc_info=True)
    finally:
        cancel_event.set()


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
