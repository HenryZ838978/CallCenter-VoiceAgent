"""VoxCPM TTS HTTP Server — standalone process, avoids fork-in-fork CUDA issue.

Usage:
    CUDA_VISIBLE_DEVICES=2 python tts_server.py --port 8200

Agent Worker calls: POST http://localhost:8200/synthesize {"text": "你好"}
Returns: binary PCM int16 44.1kHz audio
"""
import os
import sys
import time
import logging
import argparse
import asyncio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from config import TTS_MODEL_DIR, VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tts_server")

app = FastAPI(title="VoxCPM TTS Server")
engine = {}


class SynthRequest(BaseModel):
    text: str


def _do_synthesize(text: str) -> tuple:
    """Run TTS in a thread — nanovllm uses its own event loop internally."""
    t0 = time.perf_counter()
    tts = engine["tts"]
    result = tts.synthesize(text)
    audio = result["audio"]
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    ms = (time.perf_counter() - t0) * 1000
    return audio, ms


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    loop = asyncio.get_event_loop()
    audio, ms = await loop.run_in_executor(None, _do_synthesize, req.text)
    log.info("Synthesized '%s' → %d samples (%.0fms)", req.text[:30], len(audio), ms)
    return Response(content=audio.tobytes(), media_type="application/octet-stream",
                    headers={"X-TTS-Latency-Ms": str(round(ms)), "X-Sample-Rate": "44100"})


@app.get("/health")
async def health():
    return {"status": "ok", "model": "VoxCPM 1.5", "sample_rate": 44100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu-util", type=float, default=0.55)
    args = parser.parse_args()

    from engine.tts import VoxCPMTTS

    log.info("Loading VoxCPM TTS on %s (util=%.2f)...", args.device, args.gpu_util)
    tts = VoxCPMTTS(TTS_MODEL_DIR, device=args.device, gpu_memory_utilization=args.gpu_util)
    tts.load()

    if os.path.exists(VOICE_PROMPT_WAV):
        pid = tts.register_voice(VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT)
        tts.set_default_voice(pid)
        log.info("Voice clone registered")

    log.info("Warming up TTS...")
    tts.warmup()
    engine["tts"] = tts
    log.info("TTS server ready on :%d", args.port)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
