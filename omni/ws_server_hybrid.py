"""
Hybrid Voice Agent v3 — Stability hardened with voiceagent patterns

Improvements over v2:
  1. Interrupted text aggregation — barge-in context preserved, not discarded
  2. LLM streaming in thread + Queue decouple — event loop never blocked by LLM
  3. Speaker-Aware VAD (ECAPA-TDNN) — distinguishes user from TTS echo
  4. THINKING state buffers late speech — user additions merged into pipeline
"""
import os
import sys
import re
import json
import time
import base64
import asyncio
import logging
import threading
import numpy as np
from enum import Enum

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

VOICEAGENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voiceagent")
sys.path.insert(0, VOICEAGENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
import uvicorn
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hybrid")

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8200")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "MiniCPM-o-4.5-awq")
TTS_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "VoxCPM1.5")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")
VAD_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "snakers4_silero-vad")
SPEAKER_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "spkrec-ecapa-voxceleb")
USE_SPEAKER_VAD = os.environ.get("USE_SPEAKER_VAD", "1") == "1"
VOICE_PROMPT_WAV = os.path.join(VOICEAGENT_DIR, "data", "voice_prompt.wav")
VOICE_PROMPT_TEXT = "在国内引起了非常大的反响啊，我们也完全没有想到我们的这个工作会以这种方式出圈。"

EMBED_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "bge-small-zh-v1.5")
KB_DATA_PATH = os.path.join(VOICEAGENT_DIR, "data", "sample_kb.json")
RAG_DEVICE = os.environ.get("RAG_DEVICE", "cpu")
RAG_TOP_K = 3
MAX_HISTORY_TURNS = 8

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 44100
CHUNK_SAMPLES = 512
CHUNK_MS = 32
PORT = int(os.environ.get("PORT", "3001"))

TTS_SEND_CHUNK_BYTES = 8820 * 2

BARGE_IN_VAD_THRESHOLD = 0.85
BARGE_IN_RMS_THRESHOLD = 0.015
BARGE_IN_CONFIRM_CHUNKS = 3
INTERRUPT_SILENCE_CHUNKS = 10

ENDPOINT_FAST_CHUNKS = 6
ENDPOINT_DEFAULT_CHUNKS = 10
ENDPOINT_SLOW_CHUNKS = 20

SYSTEM_PROMPT_BASE = (
    "你是面壁智能的专业客服代表。只使用中文普通话。"
    "语气友好、专业、沉稳，像资深客服人员。"
    "回答简洁准确，每次1-2句话。不要用编号和列举。"
    "不要使用任何XML标签。不要复述用户的话。直接回答问题。"
)

SYSTEM_PROMPT_RAG = (
    "{base}\n\n以下是知识库，优先根据知识库内容回答。如果知识库没有相关信息，坦诚告知。"
    "\n\n知识库：\n{context}"
)

SENTENCE_RE = re.compile(r'([^。？！?!.]+[。？！?!.])')
THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
TAG_RE = re.compile(r'<[^>]+>')
HEARD_RE = re.compile(r'\[听到[】\]][^\n]*[\n]?')


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


engine = {}
metrics_history = []


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2)))


def clean_text(text):
    text = THINK_RE.sub('', text)
    text = TAG_RE.sub('', text)
    text = HEARD_RE.sub('', text)
    return text.strip()


async def _async_tts_stream(text: str, ws: WebSocket, cancel: asyncio.Event, turn: int):
    """Streaming TTS: generate and send each chunk immediately.
    Returns (ttfa_ms, chunk_count). Checks cancel between every chunk.
    """
    tts_engine = engine["tts_engine"]
    pid = engine.get("tts_prompt_id")
    kwargs = {"target_text": text, "temperature": 0.9, "cfg_value": 3.0}
    if pid:
        kwargs["prompt_id"] = pid

    t0 = time.perf_counter()
    ttfa = None
    chunk_idx = 0
    sent_count = 0
    prev_tail = None

    async for audio_chunk in tts_engine.generate(**kwargs):
        await asyncio.sleep(0)
        if cancel.is_set():
            continue  # must drain nanovllm generator fully

        if isinstance(audio_chunk, np.ndarray):
            c = audio_chunk
        elif hasattr(audio_chunk, 'numpy'):
            c = audio_chunk.numpy()
        else:
            c = np.array(audio_chunk, dtype=np.float32)

        chunk_idx += 1
        if chunk_idx == 1:
            ttfa = (time.perf_counter() - t0) * 1000
            continue  # skip VAE startup transient

        if chunk_idx == 2:
            fade = int(SAMPLE_RATE_OUT * 0.01)
            if len(c) > fade:
                c[:fade] *= np.linspace(0, 1, fade, dtype=c.dtype)

        # Cross-fade with previous chunk tail
        if prev_tail is not None and len(c) > len(prev_tail):
            c[:len(prev_tail)] = c[:len(prev_tail)] * 0.5 + prev_tail * 0.5

        xfade = min(441, len(c))
        prev_tail = c[-xfade:].copy() * np.linspace(1, 0, xfade, dtype=np.float32)
        c[-xfade:] *= np.linspace(1, 0.7, xfade, dtype=np.float32)

        if cancel.is_set():
            continue

        c16 = (c * 32767).clip(-32768, 32767).astype(np.int16)
        try:
            await ws.send_bytes(c16.tobytes())
            sent_count += 1
        except Exception:
            break
        await asyncio.sleep(0.05)  # breathing room for event loop

    return ttfa or 0, sent_count


def _sync_llm_stream(audio_b64: str, history: list, system_prompt: str,
                     sentence_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop,
                     cancel: asyncio.Event):
    """Run LLM streaming in a thread. Produces sentences into an asyncio.Queue."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}}
    ]})

    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": True,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    full_text = ""
    sentence_buf = ""
    llm_ttft = None
    t0 = time.perf_counter()

    try:
        with httpx.Client(timeout=60) as client:
            with client.stream("POST", f"{VLLM_BASE}/v1/chat/completions", json=payload) as resp:
                for line in resp.iter_lines():
                    if cancel.is_set():
                        break
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if not delta:
                        continue

                    if llm_ttft is None:
                        llm_ttft = (time.perf_counter() - t0) * 1000

                    full_text += delta
                    sentence_buf += clean_text(delta)

                    for m in SENTENCE_RE.finditer(sentence_buf):
                        sent = m.group(0).strip()
                        if sent and len(sent) >= 2:
                            asyncio.run_coroutine_threadsafe(
                                sentence_queue.put({"sentence": sent, "ttft": llm_ttft}),
                                main_loop,
                            )
                    sentence_buf = SENTENCE_RE.sub('', sentence_buf)
    except Exception as e:
        log.error("LLM thread error: %s", e)

    remaining = clean_text(sentence_buf).strip()
    if remaining and len(remaining) >= 2 and not cancel.is_set():
        asyncio.run_coroutine_threadsafe(
            sentence_queue.put({"sentence": remaining, "ttft": llm_ttft}),
            main_loop,
        )

    asyncio.run_coroutine_threadsafe(
        sentence_queue.put({"_done": True, "full_text": full_text, "llm_ttft": llm_ttft}),
        main_loop,
    )


async def load_engines():
    from nanovllm_voxcpm import VoxCPM

    log.info("=== Loading Hybrid Agent v3 engines ===")

    if USE_SPEAKER_VAD and os.path.isdir(SPEAKER_MODEL_DIR):
        from engine.speaker_vad import SpeakerAwareVAD
        log.info("Loading SpeakerAwareVAD (Silero + ECAPA-TDNN)...")
        svad = SpeakerAwareVAD(VAD_MODEL_DIR, SPEAKER_MODEL_DIR, threshold=0.5)
        svad.load(device="cpu")
        engine["vad_template"] = svad
        engine["vad_type"] = "speaker"
    else:
        from engine.vad import SileroVAD
        log.info("Loading VAD (Silero)...")
        vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
        vad.load()
        engine["vad_template"] = vad
        engine["vad_type"] = "silero"

    log.info("Loading VoxCPM TTS on %s ...", TTS_DEVICE)
    gpu_idx = int(TTS_DEVICE.split(":")[-1]) if ":" in TTS_DEVICE else 0
    tts_mem = float(os.environ.get("TTS_GPU_MEM", "0.35"))
    tts_engine = VoxCPM.from_pretrained(
        model=TTS_MODEL_DIR, gpu_memory_utilization=tts_mem, devices=[gpu_idx],
    )
    engine["tts_engine"] = tts_engine
    engine["tts_prompt_id"] = None

    if os.path.exists(VOICE_PROMPT_WAV):
        log.info("Registering voice clone...")
        with open(VOICE_PROMPT_WAV, "rb") as f:
            pid = await tts_engine.add_prompt(f.read(), "wav", VOICE_PROMPT_TEXT)
        engine["tts_prompt_id"] = pid
        log.info("Voice clone: %s", pid)

    engine["main_loop"] = asyncio.get_event_loop()
    log.info("Warming up TTS...")
    tts_engine_ref = engine["tts_engine"]
    pid_ref = engine.get("tts_prompt_id")
    wkw = {"target_text": "你好，欢迎致电。", "temperature": 0.9, "cfg_value": 3.0}
    if pid_ref:
        wkw["prompt_id"] = pid_ref
    async for _ in tts_engine_ref.generate(**wkw):
        pass
    log.info("TTS warm.")

    log.info("Loading RAG...")
    if os.path.isdir(EMBED_MODEL_DIR) and os.path.exists(KB_DATA_PATH):
        import importlib.util
        spec = importlib.util.spec_from_file_location("rag", os.path.join(VOICEAGENT_DIR, "engine", "rag.py"))
        rag_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_mod)
        rag = rag_mod.RAGEngine(EMBED_MODEL_DIR, device=RAG_DEVICE, top_k=RAG_TOP_K)
        rag.load()
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        rag.build_index(docs)
        engine["rag"] = rag
    else:
        engine["rag"] = None

    log.info("vLLM: %s | VAD: %s | TTS: temp=0.9 cfg=3.0", VLLM_BASE, engine["vad_type"])
    log.info("=== All engines loaded (v3) ===")


app = FastAPI(title="Hybrid Voice Agent v3")
_samples_dir = Path(__file__).parent / "static" / "samples"
if _samples_dir.exists():
    app.mount("/samples", StaticFiles(directory=str(_samples_dir)), name="samples")


@app.on_event("startup")
async def startup():
    await load_engines()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text("utf-8"))


@app.get("/api/info")
async def api_info():
    return {
        "architecture": "hybrid v3: vLLM Omni AWQ + VoxCPM TTS + SpeakerAwareVAD",
        "vllm_model": VLLM_MODEL, "vad_type": engine.get("vad_type"),
        "tts": "VoxCPM 1.5 (temp=0.9, cfg=3.0)", "sample_rate_out": SAMPLE_RATE_OUT,
    }


@app.get("/api/metrics")
async def api_metrics():
    return {"history": metrics_history[-50:]}


async def send_audio_chunked(ws, audio, cancel):
    if audio.dtype != np.int16:
        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    audio_bytes = audio.tobytes()
    for i in range(0, len(audio_bytes), TTS_SEND_CHUNK_BYTES):
        if cancel.is_set():
            return
        try:
            await ws.send_bytes(audio_bytes[i:i + TTS_SEND_CHUNK_BYTES])
        except Exception:
            return
        await asyncio.sleep(0.05)


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await ws.accept()
    log.info("WS connected")

    if engine["vad_type"] == "speaker":
        from engine.speaker_vad import SpeakerAwareVAD
        vad = SpeakerAwareVAD(VAD_MODEL_DIR, SPEAKER_MODEL_DIR, threshold=0.5)
        vad.load(device="cpu")
    else:
        from engine.vad import SileroVAD
        vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
        vad.load()

    state = State.IDLE
    audio_buffer = []
    silence_count = 0
    turn = 0
    cancel = asyncio.Event()
    barge_confirm = 0
    speaking_task = None
    conversation_history = []
    interrupted_texts = []
    thinking_buffer = []
    thinking_has_speech = False

    async def send_json(d):
        try: await ws.send_json(d)
        except: pass

    async def set_state(s):
        nonlocal state; state = s
        await send_json({"type": "state", "state": s.value})

    def adaptive_ep():
        d = len(audio_buffer) * CHUNK_MS
        return ENDPOINT_FAST_CHUNKS if d < 500 else (ENDPOINT_SLOW_CHUNKS if d > 3000 else ENDPOINT_DEFAULT_CHUNKS)

    async def run_pipeline(audio_np, current_turn):
        nonlocal state, thinking_buffer, thinking_has_speech
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        await set_state(State.THINKING)
        await send_json({"type": "processing", "audio_duration_s": round(len(audio_np) / SAMPLE_RATE_IN, 2)})

        # Auto-enroll speaker on first turn
        if current_turn == 1 and hasattr(vad, 'enroll_speaker') and not vad.is_enrolled:
            if len(audio_np) >= 8000:
                vad.enroll_speaker(audio_np)
                log.info("[Turn %d] Speaker auto-enrolled", current_turn)

        import soundfile as sf, io
        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE_IN, format='WAV', subtype='PCM_16')
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        # Interrupted text aggregation
        user_context = ""
        if interrupted_texts:
            user_context = "用户之前说了：" + "；".join(interrupted_texts) + "。请综合回应。\n"
            interrupted_texts.clear()

        # RAG
        rag = engine.get("rag")
        rag_context = ""
        rag_ms = 0
        if rag and conversation_history:
            last_ai = next((m["content"] for m in reversed(conversation_history) if m["role"] == "assistant"), "")
            if last_ai:
                t_rag = time.perf_counter()
                rag_result = await loop.run_in_executor(None, rag.get_context, last_ai)
                rag_context = rag_result["context"]
                rag_ms = rag_result["total_ms"]

        if rag_context:
            sys_prompt = SYSTEM_PROMPT_RAG.format(base=SYSTEM_PROMPT_BASE, context=rag_context)
        else:
            sys_prompt = SYSTEM_PROMPT_BASE
        if user_context:
            sys_prompt = sys_prompt + "\n\n" + user_context

        # LLM streaming via thread + Queue
        sentence_queue = asyncio.Queue()
        cancel.clear()
        barge_confirm_local = 0

        llm_thread = threading.Thread(
            target=_sync_llm_stream,
            args=(audio_b64, conversation_history, sys_prompt, sentence_queue, loop, cancel),
            daemon=True,
        )
        llm_thread.start()

        tts_started = False
        tts_ttfa = None
        llm_ttft = None
        sentence_count = 0
        full_text = ""

        while True:
            if cancel.is_set():
                break
            try:
                item = await asyncio.wait_for(sentence_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if item.get("_done"):
                full_text = item.get("full_text", full_text)
                llm_ttft = item.get("llm_ttft", llm_ttft)
                break

            sent = item["sentence"]
            if llm_ttft is None:
                llm_ttft = item.get("ttft")

            if cancel.is_set():
                break

            if not tts_started:
                await set_state(State.SPEAKING)
                await send_json({"type": "audio_start", "sample_rate": SAMPLE_RATE_OUT, "turn": current_turn})
                tts_started = True

            sentence_count += 1
            full_text += sent
            chunk_ttfa, chunk_count = await _async_tts_stream(sent, ws, cancel, current_turn)
            if tts_ttfa is None and chunk_ttfa > 0:
                tts_ttfa = chunk_ttfa
            log.info("[Turn %d] TTS streamed: '%s' (%d chunks)", current_turn, sent[:25], chunk_count)

        # Merge late speech from THINKING
        if thinking_has_speech and thinking_buffer:
            log.info("[Turn %d] Late speech detected during THINKING (%d chunks)", current_turn, len(thinking_buffer))
        thinking_buffer.clear()
        thinking_has_speech = False

        total_ms = (time.perf_counter() - t_start) * 1000
        ai_response = clean_text(full_text).strip()

        if ai_response:
            conversation_history.append({"role": "user", "content": "(audio)"})
            conversation_history.append({"role": "assistant", "content": ai_response})
        while len(conversation_history) > MAX_HISTORY_TURNS * 2:
            conversation_history.pop(0)

        # If interrupted, save partial response for aggregation
        if cancel.is_set() and ai_response:
            interrupted_texts.append(ai_response)
            log.info("[Turn %d] Interrupted, saved for aggregation: '%s'", current_turn, ai_response[:30])

        m = {
            "llm_ttft_ms": round(llm_ttft or 0, 1),
            "tts_ttfa_ms": round(tts_ttfa or 0, 1),
            "rag_ms": round(rag_ms, 1),
            "total_ms": round(total_ms, 1),
            "first_audio_ms": round((llm_ttft or 0) + (tts_ttfa or 0), 1),
            "text": ai_response, "timestamp": time.time(),
        }
        metrics_history.append(m)

        await send_json({
            "type": "response", "text": ai_response,
            "llm_ttft_ms": m["llm_ttft_ms"], "tts_ttfa_ms": m["tts_ttfa_ms"],
            "rag_ms": m["rag_ms"], "total_ms": m["total_ms"],
            "first_audio_ms": m["first_audio_ms"],
            "history_len": len(conversation_history),
        })
        await send_json({"type": "audio_end", "total_ms": m["total_ms"]})

        if state in (State.SPEAKING, State.THINKING):
            await set_state(State.IDLE)

    async def safe_pipeline(audio_np, current_turn):
        try:
            await run_pipeline(audio_np, current_turn)
        except Exception as e:
            log.error("[Turn %d] Pipeline error: %s", current_turn, e, exc_info=True)
            await set_state(State.IDLE)

    try:
        await set_state(State.IDLE)
        while True:
            data = await ws.receive()
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                samples = np.frombuffer(data["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                for i in range(0, len(samples), CHUNK_SAMPLES):
                    chunk = samples[i:i + CHUNK_SAMPLES]
                    if len(chunk) < CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

                    vad_result = vad.process_chunk(chunk, SAMPLE_RATE_IN)
                    speech_prob = vad_result["speech_prob"]
                    is_speech = speech_prob >= 0.5
                    rms = _rms(chunk)
                    has_speaker = vad_result.get("is_target_speaker") is not None

                    if state == State.IDLE:
                        if is_speech and rms > 0.01:
                            audio_buffer.clear()
                            audio_buffer.append(chunk.copy())
                            silence_count = 0
                            await set_state(State.LISTENING)

                    elif state == State.LISTENING:
                        audio_buffer.append(chunk.copy())
                        if is_speech:
                            silence_count = 0
                        else:
                            silence_count += 1
                            if silence_count >= adaptive_ep():
                                if len(audio_buffer) >= 6:
                                    turn += 1
                                    full_audio = np.concatenate(audio_buffer)
                                    audio_buffer.clear()
                                    silence_count = 0
                                    cancel.clear()
                                    barge_confirm = 0
                                    speaking_task = asyncio.create_task(safe_pipeline(full_audio, turn))
                                else:
                                    audio_buffer.clear()
                                    silence_count = 0
                                    await set_state(State.IDLE)

                    elif state == State.SPEAKING:
                        if has_speaker:
                            is_real = vad_result.get("is_target_speaker", False) and rms > BARGE_IN_RMS_THRESHOLD
                        else:
                            is_real = speech_prob >= BARGE_IN_VAD_THRESHOLD and rms > BARGE_IN_RMS_THRESHOLD
                        if is_real:
                            barge_confirm += 1
                            if barge_confirm >= BARGE_IN_CONFIRM_CHUNKS:
                                sim = vad_result.get("speaker_similarity", 0)
                                log.info("Barge-in confirmed (prob=%.2f, rms=%.3f, spk_sim=%.2f)",
                                         speech_prob, rms, sim)
                                cancel.set()
                                barge_confirm = 0
                                audio_buffer.clear()
                                audio_buffer.append(chunk.copy())
                                silence_count = 0
                                await send_json({"type": "barge_in", "turn": turn})
                                await set_state(State.INTERRUPTED)
                        else:
                            barge_confirm = 0

                    elif state == State.INTERRUPTED:
                        audio_buffer.append(chunk.copy())
                        if is_speech:
                            silence_count = 0
                        else:
                            silence_count += 1
                            if silence_count >= INTERRUPT_SILENCE_CHUNKS:
                                if len(audio_buffer) >= 6:
                                    turn += 1
                                    full_audio = np.concatenate(audio_buffer)
                                    audio_buffer.clear()
                                    silence_count = 0
                                    cancel.clear()
                                    barge_confirm = 0
                                    speaking_task = asyncio.create_task(safe_pipeline(full_audio, turn))
                                else:
                                    audio_buffer.clear()
                                    silence_count = 0
                                    await set_state(State.IDLE)

                    elif state == State.THINKING:
                        if is_speech and rms > 0.01:
                            thinking_buffer.append(chunk.copy())
                            thinking_has_speech = True
                        elif thinking_has_speech:
                            thinking_buffer.append(chunk.copy())

            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "reset":
                    cancel.set()
                    vad.reset()
                    audio_buffer.clear()
                    conversation_history.clear()
                    interrupted_texts.clear()
                    thinking_buffer.clear()
                    thinking_has_speech = False
                    silence_count = 0
                    barge_confirm = 0
                    turn = 0
                    await set_state(State.IDLE)
                    log.info("Session reset")

    except WebSocketDisconnect:
        log.info("Disconnected")
    except Exception as e:
        log.error("WS error: %s", e, exc_info=True)
    finally:
        cancel.set()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
