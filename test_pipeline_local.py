"""Local pipeline self-test: verify each component without browser/WebRTC.

Tests:
  1. TTS: HTTP API → PCM audio bytes
  2. STT: WAV file → transcription text
  3. LLM: prompt → streaming response
  4. Full chain: WAV → STT → LLM → TTS → output audio

Usage:
    CUDA_VISIBLE_DEVICES=2,3 ASR_DEVICE=cuda:1 python test_pipeline_local.py
"""
import os, sys, time, asyncio, struct, wave, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import httpx
from openai import AsyncOpenAI

TTS_URL = os.environ.get("TTS_SERVER_URL", "http://localhost:8200")
LLM_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("VLLM_MODEL", "MiniCPM4.1-8B-GPTQ")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda:1")
WAV_PATH = os.path.join(os.path.dirname(__file__), "static", "livekit", "test_audio", "hello.wav")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = {}


def section(name):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")


async def test_tts():
    section("TTS (VoxCPM HTTP API)")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            t0 = time.time()
            resp = await client.post(f"{TTS_URL}/synthesize", json={"text": "你好，我是语音助手"})
            latency_ms = (time.time() - t0) * 1000
            resp.raise_for_status()
            audio_bytes = resp.content
            duration_s = len(audio_bytes) / (44100 * 2)
            print(f"  Status: {resp.status_code}")
            print(f"  Audio: {len(audio_bytes)} bytes ({duration_s:.1f}s @ 44100Hz 16bit)")
            print(f"  Latency: {latency_ms:.0f}ms")
            if len(audio_bytes) > 1000:
                print(f"  Result: {PASS}")
                results['tts'] = True
                return audio_bytes
            else:
                print(f"  Result: {FAIL} — audio too short")
                results['tts'] = False
                return None
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        results['tts'] = False
        return None


async def test_llm():
    section("LLM (vLLM OpenAI API)")
    try:
        client = AsyncOpenAI(base_url=LLM_URL, api_key="dummy")
        t0 = time.time()
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个智能客服助手，用简短中文回答。"},
                {"role": "user", "content": "你好"},
            ],
            max_tokens=80, temperature=0.85, stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.15},
        )
        first_token_ms = None
        full_text = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                if first_token_ms is None:
                    first_token_ms = (time.time() - t0) * 1000
                full_text += token
        total_ms = (time.time() - t0) * 1000
        print(f"  First token: {first_token_ms:.0f}ms")
        print(f"  Total: {total_ms:.0f}ms")
        print(f"  Response: '{full_text.strip()[:100]}'")
        if full_text.strip():
            print(f"  Result: {PASS}")
            results['llm'] = True
            return full_text.strip()
        else:
            print(f"  Result: {FAIL} — empty response")
            results['llm'] = False
            return None
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        results['llm'] = False
        return None


def test_stt():
    section("STT (FireRedASR2-AED)")
    try:
        from engine.asr_firered import FireRedASR
        model_dir = os.path.join(os.path.dirname(__file__), "models", "FireRedASR2-AED")
        print(f"  Loading ASR model from {model_dir} on {ASR_DEVICE}...")
        t0 = time.time()
        asr = FireRedASR(model_dir, device=ASR_DEVICE)
        asr.load()
        load_ms = (time.time() - t0) * 1000
        print(f"  Model loaded in {load_ms:.0f}ms")

        if not os.path.exists(WAV_PATH):
            print(f"  Result: {FAIL} — WAV file not found: {WAV_PATH}")
            results['stt'] = False
            return None, None

        with wave.open(WAV_PATH, 'rb') as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            print(f"  WAV: {sr}Hz, {n_ch}ch, {sw*8}bit, {len(frames)} bytes")

        audio_i16 = np.frombuffer(frames, dtype=np.int16)
        if n_ch > 1:
            audio_i16 = audio_i16[::n_ch]
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        if sr != 16000:
            ratio = sr / 16000
            n_out = int(len(audio_f32) / ratio)
            indices = np.linspace(0, len(audio_f32) - 1, n_out).astype(int)
            audio_f32 = audio_f32[indices]
            print(f"  Resampled {sr}→16000Hz ({len(audio_f32)} samples)")

        t0 = time.time()
        result = asr.transcribe(audio_f32)
        stt_ms = (time.time() - t0) * 1000
        text = result.get("text", "").strip()
        print(f"  Transcription: '{text}'")
        print(f"  Latency: {stt_ms:.0f}ms")
        if text:
            print(f"  Result: {PASS}")
            results['stt'] = True
        else:
            print(f"  Result: {FAIL} — empty transcription")
            results['stt'] = False
        return asr, text
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        import traceback; traceback.print_exc()
        results['stt'] = False
        return None, None


async def test_full_chain(asr):
    section("FULL CHAIN: WAV → STT → LLM → TTS")
    try:
        if not os.path.exists(WAV_PATH):
            print(f"  {FAIL} — WAV not found")
            results['chain'] = False
            return

        with wave.open(WAV_PATH, 'rb') as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        audio_i16 = np.frombuffer(frames, dtype=np.int16)
        if n_ch > 1:
            audio_i16 = audio_i16[::n_ch]
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        if sr != 16000:
            ratio = sr / 16000
            n_out = int(len(audio_f32) / ratio)
            indices = np.linspace(0, len(audio_f32) - 1, n_out).astype(int)
            audio_f32 = audio_f32[indices]

        chain_t0 = time.time()

        # STT
        t0 = time.time()
        stt_result = asr.transcribe(audio_f32)
        stt_text = stt_result.get("text", "").strip()
        stt_ms = (time.time() - t0) * 1000
        print(f"  [STT] '{stt_text}' ({stt_ms:.0f}ms)")
        if not stt_text:
            print(f"  {FAIL} — STT returned empty")
            results['chain'] = False
            return

        # LLM
        t0 = time.time()
        client = AsyncOpenAI(base_url=LLM_URL, api_key="dummy")
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个智能客服助手，用简短中文回答。"},
                {"role": "user", "content": stt_text},
            ],
            max_tokens=80, temperature=0.85, stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.15},
        )
        llm_text = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                llm_text += chunk.choices[0].delta.content
        llm_ms = (time.time() - t0) * 1000
        llm_text = llm_text.strip()
        print(f"  [LLM] '{llm_text[:80]}' ({llm_ms:.0f}ms)")
        if not llm_text:
            print(f"  {FAIL} — LLM returned empty")
            results['chain'] = False
            return

        # TTS
        t0 = time.time()
        async with httpx.AsyncClient(timeout=30.0) as hc:
            resp = await hc.post(f"{TTS_URL}/synthesize", json={"text": llm_text})
            resp.raise_for_status()
            audio_bytes = resp.content
        tts_ms = (time.time() - t0) * 1000
        duration_s = len(audio_bytes) / (44100 * 2)
        print(f"  [TTS] {len(audio_bytes)} bytes ({duration_s:.1f}s) ({tts_ms:.0f}ms)")

        total_ms = (time.time() - chain_t0) * 1000
        print(f"\n  Total chain latency: {total_ms:.0f}ms")
        print(f"    STT={stt_ms:.0f}ms + LLM={llm_ms:.0f}ms + TTS={tts_ms:.0f}ms")

        if len(audio_bytes) > 1000:
            print(f"  Result: {PASS}")
            results['chain'] = True
        else:
            print(f"  Result: {FAIL} — TTS output too short")
            results['chain'] = False
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        import traceback; traceback.print_exc()
        results['chain'] = False


async def main():
    print("=" * 60)
    print("  VoxLabs Pipeline Local Self-Test")
    print("=" * 60)

    tts_audio = await test_tts()
    llm_response = await test_llm()
    asr, stt_text = test_stt()
    if asr:
        await test_full_chain(asr)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {name:12s} {status}")
    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
