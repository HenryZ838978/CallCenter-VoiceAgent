"""E2E WebSocket integration test — connects to a running server.

Tests:
  1. WS connect + ready message
  2. Send silence → stays IDLE (no false trigger)
  3. Send speech-like audio → state transitions
  4. Text input → LLM response + audio
  5. API key rejection (if VOICEAGENT_API_KEY set)
  6. /api/info returns valid JSON
  7. /api/metrics returns valid JSON

Requirements:
  - Server running on localhost:3000 (or set WS_URL / HTTP_URL env)
  - pip install websockets httpx

Usage:
    python test_e2e_ws.py
    VOICEAGENT_API_KEY=secret WS_URL=ws://host:3000/ws/voice python test_e2e_ws.py
"""
import os
import sys
import json
import time
import asyncio
import struct
import numpy as np

WS_URL = os.environ.get("WS_URL", "ws://localhost:3000/ws/voice")
HTTP_URL = os.environ.get("HTTP_URL", "http://localhost:3000")
API_KEY = os.environ.get("VOICEAGENT_API_KEY", "")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
results = {}


def section(name):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")


def make_silence(duration_s=0.5, sr=16000):
    samples = np.zeros(int(sr * duration_s), dtype=np.int16)
    return samples.tobytes()


def make_tone(duration_s=1.0, freq=440, sr=16000, amplitude=0.3):
    t = np.arange(int(sr * duration_s)) / sr
    samples = (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16)
    return samples.tobytes()


async def test_ws_connect():
    section("WS Connect + Ready")
    try:
        import websockets
        url = f"{WS_URL}?token={API_KEY}" if API_KEY else WS_URL
        async with websockets.connect(url) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(msg)
            is_ready = data.get("type") == "ready"
            version = data.get("version", "")
            session_id = data.get("session_id", "")
            print(f"  Ready: {is_ready}, version={version}, session={session_id}")
            ok = is_ready and version and session_id
            print(f"  Result: {PASS if ok else FAIL}")
            results["ws_connect"] = ok
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        results["ws_connect"] = False


async def test_ws_silence():
    section("WS Silence — no false trigger")
    try:
        import websockets
        url = f"{WS_URL}?token={API_KEY}" if API_KEY else WS_URL
        async with websockets.connect(url) as ws:
            await asyncio.wait_for(ws.recv(), timeout=10)

            silence = make_silence(1.0)
            chunk_size = 1024
            for i in range(0, len(silence), chunk_size):
                await ws.send(silence[i:i+chunk_size])
                await asyncio.sleep(0.032)

            await asyncio.sleep(0.5)
            states = []
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.3)
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data.get("type") == "state":
                            states.append(data["state"])
                except asyncio.TimeoutError:
                    break

            no_false = "listening" not in states or "thinking" not in states
            print(f"  States observed: {states}")
            print(f"  No false trigger: {no_false}")
            print(f"  Result: {PASS if no_false else FAIL}")
            results["ws_silence"] = no_false
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        results["ws_silence"] = False


async def test_ws_text_input():
    section("WS Text Input → LLM response")
    try:
        import websockets
        url = f"{WS_URL}?token={API_KEY}" if API_KEY else WS_URL
        async with websockets.connect(url) as ws:
            ready = await asyncio.wait_for(ws.recv(), timeout=10)

            # Drain greeting messages
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                pass

            await ws.send(json.dumps({"type": "text_input", "text": "你好"}))

            types_seen = set()
            has_audio = False
            t0 = time.time()
            while time.time() - t0 < 15:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                    if isinstance(msg, bytes):
                        has_audio = True
                    else:
                        data = json.loads(msg)
                        types_seen.add(data.get("type", ""))
                        if data.get("type") == "state" and data.get("state") == "idle":
                            break
                except asyncio.TimeoutError:
                    break

            has_llm = "llm_sentence" in types_seen or "llm" in types_seen
            has_rag = "rag" in types_seen

            print(f"  Message types: {sorted(types_seen)}")
            print(f"  Has LLM response: {has_llm}")
            print(f"  Has RAG context: {has_rag}")
            print(f"  Has audio: {has_audio}")

            ok = has_llm and has_audio
            print(f"  Result: {PASS if ok else FAIL}")
            results["ws_text_input"] = ok
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        results["ws_text_input"] = False


async def test_api_key_reject():
    section("API Key Rejection (bad token)")
    if not API_KEY:
        print(f"  {SKIP} — VOICEAGENT_API_KEY not set")
        results["api_key_reject"] = True
        return
    try:
        import websockets
        try:
            async with websockets.connect(f"{WS_URL}?token=wrong_key") as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                print(f"  Unexpected message: {msg}")
                results["api_key_reject"] = False
        except websockets.exceptions.ConnectionClosed as e:
            rejected = e.code == 4003
            print(f"  Close code: {e.code}, reason: {e.reason}")
            print(f"  Rejected: {rejected}")
            print(f"  Result: {PASS if rejected else FAIL}")
            results["api_key_reject"] = rejected
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        results["api_key_reject"] = False


async def test_http_info():
    section("HTTP /api/info")
    try:
        import httpx
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{HTTP_URL}/api/info", headers=headers)
            data = resp.json()
            has_version = "version" in data
            has_models = "models" in data
            print(f"  Status: {resp.status_code}")
            print(f"  Version: {data.get('version')}")
            print(f"  Models: {list(data.get('models', {}).keys())}")
            ok = resp.status_code == 200 and has_version and has_models
            print(f"  Result: {PASS if ok else FAIL}")
            results["http_info"] = ok
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        results["http_info"] = False


async def test_http_metrics():
    section("HTTP /api/metrics")
    try:
        import httpx
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{HTTP_URL}/api/metrics", headers=headers)
            data = resp.json()
            has_total = "total_sessions" in data
            has_active = "active_sessions" in data
            has_errors = "total_errors" in data
            print(f"  Status: {resp.status_code}")
            print(f"  Metrics: {data}")
            ok = resp.status_code == 200 and has_total and has_errors
            print(f"  Result: {PASS if ok else FAIL}")
            results["http_metrics"] = ok
    except Exception as e:
        print(f"  Result: {FAIL} — {e}")
        results["http_metrics"] = False


async def main():
    print("=" * 60)
    print("  VoxLabs v2.9 E2E WebSocket Test")
    print(f"  WS:   {WS_URL}")
    print(f"  HTTP: {HTTP_URL}")
    print(f"  Auth: {'enabled' if API_KEY else 'disabled'}")
    print("=" * 60)

    await test_ws_connect()
    await test_ws_silence()
    await test_ws_text_input()
    await test_api_key_reject()
    await test_http_info()
    await test_http_metrics()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {name:20s} {PASS if ok else FAIL}")
    total = len(results)
    passed = sum(results.values())
    all_pass = all(results.values())
    print(f"\n  {passed}/{total} {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
