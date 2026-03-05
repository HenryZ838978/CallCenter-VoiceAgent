"""WebSocket end-to-end test suite for VoxLabs Voice Agent.

Tests the ACTUAL WebSocket path (not direct CM.feed_audio) to catch bugs like
the v2.0 barge-in failure that only manifested in real WS deployment.

Usage:
    python test_ws_e2e.py [--url wss://localhost:3000/ws/voice] [--test all|bargein|pipeline|multiturn]

Requires: pip install websockets
"""
import sys
import json
import time
import struct
import asyncio
import argparse
import numpy as np

try:
    import websockets
    import ssl
except ImportError:
    print("Install: pip install websockets")
    sys.exit(1)


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
CHUNK_MS = 32


def generate_speech_audio(duration_s: float = 2.0, freq: float = 300.0, amplitude: float = 0.3) -> np.ndarray:
    """Generate a tone that reliably triggers VAD (sine wave simulating voice)."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * freq * t)
    noise = np.random.randn(len(audio)).astype(np.float32) * 0.05
    return audio + noise


def generate_silence(duration_s: float = 1.0) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def audio_to_chunks(audio: np.ndarray, chunk_samples: int = CHUNK_SAMPLES) -> list:
    """Split audio into chunk-sized byte packets (mimicking browser AudioWorklet)."""
    int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    chunks = []
    for i in range(0, len(int16), chunk_samples):
        chunk = int16[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk.tobytes())
    return chunks


class WSVoiceClient:
    """WebSocket voice client that mimics browser behavior."""

    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.messages = []
        self.audio_received = bytearray()
        self.states = []
        self._recv_task = None

    async def connect(self):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        self.ws = await websockets.connect(self.url, ssl=ssl_ctx, max_size=10 * 1024 * 1024)
        self._recv_task = asyncio.create_task(self._receive_loop())

        await asyncio.sleep(0.3)
        return self

    async def _receive_loop(self):
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    self.audio_received.extend(msg)
                else:
                    data = json.loads(msg)
                    self.messages.append(data)
                    if data.get("type") == "state":
                        self.states.append(data["state"])
        except websockets.ConnectionClosed:
            pass

    async def send_audio(self, audio: np.ndarray, realtime: bool = True):
        """Send audio chunks at real-time pace (like browser AudioWorklet)."""
        chunks = audio_to_chunks(audio)
        for chunk_bytes in chunks:
            await self.ws.send(chunk_bytes)
            if realtime:
                await asyncio.sleep(CHUNK_MS / 1000.0)

    async def send_audio_fast(self, audio: np.ndarray):
        """Send all audio immediately (for quick tests)."""
        await self.send_audio(audio, realtime=False)

    async def wait_for_state(self, target: str, timeout: float = 15.0) -> bool:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if target in self.states:
                return True
            await asyncio.sleep(0.05)
        return False

    async def wait_for_message(self, msg_type: str, timeout: float = 15.0) -> dict:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            for m in self.messages:
                if m.get("type") == msg_type:
                    return m
            await asyncio.sleep(0.05)
        return {}

    def get_messages(self, msg_type: str) -> list:
        return [m for m in self.messages if m.get("type") == msg_type]

    def clear(self):
        self.messages.clear()
        self.audio_received.clear()
        self.states.clear()

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
        if self.ws:
            await self.ws.close()


async def test_ws_pipeline(url: str):
    """TEST 1: Basic pipeline — send speech + silence, expect ASR + LLM + TTS audio back."""
    print("\n" + "=" * 60)
    print("TEST 1: WebSocket E2E Pipeline")
    print("=" * 60)

    client = await WSVoiceClient(url).connect()

    ready = await client.wait_for_message("ready", timeout=5)
    assert ready, "FAIL: No ready message"
    print(f"  Connected: session={ready.get('session_id')}")

    speech = generate_speech_audio(2.0, freq=300, amplitude=0.3)
    silence = generate_silence(1.0)
    audio = np.concatenate([speech, silence])

    t0 = time.perf_counter()
    await client.send_audio(audio, realtime=True)

    got_speaking = await client.wait_for_state("speaking", timeout=15)
    e2e_ms = (time.perf_counter() - t0) * 1000

    asr_msgs = client.get_messages("asr")
    metrics = client.get_messages("metrics")
    llm_msgs = client.get_messages("llm_sentence")

    print(f"  States seen: {client.states}")
    print(f"  ASR: {asr_msgs[0].get('text', '?')[:50] if asr_msgs else 'NONE'}")
    print(f"  LLM sentences: {len(llm_msgs)}")
    print(f"  TTS audio received: {len(client.audio_received)} bytes")
    print(f"  Got to SPEAKING: {got_speaking}")
    print(f"  E2E time: {e2e_ms:.0f}ms")

    if metrics:
        m = metrics[0]
        print(f"  Metrics: ASR={m.get('asr_ms',0):.0f}ms RAG={m.get('rag_ms',0):.0f}ms "
              f"LLM={m.get('llm_ms',0):.0f}ms TTS={m.get('tts_ttfa_ms',0):.0f}ms "
              f"First={m.get('first_response_ms',0):.0f}ms")

    status = "PASS" if got_speaking and len(client.audio_received) > 0 else "FAIL"
    print(f"  Result: {status}")

    await client.close()
    return status == "PASS"


async def test_ws_barge_in(url: str):
    """TEST 2: Barge-in during SPEAKING — THE critical test that v2.0 failed.
    
    Sends speech → silence → waits for SPEAKING → sends loud speech during SPEAKING.
    Expects INTERRUPTED state and barge_in message.
    """
    print("\n" + "=" * 60)
    print("TEST 2: WebSocket Barge-in (v2.0 killer bug)")
    print("=" * 60)

    client = await WSVoiceClient(url).connect()
    await client.wait_for_message("ready", timeout=5)

    speech1 = generate_speech_audio(2.0, freq=300, amplitude=0.3)
    silence = generate_silence(0.8)
    audio_turn1 = np.concatenate([speech1, silence])

    print("  Phase 1: Sending initial speech to trigger pipeline...")
    await client.send_audio(audio_turn1, realtime=True)

    got_speaking = await client.wait_for_state("speaking", timeout=15)
    if not got_speaking:
        print("  FAIL: Never reached SPEAKING state")
        await client.close()
        return False

    print(f"  Phase 2: Agent is SPEAKING, sending barge-in speech...")
    await asyncio.sleep(0.5)

    barge_speech = generate_speech_audio(1.5, freq=350, amplitude=0.5)
    t_barge = time.perf_counter()
    await client.send_audio(barge_speech, realtime=True)

    got_interrupted = await client.wait_for_state("interrupted", timeout=5)
    barge_ms = (time.perf_counter() - t_barge) * 1000

    barge_msgs = client.get_messages("barge_in")

    print(f"  States seen: {client.states}")
    print(f"  INTERRUPTED reached: {got_interrupted}")
    print(f"  Barge-in messages: {len(barge_msgs)}")
    print(f"  Barge-in latency: {barge_ms:.0f}ms")

    status = "PASS" if got_interrupted and len(barge_msgs) > 0 else "FAIL"
    print(f"  Result: {status}")

    if status == "FAIL":
        print("  !!! This is the v2.0 killer bug — frontend was not sending audio during SPEAKING.")
        print("  !!! v2.1+ fix: frontend always sends audio, server uses high VAD threshold + RMS.")

    await client.close()
    return status == "PASS"


async def test_ws_multiturn(url: str):
    """TEST 3: Multi-turn — send 3 turns of speech, verify conversation progresses."""
    print("\n" + "=" * 60)
    print("TEST 3: WebSocket Multi-turn")
    print("=" * 60)

    client = await WSVoiceClient(url).connect()
    await client.wait_for_message("ready", timeout=5)

    for turn_idx in range(3):
        client.clear()

        speech = generate_speech_audio(1.5 + turn_idx * 0.5, freq=300 + turn_idx * 50, amplitude=0.3)
        silence = generate_silence(0.8)
        audio = np.concatenate([speech, silence])

        t0 = time.perf_counter()
        await client.send_audio(audio, realtime=True)

        got_speaking = await client.wait_for_state("speaking", timeout=15)
        latency = (time.perf_counter() - t0) * 1000

        asr_msgs = client.get_messages("asr")
        asr_text = asr_msgs[0].get("text", "?")[:40] if asr_msgs else "NONE"

        await client.wait_for_state("idle", timeout=20)

        print(f"  Turn {turn_idx+1}: speaking={got_speaking} latency={latency:.0f}ms ASR='{asr_text}'")

        if not got_speaking:
            print(f"  FAIL: Turn {turn_idx+1} never reached SPEAKING")
            await client.close()
            return False

    status = "PASS"
    print(f"  Result: {status}")
    await client.close()
    return status == "PASS"


async def main():
    parser = argparse.ArgumentParser(description="WebSocket E2E tests for VoxLabs Voice Agent")
    parser.add_argument("--url", default="wss://localhost:3000/ws/voice")
    parser.add_argument("--test", default="all", choices=["all", "pipeline", "bargein", "multiturn"])
    args = parser.parse_args()

    print(f"WebSocket E2E Test Suite")
    print(f"Target: {args.url}")
    print(f"Tests: {args.test}")

    results = {}

    if args.test in ("all", "pipeline"):
        results["pipeline"] = await test_ws_pipeline(args.url)
    if args.test in ("all", "bargein"):
        results["bargein"] = await test_ws_barge_in(args.url)
    if args.test in ("all", "multiturn"):
        results["multiturn"] = await test_ws_multiturn(args.url)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
