"""Full duplex test suite: pipeline / barge-in / multi-turn.
Usage: python test_duplex.py [all|pipeline|bargein|multiturn]
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASR_MODEL_DIR, TTS_MODEL_DIR, VAD_MODEL_DIR,
    VLLM_BASE_URL, VLLM_MODEL_NAME, SYSTEM_PROMPT,
)

USE_MOCK_TTS = os.environ.get("MOCK_TTS", "0") == "1"
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")


def load_components():
    from engine.vad import SileroVAD
    from engine.asr import SenseVoiceASR
    from engine.llm import VLLMChat
    from engine.tts import VoxCPMTTS

    print("=" * 60)
    print("Loading components...")
    print("=" * 60)

    t0 = time.perf_counter()
    vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5).load()
    print(f"  VAD loaded ({(time.perf_counter()-t0)*1000:.0f}ms)")

    t0 = time.perf_counter()
    asr = SenseVoiceASR(ASR_MODEL_DIR, device=ASR_DEVICE).load()
    print(f"  ASR loaded ({(time.perf_counter()-t0)*1000:.0f}ms)")

    llm = VLLMChat(
        base_url=VLLM_BASE_URL,
        model=VLLM_MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
    )
    print("  LLM client ready (vLLM server)")

    if USE_MOCK_TTS:
        tts = MockTTS()
        print("  TTS: MOCK mode")
    else:
        t0 = time.perf_counter()
        tts = VoxCPMTTS(TTS_MODEL_DIR, device=TTS_DEVICE,
                        gpu_memory_utilization=0.55).load()
        print(f"  TTS loaded ({(time.perf_counter()-t0)*1000:.0f}ms)")
        print("  Warming up TTS...")
        tts.warmup()
        print("  TTS warmup done")

    print("=" * 60)
    return vad, asr, llm, tts


class MockTTS:
    SAMPLE_RATE = 24000
    def synthesize(self, text, prompt_id=None):
        time.sleep(0.1)
        audio = np.random.randn(24000).astype(np.float32) * 0.01
        return {"audio": audio, "sr": 24000, "ttfa_ms": 100.0,
                "total_ms": 100.0, "chunks": 1}
    def warmup(self):
        pass


def test_pipeline(asr, llm, tts, vad):
    """TEST 1: Basic E2E pipeline with synthetic audio."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic E2E Pipeline")
    print("=" * 60)

    duration_s = 3.0
    audio = np.random.randn(int(16000 * duration_s)).astype(np.float32) * 0.3

    t0 = time.perf_counter()
    asr_result = asr.transcribe(audio)
    asr_ms = asr_result["latency_ms"]
    print(f"  ASR: {asr_ms:.0f}ms | text='{asr_result['text'][:50]}'")

    user_text = asr_result["text"].strip() or "你好，请介绍一下你们的服务"
    llm_result = llm.chat(user_text)
    llm_ms = llm_result["latency_ms"]
    print(f"  LLM: {llm_ms:.0f}ms | text='{llm_result['text'][:80]}'")

    tts_result = tts.synthesize(llm_result["text"])
    tts_ms = tts_result["ttfa_ms"]
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"  TTS TTFA: {tts_ms:.0f}ms | chunks={tts_result['chunks']}")
    print(f"  E2E Total: {total_ms:.0f}ms")
    print(f"  Pipeline (ASR+LLM+TTS TTFA): {asr_ms + llm_ms + tts_ms:.0f}ms")


def test_barge_in(vad, tts):
    """TEST 2: Barge-in detection during TTS playback."""
    print("\n" + "=" * 60)
    print("TEST 2: Barge-in Detection")
    print("=" * 60)

    from engine.duplex_agent import DuplexAgent
    dummy_asr = type('MockASR', (), {'transcribe': lambda s, a, sr=16000: {'text': 'test', 'latency_ms': 0}})()
    dummy_llm = type('MockLLM', (), {
        'chat': lambda s, t: {'text': 'ok', 'latency_ms': 0},
        'reset': lambda s: None
    })()

    agent = DuplexAgent(vad, dummy_asr, dummy_llm, tts)

    result = agent.simulate_barge_in(tts_chunks_before_interrupt=6)
    print(f"  Chunks played before interrupt: {result['chunks_played']}")
    print(f"  Detection time: {result['detection_ms']:.0f}ms")
    print(f"  Barge-in triggered: {result['barge_in_triggered']}")
    status = "PASS" if result['barge_in_triggered'] else "FAIL"
    print(f"  Result: {status}")


def test_multiturn(llm, tts):
    """TEST 3: Multi-turn conversation stability."""
    print("\n" + "=" * 60)
    print("TEST 3: Multi-turn Conversation")
    print("=" * 60)

    turns = [
        "你好，我想了解你们的服务",
        "请问价格是多少？",
        "好的，谢谢，再见",
    ]

    llm.reset()
    results = []

    for i, user_text in enumerate(turns):
        t0 = time.perf_counter()
        llm_result = llm.chat(user_text)
        tts_result = tts.synthesize(llm_result["text"])
        response_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "turn": i + 1,
            "user": user_text,
            "llm_ms": llm_result["latency_ms"],
            "tts_ttfa_ms": tts_result["ttfa_ms"],
            "response_ms": response_ms,
            "reply": llm_result["text"][:60],
        })
        print(f"  Turn {i+1}: LLM={llm_result['latency_ms']:.0f}ms "
              f"TTS={tts_result['ttfa_ms']:.0f}ms "
              f"Total={response_ms:.0f}ms "
              f"| '{llm_result['text'][:60]}'")

    avg_ms = np.mean([r["response_ms"] for r in results])
    avg_llm = np.mean([r["llm_ms"] for r in results])
    avg_tts = np.mean([r["tts_ttfa_ms"] for r in results])
    pipeline_ms = avg_llm + avg_tts
    print(f"\n  Average: LLM={avg_llm:.0f}ms  TTS TTFA={avg_tts:.0f}ms")
    print(f"  Pipeline (LLM+TTS TTFA): {pipeline_ms:.0f}ms")
    print(f"  Full response (incl TTS gen): {avg_ms:.0f}ms")
    status = "PASS" if pipeline_ms < 500 else "FAIL"
    print(f"  Result: {status} (pipeline target < 500ms, actual: {pipeline_ms:.0f}ms)")


def main():
    test = sys.argv[1] if len(sys.argv) > 1 else "all"

    vad, asr, llm, tts = load_components()

    if test in ("all", "pipeline"):
        test_pipeline(asr, llm, tts, vad)
    if test in ("all", "bargein"):
        test_barge_in(vad, tts)
    if test in ("all", "multiturn"):
        test_multiturn(llm, tts)

    print("\n" + "=" * 60)
    print("All tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
