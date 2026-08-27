"""Benchmark: MiniCPM-o-4.5 end-to-end latency measurement.

Measures TTFA, total generation time, and compares with voiceagent pipeline.
Usage:
    CUDA_VISIBLE_DEVICES=4 python benchmark.py [--audio path.wav] [--rounds 5]
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OMNI_MODEL_DIR, OMNI_DEVICE, LOAD_IN_4BIT, VOICE_PROMPT_WAV, SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="Path to test audio (16kHz wav). If not given, generates TTS prompt.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    from engine.omni import OmniEngine

    print(f"Loading MiniCPM-o-4.5 on {OMNI_DEVICE} (4bit={LOAD_IN_4BIT}) ...")
    omni = OmniEngine(OMNI_MODEL_DIR, device=OMNI_DEVICE, load_in_4bit=LOAD_IN_4BIT)
    omni.load()

    if os.path.exists(VOICE_PROMPT_WAV):
        omni.set_voice(VOICE_PROMPT_WAV)

    if args.audio:
        import librosa
        audio, _ = librosa.load(args.audio, sr=16000, mono=True)
        print(f"Test audio: {args.audio} ({len(audio)/16000:.1f}s)")
    else:
        print("No audio provided, generating 3s of synthesized speech prompt...")
        t = np.linspace(0, 3, 3 * 16000)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

    prompt = "你好，请简单介绍一下自己。"

    if args.warmup > 0:
        print(f"\n--- Warmup ({args.warmup} rounds) ---")
        for i in range(args.warmup):
            r = omni.chat(audio, prompt, max_new_tokens=32)
            print(f"  Warmup {i+1}: TTFA={r['ttfa_ms']:.0f}ms total={r['total_ms']:.0f}ms text={r['text'][:30]}...")

    results = []
    print(f"\n--- Benchmark ({args.rounds} rounds) ---")
    for i in range(args.rounds):
        r = omni.chat(audio, SYSTEM_PROMPT, max_new_tokens=args.max_tokens)
        results.append(r)
        print(f"  Round {i+1}: TTFA={r['ttfa_ms']:.0f}ms total={r['total_ms']:.0f}ms "
              f"audio={r['audio_duration_s']:.1f}s text={r['text'][:40]}...")

    ttfas = [r["ttfa_ms"] for r in results]
    totals = [r["total_ms"] for r in results]

    print("\n=== Results ===")
    print(f"  TTFA   avg={np.mean(ttfas):.0f}ms  min={np.min(ttfas):.0f}ms  max={np.max(ttfas):.0f}ms  p50={np.median(ttfas):.0f}ms")
    print(f"  Total  avg={np.mean(totals):.0f}ms  min={np.min(totals):.0f}ms  max={np.max(totals):.0f}ms  p50={np.median(totals):.0f}ms")

    print("\n=== Comparison with voiceagent pipeline ===")
    print(f"  voiceagent:  ASR(117) + RAG(4) + LLM(163) + TTS_TTFA(174) = ~341ms first audio")
    print(f"  omni:        TTFA = {np.mean(ttfas):.0f}ms (single model, end-to-end)")
    diff = np.mean(ttfas) - 341
    label = "slower" if diff > 0 else "faster"
    print(f"  Difference:  {abs(diff):.0f}ms {label}")


if __name__ == "__main__":
    main()
