# VoxLabs — Full-Duplex Voice Agent

Real-time Chinese voice agent with sub-500ms response, natural barge-in, and voice cloning. Built for BPO/call-center scenarios.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Browser / SIP Client                         │
│   Mic ──→ PCM 16kHz ──→ WebSocket ──→ Server                       │
│   Speaker ←── PCM 44.1kHz ←── WebSocket ←── Server                 │
│   Client-side: instant barge-in (RMS gate, 0ms)                    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ WSS (Cloudflare Tunnel)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FastAPI WebSocket Server (:3000)                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              ConversationManager (State Machine)             │    │
│  │   IDLE → LISTENING → THINKING → SPEAKING → INTERRUPTED      │    │
│  │                                                             │    │
│  │   • Dual-layer barge-in (client 0ms + server 96ms)          │    │
│  │   • Adaptive endpointing (192ms / 320ms / 640ms)            │    │
│  │   • THINKING-state speech buffering                         │    │
│  └──────┬──────────┬──────────┬──────────┬─────────────────────┘    │
│         │          │          │          │                           │
│    ┌────▼───┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐                     │
│    │  VAD   │ │  ASR   │ │  RAG   │ │  TTS   │                     │
│    │ Silero │ │ Sense  │ │  bge + │ │ VoxCPM │                     │
│    │        │ │ Voice  │ │ FAISS  │ │  1.5   │                     │
│    │  CPU   │ │ Small  │ │        │ │nanovllm│                     │
│    └────────┘ └────────┘ └────────┘ └────────┘                     │
│                                                                     │
│         ┌──────────────────────────────┐                            │
│         │    vLLM Server (:8100)       │                            │
│         │  MiniCPM4.1-8B-GPTQ-Marlin  │                            │
│         │  Sentence streaming          │                            │
│         │  OpenAI-compatible API       │                            │
│         └──────────────────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘
```

## Performance (RTX 4090, warmup)

| Component | Latency | Model |
|---|---|---|
| ASR | **117ms** | SenseVoiceSmall (234M, FP32) |
| RAG | **3.6ms** | bge-small-zh-v1.5 + FAISS |
| LLM | **163ms** | MiniCPM4.1-8B-GPTQ via vLLM |
| TTS TTFA | **174ms** | VoxCPM 1.5 via nanovllm |
| **Pipeline Total** | **~458ms** | First audio byte to user |

Target: < 500ms &nbsp;|&nbsp; Death line: < 800ms &nbsp;|&nbsp; Actual: **458ms** ✅

### Multi-turn Conversation (3-turn average)

| Turn | User Input | LLM | TTS TTFA | Response |
|---|---|---|---|---|
| 1 | "你好，我想了解你们的服务" | 196ms | 174ms | 1239ms |
| 2 | "请问价格是多少？" | 135ms | 172ms | 959ms |
| 3 | "好的，谢谢，再见" | 159ms | 175ms | 879ms |
| **Average** | | **163ms** | **174ms** | — |

### GPU × Latency Comparison

| Hardware | ASR | LLM | TTS | Pipeline |
|---|---|---|---|---|
| RTX 4080S (32GB) | 130ms | 225ms | 115ms (mock) | 470ms |
| RTX 5090 (32GB) | 93ms | 130ms | 138ms | 342ms |
| **RTX 4090 (24GB)** | **117ms** | **163ms** | **174ms** | **337-458ms** |

## Full-Duplex & Barge-in (v2.1)

The system implements a **dual-layer barge-in** architecture:

**Layer 1 — Client-side instant response (0ms)**
- Browser continuously sends mic audio (no muting during agent speech)
- Local RMS energy detection: 3 consecutive frames above threshold → immediately flush TTS playback
- User perceives zero delay when interrupting

**Layer 2 — Server-side confirmed interrupt (~96ms)**
- High VAD threshold (0.85 vs normal 0.5) filters TTS echo/noise
- RMS energy gate (> 0.015) eliminates low-energy false triggers
- 3-chunk consecutive confirmation (~96ms) prevents transient misfires
- Confirmed barge-in cancels LLM+TTS pipeline

### Adaptive Endpointing

| User speech duration | Silence threshold | Use case |
|---|---|---|
| < 0.5s | 192ms | Short responses: "好的", "嗯" |
| 0.5–3s | 320ms | Normal conversation |
| > 3s | 640ms | Long explanations, thinking pauses |

### THINKING-state Speech Buffering

If the user speaks more while the pipeline is running (~458ms), the additional audio is buffered and merged via ASR after the initial transcription — no user input is ever dropped.

## Features

- **Voice Cloning** — 8-second reference audio → cloned voice (VoxCPM 1.5, temp=0.7, cfg=3.0)
- **Sentence Streaming** — LLM generates sentence by sentence; each sentence is sent to TTS immediately
- **RAG** — bge-small-zh-v1.5 + FAISS, hot-reload via `POST /api/rag/reload`
- **Zero Pop Audio** — First TTS chunk discarded (VAE transient), second chunk 10ms fade-in, AudioWorklet queue player
- **Utterance Aggregation** — After barge-in, interrupted segments are combined: "用户依次说了：X；Y。请综合回应"

## Architecture Advantages

### Why Pipeline > End-to-End Omni

We benchmarked against MiniCPM-o-4.5 (9B omni model) on the same hardware:

| Dimension | Pipeline | Omni (MiniCPM-o-4.5) |
|---|---|---|
| First Audio | **458ms** | 1666ms |
| VRAM | 17.6GB (2 GPU) | 21.5GB (1 GPU) |
| RAG integration | 3.6ms (vector search) | +305ms (prompt embedding) |
| Voice customization | LoRA / clone in hours | Retrain entire model |
| LLM customization | SFT in hours, ~$20 | Multi-modal SFT, ~$5000+ |
| Component upgrade | Swap one module | Retrain everything |

**For real-time scenarios (< 500ms target), pipeline wins decisively.**

### Extensibility

Each component is independently replaceable:

| Component | Current | Alternatives |
|---|---|---|
| ASR | SenseVoiceSmall / FunASR | Whisper, Paraformer, sherpa-onnx |
| LLM | MiniCPM4.1-8B-GPTQ / vLLM | Qwen, LLaMA, any OpenAI-compatible |
| TTS | VoxCPM 1.5 / nanovllm | CosyVoice, IndexTTS, GPT-SoVITS |
| RAG | bge-small + FAISS | bge-m3, Milvus, Elasticsearch |
| VAD | Silero VAD | FireRedChat pVAD, WebRTC VAD |
| Turn Detection | Adaptive silence threshold | Pipecat Smart Turn, VoTurn-80M, LiveKit EOU |

### Next Steps

- [ ] Integrate [Pipecat Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) (8M ONNX) for semantic endpointing
- [ ] Integrate [FireRedChat pVAD](https://huggingface.co/FireRedTeam/FireRedChat-pvad) for speaker-aware barge-in
- [ ] Train custom turn detector with domain-specific BPO data
- [ ] LiveKit Server + SIP trunk for telephone network integration
- [ ] Multi-session concurrency

## Project Structure

```
voiceagent/
├── engine/
│   ├── asr.py                    # SenseVoiceSmall / FunASR
│   ├── llm.py                    # vLLM streaming + sentence boundary
│   ├── tts.py                    # VoxCPM 1.5 (44.1kHz, voice clone)
│   ├── vad.py                    # Silero VAD
│   ├── rag.py                    # bge-small-zh-v1.5 + FAISS
│   ├── conversation_manager.py   # v2.1 state machine + dual-layer barge-in
│   ├── filler.py                 # Filler word cache (disabled)
│   └── duplex_agent.py           # v1 standalone agent (legacy)
│
├── static/
│   └── voice_agent.html          # AudioWorklet queue player + instant barge-in
│
├── data/
│   ├── sample_kb.json            # Knowledge base (59 FAQ items)
│   └── voice_prompt.wav          # Voice clone reference audio
│
├── models/                       # Model weights (not in repo)
│   ├── SenseVoiceSmall/          (901MB)
│   ├── MiniCPM4.1-8B-GPTQ/      (4.9GB)
│   ├── VoxCPM1.5/                (1.9GB)
│   ├── snakers4_silero-vad/
│   └── bge-small-zh-v1.5/        (91MB)
│
├── ws_server.py                  # FastAPI WebSocket server
├── config.py                     # Model paths, prompts, parameters
├── start_all.sh                  # One-click launch
├── start_vllm.sh                 # vLLM server launch
├── test_duplex.py                # Duplex integration tests
└── ab_test_rag.py                # RAG A/B benchmarks
```

## Quick Start

```bash
# 1. Start vLLM (LLM inference server)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model models/MiniCPM4.1-8B-GPTQ --served-model-name MiniCPM4.1-8B-GPTQ \
  --trust-remote-code --dtype auto --quantization gptq_marlin \
  --gpu-memory-utilization 0.85 --max-model-len 4096 --port 8100

# 2. Start Voice Agent (ASR + TTS + RAG + WebSocket)
CUDA_VISIBLE_DEVICES=2 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 \
  python ws_server.py

# 3. (Optional) Public access via Cloudflare Tunnel
./cloudflared tunnel --url https://localhost:3000 --no-tls-verify
```

Open `https://localhost:3000` → Click **Start Call** → Allow microphone → Talk.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Voice Agent Console |
| `/ws/voice` | WS | Full-duplex voice channel |
| `/api/info` | GET | Model info and config |
| `/api/rag/docs` | GET | Knowledge base documents |
| `/api/rag/query?q=` | GET | Test RAG retrieval |
| `/api/rag/reload` | POST | Hot-reload knowledge base |

## Version History

| Version | Tag | Highlights |
|---|---|---|
| v1.0 | `782f1ba` | Basic full-duplex, serial pipeline, voice cloning |
| v2.0 | `bbc11e0` | State machine, sentence streaming, zero-pop audio, utterance aggregation |
| **v2.1** | **latest** | Dual-layer barge-in, adaptive endpointing, THINKING buffer, 50ms TTS cancel |

## License

MIT
