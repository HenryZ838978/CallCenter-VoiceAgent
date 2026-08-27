<div align="center">

# 🎙️ VoiceAgent-600ms

### Voice · ASR · TTS · LLM · Infra · Posttraining

**一个仓库,两条路线:级联 Pipeline(主力)与端到端 Omni(前沿)— 全部低于 600ms 感知预算**

<br>

[![Pipeline](https://img.shields.io/badge/Cascade_Pipeline-458ms-00c853?style=for-the-badge&logo=speedtest&logoColor=white)](/)
[![Omni](https://img.shields.io/badge/E2E_Omni-250ms-2979ff?style=for-the-badge&logo=bolt&logoColor=white)](/)
[![Barge-in](https://img.shields.io/badge/Barge--in-160ms-ff6f00?style=for-the-badge&logo=bolt&logoColor=white)](/)
[![ASR CER](https://img.shields.io/badge/ASR_CER-2.89%25-9c27b0?style=for-the-badge&logo=microphone&logoColor=white)](/)
[![License](https://img.shields.io/badge/Apache-2.0-7c4dff?style=for-the-badge)](/)

<br>

> **800ms 是语音对话的图灵测试门槛**——超过它,人类会感知到"在和机器说话"。
> 本项目以 **600ms 为工程预算线**,在同一套工程底座上实现了两条架构路线,实测全部远低于预算:
> 级联 Pipeline **458ms** 首响,端到端 Omni **250ms** 首帧音频。

</div>

---

## 🗺️ 两条架构路线

```
路线 A · 级联 Pipeline (仓库根目录, 主力生产版本)
  VAD → ASR(FireRedASR2) → RAG(bge+FAISS) → LLM(Qwen3-14B/vLLM) → TTS(VoxCPM 流式)
  458ms 首响 · 组件全部可独立替换 · 话术定制 SFT ~$20

路线 B · 端到端 Omni (omni/ 目录, 延迟前沿实验线)
  VAD → MiniCPM-o 4.5 AWQ (音频原生理解, vLLM) → TTS(VoxCPM 流式)
  250ms 首帧 · 无 ASR 误差传播 · 10.5GB VRAM
```

**为什么主力是级联而不是更快的 Omni?** 可控性。级联的每个组件可独立替换/升级,换音色是小时级语音克隆,换话术是 ~$20 的 LoRA SFT;Omni 路线换话术要动整个多模态模型。延迟两者都已低于人类感知门槛,此时**可控性 > 更低的延迟**。Omni 线作为延迟前沿保留在 [omni/](omni/),完整文档见 [omni/README.md](omni/README.md)。

### 同硬件三方案实测对比 (RTX 4090)

| 方案 | 首响延迟 | 打断精度 | 换音色 | 换话术 |
|:---:|:---:|:---:|:---:|:---:|
| **🏆 级联 Pipeline (主力)** | **458ms** | **160ms** | 语音克隆 · 小时级 | SFT · ~$20 |
| 端到端 Omni ([omni/](omni/)) | **250ms** | 160ms | 语音克隆 | Omni SFT |
| 纯 Omni (raw transformers) | 1666ms ❌ | 模型原生 | 重训整个模型 | ~$5000+ |

---

## ⚡ 路线 A:级联 Pipeline 为什么快

```
                          用户说完
                             │
               ┌─────────────┼──────────────┐
               ▼             ▼              ▼
          ┌─────────┐  ┌──────────┐  ┌────────────┐
          │ 🎤 ASR  │  │ 📚 RAG   │  │            │
          │ FireRed │  │ bge+FAISS│  │            │
          │ ~200ms  │  │  ~4ms    │  │            │
          └────┬────┘  └────┬─────┘  │            │
               │            │        │            │
               ▼            ▼        │            │
          ┌──────────────────────┐   │  ⏱️ 458ms  │
          │  🧠 LLM (Streaming)  │   │  总延迟    │
          │  Qwen3-14B via vLLM  │   │            │
          │  ~163ms to 1st token │   │            │
          └──────────┬───────────┘   │            │
                     ▼               │            │
          ┌──────────────────────┐   │            │
          │  🔊 TTS (Streaming)  │   │            │
          │  VoxCPM via nanovllm │   │            │
          │  ~174ms to 1st chunk │   │            │
          └──────────────────────┘   │            │
                     │               ▼            ▼
               用户听到第一个音节 ◄──────────────┘
```

### 延迟实测

| 组件 | RTX 4090 | RTX 5090 | RTX 4080S |
|---|---|---|---|
| 🎤 ASR (FireRedASR2-AED, CER 2.89%) | **200ms** | 93ms | 130ms |
| 📚 RAG (bge-small + FAISS) | **4ms** | 3ms | 4ms |
| 🧠 LLM (Qwen3-14B-AWQ, vLLM) | **163ms** | 130ms | 225ms |
| 🔊 TTS (VoxCPM / nanovllm, 流式) | **174ms** | 138ms | 115ms |
| 🚀 **Pipeline 总计** | **458ms** | **342ms** | **470ms** |

### 核心工程优化

<details>
<summary><b>1. 流式 TTS 打断 — 160ms 精度</b></summary>

`synthesize_stream()` 逐 chunk yield (~160ms),每个 chunk 发送前检查 `_cancel_speaking`。
```
旧: TTS("整句") → 3s 音频一次推入 → 打断无效 ❌
新: TTS.stream() → chunk→send→check → 打断!→停 ✅
```
</details>

<details>
<summary><b>2. 事件循环让出 — SPEAKING 状态打断修复</b></summary>

`asyncio.sleep(0.05)` 强制 50ms 间隔,事件循环有时间处理 1-2 个麦克风 chunk (32ms/个)。
</details>

<details>
<summary><b>3. Turn 序列号音频过滤 — 消除在途帧</b></summary>

服务端每轮回复前发 `audio_start(turn=N)`,打断时前端设 `playableTurn=0`,不匹配的帧全部丢弃。
</details>

<details>
<summary><b>4. 投机式预推理 — 节省 117ms</b></summary>

用户停顿 160ms 时,Moonshine Tiny (27M, ONNX CPU) 后台启动投机 ASR,endpointing 确认后直接复用。
</details>

<details>
<summary><b>5. 自适应 Endpointing + ASR 文本累积</b></summary>

| 说话时长 | 静默阈值 | 场景 |
|---|---|---|
| < 0.5s | 640ms | "嗯..."思考中 |
| 0.5~3s | 416ms | 正常对话 |
| > 3s | 640ms | 长句 |

短句 (≤4字) 不立即送 LLM,缓冲等后续语音拼接。
</details>

<details>
<summary><b>6. PTT Demo Mode — 零 VAD 延迟</b></summary>

按住说话 → 松手 → ASR → RAG → LLM → TTS,跳过 VAD/endpointing/filler,延迟只取决于推理速度。
</details>

---

## 🔮 路线 B:端到端 Omni (omni/)

**核心思路**:MiniCPM-o 4.5 内置 Whisper-medium 音频编码器 + Qwen3-8B,直接理解音频,不需要独立 ASR 模块——消除级联误差传播,也消掉了 ASR 的 200ms。

**vLLM 是成败关键**——同一个模型,不同推理方式差 89 倍:

| | Raw Transformers | vLLM bf16 | vLLM AWQ-Marlin |
|---|---|---|---|
| TTFT | 3400ms | 48ms | **38ms** |
| Tokens/sec | 15 | 48 | **109** |
| VRAM | 19.8 GB | 22.6 GB | **10.5 GB** |

实测 (RTX 4090 x2):LLM TTFT 50ms (p50) · TTS TTFA ~190ms · **首帧音频 ~250ms** · 109 tok/s。

工程能力与级联版共享同一套底座:Speaker-Aware VAD (ECAPA-TDNN)、鲁棒打断、Turn 序列号过滤、RAG、多轮上下文。完整文档与 Quick Start 见 [omni/README.md](omni/README.md)。

---

## 🎓 Posttraining:话术定制

[sft/](sft/) 目录包含外呼话术 LoRA SFT 全流程:数据构造 → LoRA 微调 → merge + GPTQ 量化 → 偏离度评测。单卡数小时、成本 ~$20,即可把通用模型定制成领域话术,评测数据在 [data/](data/)。

---

## 🚀 Quick Start (级联主力版)

```bash
# 1. LLM 推理服务 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-14B-AWQ --served-model-name Qwen3-14B-AWQ \
  --trust-remote-code --dtype auto --quantization awq \
  --gpu-memory-utilization 0.85 --max-model-len 4096 --enforce-eager --port 8100

# 2. Voice Agent — Full-Duplex Mode
CUDA_VISIBLE_DEVICES=2,7 ASR_DEVICE=cuda:1 TTS_DEVICE=cuda:0 \
  USE_FIRERED_ASR=1 USE_SMART_TURN=1 python ws_server.py

# 3. Voice Agent — PTT Demo Mode
DEMO_MODE=1 CUDA_VISIBLE_DEVICES=2,7 ASR_DEVICE=cuda:1 TTS_DEVICE=cuda:0 \
  USE_FIRERED_ASR=1 python ws_server.py
```

Omni 版启动见 [omni/README.md](omni/README.md)。

### 环境变量

| 变量 | 说明 |
|---|---|
| `DEMO_MODE=1` | PTT 演示模式 (按住说话) |
| `USE_FIRERED_ASR=1` | FireRedASR2 (CER 2.89%) |
| `USE_MOONSHINE_ASR=1` | 投机式 ASR |
| `USE_SMART_TURN=1` | Smart Turn v3 |
| `USE_SPEAKER_VAD=1` | ECAPA-TDNN 声纹 VAD |
| `USE_DENOISE=1` | DTLN 降噪 |

---

## 🔄 可替换组件

| 组件 | 当前 | 可替换为 |
|---|---|---|
| ASR | FireRedASR2-AED | Whisper · Paraformer · SenseVoice |
| LLM | Qwen3-14B-AWQ / vLLM | MiniCPM · DeepSeek · 任何 OpenAI 兼容 |
| TTS | VoxCPM 1.5 / nanovllm | CosyVoice · IndexTTS · Fish-Speech |
| VAD | Silero / ECAPA-TDNN | FireRedVAD · WebRTC VAD |
| RAG | bge-small + FAISS | bge-m3 · Milvus · Elasticsearch |
| 传输 | WebSocket | LiveKit WebRTC · SIP |

---

## 📁 仓库结构

```
├── ws_server.py            # 级联主力版:全双工 WS 服务 (VAD→ASR→RAG→LLM→TTS)
├── engine/                 # ASR / TTS / LLM / VAD / 降噪 各组件引擎
├── sft/                    # 话术定制:LoRA SFT → 量化 → 评测全流程
├── data/                   # 外呼 benchmark 与 SFT 对比评测数据
├── static/                 # 前端 (AudioWorklet 播放器)
├── livekit_agent/          # LiveKit WebRTC 接入
└── omni/                   # 端到端 Omni 路线 (原 Hybrid-VoiceAgent, 250ms)
    ├── ws_server_hybrid.py #   vLLM Omni + VoxCPM 混合 pipeline
    ├── engine/omni.py      #   MiniCPM-o 4.5 原生引擎
    └── README.md           #   Omni 路线完整文档
```

## 📋 版本演进

| 版本 | 核心改动 |
|---|---|
| v1.0 – v2.9 | 全双工 pipeline → 5 状态机 → 双层打断 → FireRedASR2 → 崩溃恢复 |
| v3.0 | Qwen3-14B-AWQ,声纹门控,ASR 文本累积 |
| v3.1 | PTT Demo Mode,句子 Cap,Watchdog 守护 |
| **2026-08** | **合并端到端 Omni 路线 (原 Hybrid-VoiceAgent) 入 omni/,仓库更名 VoiceAgent-600ms** |

---

<div align="center">

**Apache 2.0** (根目录) · **MIT** ([omni/](omni/LICENSE)) · Built with ❤️ on RTX 4090

</div>
