<h1 align="center">
  端到端全双工语音 Agent<br>
  <small>End-to-End Full-Duplex Voice Agent</small>
</h1>

<h4 align="center">
  从模型选型到工程落地的最佳实践<br>
  ASR → RAG → LLM → TTS 全链路 sub-500ms，流式打断，语音克隆，噪声鲁棒
</h4>

<p align="center">
  <img src="https://img.shields.io/badge/首响延迟-458ms-00c853?style=for-the-badge" alt="458ms">
  <img src="https://img.shields.io/badge/打断精度-160ms-2196f3?style=for-the-badge" alt="160ms">
  <img src="https://img.shields.io/badge/ASR_CER-2.89%25-ff6f00?style=for-the-badge" alt="CER">
  <img src="https://img.shields.io/badge/v2.4-Latest-7c4dff?style=for-the-badge" alt="v2.4">
</p>

---

## 为什么做这个项目

市面上的语音 Agent 方案分两类：

1. **端到端 Omni 模型**（MiniCPM-o、GPT-4o Realtime）— 一个模型搞定 ASR+LLM+TTS，对话自然但延迟高（>1.5s）、无法定制、后训练成本极高
2. **Pipeline 级联方案**（ASR → LLM → TTS）— 各组件可独立优化，延迟可控，但打断体验差、工程复杂度高

**本项目证明：经过系统优化的 Pipeline 方案可以做到 sub-500ms 延迟 + 流式打断 + 噪声鲁棒，同时保持每个组件可独立替换/微调的灵活性。**

我们在同一硬件上对比了三种方案：

| 方案 | 首响延迟 | 打断体验 | 换音色成本 | 换话术成本 |
|---|---|---|---|---|
| **Pipeline (本项目)** | **458ms** | 160ms 流式打断 | 语音克隆，小时级 | LLM SFT，~$20 |
| Hybrid (vLLM Omni + TTS) | ~250ms | 160ms | 语音克隆 | Omni SFT |
| 纯 Omni (raw transformers) | 1666ms | 模型原生 | 重训整个模型 | ~$5000+ |

---

## 系统架构与模型选型

```
┌─────────────────────────────────────────────────────────────────────────┐
│  浏览器 (WebSocket / LiveKit WebRTC)                                    │
│  麦克风 → PCM 16kHz → 服务端  |  服务端 → PCM 44.1kHz (流式) → 扬声器  │
│  客户端: 即时 stopPlayback + Turn 序列号过滤在途帧                      │
└────────────────────────────┬────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ConversationManager v2.4 — 5 状态有限状态机                            │
│  IDLE → LISTENING → THINKING → SPEAKING → INTERRUPTED                  │
│                                                                         │
│  ┌─────┐  ┌──────────┐  ┌─────────┐  ┌─────┐  ┌───────┐  ┌─────────┐ │
│  │ VAD │→ │Turn 检测  │→ │  ASR    │→ │ RAG │→ │  LLM  │→ │TTS 流式 │ │
│  │     │  │          │  │         │  │     │  │       │  │         │ │
│  │Silero│  │Smart Turn│  │FireRed  │  │ bge │  │MiniCPM│  │ VoxCPM  │ │
│  │ CPU │  │v3 8M ONNX│  │ASR2-AED │  │small│  │4.1-8B │  │  1.5    │ │
│  │     │  │ CPU 0.7ms│  │CER 2.89%│  │FAISS│  │ vLLM  │  │nanovllm │ │
│  └─────┘  └──────────┘  └─────────┘  └─────┘  └───────┘  └─────────┘ │
│                                                                         │
│  可选组件: ECAPA-TDNN 声纹VAD │ Moonshine 投机ASR │ DTLN 降噪          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 为什么选这些模型

| 组件 | 选型 | 为什么不选其他 |
|---|---|---|
| **ASR** | **FireRedASR2-AED** (1.15B) | CER 2.89%，超越 Doubao-ASR / Qwen3-ASR / FunASR；20+ 方言含粤语；多噪声场景训练，办公/街道/车内都能用。SenseVoice 在嘈杂环境识别为空。 |
| **LLM** | **MiniCPM4.1-8B-GPTQ** | 面壁智能 8B 模型，GPTQ-Marlin 量化仅 4.9GB；支持 `enable_thinking=False` 禁用推理链；30+ 语种。Qwen-AWQ 在 Blackwell GPU 上不兼容 awq_marlin kernel。 |
| **TTS** | **VoxCPM 1.5** | 面壁智能端到端语音合成，44.1kHz 高音质；8 秒样本即可克隆音色；`generate()` 是原生 generator，支持逐 chunk 流式输出。CosyVoice 延迟更高且不支持 nanovllm 加速。 |
| **RAG** | **bge-small-zh-v1.5** (91MB) | 512 维 embedding，3.6ms 检索延迟；对 1000 条 FAQ 量级完全够用。bge-m3 (2.2GB) 多语言更强但延迟 16ms，当前场景无需。 |
| **Turn 检测** | **Pipecat Smart Turn v3** (8M) | 纯音频韵律分析，CPU 0.7ms；BSD-2 开源。VoTurn-80M 精度更高 (94.1%) 但需 GPU；LiveKit EOU 只看文本不听语音。 |
| **VAD** | **Silero VAD** | 2.2MB，CPU 推理，32ms 粒度，久经验证。FireRedVAD (F1 97.57%) 更强但对环境噪音过于敏感，需进一步调参。 |

### 为什么选这些 Infra

| Infra | 选型 | 为什么 |
|---|---|---|
| **LLM 推理** | **vLLM 0.16** | 工业级 LLM serving：PagedAttention + GPTQ-Marlin kernel + Continuous Batching + OpenAI 兼容 API。比 raw transformers 快 10-50x。比 TGI 社区更活跃，多模态支持更好。 |
| **TTS 推理** | **nanovllm-voxcpm** | VoxCPM 专用轻量推理引擎：~1200 行代码，CUDA Graph + torch.compile，单进程无 IPC 开销。vLLM 不支持 VoxCPM 架构，raw PyTorch warmup 需 155 秒。 |
| **WebSocket** | **FastAPI + uvicorn** | 原生 async WebSocket 支持，`nest_asyncio` 兼容 nanovllm 子进程事件循环。比 Flask-SocketIO 性能高 5-10x。 |
| **WebRTC** | **LiveKit** (实验) | 开源 Go SFU，Apache 2.0；原生 SIP Bridge 接电话；UDP 传输无 TCP 在途帧问题。FireRedChat 同样选择了 LiveKit。 |
| **向量检索** | **FAISS IndexFlatIP** | 1000 条文档精确搜索 <0.1ms。无需 Milvus/Elasticsearch 的运维复杂度。超过 10 万条再考虑 ANN 索引。 |

---

## 核心工程优化

### 1. 流式 TTS 打断 — 从"整句不可打断"到"160ms 精度"

**问题**：TTS `synthesize()` 一次性生成整句音频（3-5 秒），全部推入 WebSocket TCP 管道。打断时帧已在途，无法撤回。

**解法**：`synthesize_stream()` 逐 chunk yield（~160ms），每个 chunk 发送前检查 `_cancel_speaking`。

```
v2.3:  TTS("整句") → 3s 音频一次推入 → 打断无效
v2.4:  TTS.stream("整句") → chunk1→send→check → chunk2→send→check → 打断!→停
```

### 2. 事件循环让出 — SPEAKING 状态打断修复

**问题**：TTS chunk 发送循环中 `asyncio.sleep(0)` 不会真正让出事件循环（WebSocket send 非阻塞）。`feed_audio` 得不到执行，SPEAKING 状态下 VAD 无法检测用户语音。

**解法**：`asyncio.sleep(0.05)` 强制 50ms 间隔 → 事件循环有时间处理 1-2 个麦克风 chunk（32ms/个）。

### 3. Turn 序列号音频过滤

**问题**：Cloudflare tunnel 延迟不固定（100-500ms），时间窗口无法可靠过滤在途帧。

**解法**：服务端每轮回复前发 `audio_start(turn=N)`，打断时前端设 `playableTurn=0`。不匹配的帧全部丢弃，不依赖时间。

### 4. 投机式预推理

用户停顿 160ms 时，Moonshine Tiny（27M，ONNX CPU）在后台启动投机 ASR。endpointing 确认后若音频未明显增长，直接复用结果，节省 ~117ms。

### 5. 自适应 Endpointing

| 用户说话时长 | 静默阈值 | 场景 |
|---|---|---|
| < 0.5s | 192ms | "好的"、"嗯" |
| 0.5 ~ 3s | 320ms | 正常对话 |
| > 3s | 640ms | 长句、思考停顿 |

---

## 延迟实测 (RTX 4090, warmup)

| 组件 | 延迟 | 说明 |
|---|---|---|
| ASR (FireRedASR2) | **~200ms** | CER 2.89%，噪声鲁棒 |
| RAG (bge-small + FAISS) | **3.6ms** | 59 条知识库 |
| LLM (MiniCPM4.1 GPTQ via vLLM) | **~163ms** | 句级流式 |
| TTS TTFA (VoxCPM 流式) | **~174ms** | 首 chunk 延迟 |
| **Pipeline 首响** | **~458ms** | 目标 < 500ms ✅ |

### 多 GPU 横评

| 硬件 | ASR | LLM | TTS | Pipeline |
|---|---|---|---|---|
| RTX 4080S (32GB) | 130ms | 225ms | 115ms | 470ms |
| RTX 5090 (32GB) | 93ms | 130ms | 138ms | 342ms |
| **RTX 4090 (24GB)** | **200ms** | **163ms** | **174ms** | **458ms** |

---

## Quick Start

```bash
# 1. LLM 推理服务 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model models/MiniCPM4.1-8B-GPTQ --served-model-name MiniCPM4.1-8B-GPTQ \
  --trust-remote-code --dtype auto --quantization gptq_marlin \
  --gpu-memory-utilization 0.40 --max-model-len 2048 --enforce-eager --port 8100

# 2. Voice Agent (GPU 2)
CUDA_VISIBLE_DEVICES=2 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 \
  USE_FIRERED_ASR=1 USE_MOONSHINE_ASR=1 USE_SMART_TURN=1 \
  python ws_server.py

# 3. 打开浏览器 → https://localhost:3000 → Start Call
```

### 环境变量开关

所有高级组件通过环境变量开关启用，默认使用基础配置：

| 变量 | 说明 |
|---|---|
| `USE_FIRERED_ASR=1` | 启用 FireRedASR2-AED（CER 2.89%，噪声鲁棒） |
| `USE_MOONSHINE_ASR=1` | 启用 Moonshine Tiny 投机式 ASR |
| `USE_SMART_TURN=1` | 启用 Smart Turn v3 语义 endpointing |
| `USE_SPEAKER_VAD=1` | 启用 ECAPA-TDNN 声纹 VAD |
| `USE_DENOISE=1` | 启用 DTLN 降噪（4MB ONNX） |
| `TTS_GPU_UTIL=0.45` | TTS 显存占比 |

---

## 项目结构

```
voiceagent/
├── engine/                         # ML 引擎层（全部可独立替换）
│   ├── asr_firered.py              #   FireRedASR2-AED (推荐)
│   ├── asr.py                      #   SenseVoiceSmall (备选)
│   ├── asr_moonshine.py            #   Moonshine Tiny (投机)
│   ├── llm.py                      #   vLLM 流式推理
│   ├── tts.py                      #   VoxCPM 1.5 流式合成
│   ├── vad.py / speaker_vad.py     #   Silero / ECAPA-TDNN
│   ├── turn_detector.py            #   Pipecat Smart Turn v3
│   ├── rag.py                      #   bge-small + FAISS
│   ├── captioner.py                #   副语言感知
│   ├── denoiser.py                 #   DTLN 降噪 (可选)
│   └── conversation_manager.py     #   v2.4 状态机核心
│
├── livekit_agent/                  # WebRTC 传输层 (实验)
├── static/voice_agent.html         # WebSocket 前端
├── ws_server.py                    # FastAPI 服务器
└── config.py                       # 配置
```

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 语音 Agent 控制台 |
| `/ws/voice` | WS | 全双工语音通道 |
| `/api/info` | GET | 模型信息 |
| `/api/rag/docs` | GET | 知识库文档列表 |
| `/api/rag/query?q=` | GET | 检索测试 |
| `/api/rag/reload` | POST | 热更新知识库 |

---

## 版本演进

| 版本 | 核心改动 |
|---|---|
| v1.0 | 基础全双工 pipeline，语音克隆 |
| v2.0 | 5 状态机，句级流式，零爆音播放 |
| v2.1 | 双层打断，自适应 endpointing，THINKING 缓冲 |
| v2.2 | 投机式 ASR，ECAPA-TDNN 声纹 VAD，副语言感知 |
| v2.3 | Moonshine，Smart Turn v3，JitterBuffer |
| **v2.4** | **FireRedASR2，流式 TTS 打断，事件循环修复，LiveKit 实验** |

---

## 可扩展性

每个组件可独立替换，无需改动其他模块：

| 组件 | 当前 | 可替换为 |
|---|---|---|
| ASR | FireRedASR2-AED | Whisper, Paraformer, SenseVoice, Moonshine |
| LLM | MiniCPM4.1-GPTQ / vLLM | Qwen, LLaMA, DeepSeek, 任何 OpenAI 兼容 |
| TTS | VoxCPM 1.5 / nanovllm | CosyVoice, IndexTTS, GPT-SoVITS, Fish-Speech |
| VAD | Silero / ECAPA-TDNN | FireRedVAD, WebRTC VAD |
| Turn | Smart Turn v3 | VoTurn-80M, LiveKit EOU, BERT-based |
| RAG | bge-small + FAISS | bge-m3, Milvus, Elasticsearch |
| 降噪 | DTLN | RNNoise, FastEnhancer |
| 传输 | WebSocket | LiveKit WebRTC, SIP |
