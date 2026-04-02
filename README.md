<div align="center">

# 🎙️ 500ms-Voice Agent

### ASR → RAG → LLM → TTS &nbsp;·&nbsp; Sub-500ms First Response &nbsp;·&nbsp; Voice Cloning

<br>

[![Pipeline Latency](https://img.shields.io/badge/Pipeline_Latency-458ms-00c853?style=for-the-badge&logo=speedtest&logoColor=white)](/)
[![Barge-in](https://img.shields.io/badge/Barge--in-160ms-2979ff?style=for-the-badge&logo=bolt&logoColor=white)](/)
[![ASR CER](https://img.shields.io/badge/ASR_CER-2.89%25-ff6f00?style=for-the-badge&logo=microphone&logoColor=white)](/)
[![RTF](https://img.shields.io/badge/TTS_RTF-0.08x-9c27b0?style=for-the-badge&logo=waveform&logoColor=white)](/)
[![Version](https://img.shields.io/badge/v3.1-Stable-7c4dff?style=for-the-badge)](/)

<br>

> **800ms 是语音对话的图灵测试门槛** — 超过这个延迟，人类会感知到"在和机器说话"。
> 本项目通过四重工程极致优化，实现 **458ms 首响**，远低于这个门槛。

<br>

<!-- 🎬 Demo Video Placeholder -->
<table><tr><td align="center">
<br>
  
<img width="1750" height="1714" alt="超低延迟" src="https://github.com/user-attachments/assets/1c2dede4-e1b7-4cac-9f5c-59ebabfec039" />




**↑ 实机演示：PTT模式 → 458ms 首响 → 流式语音克隆回复**

<br>
</td></tr></table>

</div>

---

## ⚡ 为什么快？— 四重极致工程优化

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
          ┌──────────────────────┐   │            │
          │  🧠 LLM (Streaming)  │   │  ⏱️ 458ms  │
          │  Qwen3-14B via vLLM  │   │  总延迟    │
          │  ~163ms to 1st token │   │            │
          └──────────┬───────────┘   │            │
                     │               │            │
                     ▼               │            │
          ┌──────────────────────┐   │            │
          │  🔊 TTS (Streaming)   │   │            │
          │  VoxCPM via nanovllm  │   │            │
          │  ~174ms to 1st chunk  │   │            │
          └──────────────────────┘   │            │
                     │               │            │
                     ▼               ▼            ▼
               用户听到第一个音节 ◄──────────────┘
```

<table>
<tr>
<td width="25%" align="center">

### 🎤 ASR 加速
**FireRedASR2-AED**<br>
1.15B 参数 · CER 2.89%<br>
20+ 方言 · 噪声鲁棒<br>
<br>
<code>~200ms</code>

</td>
<td width="25%" align="center">

### 🧠 LLM 加速
**vLLM 0.16**<br>
PagedAttention · AWQ 量化<br>
Continuous Batching<br>
<br>
<code>~163ms TTFT</code>

</td>
<td width="25%" align="center">

### 🔊 TTS 加速
**nanovllm-voxcpm**<br>
CUDA Graph · torch.compile<br>
逐 chunk 流式 · 语音克隆<br>
<br>
<code>~174ms TTFA</code>

</td>
<td width="25%" align="center">

### 📚 RAG 加速
**bge-small + FAISS**<br>
512d embedding · IndexFlatIP<br>
59 docs 精确搜索<br>
<br>
<code>~4ms</code>

</td>
</tr>
</table>

---

## 📊 同硬件三方案对比

> 在同一台 RTX 4090 上实测，Pipeline 方案综合最优：

| 方案 | 首响延迟 | 打断精度 | 换音色 | 换话术 |
|:---:|:---:|:---:|:---:|:---:|
| **🏆 Pipeline (本项目)** | **458ms** | **160ms** | 语音克隆 · 小时级 | SFT · ~$20 |
| Hybrid (Omni + TTS) | ~250ms | 160ms | 语音克隆 | Omni SFT |
| 纯 Omni (raw transformers) | 1666ms ❌ | 模型原生 | 重训整个模型 | ~$5000+ |

<br>

## 🔬 延迟实测

<table>
<tr><th>组件</th><th>RTX 4090</th><th>RTX 5090</th><th>RTX 4080S</th></tr>
<tr><td>🎤 ASR</td><td><b>200ms</b></td><td>93ms</td><td>130ms</td></tr>
<tr><td>📚 RAG</td><td><b>4ms</b></td><td>3ms</td><td>4ms</td></tr>
<tr><td>🧠 LLM</td><td><b>163ms</b></td><td>130ms</td><td>225ms</td></tr>
<tr><td>🔊 TTS</td><td><b>174ms</b></td><td>138ms</td><td>115ms</td></tr>
<tr><td><b>🚀 Pipeline</b></td><td><b>458ms</b></td><td><b>342ms</b></td><td><b>470ms</b></td></tr>
</table>

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│  客户端 — 浏览器 / 原生 macOS·iPad App / WebRTC                        │
│  麦克风 → PCM 16kHz → 服务端  |  服务端 → PCM 44.1kHz (流式) → 扬声器  │
└────────────────────────────┬────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ConversationManager v3.1 — 状态机 + PTT Demo Mode                     │
│                                                                         │
│  ┌─────┐  ┌──────────┐  ┌─────────┐  ┌─────┐  ┌───────┐  ┌─────────┐ │
│  │ VAD │→ │Turn 检测  │→ │  ASR    │→ │ RAG │→ │  LLM  │→ │TTS 流式 │ │
│  │Silero│  │Smart Turn│  │FireRed  │  │ bge │  │Qwen3  │  │ VoxCPM  │ │
│  │ CPU │  │v3 · ONNX │  │ASR2-AED │  │small│  │14B-AWQ│  │  1.5    │ │
│  └─────┘  └──────────┘  └─────────┘  └─────┘  └───────┘  └─────────┘ │
│                                                                         │
│  可选: 声纹VAD · 投机ASR · DTLN降噪 · ASR文本累积器 · 句子Cap          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ 核心工程优化

<details>
<summary><b>1. 流式 TTS 打断 — 160ms 精度</b></summary>

`synthesize_stream()` 逐 chunk yield (~160ms)，每个 chunk 发送前检查 `_cancel_speaking`。
```
旧: TTS("整句") → 3s 音频一次推入 → 打断无效 ❌
新: TTS.stream() → chunk→send→check → chunk→send→check → 打断!→停 ✅
```
</details>

<details>
<summary><b>2. 事件循环让出 — SPEAKING 状态打断修复</b></summary>

`asyncio.sleep(0.05)` 强制 50ms 间隔，事件循环有时间处理 1-2 个麦克风 chunk (32ms/个)。
</details>

<details>
<summary><b>3. Turn 序列号音频过滤 — 消除在途帧</b></summary>

服务端每轮回复前发 `audio_start(turn=N)`，打断时前端设 `playableTurn=0`，不匹配的帧全部丢弃。
</details>

<details>
<summary><b>4. 投机式预推理 — 节省 117ms</b></summary>

用户停顿 160ms 时，Moonshine Tiny (27M, ONNX CPU) 后台启动投机 ASR，endpointing 确认后直接复用。
</details>

<details>
<summary><b>5. 自适应 Endpointing + ASR 文本累积</b></summary>

| 说话时长 | 静默阈值 | 场景 |
|---|---|---|
| < 0.5s | 640ms | "嗯..."思考中 |
| 0.5~3s | 416ms | 正常对话 |
| > 3s | 640ms | 长句 |

短句 (≤4字) 不立即送 LLM，缓冲等后续语音拼接。
</details>

<details>
<summary><b>6. PTT Demo Mode — 零 VAD 延迟</b></summary>

按住说话 → 松手 → ASR → RAG → LLM → TTS，跳过 VAD/endpointing/filler，延迟只取决于推理速度。
</details>

---

## 🚀 Quick Start

```bash
# 1. LLM 推理服务 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-14B-AWQ --served-model-name Qwen3-14B-AWQ \
  --trust-remote-code --dtype auto --quantization awq \
  --gpu-memory-utilization 0.85 --max-model-len 4096 --enforce-eager --port 8100

# 2. Voice Agent — Full-Duplex Mode (GPU 2+7)
CUDA_VISIBLE_DEVICES=2,7 ASR_DEVICE=cuda:1 TTS_DEVICE=cuda:0 \
  USE_FIRERED_ASR=1 USE_SMART_TURN=1 python ws_server.py

# 3. Voice Agent — PTT Demo Mode
DEMO_MODE=1 CUDA_VISIBLE_DEVICES=2,7 ASR_DEVICE=cuda:1 TTS_DEVICE=cuda:0 \
  USE_FIRERED_ASR=1 python ws_server.py
```

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

## 🔌 API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/ws/voice` | WS | 全双工 / PTT 语音通道 |
| `/api/info` | GET | 模型与配置信息 |
| `/api/rag/docs` | GET | 知识库文档列表 |
| `/api/rag/query?q=` | GET | 检索测试 |
| `/api/rag/reload` | POST | 热更新知识库 |

---

## 🔄 可替换组件

> 每个组件可独立替换，无需改动其他模块：

| 组件 | 当前 | 可替换为 |
|---|---|---|
| ASR | FireRedASR2-AED | Whisper · Paraformer · SenseVoice |
| LLM | Qwen3-14B-AWQ / vLLM | MiniCPM · DeepSeek · 任何 OpenAI 兼容 |
| TTS | VoxCPM 1.5 / nanovllm | CosyVoice · IndexTTS · Fish-Speech |
| VAD | Silero / ECAPA-TDNN | FireRedVAD · WebRTC VAD |
| RAG | bge-small + FAISS | bge-m3 · Milvus · Elasticsearch |
| 传输 | WebSocket | LiveKit WebRTC · SIP |

---

## 📋 版本演进

| 版本 | 核心改动 |
|---|---|
| v1.0 | 基础全双工 pipeline，语音克隆 |
| v2.0 | 5 状态机，句级流式，零爆音播放 |
| v2.1 | 双层打断，自适应 endpointing |
| v2.4 | FireRedASR2，流式 TTS 打断，事件循环修复 |
| v2.9 | 崩溃恢复，Filler 激活，状态竞态修复 |
| v3.0 | Qwen3-14B-AWQ，声纹门控，ASR 文本累积 |
| **v3.1** | **PTT Demo Mode，句子 Cap，Watchdog 守护** |

---

<div align="center">

**Apache 2.0 Licensed** · Built with ❤️ on RTX 4090

</div>
