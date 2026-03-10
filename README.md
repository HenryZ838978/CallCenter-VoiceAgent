<h1 align="center">VoxLabs — Full-Duplex Voice Agent</h1>

<p align="center">
  <b>实时中文语音客服 Agent</b> — sub-500ms 首响、流式打断、语音克隆、噪声鲁棒<br>
  面向 BPO / 智能外呼场景
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pipeline-458ms-00c853?style=for-the-badge" alt="Pipeline 458ms">
  <img src="https://img.shields.io/badge/Barge--in-160ms_max-2196f3?style=for-the-badge" alt="Barge-in 160ms">
  <img src="https://img.shields.io/badge/ASR_CER-2.89%25-ff6f00?style=for-the-badge" alt="CER 2.89%">
  <img src="https://img.shields.io/badge/Version-v2.4-7c4dff?style=for-the-badge" alt="v2.4">
  <img src="https://img.shields.io/badge/GPU-RTX_4090-76ff03?style=for-the-badge" alt="RTX 4090">
</p>

---

## 系统架构

<table align="center" style="border-collapse:collapse;">
<tr><td align="center" colspan="6" style="padding:16px;background:#1a237e;color:#fff;border-radius:12px 12px 0 0;">
<b>Browser / SIP Client</b><br>
<small>Mic → PCM 16kHz → WebSocket → Server &nbsp;|&nbsp; Speaker ← PCM 44.1kHz ← Server</small><br>
<small>客户端打断: RMS 检测 → stopPlayback() (0ms) &nbsp;|&nbsp; Turn 序列号过滤在途帧</small>
</td></tr>
<tr><td align="center" colspan="6" style="padding:4px;color:#90caf9;">▼ WSS / WebRTC (LiveKit 实验性)</td></tr>
<tr><td align="center" colspan="6" style="padding:12px;background:#0d47a1;color:#fff;">
<b>FastAPI WebSocket Server (:3000)</b>
</td></tr>
<tr><td align="center" colspan="6" style="padding:12px;background:#1565c0;color:#e3f2fd;border-bottom:2px solid #0d47a1;">
<b>ConversationManager v2.4 — 5 状态机</b><br>
<code>IDLE → LISTENING → THINKING → SPEAKING → INTERRUPTED</code><br>
<small>流式 TTS 打断 · 投机式 ASR · 语义 Endpointing · 副语言感知 · Turn 序列号</small>
</td></tr>
<tr>
<td align="center" style="padding:10px;background:#004d40;color:#a7ffeb;width:16%;border-radius:0 0 0 12px;">
<b>VAD</b><br><small>Silero / ECAPA<br>声纹识别</small><br>
<code>CPU</code>
</td>
<td align="center" style="padding:10px;background:#1b5e20;color:#b9f6ca;width:16%;">
<b>ASR</b><br><small>FireRedASR2-AED<br>1.15B FP16</small><br>
<code>~200ms</code>
</td>
<td align="center" style="padding:10px;background:#006064;color:#b2ebf2;width:16%;">
<b>Turn</b><br><small>Pipecat Smart<br>Turn v3</small><br>
<code>CPU 0.7ms</code>
</td>
<td align="center" style="padding:10px;background:#e65100;color:#fff3e0;width:16%;">
<b>RAG</b><br><small>bge-small-zh<br>+ FAISS</small><br>
<code>3.6ms</code>
</td>
<td align="center" style="padding:10px;background:#4a148c;color:#e1bee7;width:16%;">
<b>LLM</b><br><small>MiniCPM4.1-8B<br>GPTQ-Marlin</small><br>
<code>163ms</code>
</td>
<td align="center" style="padding:10px;background:#b71c1c;color:#ffcdd2;width:16%;border-radius:0 0 12px 0;">
<b>TTS</b><br><small>VoxCPM 1.5<br>流式 nanovllm</small><br>
<code>174ms</code>
</td>
</tr>
</table>

---

## v2.4 核心改进

### 流式 TTS 打断 — 从"整句不可打断"到"160ms 精度"

```
v2.3 (批量TTS):
  synthesize("一整句") → 3秒音频一次性推入WebSocket → 打断时已全在TCP管道 → 无法撤回

v2.4 (流式TTS):
  synthesize_stream() → chunk1(160ms) → 发送 → check cancel
                      → chunk2(160ms) → 发送 → 用户说"停" → cancel! → 立即停止
```

### FireRedASR2 — 噪声鲁棒 ASR (CER 2.89%)

<table align="center">
<tr><th>指标</th><th>SenseVoice (v2.3)</th><th>FireRedASR2-AED (v2.4)</th></tr>
<tr><td>普通话 CER</td><td>~5%</td><td><b>2.89%</b></td></tr>
<tr><td>方言</td><td>5 语种</td><td><b>20+ 方言 (含粤语)</b></td></tr>
<tr><td>噪声鲁棒</td><td>弱</td><td><b>多噪声场景训练</b></td></tr>
<tr><td>延迟 (warmup)</td><td>117ms</td><td>~200ms</td></tr>
</table>

### 事件循环修复 — SPEAKING 状态可持续检测打断

TTS chunk 发送间加 `asyncio.sleep(0.05)` 强制让出事件循环 → `feed_audio` 持续处理麦克风 → VAD 可在 SPEAKING 状态检测到用户语音并触发打断。

### Turn 序列号音频过滤

```
正常: audio_start(turn=N) → playableTurn=N → 播放
打断: barge_in → playableTurn=0 → 丢弃所有在途帧 (无论积压多少)
新轮: audio_start(turn=N+1) → playableTurn=N+1 → 恢复播放
```

---

## 全双工打断体系

<table align="center">
<tr>
<td align="center" style="padding:16px;background:#e3f2fd;border:2px solid #2196f3;border-radius:12px;">
<h3>Layer 1 — 客户端即时层</h3>
<b style="font-size:28px;color:#2196f3;">0ms</b><br><br>
RMS 检测 → stopPlayback()<br>
Turn 序列号 → 丢弃在途帧<br>
<b>用户感知零延迟</b>
</td>
<td align="center" style="padding:8px;font-size:24px;">→</td>
<td align="center" style="padding:16px;background:#f3e5f5;border:2px solid #9c27b0;border-radius:12px;">
<h3>Layer 2 — 服务端流式取消</h3>
<b style="font-size:28px;color:#9c27b0;">~160ms</b><br><br>
VAD ≥ 0.6 + RMS > 0.008<br>
2 chunks 确认 (64ms)<br>
<b>流式 TTS 逐 chunk cancel</b>
</td>
</tr>
</table>

### 语义 Endpointing (Smart Turn v3)

Pipecat Smart Turn v3 (8M ONNX, CPU 0.7ms) 分析音频韵律判断用户是否说完。替代固定静默阈值。

### 投机式 ASR (Moonshine Tiny)

静默 160ms 即启动 Moonshine Tiny (27M, ONNX CPU) 预推理。endpointing 确认时若音频未变化，直接复用结果。

---

## 核心特性

<table>
<tr>
<td width="50%" valign="top">

**语音克隆** — 8s 参考音频 → 克隆音色 (VoxCPM 1.5)

**句级流式** — LLM 逐句生成 → TTS 逐 chunk 流式发送

**RAG 知识库** — bge-small + FAISS, `POST /api/rag/reload` 热更新

**副语言感知** — 语气/情绪分析注入 LLM prompt

</td>
<td width="50%" valign="top">

**噪声鲁棒** — FireRedASR2 多噪声场景训练 (CER 2.89%)

**打断聚合** — "用户依次说了：X；Y。请综合回应"

**THINKING 缓冲** — pipeline 运行期间用户追加语音不丢失

**LiveKit 实验** — WebRTC Agent Worker (VAD+STT+LLM 已通)

</td>
</tr>
</table>

---

## 三种方案对比

<table align="center">
<tr><th>维度</th><th>Pipeline (本项目)</th><th>Hybrid (姊妹项目)</th><th>纯 Omni</th></tr>
<tr><td>首响延迟</td><td>458ms ✅</td><td><b>~250ms ✅✅</b></td><td>1666ms ❌</td></tr>
<tr><td>打断精度</td><td><b>160ms (流式TTS)</b></td><td>160ms</td><td>模型原生</td></tr>
<tr><td>噪声鲁棒</td><td><b>FireRedASR2 CER 2.89%</b></td><td>Whisper-medium</td><td>内置</td></tr>
<tr><td>换音色</td><td><b>LoRA / clone, 小时级</b></td><td>VoxCPM clone</td><td>重训模型</td></tr>
<tr><td>换话术</td><td><b>LLM SFT, ~$20</b></td><td>Omni SFT</td><td>~$5000+</td></tr>
<tr><td>换 ASR/LLM/TTS</td><td><b>替换单个模块</b></td><td>绑定</td><td>绑定</td></tr>
</table>

### 可扩展性

<table align="center">
<tr><th>组件</th><th>当前方案</th><th>可替换为</th></tr>
<tr><td>ASR</td><td>FireRedASR2-AED / SenseVoice</td><td>Whisper, Paraformer, Moonshine</td></tr>
<tr><td>LLM</td><td>MiniCPM4.1-8B-GPTQ / vLLM</td><td>Qwen, LLaMA, 任何 OpenAI 兼容</td></tr>
<tr><td>TTS</td><td>VoxCPM 1.5 流式 / nanovllm</td><td>CosyVoice, IndexTTS, GPT-SoVITS</td></tr>
<tr><td>VAD</td><td>Silero + ECAPA-TDNN</td><td>FireRedVAD, WebRTC VAD</td></tr>
<tr><td>Turn Detection</td><td>Pipecat Smart Turn v3 (8M)</td><td>VoTurn-80M, LiveKit EOU</td></tr>
<tr><td>降噪</td><td>DTLN (4MB ONNX, 可选)</td><td>RNNoise, FastEnhancer</td></tr>
</table>

---

## 项目结构

```
voiceagent/
├── engine/
│   ├── asr.py                    # SenseVoiceSmall / FunASR
│   ├── asr_firered.py            # FireRedASR2-AED (CER 2.89%)
│   ├── asr_moonshine.py          # Moonshine Tiny (投机式 ASR)
│   ├── fireredasr2/              # FireRedASR2 源码
│   ├── llm.py                    # vLLM streaming + sentence boundary
│   ├── tts.py                    # VoxCPM 1.5 流式 (synthesize_stream)
│   ├── vad.py                    # Silero VAD
│   ├── speaker_vad.py            # SpeakerAwareVAD (ECAPA-TDNN)
│   ├── firered_vad.py            # FireRedVAD wrapper
│   ├── turn_detector.py          # Pipecat Smart Turn v3
│   ├── rag.py                    # bge-small-zh-v1.5 + FAISS
│   ├── captioner.py              # 副语言感知
│   ├── denoiser.py               # DTLN 降噪 (可选)
│   ├── conversation_manager.py   # v2.4 状态机
│   └── filler.py                 # Filler word cache
│
├── livekit_agent/                # LiveKit WebRTC Agent (实验性)
│   ├── run.py                    # Agent Worker
│   ├── silero_vad_lk.py          # LiveKit VAD plugin
│   ├── playground.html           # WebRTC 前端
│   └── token_gen.py              # Token 生成器
│
├── static/
│   └── voice_agent.html          # WebSocket 前端 + Turn 序列号过滤
│
├── data/
│   ├── sample_kb.json            # 知识库 (59 FAQ)
│   └── voice_prompt.wav          # 语音克隆参考音频
│
├── models/                       # 模型权重 (不含在 repo 中)
│   ├── FireRedASR2-AED/          (4.5GB)
│   ├── MiniCPM4.1-8B-GPTQ/      (4.9GB)
│   ├── VoxCPM1.5/                (1.9GB)
│   ├── SenseVoiceSmall/          (901MB, 备选 ASR)
│   ├── spkrec-ecapa-voxceleb/    (ECAPA-TDNN)
│   ├── snakers4_silero-vad/
│   ├── bge-small-zh-v1.5/        (91MB)
│   └── dtln/                     (4MB, 可选降噪)
│
├── ws_server.py                  # FastAPI WebSocket 服务器
├── config.py                     # 模型路径、提示词、参数
├── start_all.sh                  # WebSocket 版一键启动
├── start_livekit.sh              # LiveKit 版启动
├── test_duplex.py                # 双工集成测试
├── test_ws_e2e.py                # WebSocket 端到端测试
└── ab_test_rag.py                # RAG A/B 测试
```

## Quick Start

```bash
# 1. vLLM (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model models/MiniCPM4.1-8B-GPTQ --served-model-name MiniCPM4.1-8B-GPTQ \
  --trust-remote-code --dtype auto --quantization gptq_marlin \
  --gpu-memory-utilization 0.40 --max-model-len 2048 --enforce-eager --port 8100

# 2. Voice Agent (GPU 2) — 推荐配置
CUDA_VISIBLE_DEVICES=2 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 \
  USE_FIRERED_ASR=1 USE_MOONSHINE_ASR=1 USE_SMART_TURN=1 \
  python ws_server.py

# 3. (可选) 公网访问
./cloudflared tunnel --url https://localhost:3000 --no-tls-verify
```

### 环境变量开关

| 变量 | 默认 | 说明 |
|---|---|---|
| `USE_FIRERED_ASR=1` | 0 | 启用 FireRedASR2 替代 SenseVoice |
| `USE_MOONSHINE_ASR=1` | 0 | 启用 Moonshine 投机式 ASR |
| `USE_SMART_TURN=1` | 0 | 启用 Smart Turn 语义 endpointing |
| `USE_SPEAKER_VAD=1` | 0 | 启用 ECAPA-TDNN 声纹 VAD |
| `USE_FIRERED_VAD=1` | 0 | 启用 FireRedVAD |
| `USE_DENOISE=1` | 0 | 启用 DTLN 降噪 |
| `TTS_GPU_UTIL` | 0.55 | TTS GPU 显存占比 |

## API

| 端点 | 方法 | 说明 |
|---|---|---|
| `/` | GET | Voice Agent Console |
| `/ws/voice` | WS | 全双工语音通道 |
| `/api/info` | GET | 模型信息 |
| `/api/rag/docs` | GET | 知识库文档 |
| `/api/rag/query?q=` | GET | RAG 检索测试 |
| `/api/rag/reload` | POST | 热更新知识库 |

## 版本历史

| 版本 | Commit | 特性 |
|---|---|---|
| v1.0 | `782f1ba` | 基础全双工, 串行 pipeline, 语音克隆 |
| v2.0 | `bbc11e0` | 状态机, 句级流式, 零爆音, 打断聚合 |
| v2.1 | `eb2e1a0` | 双层打断, 自适应 endpointing, THINKING 缓冲 |
| v2.2 | `ebe4872` | 投机式 ASR, 声纹 VAD, 副语言感知 |
| v2.3 | `2e95c76` | Moonshine, Smart Turn, FireRedVAD, JitterBuffer |
| **v2.4** | **`ea46912`** | **FireRedASR2, 流式 TTS 打断, 事件循环修复, LiveKit 实验** |
