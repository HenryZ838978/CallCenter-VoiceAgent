<h1 align="center">VoxLabs — Full-Duplex Voice Agent</h1>

<p align="center">
  <b>实时中文语音客服 Agent</b> — sub-500ms 首响、自然打断、语音克隆<br>
  面向 BPO / 智能外呼场景
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pipeline-458ms-00c853?style=for-the-badge" alt="Pipeline 458ms">
  <img src="https://img.shields.io/badge/Barge--in-0ms_client-2196f3?style=for-the-badge" alt="Barge-in 0ms">
  <img src="https://img.shields.io/badge/Version-v2.2-7c4dff?style=for-the-badge" alt="v2.2">
  <img src="https://img.shields.io/badge/GPU-RTX_4090-76ff03?style=for-the-badge" alt="RTX 4090">
</p>

---

## 系统架构

<!-- Architecture diagram rendered as HTML for better visual clarity -->
<table align="center" style="border-collapse:collapse;">
<tr><td align="center" colspan="5" style="padding:16px;background:#1a237e;color:#fff;border-radius:12px 12px 0 0;">
<b>Browser / SIP Client</b><br>
<small>Mic → PCM 16kHz → WebSocket → Server &nbsp;|&nbsp; Speaker ← PCM 44.1kHz ← Server</small><br>
<small>客户端即时打断: RMS > 0.03 连续 3 帧 → stopPlayback() <b>(0ms)</b></small>
</td></tr>
<tr><td align="center" colspan="5" style="padding:4px;color:#90caf9;">▼ WSS</td></tr>
<tr><td align="center" colspan="5" style="padding:12px;background:#0d47a1;color:#fff;">
<b>FastAPI WebSocket Server (:3000)</b>
</td></tr>
<tr><td align="center" colspan="5" style="padding:12px;background:#1565c0;color:#e3f2fd;border-bottom:2px solid #0d47a1;">
<b>ConversationManager v2.2 — 5 状态机</b><br>
<code>IDLE → LISTENING → THINKING → SPEAKING → INTERRUPTED</code><br>
<small>双层打断 (0ms+96ms) · 投机式 ASR · 自适应 Endpointing · 声纹 VAD · 副语言感知</small>
</td></tr>
<tr>
<td align="center" style="padding:10px;background:#004d40;color:#a7ffeb;width:20%;border-radius:0 0 0 12px;">
<b>VAD</b><br><small>Silero + ECAPA-TDNN<br>声纹识别</small><br>
<code>CPU ~1.7ms</code>
</td>
<td align="center" style="padding:10px;background:#1b5e20;color:#b9f6ca;width:20%;">
<b>ASR</b><br><small>SenseVoiceSmall<br>234M FP32</small><br>
<code>117ms</code>
</td>
<td align="center" style="padding:10px;background:#e65100;color:#fff3e0;width:20%;">
<b>RAG</b><br><small>bge-small-zh<br>+ FAISS</small><br>
<code>3.6ms</code>
</td>
<td align="center" style="padding:10px;background:#4a148c;color:#e1bee7;width:20%;">
<b>LLM</b><br><small>MiniCPM4.1-8B<br>GPTQ-Marlin</small><br>
<code>163ms</code>
</td>
<td align="center" style="padding:10px;background:#b71c1c;color:#ffcdd2;width:20%;border-radius:0 0 12px 0;">
<b>TTS</b><br><small>VoxCPM 1.5<br>nanovllm</small><br>
<code>174ms</code>
</td>
</tr>
</table>

<br>

## 延迟实测 (RTX 4090, warmup)

<table align="center">
<tr>
<th>组件</th><th>延迟</th><th>模型</th>
</tr>
<tr><td>ASR</td><td><b>117ms</b></td><td>SenseVoiceSmall (234M, FP32)</td></tr>
<tr><td>RAG</td><td><b>3.6ms</b></td><td>bge-small-zh-v1.5 + FAISS</td></tr>
<tr><td>LLM</td><td><b>163ms</b></td><td>MiniCPM4.1-8B-GPTQ via vLLM</td></tr>
<tr><td>TTS TTFA</td><td><b>174ms</b></td><td>VoxCPM 1.5 via nanovllm</td></tr>
<tr><td><b>Pipeline 首响</b></td><td><b>~458ms ✅</b></td><td>目标 &lt; 500ms · 红线 &lt; 800ms</td></tr>
</table>

### 多轮对话

<table align="center">
<tr><th>轮次</th><th>用户输入</th><th>LLM</th><th>TTS</th><th>总响应</th></tr>
<tr><td>Turn 1</td><td>你好，我想了解你们的服务</td><td>196ms</td><td>174ms</td><td>1239ms</td></tr>
<tr><td>Turn 2</td><td>请问价格是多少？</td><td>135ms</td><td>172ms</td><td>959ms</td></tr>
<tr><td>Turn 3</td><td>好的，谢谢，再见</td><td>159ms</td><td>175ms</td><td>879ms</td></tr>
<tr><td><b>平均</b></td><td></td><td><b>163ms</b></td><td><b>174ms</b></td><td>—</td></tr>
</table>

### 多 GPU 横评

<table align="center">
<tr><th>硬件</th><th>ASR</th><th>LLM</th><th>TTS</th><th>Pipeline</th></tr>
<tr><td>RTX 4080S (32GB)</td><td>130ms</td><td>225ms</td><td>115ms</td><td>470ms</td></tr>
<tr><td>RTX 5090 (32GB)</td><td>93ms</td><td>130ms</td><td>138ms</td><td>342ms</td></tr>
<tr><td><b>RTX 4090 (24GB)</b></td><td><b>117ms</b></td><td><b>163ms</b></td><td><b>174ms</b></td><td><b>337-458ms</b></td></tr>
</table>

---

## 全双工打断 (v2.1+)

系统实现 **双层打断架构**：

<table align="center">
<tr>
<td align="center" style="padding:16px;background:#e3f2fd;border:2px solid #2196f3;border-radius:12px;">
<h3>Layer 1 — 客户端即时层</h3>
<b style="font-size:28px;color:#2196f3;">0ms</b><br><br>
浏览器始终发送麦克风音频<br>
本地 RMS 检测: 连续 3 帧 > 阈值<br>
→ 立即 <code>stopPlayback()</code><br>
<b>用户感知零延迟打断</b>
</td>
<td align="center" style="padding:8px;font-size:24px;">→</td>
<td align="center" style="padding:16px;background:#f3e5f5;border:2px solid #9c27b0;border-radius:12px;">
<h3>Layer 2 — 服务端确认层</h3>
<b style="font-size:28px;color:#9c27b0;">96ms</b><br><br>
VAD ≥ 0.85 (高阈值过滤回声)<br>
RMS > 0.015 (能量门限)<br>
连续 3 chunks 确认<br>
<b>确认后 cancel LLM+TTS</b>
</td>
</tr>
</table>

### 声纹 VAD (v2.2)

<table align="center">
<tr>
<td align="center" style="padding:16px;background:#e8f5e9;border:2px solid #4caf50;border-radius:12px;">
<h3>SpeakerAwareVAD</h3>
<b>ECAPA-TDNN (192维声纹) + Silero VAD</b><br><br>
第一轮说话 → 自动注册声纹<br>
后续打断 → 声纹匹配确认<br><br>
<b>TTS 回声</b>: similarity ~0.3 → ❌ 拒绝<br>
<b>用户本人</b>: similarity ~0.95 → ✅ 通过<br><br>
<small>每 chunk 1.7ms (CPU) · 每 96ms 验证一次</small>
</td>
</tr>
</table>

### 自适应 Endpointing

<table align="center">
<tr>
<td align="center" style="padding:14px;background:#e8f5e9;border-radius:10px;width:33%;">
<b style="font-size:24px;color:#4caf50;">192ms</b><br>
<b>短语/确认</b><br>
<small>用户说话 &lt; 0.5s</small><br>
<small>"好的" "嗯" "是的"</small>
</td>
<td align="center" style="padding:14px;background:#fff8e1;border-radius:10px;width:33%;">
<b style="font-size:24px;color:#ff8f00;">320ms</b><br>
<b>正常对话</b><br>
<small>用户说话 0.5~3s</small><br>
<small>"我想了解你们的服务"</small>
</td>
<td align="center" style="padding:14px;background:#f3e5f5;border-radius:10px;width:33%;">
<b style="font-size:24px;color:#7b1fa2;">640ms</b><br>
<b>长句/思考</b><br>
<small>用户说话 &gt; 3s</small><br>
<small>"我之前用过另一家……嗯……"</small>
</td>
</tr>
</table>

### 投机式预推理 (v2.2)

```
v2.1:  [说话 2s] [静默 320ms] [ASR 117ms] [RAG] [LLM] [TTS]
                               ↑ endpointing 后才开始 ASR

v2.2:  [说话 2s] [静默 160ms → 投机 ASR 启动]
                 [静默 320ms → endpointing 确认 → ASR 已完成!] [RAG] [LLM] [TTS]
                                                                ↑ 省了 ~117ms
```

### 副语言感知 (v2.2)

用户语音 → 语气/情绪/语速分析 → 注入 LLM prompt:

```
[语音观察：语气较激动，语速偏快]
用户说：我等了很久了，到底什么时候能解决？
```

LLM 根据用户 **怎么说**（不只是说了什么）调整回复策略。

---

## 核心特性

<table>
<tr>
<td width="50%" valign="top">

**语音克隆** — 8s 参考音频 → 克隆音色 (VoxCPM 1.5, temp=0.7, cfg=3.0)

**句级流式** — LLM 逐句生成，每句立即送 TTS

**RAG 知识库** — bge-small + FAISS, 支持 `POST /api/rag/reload` 热更新

</td>
<td width="50%" valign="top">

**零爆音播放** — 首 chunk 丢弃 (VAE transient) + 10ms fade-in + AudioWorklet 队列

**打断聚合** — "用户依次说了：X；Y。请综合回应"

**THINKING 缓冲** — pipeline 运行期间用户追加的语音不丢失

</td>
</tr>
</table>

---

## 三种方案对比

同硬件 (RTX 4090) 实测，三种架构各有优势：

<table align="center">
<tr><th>维度</th><th>Pipeline (本项目)</th><th>Hybrid (姊妹项目)</th><th>纯 Omni (raw)</th></tr>
<tr><td>架构</td><td>ASR → RAG → LLM → TTS</td><td><b>vLLM Omni AWQ → VoxCPM TTS</b></td><td>MiniCPM-o-4.5 端到端</td></tr>
<tr><td>首响延迟</td><td>458ms ✅</td><td><b>~250ms ✅✅</b></td><td>1666ms ❌</td></tr>
<tr><td>VRAM</td><td>17.6GB (2 GPU)</td><td><b>10.5+7 GB (2 GPU)</b></td><td>21.5GB (1 GPU)</td></tr>
<tr><td>RAG 集成</td><td><b>3.6ms (向量检索)</b></td><td>3.6ms (向量检索)</td><td>+305ms (prompt 嵌入)</td></tr>
<tr><td>打断自然度</td><td>双层打断 + 声纹 VAD</td><td>双层打断 + 声纹 VAD</td><td><b>模型原生理解</b></td></tr>
<tr>
<td colspan="4" style="padding:8px 0;"><b>灵活性 — Pipeline 核心优势</b></td>
</tr>
<tr><td>换音色</td><td><b>LoRA / clone, 小时级</b></td><td><b>VoxCPM clone, 小时级</b></td><td>重训整个模型</td></tr>
<tr><td>换话术</td><td><b>LLM SFT, ~$20</b></td><td>需 Omni 多模态 SFT</td><td>多模态 SFT, ~$5000+</td></tr>
<tr><td>换 ASR</td><td><b>替换 engine/asr.py</b></td><td>绑定 Whisper-medium</td><td>绑定内置 ASR</td></tr>
<tr><td>换 LLM</td><td><b>任何 OpenAI 兼容</b></td><td>绑定 MiniCPM-o</td><td>绑定内置 LLM</td></tr>
<tr><td>换 TTS</td><td><b>替换 engine/tts.py</b></td><td><b>替换 TTS 模块</b></td><td>绑定内置 TTS</td></tr>
<tr><td>多客户定制</td><td><b>配置级, 分钟上线</b></td><td>需切换 Omni 模型</td><td>每客户一个模型变体</td></tr>
</table>

**选型建议：**
- **追求极致延迟** → Hybrid (250ms)：[Hybrid-VoiceAgent](https://github.com/HenryZ838978/Hybrid-VoiceAgent)
- **追求灵活定制** → Pipeline (458ms)：本项目，各组件独立替换、LLM 自由 SFT、多客户配置化
- **研究/Demo** → 纯 Omni：一个模型搞定一切，但延迟和定制成本高

### 可扩展性

<table align="center">
<tr><th>组件</th><th>当前方案</th><th>可替换为</th></tr>
<tr><td>ASR</td><td>SenseVoiceSmall / FunASR</td><td>Whisper, Paraformer, sherpa-onnx</td></tr>
<tr><td>LLM</td><td>MiniCPM4.1-8B-GPTQ / vLLM</td><td>Qwen, LLaMA, 任何 OpenAI 兼容</td></tr>
<tr><td>TTS</td><td>VoxCPM 1.5 / nanovllm</td><td>CosyVoice, IndexTTS, GPT-SoVITS</td></tr>
<tr><td>RAG</td><td>bge-small + FAISS</td><td>bge-m3, Milvus, Elasticsearch</td></tr>
<tr><td>VAD</td><td>Silero + ECAPA-TDNN</td><td>FireRedChat pVAD, WebRTC VAD</td></tr>
<tr><td>Turn Detection</td><td>自适应静默阈值</td><td>Pipecat Smart Turn, VoTurn-80M</td></tr>
</table>

---

## 项目结构

```
voiceagent/
├── engine/
│   ├── asr.py                    # SenseVoiceSmall / FunASR
│   ├── llm.py                    # vLLM streaming + sentence boundary
│   ├── tts.py                    # VoxCPM 1.5 (44.1kHz, voice clone)
│   ├── vad.py                    # Silero VAD
│   ├── speaker_vad.py            # SpeakerAwareVAD (ECAPA-TDNN + Silero)
│   ├── rag.py                    # bge-small-zh-v1.5 + FAISS
│   ├── captioner.py              # 副语言感知 (heuristic / omni)
│   ├── conversation_manager.py   # v2.2 状态机 + 全部优化
│   ├── filler.py                 # Filler word cache
│   └── duplex_agent.py           # v1 standalone agent (legacy)
│
├── static/
│   └── voice_agent.html          # AudioWorklet + instant barge-in
│
├── data/
│   ├── sample_kb.json            # 知识库 (59 FAQ)
│   └── voice_prompt.wav          # 语音克隆参考音频
│
├── models/                       # 模型权重 (不含在 repo 中)
│   ├── SenseVoiceSmall/          (901MB)
│   ├── MiniCPM4.1-8B-GPTQ/      (4.9GB)
│   ├── VoxCPM1.5/                (1.9GB)
│   ├── spkrec-ecapa-voxceleb/    (ECAPA-TDNN speaker encoder)
│   ├── snakers4_silero-vad/
│   └── bge-small-zh-v1.5/        (91MB)
│
├── ws_server.py                  # FastAPI WebSocket 服务器
├── config.py                     # 模型路径、提示词、参数
├── start_all.sh                  # 一键启动
├── test_duplex.py                # 双工集成测试
├── test_ws_e2e.py                # WebSocket 端到端测试
└── ab_test_rag.py                # RAG A/B 测试
```

## Quick Start

```bash
# 1. vLLM (LLM 推理服务)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model models/MiniCPM4.1-8B-GPTQ --served-model-name MiniCPM4.1-8B-GPTQ \
  --trust-remote-code --dtype auto --quantization gptq_marlin \
  --gpu-memory-utilization 0.85 --max-model-len 4096 --port 8100

# 2. Voice Agent (ASR + TTS + RAG + WebSocket)
CUDA_VISIBLE_DEVICES=2 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 \
  USE_SPEAKER_VAD=1 python ws_server.py

# 3. (可选) 公网访问
./cloudflared tunnel --url https://localhost:3000 --no-tls-verify
```

打开 `https://localhost:3000` → **Start Call** → 允许麦克风 → 开始对话

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
| **v2.2** | **`ebe4872`** | **投机式 ASR, 声纹 VAD, 副语言感知, WS E2E 测试** |
