<div align="center">

# 🚀 Hybrid Voice Agent — 王牌方案

### 面壁智能 · 智能语音外呼系统 · 新一代架构

<br>

<table>
<tr>
<td align="center" width="25%">
<h1>250ms</h1>
<b>首帧音频</b><br>
<sub>比 Pipeline 快 1.8x</sub>
</td>
<td align="center" width="25%">
<h1>50ms</h1>
<b>LLM TTFT</b><br>
<sub>vLLM AWQ-Marlin</sub>
</td>
<td align="center" width="25%">
<h1>10.5GB</h1>
<b>模型显存</b><br>
<sub>单 4090 可跑</sub>
</td>
<td align="center" width="25%">
<h1>109</h1>
<b>tok/s</b><br>
<sub>生成速度</sub>
</td>
</tr>
</table>

</div>

---

## 🔥 为什么要做这个？

我们的 Pipeline v2.0 方案（ASR+LLM+TTS）已经做到 454ms 首帧音频。但链路上有 **两个根本瓶颈**：

1. **ASR 误差传播** — SenseVoice 把"开源"听成"开元"→ RAG 检索不到 → LLM 瞎答
2. **串行延迟堆叠** — 4 个模型一个接一个跑，优化到头了

**Hybrid 方案的核心思路**：让 LLM 直接听懂音频，跳过 ASR。

---

## 📊 架构对比：Pipeline vs Hybrid

<table>
<tr>
<th width="50%">Pipeline v2.0（现有方案）</th>
<th width="50%">Hybrid v3.0（王牌方案）</th>
</tr>
<tr>
<td>

```
用户语音
  ↓
VAD (Silero)
  ↓
ASR (SenseVoice) ──── 117ms
  ↓
RAG (bge-small) ───── 4ms
  ↓
LLM (MiniCPM4.1) ─── 163ms
  ↓
TTS (VoxCPM) ──────── 174ms
  ↓
播放  ═══════════════ 458ms
```

**5 个模型 · 2 GPU · 17.6 GB**

</td>
<td>

```
用户语音
  ↓
Speaker-Aware VAD (Silero + ECAPA-TDNN)
  ↓
vLLM Omni (MiniCPM-o 4.5 AWQ) ── 50ms
  │  内置 Whisper 音频编码
  │  内置 Qwen3-8B 推理
  │  不需要 ASR！无误差传播！
  ↓
VoxCPM TTS ────────────────────── 190ms
  ↓
播放  ════════════════════════════ 250ms ⚡
```

**2 个模型 · 2 GPU · 17.5 GB**

</td>
</tr>
</table>

---

## 💡 关键技术突破

### 1. vLLM 加速：68x 提速

同一个 MiniCPM-o 4.5 模型，不同推理方式差距巨大：

<table>
<tr>
<th></th>
<th>Raw Transformers<br><sub>（HuggingFace 原生）</sub></th>
<th>vLLM bf16</th>
<th>vLLM AWQ-Marlin int4<br><sub>（我们用的）</sub></th>
</tr>
<tr>
<td><b>首 Token 延迟</b></td>
<td>3,400ms 😱</td>
<td>48ms</td>
<td><b>38ms</b> ⚡</td>
</tr>
<tr>
<td><b>生成速度</b></td>
<td>15 tok/s</td>
<td>48 tok/s</td>
<td><b>109 tok/s</b></td>
</tr>
<tr>
<td><b>显存占用</b></td>
<td>19.8 GB</td>
<td>22.6 GB</td>
<td><b>10.5 GB</b></td>
</tr>
<tr>
<td><b>对比基线</b></td>
<td>1x</td>
<td>71x</td>
<td><b>89x</b></td>
</tr>
</table>

> 关键：vLLM 的 PagedAttention + CUDA Graph + AWQ-Marlin kernel 把 MiniCPM-o 从"只能做研究"变成"可以做产品"。

### 2. Speaker-Aware VAD：从根本解决回声打断

<table>
<tr>
<th>问题</th>
<th>传统 VAD</th>
<th>Speaker-Aware VAD（我们用的）</th>
</tr>
<tr>
<td>TTS 回声触发打断</td>
<td>❌ 经常误触</td>
<td>✅ ECAPA-TDNN 声纹验证，回声被过滤</td>
</tr>
<tr>
<td>旁人说话</td>
<td>❌ 无法区分</td>
<td>✅ 只响应注册用户的声纹</td>
</tr>
<tr>
<td>环境噪音</td>
<td>⚠️ 依赖阈值调参</td>
<td>✅ VAD 0.85 + RMS + 3-chunk 确认 + 声纹</td>
</tr>
</table>

### 3. LLM 线程解耦：事件循环不阻塞

```
旧方案：async for token in httpx_stream → 事件循环被 LLM 占用 → WebSocket 卡顿
新方案：独立线程跑 LLM → asyncio.Queue 传递句子 → 事件循环始终自由 → 打断响应 <50ms
```

### 4. 打断文本聚合：对话不断裂

```
用户：你们公司做什么的？
AI：  我们是面壁智能，专注于——
用户：（打断）那你们 CEO 是谁？

旧方案：丢弃"面壁智能"的回答，独立处理新问题
新方案：保留"用户之前说了：面壁智能..."，下一轮综合回应
```

---

## 📈 实测数据

<table>
<tr>
<th>轮次</th>
<th>LLM TTFT</th>
<th>TTS TTFA</th>
<th>首帧音频</th>
</tr>
<tr><td>Turn 1</td><td>106ms</td><td>199ms</td><td><b>304ms</b></td></tr>
<tr><td>Turn 2</td><td>63ms</td><td>192ms</td><td><b>255ms</b></td></tr>
<tr><td>Turn 3</td><td>65ms</td><td>186ms</td><td><b>251ms</b></td></tr>
<tr><td>Turn 4</td><td>63ms</td><td>191ms</td><td><b>254ms</b></td></tr>
<tr><td><b>平均</b></td><td><b>74ms</b></td><td><b>192ms</b></td><td><b>266ms</b></td></tr>
</table>

> 对比：Pipeline v2.0 平均 454ms，Hybrid v3.0 平均 266ms，**快 1.7x**。
>
> 生产环境（H100 集群）预估可降至 **<200ms**，且有回退到 bf16 的精度余量。

---

## 🏗️ 技术栈

| 组件 | 技术选型 | 说明 |
|---|---|---|
| 音频理解 + LLM | MiniCPM-o 4.5 AWQ int4 | 9B omni 模型，Whisper-medium + Qwen3-8B |
| 推理引擎 | vLLM 0.16.0 + AWQ-Marlin | PagedAttention + CUDA Graph |
| 语音合成 | VoxCPM 1.5 | 声音克隆，temp=0.9 cfg=3.0 |
| 打断检测 | Speaker-Aware VAD | Silero VAD + ECAPA-TDNN 声纹 |
| 知识库 | bge-small-zh-v1.5 + FAISS | 59 条 FAQ，~3.6ms |
| 前端 | AudioWorklet queue player | 零爆音，交叉淡化 |
| 通信 | WebSocket + Cloudflare Tunnel | 公网可访问 |

---

## 🚀 快速启动

```bash
# 1. 启动 vLLM (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model models/MiniCPM-o-4_5-awq --served-model-name MiniCPM-o-4.5-awq \
  --trust-remote-code --dtype half --quantization awq_marlin \
  --gpu-memory-utilization 0.90 --max-model-len 4096 --port 8200

# 2. 启动 Hybrid Server (GPU 1)
CUDA_VISIBLE_DEVICES=1 TTS_DEVICE=cuda:0 PORT=3001 python ws_server_hybrid.py

# 3. 公网演示
./cloudflared tunnel --url http://localhost:3001
```

---

## 📋 版本演进

| 版本 | 日期 | 里程碑 |
|---|---|---|
| v1.0 | 2026-03-04 | 基础 vLLM + VoxCPM 混合 pipeline |
| v2.0 | 2026-03-04 | 5 状态机 + 鲁棒打断 + 自适应 endpointing |
| v2.1 | 2026-03-05 | 多轮上下文 + RAG + TTS 调优 (temp=0.9) |
| v3.0 | 2026-03-05 | LLM 线程解耦 + Speaker VAD + 打断聚合 |
| **v3.1** | **2026-03-10** | **流式 TTS + 事件循环呼吸 + Turn 序列号过滤** |

---

## 👥 关联项目

| 项目 | 说明 |
|---|---|
| [CallCenter-VoiceAgent](https://github.com/HenryZ838978/CallCenter-VoiceAgent) | Pipeline 方案 (ASR+LLM+TTS)，v2.0 |
| **Hybrid-VoiceAgent（本项目）** | 王牌方案 (Omni+TTS)，v3.0 |

---

<div align="center">
<sub>面壁智能 · AI 大模型技术创新与应用落地</sub>
</div>
