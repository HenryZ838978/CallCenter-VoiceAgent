<div align="center">

# Hybrid Voice Agent

### 超快全双工语音对话系统 / Ultra-fast Full-Duplex Voice Conversation

**Audio In → Omni Model → Audio Out — ~250ms 首帧音频**

<br>

<table>
<tr>
<td align="center"><b>250ms</b><br><sub>首帧音频 First Audio</sub></td>
<td align="center"><b>1.8x</b><br><sub>快于传统 Pipeline</sub></td>
<td align="center"><b>10.5GB</b><br><sub>AWQ int4 VRAM</sub></td>
<td align="center"><b>109</b><br><sub>tokens/sec</sub></td>
</tr>
</table>

<br>

基于 [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) (音频理解) + [VoxCPM 1.5](https://github.com/openbmb) (语音合成)，通过 [vLLM](https://github.com/vllm-project/vllm) AWQ-Marlin 量化推理。

</div>

---

## 为什么选择这个架构？

传统语音 Agent 串联 4-5 个独立模型，每个都增加延迟，且 ASR 误差会逐级传播：

<table>
<tr>
<th>传统 Pipeline</th>
<th>Hybrid（本项目）</th>
</tr>
<tr>
<td>

```
VAD → ASR → RAG → LLM → TTS
 │    117ms  4ms  163ms  174ms
 │
 └──── 总计 ~458ms ────
```

5 个模型 · 2 GPU · ~17.6 GB

</td>
<td>

```
VAD → vLLM Omni AWQ → VoxCPM TTS
 │      50ms TTFT       190ms TTFA
 │
 └───── 总计 ~250ms ─────
```

2 个模型 · 2 GPU · ~17.5 GB

</td>
</tr>
</table>

**核心思路**：MiniCPM-o 4.5 内置 Whisper-medium（音频编码器）+ Qwen3-8B（LLM），可以直接理解音频——不需要单独的 ASR 模块。配合 vLLM 的高效推理，完全消除了 ASR 瓶颈。

## vLLM 为什么关键

同一个模型，不同推理方式的延迟天差地别：

<table>
<tr>
<th></th>
<th>Raw Transformers</th>
<th>vLLM bf16</th>
<th>vLLM AWQ-Marlin</th>
</tr>
<tr>
<td><b>TTFT</b></td>
<td>3400ms</td>
<td>48ms</td>
<td><b>38ms</b></td>
</tr>
<tr>
<td><b>Tokens/sec</b></td>
<td>15</td>
<td>48</td>
<td><b>109</b></td>
</tr>
<tr>
<td><b>VRAM</b></td>
<td>19.8 GB</td>
<td>22.6 GB</td>
<td><b>10.5 GB</b></td>
</tr>
<tr>
<td><b>加速比</b></td>
<td>1x</td>
<td>71x</td>
<td><b>89x</b></td>
</tr>
</table>

vLLM 提供 PagedAttention、CUDA Graph 和原生 AWQ-Marlin kernel——把研究模型变成生产级推理引擎。

## 功能特性

<table>
<tr>
<td width="50%">

**音频能力**
- ~250ms 首帧音频（快于人类对话切换 ~300ms）
- 音频原生理解（无 ASR 中间步骤）
- 声音克隆（VoxCPM 1.5 参考音色）
- 句级流式（LLM 出一句 TTS 说一句）

</td>
<td width="50%">

**工程能力**
- Speaker-Aware VAD（ECAPA-TDNN 说话人验证）
- 鲁棒打断（VAD 0.85 + RMS + 3-chunk 确认）
- 打断文本聚合（上下文不断裂）
- LLM 线程解耦（Queue 架构）
- 多轮对话（8 轮历史）
- RAG 知识库（bge-small + FAISS）

</td>
</tr>
</table>

## 系统架构

```
浏览器（麦克风 + 扬声器）
    │  Turn 序列号过滤 — 丢弃旧 turn 的在途音频帧
    │
    └── WebSocket ──→ FastAPI Server (:3002)
                        │
                        ├── Speaker-Aware VAD (Silero + ECAPA-TDNN, CPU)
                        │     说话人验证，区分用户 vs TTS回声 vs 旁人
                        │
                        ├── vLLM API (:8200, GPU)      ← MiniCPM-o 4.5 AWQ int4
                        │     音频理解 + 文本生成          10.5 GB, TTFT ~50ms
                        │     LLM 在独立线程运行          句子通过 Queue 传递
                        │
                        ├── RAG (bge-small + FAISS)    ← 59 条知识库
                        │     上轮对话检索，~3.6ms
                        │
                        └── 流式 VoxCPM TTS (GPU)      ← 声音克隆
                              逐 chunk 生成 + 立即发送      temp=0.9, cfg=3.0
                              44.1 kHz · asyncio.sleep(0.05) 事件循环呼吸
```

## 快速开始

### 环境要求

- NVIDIA GPU 24GB+ (如 RTX 4090) x2
- Conda / Python 3.10+
- vLLM 0.16.0+

### 1. 环境搭建

```bash
bash setup_env.sh
```

### 2. 下载模型

| 模型 | 来源 | 大小 |
|---|---|---|
| MiniCPM-o 4.5 AWQ | `openbmb/MiniCPM-o-4_5-awq` | ~7.9 GB |
| VoxCPM 1.5 | 面壁智能 | ~1.9 GB |
| Silero VAD | snakers4 | ~2 MB |
| ECAPA-TDNN | speechbrain | ~80 MB |
| bge-small-zh-v1.5 | BAAI | ~91 MB |

### 3. 启动 vLLM

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/MiniCPM-o-4_5-awq \
  --served-model-name MiniCPM-o-4.5-awq \
  --trust-remote-code --dtype half --quantization awq_marlin \
  --gpu-memory-utilization 0.90 --max-model-len 4096 \
  --host 0.0.0.0 --port 8200
```

### 4. 启动 Hybrid Server

```bash
CUDA_VISIBLE_DEVICES=1 TTS_DEVICE=cuda:0 PORT=3001 python ws_server_hybrid.py
```

### 5. 公网演示（可选）

```bash
./cloudflared tunnel --url http://localhost:3001
```

## 性能实测

RTX 4090 x2, MiniCPM-o 4.5 AWQ int4:

<table>
<tr>
<td align="center"><b>LLM TTFT</b><br>50ms (p50)</td>
<td align="center"><b>TTS TTFA</b><br>~190ms</td>
<td align="center"><b>首帧音频</b><br><b>~250ms</b></td>
<td align="center"><b>生成速度</b><br>109 tok/s</td>
</tr>
<tr>
<td align="center"><b>Omni VRAM</b><br>10.5 GB</td>
<td align="center"><b>TTS VRAM</b><br>~7 GB</td>
<td align="center"><b>RAG</b><br>~3.6ms</td>
<td align="center"><b>多轮</b><br>8 轮历史</td>
</tr>
</table>

## 版本历史

| 版本 | 特性 |
|---|---|
| v1.0 | 基础混合 pipeline (vLLM + VoxCPM) |
| v2.0 | 5 状态机 + 鲁棒打断 + 自适应 endpointing |
| v2.1 | 多轮上下文 + RAG + 音频平滑 + TTS 调优 |
| v3.0 | LLM 线程解耦 + Speaker VAD + 打断聚合 |
| **v3.1** | **流式 TTS + 事件循环呼吸 + Turn 序列号过滤** |

## 项目结构

```
├── ws_server_hybrid.py     # 核心：vLLM Omni + VoxCPM TTS 混合 pipeline
├── ws_server.py            # 备选：纯 Omni 端到端服务器
├── static/index.html       # 前端（对比仪表板 + AudioWorklet 播放器）
├── engine/
│   ├── omni.py             # MiniCPM-o 4.5 原生引擎（基准测试用）
│   ├── vad.py              # Silero VAD
│   └── speaker_vad.py      # Speaker-Aware VAD (ECAPA-TDNN)
├── config.py               # 模型路径、提示词、知识库
├── benchmark.py            # 延迟测量脚本
├── setup_env.sh            # 环境搭建
└── start_all.sh            # 一键启动
```

## License

MIT
