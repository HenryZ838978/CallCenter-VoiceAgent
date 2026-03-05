"""Paralinguistic audio captioner — generates natural language descriptions of user's vocal tone,
emotion, speaking speed, and audio scene. Injected into LLM system prompt to enable empathetic responses.

Uses vLLM-served MiniCPM-o-4.5 (or any audio-capable model) via OpenAI-compatible API.
Falls back to a lightweight heuristic captioner when no model server is available.
"""
import time
import logging
import struct
import base64
import io
import numpy as np

log = logging.getLogger("voxlabs.captioner")

CAPTION_SYSTEM_PROMPT = (
    "你是一个语音分析助手。请用一句简短的中文描述这段语音的特征，包括：说话人性别、语速(快/正常/慢)、"
    "情绪(平静/高兴/着急/生气/犹豫/悲伤)、以及背景环境(安静/嘈杂/有音乐等)。"
    "只输出一句描述，不超过30字。"
)


def _audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Convert float32 PCM array to WAV bytes for API transmission."""
    int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    num_samples = len(int16)
    data_size = num_samples * 2
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(int16.tobytes())
    return buf.getvalue()


class HeuristicCaptioner:
    """Lightweight captioner using audio statistics — no model required."""

    def describe(self, audio: np.ndarray, sr: int = 16000) -> str:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        duration = len(audio) / sr

        zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))

        parts = []

        if rms > 0.15:
            parts.append("语气较激动")
        elif rms > 0.05:
            parts.append("语气正常")
        elif rms > 0.01:
            parts.append("语气平缓")
        else:
            parts.append("语气很轻")

        words_est = duration * 3.5
        if duration > 0:
            if words_est / duration > 4.5:
                parts.append("语速偏快")
            elif words_est / duration < 2.0:
                parts.append("语速偏慢")

        if rms < 0.005:
            parts.append("背景安静")
        elif zcr > 0.3 and rms < 0.03:
            parts.append("可能有背景噪音")

        return "，".join(parts) if parts else "语气正常，背景安静"


class OmniCaptioner:
    """Captioner using vLLM-served audio-capable model (MiniCPM-o, Qwen-Audio, etc.)."""

    def __init__(self, base_url: str = "http://localhost:8200/v1",
                 model: str = "MiniCPM-o-4.5-awq",
                 api_key: str = "dummy"):
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._client = None

    def load(self):
        from openai import OpenAI
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self

    def describe(self, audio: np.ndarray, sr: int = 16000) -> str:
        if self._client is None:
            return ""

        trim_samples = min(len(audio), sr * 10)
        audio_trimmed = audio[-trim_samples:]

        wav_bytes = _audio_to_wav_bytes(audio_trimmed, sr)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        audio_url = f"data:audio/wav;base64,{audio_b64}"

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "input_audio", "input_audio": {"data": audio_url, "format": "wav"}},
                        {"type": "text", "text": "请描述这段语音的特征。"},
                    ]},
                ],
                max_tokens=60,
                temperature=0.3,
            )
            text = resp.choices[0].message.content or ""
            latency = (time.perf_counter() - t0) * 1000
            log.info("OmniCaptioner: '%s' (%.0fms)", text.strip(), latency)
            return text.strip()
        except Exception as e:
            log.warning("OmniCaptioner failed: %s", e)
            return ""
