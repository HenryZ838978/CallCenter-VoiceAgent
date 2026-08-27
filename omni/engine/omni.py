"""MiniCPM-o-4.5 end-to-end speech engine.

Replaces separate ASR + LLM + TTS with a single omni model.
Audio in (16kHz float32) -> Audio out (24kHz float32) with streaming.
Vision module disabled — audio-only mode.
"""
import time
import logging
import tempfile
import numpy as np

log = logging.getLogger("omni_agent.omni")

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
CHUNK_DURATION_S = 1  # prefill in 1-second chunks
MIN_AUDIO_SAMPLES = 16000


class OmniEngine:
    def __init__(self, model_dir: str, device: str = "cuda:0",
                 load_in_4bit: bool = False):
        self.model_dir = model_dir
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.ref_audio = None
        self.ref_audio_path = None
        self._session_counter = 0
        self._session_active = False

    def load(self):
        import torch
        from transformers import AutoModel

        log.info("Loading MiniCPM-o-4.5 (vision=OFF, audio=ON, tts=ON) ...")
        t0 = time.perf_counter()

        kwargs = dict(
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=True,
            init_tts=True,
        )

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            log.info("Using int4 quantization (bitsandbytes)")

        self.model = AutoModel.from_pretrained(self.model_dir, **kwargs)
        self.model.eval()

        if not self.load_in_4bit:
            self.model = self.model.to(self.device)

        self.model.init_tts(streaming=True)
        self._tts_ready = True
        load_s = time.perf_counter() - t0
        log.info("Model loaded in %.1fs", load_s)

        import torch
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        log.info("VRAM used: %.1f GB", mem_gb)
        return self

    def set_voice(self, ref_audio_path: str):
        """Set reference audio for voice cloning."""
        import librosa
        self.ref_audio_path = ref_audio_path
        self.ref_audio, _ = librosa.load(ref_audio_path, sr=SAMPLE_RATE_IN, mono=True)
        log.info("Voice reference set: %s (%.1fs)", ref_audio_path, len(self.ref_audio) / SAMPLE_RATE_IN)
        return self

    def _next_session_id(self) -> str:
        self._session_counter += 1
        return f"s{self._session_counter}"

    def _build_sys_msg(self, system_prompt: str) -> dict:
        if self.ref_audio is not None:
            return {
                "role": "system",
                "content": [
                    "模仿输入音频中的声音特征。",
                    self.ref_audio,
                    system_prompt,
                ],
            }
        return {"role": "system", "content": [system_prompt]}

    def _split_audio_chunks(self, audio: np.ndarray):
        """Split audio into 1-second chunks for streaming prefill."""
        chunk_size = SAMPLE_RATE_IN * CHUNK_DURATION_S
        total = len(audio)
        num = max(1, (total + chunk_size - 1) // chunk_size)
        chunks = []
        for i in range(num):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total)
            chunk = audio[start:end]
            if i == num - 1 and len(chunk) < MIN_AUDIO_SAMPLES:
                chunk = np.concatenate([
                    chunk,
                    np.zeros(MIN_AUDIO_SAMPLES - len(chunk), dtype=chunk.dtype),
                ])
            chunks.append(chunk)
        return chunks

    def reset_session(self):
        """Reset for a new conversation. Do NOT re-init TTS (causes OOM)."""
        if not getattr(self, '_tts_ready', False):
            self.model.init_tts(streaming=True)
            self._tts_ready = True
        self.model.reset_session(reset_token2wav_cache=True)
        if self.ref_audio is not None:
            self.model.init_token2wav_cache(prompt_speech_16k=self.ref_audio)
        self._session_active = False

    def stream_chat(self, user_audio: np.ndarray, system_prompt: str,
                    max_new_tokens: int = 512):
        """Streaming speech conversation: audio in -> yields (wav_chunk, text_chunk).

        Args:
            user_audio: float32 numpy array at 16kHz
            system_prompt: text system prompt (may include KB)
            max_new_tokens: generation length limit

        Yields:
            (wav_chunk, text_chunk) — wav_chunk is a torch Tensor, text_chunk is str
        """
        self.reset_session()
        sid = self._next_session_id()

        sys_msg = self._build_sys_msg(system_prompt)
        self.model.streaming_prefill(
            session_id=sid, msgs=[sys_msg],
            omni_mode=False, is_last_chunk=True,
        )

        chunks = self._split_audio_chunks(user_audio)
        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            self.model.streaming_prefill(
                session_id=sid,
                msgs=[{"role": "user", "content": [chunk]}],
                omni_mode=False,
                is_last_chunk=is_last,
            )

        gen = self.model.streaming_generate(
            session_id=sid,
            generate_audio=True,
            use_tts_template=True,
            enable_thinking=False,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            length_penalty=1.1,
        )

        self._session_active = True
        for wav_chunk, text_chunk in gen:
            yield wav_chunk, text_chunk
        self._session_active = False

    def chat(self, user_audio: np.ndarray, system_prompt: str,
             output_audio_path: str = None, max_new_tokens: int = 512) -> dict:
        """Non-streaming: audio in -> dict with text + audio info.

        Uses the streaming API internally and collects all chunks.
        Returns dict: {text, audio_path, ttfa_ms, total_ms, audio_duration_s}
        """
        import torch, soundfile as sf

        t_total_start = time.perf_counter()
        ttfa = None
        audios = []
        text = ""

        for wav_chunk, text_chunk in self.stream_chat(user_audio, system_prompt, max_new_tokens):
            if ttfa is None:
                ttfa = (time.perf_counter() - t_total_start) * 1000
            if wav_chunk is not None:
                audios.append(wav_chunk)
            text += text_chunk

        total_ms = (time.perf_counter() - t_total_start) * 1000

        if audios:
            waveform = torch.cat(audios, dim=-1)
            if waveform.dim() > 1:
                waveform = waveform[0]
            if output_audio_path is None:
                output_audio_path = tempfile.mktemp(suffix=".wav")
            sf.write(output_audio_path, waveform.cpu().float().numpy(),
                     samplerate=SAMPLE_RATE_OUT)
            audio_dur = len(waveform) / SAMPLE_RATE_OUT
        else:
            audio_dur = 0.0

        return {
            "text": text,
            "audio_path": output_audio_path,
            "ttfa_ms": ttfa or 0.0,
            "total_ms": total_ms,
            "audio_duration_s": audio_dur,
        }

    def transcribe(self, audio_16k: np.ndarray, language: str = "zh") -> str:
        """ASR mode: audio -> text transcription."""
        if language == "zh":
            prompt = "请仔细听这段音频片段，并将其内容逐字记录。"
        else:
            prompt = "Please listen to the audio snippet carefully and transcribe the content."
        msgs = [{"role": "user", "content": [prompt, audio_16k]}]
        return self.model.chat(
            msgs=msgs, do_sample=False, max_new_tokens=512,
            use_tts_template=True, generate_audio=False,
        )

    def warmup(self):
        """Run a dummy inference to warm up CUDA kernels."""
        log.info("Warming up omni engine...")
        t0 = time.perf_counter()
        dummy = np.zeros(SAMPLE_RATE_IN * 2, dtype=np.float32)  # 2s silence
        try:
            result = self.chat(dummy, "你好", max_new_tokens=32)
            log.info("Warmup done in %.1fs, text=%r", time.perf_counter() - t0, result["text"][:50])
        except Exception as e:
            log.warning("Warmup failed (non-fatal): %s", e)
