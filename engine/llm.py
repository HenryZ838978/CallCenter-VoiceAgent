"""LLM inference via vLLM OpenAI-compatible API with streaming support."""
import re
import time
import httpx
from openai import OpenAI

_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_INCOMPLETE_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_ANY_XML_TAG_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")
_SENTENCE_END_RE = re.compile(r"[。！？!?；;，,\n]")


class VLLMChat:
    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "MiniCPM4.1-8B-GPTQ",
                 system_prompt: str = "", max_history: int = 20):
        self._client = OpenAI(
            base_url=base_url, api_key="dummy",
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=10.0),
        )
        self._model = model
        self._system_prompt = system_prompt
        self._history = []
        self._max_history = max_history
        self._summary = ""

    def reset(self):
        self._history = []
        self._summary = ""

    def _trim_history(self):
        """Keep last max_history messages; summarize evicted turns into _summary."""
        if len(self._history) <= self._max_history:
            return
        evicted = self._history[:len(self._history) - self._max_history]
        self._history = self._history[-self._max_history:]
        turns = []
        for i in range(0, len(evicted) - 1, 2):
            u = evicted[i].get("content", "")[:60]
            a = evicted[i + 1].get("content", "")[:60] if i + 1 < len(evicted) else ""
            turns.append(f"用户:{u} → AI:{a}")
        if turns:
            self._summary = "之前的对话摘要：" + "；".join(turns[-5:])

    def _build_messages(self, user_text: str, rag_context: str = None):
        messages = []
        system = self._system_prompt
        if rag_context:
            system = system.replace("{context}", rag_context) if "{context}" in system else (
                system + f"\n\n知识库信息：\n{rag_context}"
            )
        if self._summary:
            system = system + f"\n\n{self._summary}"
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def chat(self, user_text: str, rag_context: str = None) -> dict:
        """Non-streaming chat. Returns full response."""
        messages = self._build_messages(user_text, rag_context)
        t0 = time.perf_counter()
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.15}

        resp = self._client.chat.completions.create(
            model=self._model, messages=messages,
            max_tokens=120, temperature=0.85, top_p=0.9, extra_body=extra_body,
        )
        latency = (time.perf_counter() - t0) * 1000
        text = self._clean(resp.choices[0].message.content or "")
        tokens = resp.usage.completion_tokens if resp.usage else 0

        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": text})
        self._trim_history()

        return {"text": text, "latency_ms": latency, "tokens": tokens}

    def stream_sentences(self, user_text: str, rag_context: str = None):
        """Streaming chat that yields complete sentences as they form.
        Yields: {'sentence': str, 'ttfs_ms': float, 'is_first': bool, 'is_last': bool}
        """
        messages = self._build_messages(user_text, rag_context)
        t0 = time.perf_counter()
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.15}

        stream = self._client.chat.completions.create(
            model=self._model, messages=messages,
            max_tokens=120, temperature=0.85, top_p=0.9,
            stream=True, extra_body=extra_body,
        )

        buffer = ""
        full_text = ""
        first_sentence = True
        first_token_time = None

        for chunk in stream:
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue
            token = chunk.choices[0].delta.content
            if first_token_time is None:
                first_token_time = time.perf_counter()

            buffer += token

            while True:
                match = _SENTENCE_END_RE.search(buffer)
                if not match:
                    break
                idx = match.end()
                sentence = self._clean(buffer[:idx])
                buffer = buffer[idx:].lstrip()

                if sentence and len(sentence) >= 2:
                    elapsed = (time.perf_counter() - t0) * 1000
                    yield {
                        "sentence": sentence,
                        "ttfs_ms": elapsed if first_sentence else 0,
                        "is_first": first_sentence,
                        "is_last": False,
                    }
                    full_text += sentence
                    first_sentence = False

        remainder = self._clean(buffer)
        if remainder and len(remainder) >= 2:
            elapsed = (time.perf_counter() - t0) * 1000
            yield {
                "sentence": remainder,
                "ttfs_ms": elapsed if first_sentence else 0,
                "is_first": first_sentence,
                "is_last": True,
            }
            full_text += remainder
        elif full_text:
            pass  # last sentence already yielded

        final_text = self._clean(full_text)
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": final_text})
        self._trim_history()

    @staticmethod
    def _clean(text: str) -> str:
        text = _TAG_RE.sub("", text)
        text = _INCOMPLETE_THINK_RE.sub("", text)
        text = _ANY_XML_TAG_RE.sub("", text)
        return text.strip()
