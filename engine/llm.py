"""LLM inference via vLLM OpenAI-compatible API."""
import re
import time
from openai import OpenAI

_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_INCOMPLETE_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_ANY_XML_TAG_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")


class VLLMChat:
    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "MiniCPM4.1-8B-GPTQ",
                 system_prompt: str = ""):
        self._client = OpenAI(base_url=base_url, api_key="dummy")
        self._model = model
        self._system_prompt = system_prompt
        self._history = []

    def reset(self):
        self._history = []

    def chat(self, user_text: str, stream: bool = False,
             rag_context: str = None) -> dict:
        messages = []
        system = self._system_prompt
        if rag_context:
            system = system.replace("{context}", rag_context) if "{context}" in system else (
                system + f"\n\n知识库信息：\n{rag_context}"
            )
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})

        t0 = time.perf_counter()
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        extra_body["repetition_penalty"] = 1.2
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            extra_body=extra_body,
        )
        latency = (time.perf_counter() - t0) * 1000
        text = self._clean(resp.choices[0].message.content or "")
        tokens = resp.usage.completion_tokens if resp.usage else 0

        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": text})
        if len(self._history) > 10:
            self._history = self._history[-10:]

        return {"text": text, "latency_ms": latency, "tokens": tokens}

    @staticmethod
    def _clean(text: str) -> str:
        text = _TAG_RE.sub("", text)
        text = _INCOMPLETE_THINK_RE.sub("", text)
        text = _ANY_XML_TAG_RE.sub("", text)
        text = text.strip()
        return text
