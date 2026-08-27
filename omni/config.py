import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

VOICEAGENT_DIR = os.path.join(os.path.dirname(BASE_DIR), "voiceagent")

OMNI_MODEL_DIR = os.environ.get(
    "OMNI_MODEL_DIR",
    os.path.join(MODEL_DIR, "MiniCPM-o-4_5"),
)

VAD_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "snakers4_silero-vad")
EMBED_MODEL_DIR = os.path.join(VOICEAGENT_DIR, "models", "bge-small-zh-v1.5")
KB_DATA_PATH = os.path.join(VOICEAGENT_DIR, "data", "sample_kb.json")
VOICE_PROMPT_WAV = os.path.join(VOICEAGENT_DIR, "data", "voice_prompt.wav")

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
CHUNK_MS = 32
CHUNK_SAMPLES = int(SAMPLE_RATE_IN * CHUNK_MS / 1000)  # 512

OMNI_DEVICE = os.environ.get("OMNI_DEVICE", "cuda:0")
RAG_DEVICE = os.environ.get("RAG_DEVICE", "cpu")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "1") == "1"

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "3001"))

VAD_THRESHOLD = 0.5
VAD_SILENCE_CHUNKS = 15       # ~480ms silence to trigger speech_end
ENDPOINTING_CHUNKS = 8        # ~256ms for faster endpointing
MIN_SPEECH_CHUNKS = 10        # ~320ms minimum speech to process


def build_system_prompt() -> str:
    """Build system prompt with full KB embedded (no separate RAG lookup needed)."""
    prompt = (
        "你是面壁智能的专业客服代表。只使用中文普通话。"
        "语气友好、专业、沉稳，像资深客服人员。"
        "回答简洁准确，每次1-2句话。不要用编号和列举。"
        "如果知识库没有相关信息，坦诚告知并表示会跟进。\n\n"
        "知识库：\n"
    )
    if os.path.exists(KB_DATA_PATH):
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        for doc in docs:
            prompt += f"Q: {doc['question']}\nA: {doc['answer']}\n\n"
    return prompt


SYSTEM_PROMPT = build_system_prompt()
