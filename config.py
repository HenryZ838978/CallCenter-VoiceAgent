import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

ASR_MODEL_DIR = os.path.join(MODEL_DIR, "SenseVoiceSmall")
TTS_MODEL_DIR = os.path.join(MODEL_DIR, "VoxCPM1.5")
LLM_MODEL_DIR = os.path.join(MODEL_DIR, "MiniCPM4.1-8B-GPTQ")
VAD_MODEL_DIR = os.path.join(MODEL_DIR, "snakers4_silero-vad")

VLLM_BASE_URL = "http://localhost:8100/v1"
VLLM_MODEL_NAME = "MiniCPM4.1-8B-GPTQ"

LLM_GPU = int(os.environ.get("LLM_GPU", "1"))
ASR_GPU = int(os.environ.get("ASR_GPU", "2"))
TTS_GPU = int(os.environ.get("TTS_GPU", "2"))

SAMPLE_RATE = 16000
CHUNK_MS = 32
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 512

EMBED_MODEL_DIR = os.path.join(MODEL_DIR, "bge-small-zh-v1.5")
EMBED_MODEL_LARGE_DIR = os.path.join(MODEL_DIR, "bge-m3")
RERANKER_MODEL_DIR = os.path.join(MODEL_DIR, "bge-reranker-v2-m3")
KB_DATA_PATH = os.path.join(BASE_DIR, "data", "sample_kb.json")
RAG_INDEX_PATH = os.path.join(BASE_DIR, "data", "rag_index")
RAG_TOP_K = 3
RAG_GPU = int(os.environ.get("RAG_GPU", "2"))

VOICE_PROMPT_WAV = os.path.join(BASE_DIR, "data", "voice_prompt.wav")
VOICE_PROMPT_TEXT = "在国内引起了非常大的反响啊，我们也完全没有想到我们的这个工作会以这种方式出圈。"

SYSTEM_PROMPT = (
    "你是 VoxLabs 智能客服助手。你需要礼貌、简洁地回答用户的问题。"
    "回答尽量控制在2-3句话内。使用中文回答。"
)

SYSTEM_PROMPT_RAG = (
    "你是 VoxLabs 智能客服助手。根据以下知识库信息回答用户的问题。"
    "如果知识库中有相关信息，优先使用知识库内容回答。"
    "回答简洁，控制在2-3句话内。使用中文回答。\n\n"
    "知识库信息：\n{context}"
)
