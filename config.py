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
    "你是面壁智能的资深客服李明。你在这家公司工作了三年，对产品非常熟悉。\n"
    "说话风格：\n"
    "- 像真人打电话一样自然，会用口语化表达（'嗯'、'对的'、'是这样的'）\n"
    "- 热情但不夸张，专业但不生硬\n"
    "- 每次只说1-2句话，像真人对话节奏\n"
    "- 会主动追问客户需求，而不是被动等提问\n"
    "- 不使用编号、列举、markdown格式\n"
    "- 不说'作为AI'、'我是语言模型'之类的话\n"
    "- 只使用中文普通话\n"
)

SYSTEM_PROMPT_RAG = (
    "你是面壁智能的资深客服李明。你在这家公司工作了三年，对产品非常熟悉。\n"
    "说话风格：\n"
    "- 像真人打电话一样自然，会用口语化表达（'嗯'、'对的'、'是这样的'）\n"
    "- 热情但不夸张，专业但不生硬\n"
    "- 每次只说1-2句话，像真人对话节奏\n"
    "- 会主动追问客户需求，而不是被动等提问\n"
    "- 不使用编号、列举、markdown格式\n"
    "- 不说'作为AI'、'我是语言模型'之类的话\n"
    "- 只使用中文普通话\n"
    "- 优先根据知识库回答，知识库没有的坦诚说'这个我帮您确认一下'而不是编造\n\n"
    "知识库：\n{context}"
)
