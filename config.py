import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

ASR_MODEL_DIR = os.path.join(MODEL_DIR, "SenseVoiceSmall")
TTS_MODEL_DIR = os.path.join(MODEL_DIR, "VoxCPM1.5")
LLM_MODEL_DIR = os.path.join(MODEL_DIR, "Qwen3-14B-AWQ")
VAD_MODEL_DIR = os.path.join(MODEL_DIR, "snakers4_silero-vad")

VLLM_BASE_URL = "http://localhost:8100/v1"
VLLM_MODEL_NAME = "Qwen3-14B-AWQ"

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

# ---------------------------------------------------------------------------
# LLM context
# ---------------------------------------------------------------------------
LLM_MAX_HISTORY = int(os.environ.get("LLM_MAX_HISTORY", "30"))

# ---------------------------------------------------------------------------
# API key (empty = auth disabled)
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("VOICEAGENT_API_KEY", "")

# ---------------------------------------------------------------------------
# RAG confidence threshold (cosine similarity, 0-1)
# ---------------------------------------------------------------------------
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.35"))

VOICE_PROMPT_WAV = os.environ.get(
    "VOICE_PROMPT_WAV",
    os.path.join(BASE_DIR, "data", "lipsync_real_clone_sample_5s.wav"),
)
VOICE_PROMPT_TEXT = os.environ.get(
    "VOICE_PROMPT_TEXT",
    "hello大家好呀我是毒舌表妹豆角给你拆解AI圈儿又有什么专门吓唬你的大新闻",
)

# ---------------------------------------------------------------------------
# D1: Greeting
# ---------------------------------------------------------------------------
GREETING_TEXT = os.environ.get(
    "GREETING_TEXT",
    "您好，我是面壁智能的Voice services Agent。有什么可以帮您的呢？",
)

# ---------------------------------------------------------------------------
# D2: Idle timeout (seconds)
# ---------------------------------------------------------------------------
IDLE_TIMEOUT_S = float(os.environ.get("IDLE_TIMEOUT_S", "15"))
IDLE_GOODBYE_S = float(os.environ.get("IDLE_GOODBYE_S", "30"))
IDLE_PROMPT_TEXT = os.environ.get("IDLE_PROMPT_TEXT", "您还在吗？有什么需要帮助的吗？")
IDLE_GOODBYE_TEXT = os.environ.get("IDLE_GOODBYE_TEXT", "好的，如果后续有问题随时找我")

# ---------------------------------------------------------------------------
# D3: Error recovery
# ---------------------------------------------------------------------------
ASR_EMPTY_RETRY_TEXT = "抱歉没太听清，您能再说一次吗？"
ASR_NOISY_SUGGEST_TEXT = "您那边好像有点吵，要不，您换个安静点环境？"
MAX_CONSECUTIVE_EMPTY_ASR = 3

# ---------------------------------------------------------------------------
# D4: Farewell detection
# ---------------------------------------------------------------------------
FAREWELL_KEYWORDS = [
    "再见", "拜拜", "拜了", "没了", "就这些", "没有了",
    "没其他", "没别的", "谢谢再见", "好的再见", "挂了",
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是面壁智能的资深客服lucy。你在这家公司工作了三年，对产品非常熟悉。\n"
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
    "你是面壁智能的资深客服lucy。你在这家公司工作了三年，对产品非常熟悉。\n"
    "说话风格：\n"
    "- 像真人打电话一样自然，会用口语化表达（'嗯'、'对的'、'是这样的'）\n"
    "- 热情但不夸张，专业但不生硬\n"
    "- 每次只说1-2句话，像真人对话节奏\n"
    "- 会主动追问客户需求，而不是被动等提问\n"
    "- 不使用编号、列举、markdown格式\n"
    "- 不说'作为AI'、'我是语言模型'之类的话\n"
    "- 只使用中文普通话\n"
    "- 优先根据知识库回答，知识库没有的坦诚说'这个我问下同事，再给您回复哈'而不是编造\n\n"
    "知识库：\n{context}"
)
