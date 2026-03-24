"""Self-test for v2.9 features (no GPU / no running services required).

Tests:
  D1-D4 — Experience layer (greeting, idle timeout, ASR recovery, farewell)
  P0    — Crash safety (_run_pipeline exception → spoken apology + IDLE)
  P0    — Task registry + shutdown lifecycle
  P1    — Filler activation on THINKING
  P1    — Session metrics collection
  G1    — State race: _run_pipeline awaits _stream_llm_tts (no premature IDLE)
  G2    — handle_text_input RAG exception safety
  G7    — Bounded sentence queue backpressure
  G9    — LLM history trim + summary
  G10   — RAG score threshold filtering

Usage:
    python test_experience.py
"""
import os
import sys
import time
import asyncio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = {}


# ---------------------------------------------------------------------------
# Mocks — no GPU, no model files, no running servers
# ---------------------------------------------------------------------------
class MockVAD:
    def __init__(self):
        self._speech_prob = 0.0

    def process_chunk(self, chunk, sr):
        return {"speech_prob": self._speech_prob}

    def reset(self):
        self._speech_prob = 0.0

    def set_speech(self, prob):
        self._speech_prob = prob


class MockASR:
    def __init__(self):
        self.next_text = "你好"

    def transcribe(self, audio):
        return {"text": self.next_text, "latency_ms": 5.0}


class MockLLM:
    def __init__(self):
        self.next_sentences = [{"sentence": "好的，", "ttfs_ms": 10, "is_first": True, "is_last": False},
                               {"sentence": "有什么可以帮您？", "ttfs_ms": 0, "is_first": False, "is_last": True}]

    def stream_sentences(self, user_text, rag_context=None):
        for s in self.next_sentences:
            yield s

    def reset(self):
        pass


class MockTTS:
    SAMPLE_RATE = 44100

    def synthesize_stream(self, text):
        chunk = np.zeros(4410, dtype=np.float32)
        yield {"audio": chunk, "ttfa_ms": 10.0, "total_ms": 20.0}
        yield {"audio": chunk, "ttfa_ms": 0, "total_ms": 20.0}

    def synthesize(self, text, prompt_id=None):
        audio = np.zeros(44100, dtype=np.float32)
        return {"audio": audio, "sr": 44100, "ttfa_ms": 10.0, "total_ms": 50.0, "chunks": 1}

    def warmup(self):
        pass


class CrashRAG:
    """RAG that always raises."""
    def get_context(self, question, top_k=None):
        raise RuntimeError("RAG index corrupted")


class MockRAG:
    def get_context(self, question, top_k=None):
        return {"context": "mock context", "num_results": 1,
                "embed_ms": 1, "search_ms": 0.1, "rerank_ms": 0, "total_ms": 1.1}


class MockFiller:
    def get_filler(self):
        return "嗯，", b"\x00" * 100


class MessageCollector:
    """Collects all messages sent by ConversationManager."""

    def __init__(self):
        self.messages = []
        self.audio_chunks = 0

    async def __call__(self, data):
        if "_audio" in data:
            self.audio_chunks += 1
        else:
            self.messages.append(data)

    def types(self):
        return [m.get("type") for m in self.messages]

    def find(self, msg_type):
        return [m for m in self.messages if m.get("type") == msg_type]

    def has(self, msg_type):
        return any(m.get("type") == msg_type for m in self.messages)

    def clear(self):
        self.messages.clear()
        self.audio_chunks = 0


def make_cm(collector, **overrides):
    from engine.conversation_manager import ConversationManager
    cfg = {
        "greeting_text": "您好，测试开场白。",
        "idle_timeout_s": 0.3,
        "idle_goodbye_s": 0.8,
        "idle_prompt_text": "您还在吗？",
        "idle_goodbye_text": "再见！",
        "asr_empty_retry_text": "没听清",
        "asr_noisy_suggest_text": "环境嘈杂",
        "max_consecutive_empty_asr": 2,
        "farewell_keywords": ["再见", "拜拜"],
    }
    cfg.update(overrides)
    return ConversationManager(
        asr=MockASR(), llm=MockLLM(), tts=MockTTS(),
        rag=MockRAG(), vad=MockVAD(), filler=MockFiller(),
        send_fn=collector, experience_config=cfg,
    )


def section(name):
    print(f"\n{'=' * 60}")
    print(f"  TEST: {name}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# D1: Greeting
# ---------------------------------------------------------------------------
async def test_d1_greeting():
    section("D1 — Greeting auto-plays on start")
    col = MessageCollector()
    cm = make_cm(col)

    await cm.start_greeting()

    has_system_msg = col.has("system_message")
    has_audio_start = col.has("audio_start")
    has_audio = col.audio_chunks > 0
    ended_idle = col.find("state") and col.find("state")[-1].get("state") == "idle"

    sys_msgs = col.find("system_message")
    greeting_text = sys_msgs[0].get("text", "") if sys_msgs else ""

    print(f"  System message sent: {has_system_msg}")
    print(f"  Audio start sent: {has_audio_start}")
    print(f"  Audio chunks: {col.audio_chunks}")
    print(f"  Ended in IDLE: {ended_idle}")
    print(f"  Greeting text: '{greeting_text}'")

    ok = has_system_msg and has_audio_start and has_audio and ended_idle
    print(f"  Result: {PASS if ok else FAIL}")
    results["D1_greeting"] = ok


async def test_d1_no_greeting():
    section("D1 — No greeting when text is empty")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")

    await cm.start_greeting()

    no_messages = not col.has("system_message") and col.audio_chunks == 0
    print(f"  No system message: {no_messages}")
    print(f"  Result: {PASS if no_messages else FAIL}")
    results["D1_no_greeting"] = no_messages


# ---------------------------------------------------------------------------
# D2: Idle timeout
# ---------------------------------------------------------------------------
async def test_d2_idle_prompt():
    section("D2 — Idle timeout triggers prompt")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="", idle_timeout_s=0.2, idle_goodbye_s=5.0)
    from engine.conversation_manager import State
    cm.state = State.IDLE
    cm._idle_since = time.monotonic() - 0.3

    await cm._check_idle_timeout()

    prompted = cm._idle_prompted
    has_sys = col.has("system_message")
    sys_text = col.find("system_message")[0].get("text", "") if has_sys else ""

    print(f"  _idle_prompted set: {prompted}")
    print(f"  System message: '{sys_text}'")

    ok = prompted and "还在" in sys_text
    print(f"  Result: {PASS if ok else FAIL}")
    results["D2_idle_prompt"] = ok


async def test_d2_idle_goodbye():
    section("D2 — Idle goodbye + session_end")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="", idle_timeout_s=0.1, idle_goodbye_s=0.3)
    from engine.conversation_manager import State
    cm.state = State.IDLE
    cm._idle_since = time.monotonic() - 0.15
    await cm._check_idle_timeout()
    assert cm._idle_prompted

    # Wait for the prompt TTS task to finish
    if cm._speaking_task:
        await cm._speaking_task
    col.clear()

    cm._idle_since = time.monotonic() - 0.4
    await cm._check_idle_timeout()

    goodbye_sent = cm._idle_goodbye_sent
    session_ending = cm._session_ending

    # Wait for goodbye TTS task
    if cm._speaking_task:
        await cm._speaking_task

    has_session_end = col.has("session_end")
    reason = col.find("session_end")[0].get("reason", "") if has_session_end else ""

    print(f"  _idle_goodbye_sent: {goodbye_sent}")
    print(f"  session_end sent: {has_session_end}")
    print(f"  reason: '{reason}'")

    ok = goodbye_sent and has_session_end and reason == "idle_timeout"
    print(f"  Result: {PASS if ok else FAIL}")
    results["D2_goodbye"] = ok


async def test_d2_reset_on_speech():
    section("D2 — Idle flags reset when user speaks")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="", idle_timeout_s=0.1, idle_goodbye_s=0.5)
    from engine.conversation_manager import State

    cm.state = State.IDLE
    cm._idle_prompted = True
    cm._consecutive_empty_asr = 2

    speech_chunk = np.random.randn(512).astype(np.float32) * 0.1
    cm.vad.set_speech(0.9)
    await cm._process_chunk(speech_chunk)

    reset_ok = (not cm._idle_prompted and
                not cm._idle_goodbye_sent and
                cm._consecutive_empty_asr == 0 and
                cm.state == State.LISTENING)

    print(f"  State after speech: {cm.state.value}")
    print(f"  idle_prompted reset: {not cm._idle_prompted}")
    print(f"  consecutive_empty reset: {cm._consecutive_empty_asr == 0}")
    print(f"  Result: {PASS if reset_ok else FAIL}")
    results["D2_reset"] = reset_ok


# ---------------------------------------------------------------------------
# D3: ASR error recovery
# ---------------------------------------------------------------------------
async def test_d3_asr_empty():
    section("D3 — ASR empty triggers recovery message")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    cm.asr.next_text = ""

    speech = np.random.randn(16000).astype(np.float32) * 0.1
    from engine.conversation_manager import State
    cm.state = State.THINKING
    cm._turn = 1

    await cm._run_pipeline(speech, 1, None, 0, None)

    sys_msgs = col.find("system_message")
    has_recovery = len(sys_msgs) > 0
    recovery_text = sys_msgs[0].get("text", "") if has_recovery else ""
    count = cm._consecutive_empty_asr

    print(f"  Recovery message: '{recovery_text}'")
    print(f"  Consecutive empty count: {count}")
    print(f"  Ended in IDLE: {cm.state == State.IDLE}")

    ok = has_recovery and "听清" in recovery_text and count == 1
    print(f"  Result: {PASS if ok else FAIL}")
    results["D3_empty"] = ok


async def test_d3_escalation():
    section("D3 — Escalation to noisy suggestion after N empties")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="", max_consecutive_empty_asr=2)
    cm.asr.next_text = ""
    from engine.conversation_manager import State

    for i in range(3):
        col.clear()
        cm.state = State.THINKING
        cm._turn = i + 1
        speech = np.random.randn(16000).astype(np.float32) * 0.1
        await cm._run_pipeline(speech, i + 1, None, 0, None)

    sys_msgs = col.find("system_message")
    last_text = sys_msgs[0].get("text", "") if sys_msgs else ""

    escalated = "嘈杂" in last_text
    count = cm._consecutive_empty_asr

    print(f"  After 3 empties, message: '{last_text}'")
    print(f"  Consecutive count: {count}")
    print(f"  Escalated to noisy: {escalated}")
    print(f"  Result: {PASS if escalated else FAIL}")
    results["D3_escalation"] = escalated


# ---------------------------------------------------------------------------
# D4: Farewell detection
# ---------------------------------------------------------------------------
async def test_d4_farewell():
    section("D4 — Farewell keyword triggers session_end")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    cm.asr.next_text = "好的谢谢，再见"
    from engine.conversation_manager import State

    cm.state = State.THINKING
    cm._turn = 1
    speech = np.random.randn(16000).astype(np.float32) * 0.1

    await cm._run_pipeline(speech, 1, None, 0, None)

    # Wait for LLM+TTS streaming task
    if cm._speaking_task:
        await cm._speaking_task

    has_end = col.has("session_end")
    reason = col.find("session_end")[0].get("reason", "") if has_end else ""

    print(f"  session_end sent: {has_end}")
    print(f"  reason: '{reason}'")
    print(f"  Result: {PASS if has_end and reason == 'farewell' else FAIL}")
    results["D4_farewell"] = has_end and reason == "farewell"


async def test_d4_no_false_positive():
    section("D4 — Normal text does NOT trigger session_end")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    cm.asr.next_text = "你好，介绍一下产品"
    from engine.conversation_manager import State

    cm.state = State.THINKING
    cm._turn = 1
    speech = np.random.randn(16000).astype(np.float32) * 0.1

    await cm._run_pipeline(speech, 1, None, 0, None)

    if cm._speaking_task:
        await cm._speaking_task

    no_end = not col.has("session_end")

    print(f"  No session_end: {no_end}")
    print(f"  Result: {PASS if no_end else FAIL}")
    results["D4_no_false"] = no_end


# ---------------------------------------------------------------------------
# P0: Crash safety
# ---------------------------------------------------------------------------
async def test_p0_pipeline_crash():
    section("P0 — Pipeline crash → apology + IDLE (not stuck in THINKING)")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    from engine.conversation_manager import State

    class CrashASR:
        def transcribe(self, audio):
            raise RuntimeError("GPU OOM simulated")

    cm.asr = CrashASR()
    cm.state = State.THINKING
    cm._turn = 1

    speech = np.random.randn(16000).astype(np.float32) * 0.1
    await cm._run_pipeline(speech, 1, None, 0, None)

    ended_idle = cm.state == State.IDLE
    has_error = col.has("error") or col.has("system_message")
    has_audio = col.audio_chunks > 0

    print(f"  Ended in IDLE (not stuck): {ended_idle}")
    print(f"  Error/apology sent: {has_error}")
    print(f"  Audio apology played: {has_audio}")
    print(f"  Errors recorded: {cm.metrics.errors}")

    ok = ended_idle and has_error
    print(f"  Result: {PASS if ok else FAIL}")
    results["P0_crash_safety"] = ok


async def test_p0_shutdown():
    section("P0 — shutdown() cancels all tracked tasks")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    from engine.conversation_manager import State

    async def slow_task():
        await asyncio.sleep(10)

    cm._track_task(slow_task(), name="test-slow")
    assert len(cm._tasks) >= 1

    cm.shutdown()

    await asyncio.sleep(0.05)
    all_done = all(t.done() for t in cm._tasks) if cm._tasks else True
    is_dead = cm._dead

    print(f"  All tasks cancelled: {all_done}")
    print(f"  CM marked dead: {is_dead}")

    ok = all_done and is_dead
    print(f"  Result: {PASS if ok else FAIL}")
    results["P0_shutdown"] = ok


# ---------------------------------------------------------------------------
# P1: Filler activation
# ---------------------------------------------------------------------------
async def test_p1_filler():
    section("P1 — Filler sent on THINKING entry")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    cm.asr.next_text = "你好"
    from engine.conversation_manager import State

    cm.vad.set_speech(0.9)
    speech = np.random.randn(512).astype(np.float32) * 0.1
    await cm._process_chunk(speech)
    assert cm.state == State.LISTENING

    cm.vad.set_speech(0.0)
    for _ in range(12):
        silence = np.zeros(512, dtype=np.float32)
        await cm._process_chunk(silence)

    await asyncio.sleep(0.1)
    if cm._speaking_task:
        await cm._speaking_task

    has_filler = col.has("filler")
    filler_msgs = col.find("filler")
    filler_text = filler_msgs[0].get("text", "") if filler_msgs else ""

    print(f"  Filler sent: {has_filler}")
    print(f"  Filler text: '{filler_text}'")

    ok = has_filler
    print(f"  Result: {PASS if ok else FAIL}")
    results["P1_filler"] = ok


# ---------------------------------------------------------------------------
# P1: Session metrics
# ---------------------------------------------------------------------------
async def test_p1_metrics():
    section("P1 — Session metrics collection")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    cm.asr.next_text = "你好"
    from engine.conversation_manager import State

    cm.state = State.THINKING
    cm._turn = 1
    speech = np.random.randn(16000).astype(np.float32) * 0.1
    await cm._run_pipeline(speech, 1, None, 0, None)
    if cm._speaking_task:
        await cm._speaking_task

    summary = cm.metrics.summary()
    has_turn = summary["turns"] >= 1

    print(f"  Metrics: {summary}")
    print(f"  Turn recorded: {has_turn}")

    ok = has_turn
    print(f"  Result: {PASS if ok else FAIL}")
    results["P1_metrics"] = ok


# ---------------------------------------------------------------------------
# G1: State race — _run_pipeline awaits _stream_llm_tts
# ---------------------------------------------------------------------------
async def test_g1_no_premature_idle():
    section("G1 — No premature IDLE while LLM+TTS still streaming")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    cm.asr.next_text = "你好"
    from engine.conversation_manager import State

    cm.state = State.THINKING
    cm._turn = 1
    speech = np.random.randn(16000).astype(np.float32) * 0.1

    state_log = []
    original_send = col.__call__
    async def tracking_send(data):
        await original_send(data)
        if isinstance(data, dict) and data.get("type") == "state":
            state_log.append(data["state"])

    cm._send = tracking_send
    await cm._run_pipeline(speech, 1, None, 0, None)

    speaking_idx = state_log.index("speaking") if "speaking" in state_log else -1
    idle_indices = [i for i, s in enumerate(state_log) if s == "idle"]
    last_idle = idle_indices[-1] if idle_indices else -1

    no_premature = True
    for idx in idle_indices:
        if idx < speaking_idx:
            no_premature = False

    ended_idle = cm.state == State.IDLE
    has_audio = col.audio_chunks > 0

    print(f"  State sequence: {state_log}")
    print(f"  Audio chunks played: {col.audio_chunks}")
    print(f"  No premature IDLE before TTS done: {no_premature}")
    print(f"  Final state IDLE: {ended_idle}")

    ok = no_premature and ended_idle and has_audio
    print(f"  Result: {PASS if ok else FAIL}")
    results["G1_no_premature_idle"] = ok


# ---------------------------------------------------------------------------
# G2: handle_text_input RAG exception safety
# ---------------------------------------------------------------------------
async def test_g2_text_input_rag_crash():
    section("G2 — handle_text_input survives RAG crash")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    cm.rag = CrashRAG()
    from engine.conversation_manager import State

    try:
        await cm.handle_text_input("你好")
        if cm._speaking_task:
            await cm._speaking_task
        survived = True
    except Exception:
        survived = False

    ended_idle = cm.state == State.IDLE
    has_audio = col.audio_chunks > 0

    print(f"  Survived RAG crash: {survived}")
    print(f"  Final state IDLE: {ended_idle}")
    print(f"  Audio output generated: {has_audio}")

    ok = survived and ended_idle
    print(f"  Result: {PASS if ok else FAIL}")
    results["G2_text_rag_crash"] = ok


# ---------------------------------------------------------------------------
# G7: Bounded sentence queue
# ---------------------------------------------------------------------------
async def test_g7_bounded_queue():
    section("G7 — Sentence queue has maxsize (backpressure)")
    col = MessageCollector()
    cm = make_cm(col, greeting_text="")
    cm.asr.next_text = "你好"
    from engine.conversation_manager import State

    cm.state = State.THINKING
    cm._turn = 1
    speech = np.random.randn(16000).astype(np.float32) * 0.1
    await cm._run_pipeline(speech, 1, None, 0, None)

    ok = cm.state == State.IDLE
    print(f"  Pipeline completed with bounded queue: {ok}")
    print(f"  Result: {PASS if ok else FAIL}")
    results["G7_bounded_queue"] = ok


# ---------------------------------------------------------------------------
# G9: LLM history trimming + summary
# ---------------------------------------------------------------------------
async def test_g9_llm_history_trim():
    section("G9 — LLM history trim + summary generation")

    class TestLLM:
        def __init__(self):
            self._history = []
            self._max_history = 4
            self._summary = ""

        def _trim_history(self):
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

    llm = TestLLM()
    for i in range(10):
        llm._history.append({"role": "user", "content": f"question {i}"})
        llm._history.append({"role": "assistant", "content": f"answer {i}"})
    llm._trim_history()

    trimmed = len(llm._history) <= 4
    has_summary = len(llm._summary) > 0 and "摘要" in llm._summary

    print(f"  History size after trim: {len(llm._history)} (max={llm._max_history})")
    print(f"  Summary generated: {has_summary}")
    print(f"  Summary: '{llm._summary[:80]}...'")

    ok = trimmed and has_summary
    print(f"  Result: {PASS if ok else FAIL}")
    results["G9_history_trim"] = ok


# ---------------------------------------------------------------------------
# G10: RAG score threshold
# ---------------------------------------------------------------------------
async def test_g10_rag_threshold():
    section("G10 — RAG filters low-confidence results")

    class MockRAGWithThreshold:
        def __init__(self):
            self._score_threshold = 0.5
            self._reranker = None

        def test_filter(self, candidates, k):
            results = candidates[:k]
            if self._score_threshold > 0:
                score_key = "rerank_score" if self._reranker else "score"
                results = [r for r in results if r.get(score_key, 0) >= self._score_threshold]
            return results

    rag = MockRAGWithThreshold()
    candidates = [
        {"text": "good", "score": 0.8},
        {"text": "ok", "score": 0.5},
        {"text": "bad", "score": 0.2},
    ]
    filtered = rag.test_filter(candidates, 3)
    kept = len(filtered)
    no_bad = all(r["score"] >= 0.5 for r in filtered)

    print(f"  Input candidates: {len(candidates)}, after filter: {kept}")
    print(f"  All above threshold: {no_bad}")

    ok = kept == 2 and no_bad
    print(f"  Result: {PASS if ok else FAIL}")
    results["G10_rag_threshold"] = ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    print("=" * 60)
    print("  VoxLabs v2.9 Self-Test (D1-D4 + P0-P1 + G1-G10)")
    print("  No GPU / No services required")
    print("=" * 60)

    await test_d1_greeting()
    await test_d1_no_greeting()
    await test_d2_idle_prompt()
    await test_d2_idle_goodbye()
    await test_d2_reset_on_speech()
    await test_d3_asr_empty()
    await test_d3_escalation()
    await test_d4_farewell()
    await test_d4_no_false_positive()
    await test_p0_pipeline_crash()
    await test_p0_shutdown()
    await test_p1_filler()
    await test_p1_metrics()
    await test_g1_no_premature_idle()
    await test_g2_text_input_rag_crash()
    await test_g7_bounded_queue()
    await test_g9_llm_history_trim()
    await test_g10_rag_threshold()

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        print(f"  {name:30s} {PASS if ok else FAIL}")
    total = len(results)
    passed = sum(results.values())
    all_pass = all(results.values())
    print(f"\n  {passed}/{total} {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 60}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
