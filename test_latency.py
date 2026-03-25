"""延迟精测: SFT-GPTQ 模型 max_tokens=80 + TTFT streaming"""
import time
import re
from openai import OpenAI
import httpx

client = OpenAI(
    base_url="http://localhost:8199/v1", api_key="dummy",
    timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=10.0),
)

SYSTEM = (
    "你是面壁智能的资深外呼坐席李明。你在这家公司工作了三年，负责给潜在客户打电话介绍AI智能客服产品。\n"
    "说话风格：像真人打电话一样自然，口语化，每次1-3句话。\n"
    "会主动引导对话、挖掘需求。遇拒绝礼貌退出不纠缠，遇犹豫适度引导给方案。\n"
    "不用编号列举markdown，不暴露AI身份，只说中文。不要输出思考过程。"
)

tests = [
    ("开场白", "喂？"),
    ("愿意听", "你说吧，什么事？"),
    ("产品介绍", "AI客服？具体能做什么呀？"),
    ("直接拒绝", "不感兴趣，别打了"),
    ("投诉", "你们这是骚扰电话吧？我要投诉"),
    ("犹豫", "听起来还行，但我得考虑考虑"),
    ("问价格", "这个要多少钱？"),
    ("约时间", "行，那你约个时间给我详细讲讲"),
    ("结束", "好的，那就这样吧"),
    ("辱骂", "滚，别烦我"),
]

_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Warmup
client.chat.completions.create(
    model="MiniCPM4.1-8B-SFT-GPTQ",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=10,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

header = f"{'场景':<12} {'总延迟':>8} {'TTFT':>8} {'tokens':>7}  回复"
print(header)
print("-" * 90)

latencies = []
ttfts = []

for label, user_input in tests:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_input},
    ]
    extra = {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.15}

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="MiniCPM4.1-8B-SFT-GPTQ", messages=messages,
        max_tokens=80, temperature=0.7, top_p=0.9,
        extra_body=extra,
    )
    total_ms = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content or ""
    text = _TAG_RE.sub("", raw).strip()
    tokens = resp.usage.completion_tokens if resp.usage else 0
    latencies.append(total_ms)

    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model="MiniCPM4.1-8B-SFT-GPTQ", messages=messages,
        max_tokens=80, temperature=0.7, top_p=0.9,
        stream=True, extra_body=extra,
    )
    ttft = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            ttft = (time.perf_counter() - t0) * 1000
            break
    for chunk in stream:
        pass
    if ttft:
        ttfts.append(ttft)
    else:
        ttft = 0
        ttfts.append(0)

    print(f"{label:<12} {total_ms:>7.0f}ms {ttft:>7.0f}ms {tokens:>6}  {text[:55]}...")

print("-" * 90)
avg_lat = sum(latencies) / len(latencies)
p50_lat = sorted(latencies)[len(latencies) // 2]
p90_lat = sorted(latencies)[int(len(latencies) * 0.9)]
avg_ttft = sum(ttfts) / len(ttfts)
p50_ttft = sorted(ttfts)[len(ttfts) // 2]
p90_ttft = sorted(ttfts)[int(len(ttfts) * 0.9)]

print(f"总延迟:  avg={avg_lat:.0f}ms  p50={p50_lat:.0f}ms  p90={p90_lat:.0f}ms")
print(f"TTFT:    avg={avg_ttft:.0f}ms  p50={p50_ttft:.0f}ms  p90={p90_ttft:.0f}ms")
print(f"总延迟 <400ms: {sum(1 for l in latencies if l < 400)}/{len(latencies)}")
print(f"总延迟 <800ms: {sum(1 for l in latencies if l < 800)}/{len(latencies)}")
print(f"TTFT <200ms:   {sum(1 for t in ttfts if t < 200)}/{len(ttfts)}")
print(f"TTFT <400ms:   {sum(1 for t in ttfts if t < 400)}/{len(ttfts)}")
