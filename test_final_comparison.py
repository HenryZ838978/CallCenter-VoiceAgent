"""最终对比: 原始 GPTQ+fewshot vs SFT-GPTQ+简单prompt

重点关注:
1. 延迟 (目标全链路 <800ms, LLM 部分 <400ms)
2. 关键要素命中率
3. 量化后 SFT 效果是否保留
"""
import json
import os
import re
import sys
import time

from openai import OpenAI
import httpx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "outbound_call_benchmark.json")

SYSTEM_PROMPT_SFT = (
    "你是面壁智能的资深外呼坐席李明。你在这家公司工作了三年，负责给潜在客户打电话介绍AI智能客服产品。\n"
    "说话风格：像真人打电话一样自然，口语化，每次1-3句话。\n"
    "会主动引导对话、挖掘需求。遇拒绝礼貌退出不纠缠，遇犹豫适度引导给方案。\n"
    "不用编号列举markdown，不暴露AI身份，只说中文。不要输出思考过程。"
)

SYSTEM_PROMPT_FEWSHOT = """\
你是面壁智能的资深外呼坐席李明。你在这家公司工作了三年，负责给潜在客户打电话介绍AI智能客服产品。

【核心原则】
1. 每句话必须像真人打电话，口语化、短句、1-3句。绝不用编号、列举、markdown。
2. 每轮回复结尾必须有一个互动提问或明确的下一步动作。
3. 绝不说"作为AI"、"我是语言模型"之类的话。
4. 只说中文普通话。

【销售话术行为规范】
■ 开场白：先确认身份→自报家门→说明来意→征求同意。
■ 客户说在忙：立即尊重→提供替代方案→留触点。
■ 客户愿意听：一句话说价值+具体数字→抛互动问题。
■ 介绍产品：通俗+场景举例+量化对比。
■ 客户拒绝：先认同→不纠缠→埋种子。
■ 客户投诉：诚恳道歉→承诺不再打扰→快速结束。
■ 客户犹豫：不施压→发资料→索取联系方式。
■ 问价格：不直接报价→范围锚定→反问需求。
■ 约时间：二选一提问法。
■ 结束通话：确认后续动作→留联系方式→温暖告别。

产品知识库：
Q: 核心产品？A: MiniCPM系列大模型、面壁露卡Luca、XAgent、松果派。
Q: 智能客服？A: 意图识别+多轮对话+知识库检索。
Q: 汽车行业案例？A: 与吉利、长安、大众合作智能座舱。
Q: 数据安全？A: 支持端侧/私有化部署，数据不出域。"""

_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_INCOMPLETE_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_ANY_XML_TAG_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")


def clean(text):
    text = _TAG_RE.sub("", text)
    text = _INCOMPLETE_THINK_RE.sub("", text)
    text = _ANY_XML_TAG_RE.sub("", text)
    return text.strip()


def call_llm(client, model_id, messages):
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id, messages=messages,
        max_tokens=150, temperature=0.7, top_p=0.9,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "repetition_penalty": 1.15,
        },
    )
    latency = (time.perf_counter() - t0) * 1000
    text = clean(resp.choices[0].message.content or "")
    tokens = resp.usage.completion_tokens if resp.usage else 0
    return text, latency, tokens


def _semantic_contains(text, concept):
    concept_map = {
        "确认身份": ["请问是", "张先生", "王先生", "您好"],
        "自报家门": ["面壁智能", "李明", "我是", "我这边"],
        "说明来电原因": ["之前", "咨询", "关注", "了解", "留过"],
        "征求同意": ["方便吗", "可以吗", "耽误", "花您"],
        "先共情": ["理解", "明白", "懂您"],
        "合理解释来源": ["之前", "网上", "平台", "咨询"],
        "给退出选项": ["不打扰", "不需要的话"],
        "立即尊重": ["好的", "不好意思", "抱歉"],
        "提供替代方案": ["微信", "资料", "邮箱", "再打"],
        "留下触点": ["微信", "电话", "邮箱", "联系"],
        "简洁说重点": True,
        "给出具体数字吸引注意": ["60%", "一半", "三分之一", "五分之一", "%", "倍"],
        "抛出互动问题": ["？", "?"],
        "通俗易懂": True,
        "具体场景举例": ["比如", "例如", "像", "售后", "回访"],
        "突出核心价值": ["省", "降", "成本", "效率", "7x24", "不休息"],
        "量化对比": ["%", "倍", "个人", "分之"],
        "先认同": ["明白", "理解", "是的", "对的", "挺好"],
        "不纠缠但埋种子": ["后面", "需要", "随时", "如果"],
        "立即退出": ["好的", "不打扰", "再见"],
        "诚恳道歉": ["抱歉", "对不起", "不好意思"],
        "承诺不再打扰": ["不会再", "备注", "不再"],
        "不施压": ["没问题", "理解", "不急"],
        "索取联系方式": ["微信", "电话", "邮箱", "加个"],
        "认同顾虑": ["正常", "理解", "担心", "顾虑"],
        "提出低风险试用方案": ["试用", "试试", "小范围", "测试"],
        "不直接报价(先了解需求)": ["取决于", "根据", "看您", "多少"],
        "TCO对比计算": ["算", "工资", "五险", "成本", "一个月"],
        "二选一提问法": ["还是", "或者"],
        "确认具体时间": ["周", "上午", "下午", "点"],
        "确认后续动作": ["资料", "发给您", "方案"],
        "礼貌告别": ["再见", "顺利", "祝您"],
        "不还口": ["好的", "抱歉", "不好意思"],
        "主动确认连接": ["听到", "能听", "您好"],
        "重新自报家门": ["面壁智能", "李明"],
        "回应安全顾虑": ["安全", "加密", "保护"],
        "说明私有化部署": ["私有化", "本地", "内网", "服务器"],
        "强调加密": ["加密", "安全"],
        "用行业案例背书": ["金融", "客户", "在用"],
        "肯定能对接": ["可以", "能", "支持"],
        "举例说明": ["Salesforce", "企微", "钉钉"],
        "给时间预期": ["一周", "几天"],
        "反问了解客户": ["您", "哪家", "什么"],
        "专属对接人": ["专属", "对接"],
        "7x24支持": ["7x24", "24小时"],
        "响应时效承诺": ["2小时", "及时"],
        "消除顾虑": ["不用担心", "放心"],
        "认同行业特殊性": ["理解", "确实", "特殊"],
        "说明可定制训练": ["训练", "定制"],
        "举行业具体场景": ["预约", "挂号", "查询"],
        "区分AI和人工边界": ["医生", "人工", "专业"],
        "转换思路-小公司更需要": ["更需要", "更划算"],
        "切入痛点": ["忙不过来", "流失", "没人接"],
        "强调性价比": ["几百块", "便宜"],
        "具体费用对比": ["招人", "实习生"],
        "转换框架-减负不裁员": ["减负", "分担", "辅助"],
        "说正面案例": ["欢迎", "反而"],
        "强调KPI提升": ["KPI", "更好"],
        "打消合同顾虑": ["灵活", "随时"],
        "灵活付费方式": ["按月", "月付"],
        "低风险承诺": ["没有风险", "试用"],
        "推进下一步": ["模板", "发给您"],
        "肯定有折扣空间": ["优惠", "折扣"],
        "给出折扣范围": ["八折", "打折"],
        "长期合作激励": ["续约", "老客户"],
        "推进面谈": ["约", "面对面", "聊"],
        "不回避价格对比": ["对比", "价格"],
        "强调差异化价值": ["大模型", "准确率", "定制"],
        "了解竞品信息": ["哪家", "对比"],
        "提供对比": ["对比", "评估"],
        "立即响应": ["没问题", "马上"],
        "说明方案内容": ["产品介绍", "案例", "报价"],
        "预留下次沟通": ["随时", "联系"],
        "配合客户节奏": ["好的", "不耽误"],
        "简短告别": ["再见"],
        "留联系方式": ["联系我", "找我"],
        "轻松回应": ["哈哈", "不太懂"],
        "巧妙过渡到主题": ["投资", "客服", "回本"],
        "不生硬": True,
        "确认信号问题": ["信号", "听清"],
        "不直接放弃": True,
        "再次认同": ["理解", "谨慎"],
        "用实证替代口头保证": ["demo", "测试", "试用", "看效果"],
        "免费测试降低风险": ["免费", "不花钱"],
        "明确下一步": ["帮您", "安排"],
        "简洁回答技术问题": ["MiniCPM", "自研"],
        "不过度深入": True,
        "转接技术团队": ["技术", "同事", "详细"],
        "突出成本优势": ["成本低", "便宜", "省"],
        "不尴尬不回避": ["理解", "确实"],
        "用案例证明实力": ["吉利", "大众", "长安"],
        "用户数据背书": ["万", "用户"],
        "引导体验": ["体验", "试试"],
        "给出回本周期": ["三到六", "3到6", "半年"],
        "具体数字算账": ["四五万", "五六千", "两三万"],
        "对比人工成本": ["工资", "人工", "成本"],
        "量化节省金额": ["省", "万", "赚回"],
    }
    if concept in concept_map:
        val = concept_map[concept]
        if val is True:
            return True
        return any(w in text for w in val)
    return False


def evaluate(d, actual, latency, tokens):
    expected = d["expected_response"]
    key_elements = d.get("key_elements", [])
    anti_patterns = d.get("anti_patterns", [])

    actual_lower = actual.lower()
    ke_hits = [(e, e.lower() in actual_lower or _semantic_contains(actual, e)) for e in key_elements]
    ap_hits = [(a, a.lower() in actual_lower or _semantic_contains(actual, a)) for a in anti_patterns]
    has_question = "？" in actual or "?" in actual
    is_proactive = has_question or any(w in actual for w in ["您看", "要不", "您觉得", "怎么样", "可以吗", "方便吗"])

    return {
        "id": d["id"], "category": d["category"],
        "ke_hits": ke_hits, "ap_hits": ap_hits,
        "has_question": has_question, "is_proactive": is_proactive,
        "latency_ms": latency, "tokens": tokens,
        "length_ratio": len(actual) / max(len(expected), 1),
        "actual": actual[:100],
    }


def run_and_summarize(client, model_id, dialogues, system_prompt, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Model: {model_id}")
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"{'='*60}")

    results = []
    latencies = []
    for d in dialogues:
        messages = [{"role": "system", "content": system_prompt}]
        if d.get("context"):
            messages.append({"role": "user", "content": f"[对话场景：{d['context']}]\n{d['user_input']}"})
        else:
            messages.append({"role": "user", "content": d["user_input"]})

        actual, latency, tokens = call_llm(client, model_id, messages)
        r = evaluate(d, actual, latency, tokens)
        results.append(r)
        latencies.append(latency)

        ke_hit = sum(1 for _, f in r["ke_hits"] if f)
        ke_total = len(r["ke_hits"])
        marker = "✅" if ke_hit / max(ke_total, 1) >= 0.5 else "❌"
        print(f"  {marker} [{d['id']}] {latency:6.0f}ms | {actual[:60]}...")

    n = len(results)
    ke_total = sum(len(r["ke_hits"]) for r in results)
    ke_hit = sum(sum(1 for _, f in r["ke_hits"] if f) for r in results)
    ap_hit = sum(sum(1 for _, f in r["ap_hits"] if f) for r in results)
    proactive = sum(1 for r in results if r["is_proactive"])

    p50 = sorted(latencies)[n // 2]
    p90 = sorted(latencies)[int(n * 0.9)]
    p99 = sorted(latencies)[int(n * 0.99)]
    avg = sum(latencies) / n

    print(f"\n  --- {label} 汇总 ---")
    print(f"  关键要素命中率: {ke_hit}/{ke_total} ({ke_hit/max(ke_total,1)*100:.1f}%)")
    print(f"  反模式触发:     {ap_hit}")
    print(f"  主动引导率:     {proactive}/{n} ({proactive/n*100:.1f}%)")
    print(f"  延迟 avg/p50/p90/p99: {avg:.0f} / {p50:.0f} / {p90:.0f} / {p99:.0f} ms")

    return {
        "label": label, "results": results,
        "ke_hit": ke_hit, "ke_total": ke_total,
        "ap_hit": ap_hit, "proactive": proactive, "n": n,
        "latency_avg": avg, "latency_p50": p50,
        "latency_p90": p90, "latency_p99": p99,
    }


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dialogues = json.load(f)["dialogues"]
    print(f"基准对话: {len(dialogues)} 条")

    orig_client = OpenAI(
        base_url="http://localhost:8100/v1", api_key="dummy",
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=10.0),
    )
    sft_client = OpenAI(
        base_url="http://localhost:8199/v1", api_key="dummy",
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=10.0),
    )

    # Warmup
    print("[Warmup]")
    call_llm(orig_client, "MiniCPM4.1-8B-GPTQ", [{"role": "user", "content": "你好"}])
    call_llm(sft_client, "MiniCPM4.1-8B-SFT-GPTQ", [{"role": "user", "content": "你好"}])
    print("  Done.\n")

    s1 = run_and_summarize(
        orig_client, "MiniCPM4.1-8B-GPTQ", dialogues,
        SYSTEM_PROMPT_FEWSHOT, "原始 GPTQ + fewshot prompt (baseline)",
    )
    s2 = run_and_summarize(
        sft_client, "MiniCPM4.1-8B-SFT-GPTQ", dialogues,
        SYSTEM_PROMPT_SFT, "SFT-GPTQ + 简单 prompt",
    )

    print(f"\n\n{'#'*60}")
    print("  最终对比")
    print(f"{'#'*60}")
    print(f"{'指标':<20} {'原始GPTQ+fewshot':>20} {'SFT-GPTQ+simple':>20} {'变化':>12}")
    print("-" * 72)

    gke = s1["ke_hit"] / max(s1["ke_total"], 1) * 100
    ske = s2["ke_hit"] / max(s2["ke_total"], 1) * 100
    print(f"{'关键要素命中率':<16} {gke:>18.1f}% {ske:>18.1f}% {ske-gke:>+10.1f}pp")
    print(f"{'反模式触发':<16} {s1['ap_hit']:>19} {s2['ap_hit']:>19}")
    gp = s1["proactive"] / s1["n"] * 100
    sp = s2["proactive"] / s2["n"] * 100
    print(f"{'主动引导率':<16} {gp:>18.1f}% {sp:>18.1f}% {sp-gp:>+10.1f}pp")
    print(f"{'延迟 avg':<16} {s1['latency_avg']:>17.0f}ms {s2['latency_avg']:>17.0f}ms {s2['latency_avg']-s1['latency_avg']:>+10.0f}ms")
    print(f"{'延迟 p50':<16} {s1['latency_p50']:>17.0f}ms {s2['latency_p50']:>17.0f}ms {s2['latency_p50']-s1['latency_p50']:>+10.0f}ms")
    print(f"{'延迟 p90':<16} {s1['latency_p90']:>17.0f}ms {s2['latency_p90']:>17.0f}ms {s2['latency_p90']-s1['latency_p90']:>+10.0f}ms")
    print(f"{'prompt长度':<16} {len(SYSTEM_PROMPT_FEWSHOT):>18}ch {len(SYSTEM_PROMPT_SFT):>18}ch")

    under_400 = sum(1 for r in s2["results"] if r["latency_ms"] < 400)
    under_800 = sum(1 for r in s2["results"] if r["latency_ms"] < 800)
    print(f"\nSFT-GPTQ 延迟分布: <400ms={under_400}/{s2['n']}  <800ms={under_800}/{s2['n']}")

    out_path = os.path.join(os.path.dirname(__file__), "data", "final_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"baseline": s1, "sft_gptq": s2}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
