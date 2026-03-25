"""SFT 前后对比偏离度测试

对比两个模型:
  - 原始 GPTQ 模型 (port 8100) + Layer 1 fewshot + RAG
  - SFT 微调模型 (port 8199) + 简单 system prompt

使用相同的 44 条基准对话，输出对比报告。
"""
import os
import sys
import json
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
2. 每轮回复结尾必须有一个互动提问或明确的下一步动作（加微信、发资料、约时间）。
3. 绝不说"作为AI"、"我是语言模型"之类的话。
4. 只说中文普通话。

【销售话术行为规范 — 严格遵守】

■ 开场白：先确认身份 → 自报家门 → 一句话说明来意 → 征求同意。
  示范：客户说"喂？" → "您好，请问是张先生吗？我这边是面壁智能的李明，之前您在我们官网留过咨询信息，今天打给您简单聊几句，方便吗？"

■ 客户说"在忙/没时间"：立即尊重 → 提供替代方案(加微信/发资料) → 留触点。
■ 客户愿意听：一句话说价值+具体数字 → 抛出互动问题。
■ 介绍产品：通俗+场景举例+量化对比，不说技术黑话。
■ 客户说"不需要"：先认同 → 转换框架(不是替代而是辅助) → 不纠缠但埋种子。
■ 客户说"不感兴趣/别打了"：立即停止 → 留品牌记忆 → 礼貌告别。
■ 客户说"骚扰电话/要投诉"：诚恳道歉 → 承诺不再打扰 → 快速结束。
■ 客户犹豫"考虑考虑"：不施压 → 发资料降低决策成本 → 索取联系方式。
■ 问价格：不直接报价 → 给出范围锚定 → 反问了解需求。
■ 约时间：二选一提问法+说明形式和时长。
■ 结束通话：确认后续动作 → 留联系方式 → 温暖告别。

产品知识库：
Q: 核心产品？ A: MiniCPM系列大模型、面壁露卡Luca、XAgent、松果派。
Q: 智能客服？ A: 支持意图识别、多轮对话、知识库检索，大幅提升解决率。
Q: 与传统客服区别？ A: 大模型理解自然语言，更灵活准确。
Q: 汽车行业案例？ A: 与吉利、长安、大众合作智能座舱。
Q: 收费？ A: 按Token计费，有免费测试额度。
Q: 微调服务？ A: 提供行业定制微调。
Q: 数据安全？ A: 支持端侧/私有化部署，数据不出域。"""

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
        max_tokens=200, temperature=0.7, top_p=0.9,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "repetition_penalty": 1.15,
        },
    )
    latency = (time.perf_counter() - t0) * 1000
    text = clean(resp.choices[0].message.content or "")
    return text, latency


def _semantic_contains(text, concept):
    concept_map = {
        "确认身份": ["请问是", "张先生", "王先生", "您好"],
        "自报家门": ["面壁智能", "李明", "我是", "我这边"],
        "说明来电原因": ["之前", "咨询", "关注", "了解", "留过"],
        "征求同意": ["方便吗", "可以吗", "耽误", "花您"],
        "先共情": ["理解", "明白", "懂您", "您说的对"],
        "合理解释来源": ["之前", "网上", "平台", "咨询"],
        "给退出选项": ["不打扰", "不需要的话", "如果不方便"],
        "立即尊重": ["好的", "不好意思", "抱歉"],
        "提供替代方案": ["微信", "资料", "邮箱", "再打"],
        "留下触点": ["微信", "电话", "邮箱", "联系"],
        "简洁说重点": True,
        "给出具体数字吸引注意": ["60%", "一半", "三分之一", "五分之一", "%", "倍"],
        "抛出互动问题": ["？", "?"],
        "通俗易懂": True,
        "具体场景举例": ["比如", "例如", "像", "售后", "回访", "查订单"],
        "突出核心价值": ["省", "降", "成本", "效率", "7x24", "不休息"],
        "量化对比": ["%", "倍", "个人", "分之"],
        "消除'机器人'偏见": ["不是以前", "不是传统", "大模型"],
        "对比旧技术差异": ["以前", "传统", "关键词", "规则"],
        "强调可定制": ["训练", "定制", "根据", "专属"],
        "兜底方案": ["转人工", "人工", "搞不定"],
        "先认同": ["明白", "理解", "是的", "对的", "挺好"],
        "不纠缠但埋种子": ["后面", "需要", "随时", "如果"],
        "立即退出": ["好的", "不打扰", "再见"],
        "诚恳道歉": ["抱歉", "对不起", "不好意思", "很抱歉"],
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
        "承诺不再打扰": ["不会再", "不再", "备注"],
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
        "提供替代方案": ["微信", "改天", "再打"],
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


def evaluate(d, actual, mode, latency):
    expected = d["expected_response"]
    key_elements = d.get("key_elements", [])
    anti_patterns = d.get("anti_patterns", [])

    actual_lower = actual.lower()
    ke_hits = []
    for elem in key_elements:
        found = elem.lower() in actual_lower or _semantic_contains(actual, elem)
        ke_hits.append((elem, found))

    ap_hits = []
    for ap in anti_patterns:
        found = ap.lower() in actual_lower or _semantic_contains(actual, ap)
        ap_hits.append((ap, found))

    has_question = "？" in actual or "?" in actual
    has_markdown = bool(re.search(r"[#\*\-]\s|^\d+\.", actual, re.MULTILINE))
    has_ai_reveal = any(w in actual for w in [
        "作为AI", "我是语言模型", "作为一个", "我是人工智能", "作为大模型",
    ])
    is_proactive = has_question or any(
        w in actual for w in ["您看", "要不", "您觉得", "怎么样", "可以吗", "方便吗"]
    )

    return {
        "id": d["id"], "category": d["category"], "mode": mode,
        "user_input": d["user_input"],
        "expected": expected[:80], "actual": actual[:80],
        "key_elements": ke_hits, "anti_patterns": ap_hits,
        "has_question": has_question, "has_markdown": has_markdown,
        "has_ai_reveal": has_ai_reveal, "is_proactive": is_proactive,
        "latency_ms": latency,
        "length_ratio": len(actual) / max(len(expected), 1),
    }


def run_test(client, model_id, dialogues, mode, system_prompt):
    results = []
    for d in dialogues:
        messages = [{"role": "system", "content": system_prompt}]
        if d.get("context"):
            prefix = f"[对话场景：{d['context']}]\n"
        else:
            prefix = ""
        messages.append({"role": "user", "content": prefix + d["user_input"]})
        actual, latency = call_llm(client, model_id, messages)
        results.append(evaluate(d, actual, mode, latency))
        print(f"  [{d['id']}] {d['user_input'][:20]}... → {latency:.0f}ms | {actual[:60]}...")
    return results


def summarize(results, label):
    n = len(results)
    ke_total = sum(len(r["key_elements"]) for r in results)
    ke_hit = sum(sum(1 for _, f in r["key_elements"] if f) for r in results)
    ap_total = sum(len(r["anti_patterns"]) for r in results)
    ap_hit = sum(sum(1 for _, f in r["anti_patterns"] if f) for r in results)
    avg_lat = sum(r["latency_ms"] for r in results) / n
    avg_len = sum(r["length_ratio"] for r in results) / n
    proactive = sum(1 for r in results if r["is_proactive"])
    question = sum(1 for r in results if r["has_question"])
    markdown = sum(1 for r in results if r["has_markdown"])
    ai_reveal = sum(1 for r in results if r["has_ai_reveal"])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  关键要素命中率:  {ke_hit}/{ke_total} ({ke_hit/max(ke_total,1)*100:.1f}%)")
    print(f"  反模式触发率:    {ap_hit}/{ap_total} ({ap_hit/max(ap_total,1)*100:.1f}%)")
    print(f"  主动引导率:      {proactive}/{n} ({proactive/n*100:.1f}%)")
    print(f"  包含互动提问:    {question}/{n} ({question/n*100:.1f}%)")
    print(f"  Markdown泄漏:    {markdown}/{n}")
    print(f"  AI身份暴露:      {ai_reveal}/{n}")
    print(f"  平均延迟:        {avg_lat:.0f}ms")
    print(f"  平均长度比:      {avg_len:.2f}x")

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    print(f"\n  分类别命中:")
    for cat in sorted(by_cat):
        cr = by_cat[cat]
        ct = sum(len(r["key_elements"]) for r in cr)
        ch = sum(sum(1 for _, f in r["key_elements"] if f) for r in cr)
        cp = sum(1 for r in cr if r["is_proactive"])
        print(f"    [{cat}] 要素 {ch}/{ct}={ch/max(ct,1)*100:.0f}%  引导 {cp}/{len(cr)}")

    return {
        "label": label, "ke_hit": ke_hit, "ke_total": ke_total,
        "ap_hit": ap_hit, "ap_total": ap_total,
        "proactive": proactive, "question": question,
        "markdown": markdown, "ai_reveal": ai_reveal,
        "avg_latency": avg_lat, "avg_length_ratio": avg_len,
        "n": n,
    }


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dialogues = json.load(f)["dialogues"]
    print(f"加载 {len(dialogues)} 条基准对话\n")

    gptq_client = OpenAI(
        base_url="http://localhost:8100/v1", api_key="dummy",
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=10.0),
    )
    sft_client = OpenAI(
        base_url="http://localhost:8199/v1", api_key="dummy",
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=10.0),
    )

    gptq_model = gptq_client.models.list().data[0].id
    sft_model = sft_client.models.list().data[0].id
    print(f"GPTQ model: {gptq_model} (port 8100)")
    print(f"SFT model:  {sft_model} (port 8199)\n")

    print("=" * 60)
    print("Group A: 原始 GPTQ + fewshot + RAG")
    print("=" * 60)
    gptq_results = run_test(gptq_client, gptq_model, dialogues, "GPTQ+fewshot", SYSTEM_PROMPT_FEWSHOT)

    print("\n" + "=" * 60)
    print("Group B: SFT 微调模型 + 简单 prompt")
    print("=" * 60)
    sft_results = run_test(sft_client, sft_model, dialogues, "SFT", SYSTEM_PROMPT_SFT)

    gptq_summary = summarize(gptq_results, "原始 GPTQ + fewshot + RAG (prompt engineering)")
    sft_summary = summarize(sft_results, "SFT 微调模型 + 简单 prompt")

    print(f"\n\n{'#'*60}")
    print("   SFT 提升对比")
    print(f"{'#'*60}")
    gke = gptq_summary["ke_hit"] / max(gptq_summary["ke_total"], 1) * 100
    ske = sft_summary["ke_hit"] / max(sft_summary["ke_total"], 1) * 100
    print(f"  关键要素命中率:  {gke:.1f}% → {ske:.1f}% ({'↑' if ske > gke else '↓'}{abs(ske-gke):.1f}pp)")
    gap = gptq_summary["ap_hit"] / max(gptq_summary["ap_total"], 1) * 100
    sap = sft_summary["ap_hit"] / max(sft_summary["ap_total"], 1) * 100
    print(f"  反模式触发率:    {gap:.1f}% → {sap:.1f}% ({'↓' if sap < gap else '↑'}{abs(gap-sap):.1f}pp)")
    gp = gptq_summary["proactive"] / gptq_summary["n"] * 100
    sp = sft_summary["proactive"] / sft_summary["n"] * 100
    print(f"  主动引导率:      {gp:.1f}% → {sp:.1f}% ({'↑' if sp > gp else '↓'}{abs(sp-gp):.1f}pp)")
    print(f"  Markdown泄漏:    {gptq_summary['markdown']} → {sft_summary['markdown']}")
    print(f"  AI身份暴露:      {gptq_summary['ai_reveal']} → {sft_summary['ai_reveal']}")
    print(f"  平均延迟:        {gptq_summary['avg_latency']:.0f}ms → {sft_summary['avg_latency']:.0f}ms")

    output = {"gptq": gptq_results, "sft": sft_results, "summary": {"gptq": gptq_summary, "sft": sft_summary}}
    out_path = os.path.join(os.path.dirname(__file__), "data", "sft_comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
