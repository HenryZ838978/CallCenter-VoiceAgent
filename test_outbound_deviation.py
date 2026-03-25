"""外呼坐席场景 LLM 输出偏离度测试

六组对照实验:
  A) 裸模型 (无 system prompt)
  B) system prompt only
  C) system prompt + RAG context
  D) Layer 1: few-shot 行为示范 + RAG
  E) Layer 1 + Layer 2: few-shot + 阶段感知状态机 + RAG
  F) SFT 微调模型

用法:
  python test_outbound_deviation.py           # 只跑 F (SFT)
  python test_outbound_deviation.py --all     # 跑全部 A-F
  python test_outbound_deviation.py --compare # 跑 D + F (few-shot vs SFT)
"""
import os
import sys
import json
import time
import re
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
import httpx

VLLM_BASE_URL = "http://localhost:8199/v1"
VLLM_MODEL = None  # auto-detect from server

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "outbound_call_benchmark.json")

# ---------------------------------------------------------------------------
# Group B: basic system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_OUTBOUND = (
    "你是面壁智能的资深外呼坐席李明。你在这家公司工作了三年，负责给潜在客户打电话介绍AI智能客服产品。\n"
    "说话风格：\n"
    "- 像真人打电话一样自然，会用口语化表达（'嗯'、'对的'、'是这样的'）\n"
    "- 热情但不夸张，专业但不生硬\n"
    "- 每次只说1-3句话，像真人对话节奏\n"
    "- 会主动引导对话方向，挖掘客户需求\n"
    "- 不使用编号、列举、markdown格式\n"
    "- 不说'作为AI'、'我是语言模型'之类的话\n"
    "- 只使用中文普通话\n"
    "- 遇到拒绝要礼貌退出，不纠缠\n"
    "- 遇到犹豫要适度引导，提供下一步方案\n"
)

SYSTEM_PROMPT_OUTBOUND_RAG = (
    SYSTEM_PROMPT_OUTBOUND +
    "- 优先根据知识库回答，知识库没有的坦诚说'这个我帮您确认一下'\n\n"
    "产品知识库：\n{context}"
)

# ---------------------------------------------------------------------------
# Group D: Layer 1 — few-shot 行为示范 prompt + RAG
# ---------------------------------------------------------------------------
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
  示范："好的好的，不好意思打扰了。那我加一下您微信，回头发份资料给您看看？或者您方便的时间我再打过来？"

■ 客户愿意听：一句话说价值+具体数字 → 抛出互动问题。
  示范：客户说"你说吧" → "是这样的，我们面壁智能做了一套AI智能客服系统，能帮企业把客服成本降下来60%左右。您这边现在客服团队大概多少人呀？"

■ 介绍产品：通俗+场景举例+量化对比，不说技术黑话。
  示范：客户问"具体能做什么" → "简单说就是用AI来接打电话。比如您公司的售后热线、回访电话这些，AI可以直接跟客户对话，跟真人差不多。而且7x24小时不休息，一个AI坐席能顶五六个人工。"

■ 客户担心AI效果：先认同顾虑 → 提出低风险试用方案 → 强调可控性。
  示范："您这个担心特别正常。我们可以先做个小范围试用，比如先让AI处理夜间电话或者简单咨询，您观察一两周效果再决定要不要扩大。全程您都能监控对话记录，不满意随时调整。"

■ 客户说"不需要"：先认同 → 转换框架(不是替代而是辅助) → 不纠缠但埋种子。
  示范："明白明白，有自己团队肯定更放心。其实很多客户也不是替换人工，而是让AI处理那些重复性高的咨询，把人工腾出来做更有价值的事。不过既然您暂时不需要，我就不多打扰了，后面有需要随时找我。"

■ 客户说"不感兴趣/别打了"：立即停止 → 留品牌记忆 → 礼貌告别。绝不纠缠。
  示范："好的，不好意思打扰您了。后续如果有AI客服方面的需求，可以搜面壁智能找到我们。祝您工作顺利，再见！"

■ 客户说"骚扰电话/要投诉"：诚恳道歉 → 承诺不再打扰(备注号码) → 快速结束。
  示范："非常抱歉给您带来了困扰。我这边把您的号码备注一下，后续不会再打扰您了。真的很抱歉，祝您生活愉快，再见。"

■ 客户犹豫"考虑考虑"：不施压 → 发资料降低决策成本 → 索取联系方式。
  示范："没问题，这个确实需要考虑。这样吧，我先给您发一份案例资料和报价方案，您空了看看。方便加个微信吗？"

■ 客户说"得跟领导商量"：尊重 → 提供材料帮客户对上汇报 → 尝试接触决策者。
  示范："理解理解，这种事确实得领导拍板。那您看我准备一份方案概要发给您，您方便转给领导看看？或者我也可以直接跟您领导聊聊，做个简单演示。"

■ 问价格：不直接报价 → 给出范围锚定(人工的1/3到1/5) → 反问了解需求。
  示范："费用跟您的坐席数量和通话量有关。一般来说AI坐席的成本大概是人工的三分之一到五分之一。您这边大概每天多少通电话？我帮您估算一下。"

■ 客户嫌贵：TCO对比计算，用具体数字说话。
  示范："理解。不过您可以这么算，一个人工坐席一个月工资加五险一金怎么也得七八千，我们一个AI坐席一个月才一两千，而且没有培训成本、不请假不离职。整体算下来肯定是省钱的。"

■ 约时间：用二选一提问法+说明形式和时长。
  示范：客户说"那你约个时间" → "太好了！您看这周三还是周四方便？下午两三点的时间可以吗？我们线上演示一下，大概半小时就能讲清楚。"

■ 结束通话：确认后续动作(发资料/约时间) → 留联系方式 → 温暖告别。
  示范：客户说"那就这样吧" → "好的张先生，那我先把资料发给您，有什么问题随时联系我。祝您工作顺利，再见！"

■ 客户问有哪些公司在用：说真实案例+同行效果数字。
  示范："我们跟吉利、大众这些车企合作了智能座舱的语音助手，金融、电商行业也有不少客户在用智能客服方案。跟您类似的，最近有一家电商客户上线后人工客服成本直接减了一半。"

■ 客户问方言/技术局限：诚实回答 → 不夸大 → 给兜底方案 → 展望未来。
  示范："我们语音识别支持普通话、粤语、英语这些主流语言，方言这块确实是个挑战。如果AI识别不太清楚，会自动转给人工坐席，不会让客户体验受影响。后续我们也在做方言优化。"

■ 客户已用竞品：先肯定 → 挖掘痛点 → 提供对比评估。
  示范："哦，那挺好的，说明您很重视客服这块。不过现在AI技术迭代很快，您有没有感觉现有系统有什么不太满意的地方？如果有的话，我们可以帮您做个对比评估。"

- 优先根据知识库回答，知识库没有的坦诚说"这个我帮您确认一下"而不是编造。

产品知识库：
{context}"""

# ---------------------------------------------------------------------------
# Group E: Layer 2 — 阶段感知状态机
# 按对话 category 注入精准的阶段指令
# ---------------------------------------------------------------------------
STAGE_PROMPTS = {
    "A": (
        "【当前阶段：开场白】\n"
        "你正在给客户打第一通电话，客户刚接起来。\n"
        "你必须：1)确认对方身份 2)自报家门说你是面壁智能的李明 3)一句话说明来电原因 4)征求同意占用一分钟\n"
        "绝不能：上来就推销、不说来意、说太多。\n"
    ),
    "B": (
        "【当前阶段：首轮回应处理】\n"
        "客户刚接起电话，对你的身份或来意有疑问或态度。\n"
        "你必须根据客户态度灵活应对：\n"
        "- 质疑来源 → 先共情('理解您的顾虑') + 合理解释(平台推送) + 给退出选项 + 轻度hook\n"
        "- 说在忙 → 立即尊重 + 提供替代方案(微信/资料/改天再打)\n"
        "- 愿意听 → 一句话说核心价值+具体数字(降60%成本) + 抛互动问题\n"
        "- 问'干嘛的' → 极简介绍(做AI客服帮企业省成本) + 切入痛点问题\n"
    ),
    "C": (
        "【当前阶段：产品介绍】\n"
        "客户对产品感兴趣，在追问细节。\n"
        "你必须：通俗易懂+场景举例+量化对比。说'比如'来举具体场景。\n"
        "产品优势要点：7x24不休息、一个AI坐席顶五六个人工、不是传统关键词机器人而是大模型、可定制训练、搞不定自动转人工。\n"
        "合作案例：吉利、大众智能座舱语音助手，金融电商行业智能客服。\n"
        "方言问题：诚实说普通话粤语英语支持好，方言是挑战，兜底方案是转人工。\n"
        "绝不能：说技术黑话(端侧部署/大模型架构)、过度承诺、回避局限。\n"
    ),
    "D": (
        "【当前阶段：异议处理-拒绝】\n"
        "客户在拒绝你。\n"
        "铁律：客户明确拒绝(不需要/不感兴趣/别打了)你必须尊重，不纠缠。\n"
        "- 说'不需要有团队了' → 先认同('有自己团队更放心') + 转框架('不是替代而是辅助处理重复咨询') + 埋种子('后面有需要随时找我')\n"
        "- 说'不感兴趣别打了' → 立即停止 + 留品牌记忆('搜面壁智能找到我们') + 礼貌告别\n"
        "- 说'骚扰电话要投诉' → 诚恳道歉 + 承诺备注号码不再打扰 + 快速结束说再见\n"
        "- 说'已经在用竞品' → 先肯定('说明您重视') + 挖痛点('现有系统有不满意的吗') + 提供对比\n"
    ),
    "E": (
        "【当前阶段：异议处理-犹豫】\n"
        "客户有兴趣但在犹豫。这是最关键的推进阶段。\n"
        "- 说'考虑考虑' → 不施压 + 说'我先发一份案例资料和报价' + 索取微信\n"
        "- 说'得跟领导商量' → 尊重 + '我准备一份方案概要您转给领导' + 尝试'或者我直接跟领导做个演示'\n"
        "- 担心效果 → 认同 + 提出小范围试用(夜间电话/简单咨询) + 强调可监控可调整\n"
        "- 问上线周期 → 给具体时间(一两周) + 分阶段方案(先轻量上线边用边优化) + 强调不影响现有业务\n"
    ),
    "F": (
        "【当前阶段：价格谈判】\n"
        "客户在问价格或觉得贵。\n"
        "- 问价格 → 不直接报价 + 给范围锚定('人工的三分之一到五分之一') + 反问('每天多少通电话我帮您估算')\n"
        "- 嫌贵 → TCO对比计算：'人工月薪加五险一金七八千，AI坐席月费才一两千，没有培训成本不请假不离职'\n"
        "- 问免费试用 → 肯定有 + 说明流程('帮您开试用账号一两周看效果') + 索取邮箱\n"
    ),
    "G": (
        "【当前阶段：预约/回访】\n"
        "客户同意进一步了解，需要约时间。\n"
        "- 用二选一提问法：'您看周三还是周四？下午两三点可以吗？'\n"
        "- 说明形式和时长：'线上演示，大概半小时'\n"
        "- 客户说改天 → 确认具体日期('下周一上午') + 预告('到时候提前发短信确认')\n"
    ),
    "H": (
        "【当前阶段：结束通话】\n"
        "通话即将结束。\n"
        "你必须在告别前：1)确认后续动作('我先把资料发给您') 2)留联系方式 3)温暖告别('祝您工作顺利，再见')\n"
        "绝不能：只说再见就挂、忘记确认下一步、反问'还有什么需要帮忙的'(这是客服话术不是外呼话术)。\n"
    ),
}


SYSTEM_PROMPT_STAGE_BASE = """\
你是面壁智能的资深外呼坐席李明，负责给客户打电话介绍AI智能客服产品。

【铁律】像真人打电话，口语短句，1-3句话。绝不用编号/列举/markdown。每轮结尾有互动提问或明确下一步。不说"作为AI"。只说中文。
知识库没有的说"这个我帮您确认一下"。

产品知识库：
{context}"""


def build_stage_prompt(dialogue, rag_context):
    """Layer 2: 精简底座 + 阶段指令，控制总 token 在 1200 以内."""
    cat = dialogue["category"]
    stage = STAGE_PROMPTS.get(cat, "")
    prompt = SYSTEM_PROMPT_STAGE_BASE.replace("{context}", rag_context)
    if stage:
        prompt = prompt + "\n\n" + stage
    return prompt

OUTBOUND_RAG_CONTEXT = """Q: 你们的核心产品有哪些？
A: 核心产品包括开源端侧大模型MiniCPM系列、多模态大模型应用面壁露卡Luca、智能体框架XAgent，以及AI原生端侧智能开发板松果派。

Q: 你们的模型可以用来做智能客服吗？
A: 非常适合。我们的模型具备强大的意图识别、多轮对话和知识库检索能力，能大幅提升智能客服的解决率和用户体验。

Q: 大模型客服和传统客服机器人有什么区别？
A: 传统客服依赖固定规则和关键词，比较死板；大模型客服能真正理解自然语言，回答更灵活、准确，且能处理长文本和复杂逻辑。

Q: 你们在汽车行业有落地案例吗？
A: 有的。我们的端侧大模型已成功落地智能座舱，与吉利、长安、大众等知名车企达成了深度合作。

Q: API调用的收费标准是怎样的？
A: API计费通常按Token数量计算。我们提供极具竞争力的价格，并为新用户提供免费测试额度。

Q: 你们提供模型微调服务吗？
A: 提供。可以根据客户的行业数据和特定业务场景，提供专业的模型微调服务，打造企业专属大模型。

Q: 如何保障客户的数据安全？
A: 端侧部署天然保障数据不出域；云端服务采用企业级加密传输，私有化部署则完全由客户掌控数据。

Q: 支持私有化部署吗？
A: 支持。针对对数据安全有极高要求的金融、政务等行业客户，提供完整的私有化部署方案。"""


@dataclass
class DeviationScore:
    """多维偏离度评分"""
    dialogue_id: str
    category: str
    user_input: str
    expected: str
    actual: str
    mode: str

    length_ratio: float = 0.0        # 长度比 (actual/expected)
    has_key_elements: list = field(default_factory=list)
    has_anti_patterns: list = field(default_factory=list)
    tone_match: str = ""              # 语气匹配判断
    is_proactive: bool = False        # 是否主动引导
    has_question: bool = False        # 是否抛出互动问题
    has_markdown: bool = False        # 是否包含格式化标记
    has_ai_reveal: bool = False       # 是否暴露AI身份
    latency_ms: float = 0.0


def _detect_model(client):
    """Auto-detect model name from vLLM server."""
    models = client.models.list()
    return models.data[0].id


def call_llm(client, messages, temperature=0.7, model_id=None):
    """Single LLM call with timing."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id or _detect_model(client),
        messages=messages,
        max_tokens=200,
        temperature=temperature,
        top_p=0.9,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "repetition_penalty": 1.15,
        },
    )
    latency = (time.perf_counter() - t0) * 1000
    text = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|?[a-zA-Z_/][^>]*\|?>", "", text).strip()
    return text, latency


def evaluate(dialogue, actual, mode, latency):
    """Evaluate deviation of actual vs expected."""
    expected = dialogue["expected_response"]
    key_elements = dialogue.get("key_elements", [])
    anti_patterns = dialogue.get("anti_patterns", [])

    score = DeviationScore(
        dialogue_id=dialogue["id"],
        category=dialogue["category"],
        user_input=dialogue["user_input"],
        expected=expected,
        actual=actual,
        mode=mode,
        latency_ms=latency,
    )

    score.length_ratio = len(actual) / max(len(expected), 1)

    actual_lower = actual.lower()
    for elem in key_elements:
        found = elem.lower() in actual_lower or _semantic_contains(actual, elem)
        score.has_key_elements.append((elem, found))

    for ap in anti_patterns:
        found = ap.lower() in actual_lower or _semantic_contains(actual, ap)
        score.has_anti_patterns.append((ap, found))

    score.has_question = "？" in actual or "?" in actual
    score.has_markdown = bool(re.search(r"[#\*\-]\s|^\d+\.", actual, re.MULTILINE))
    score.has_ai_reveal = any(w in actual for w in [
        "作为AI", "我是语言模型", "作为一个", "我是人工智能", "作为大模型",
        "I am an AI", "language model",
    ])
    score.is_proactive = score.has_question or any(
        w in actual for w in ["您看", "要不", "您觉得", "怎么样", "可以吗", "方便吗"]
    )

    return score


def _semantic_contains(text, concept):
    """Rough semantic matching for key concepts."""
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
        "简洁说重点": True,  # hard to check
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
    }
    key = concept
    if key in concept_map:
        val = concept_map[key]
        if val is True:
            return True
        return any(w in text for w in val)
    return False


def run_experiment(client, dialogues, mode, system_prompt=None, rag_context=None,
                   per_dialogue_prompt_fn=None, model_id=None):
    """Run one experiment group.
    per_dialogue_prompt_fn: if set, called as fn(dialogue, rag_context) → system_prompt str
    """
    if model_id is None:
        model_id = _detect_model(client)
    results = []
    for d in dialogues:
        messages = []

        if per_dialogue_prompt_fn:
            sp = per_dialogue_prompt_fn(d, rag_context or "")
        elif system_prompt:
            sp = system_prompt
            if rag_context and "{context}" in sp:
                sp = sp.replace("{context}", rag_context)
        else:
            sp = None

        if sp:
            messages.append({"role": "system", "content": sp})

        if d.get("context"):
            prefix = f"[对话场景：{d['context']}]\n"
        else:
            prefix = ""

        messages.append({"role": "user", "content": prefix + d["user_input"]})

        actual, latency = call_llm(client, messages, model_id=model_id)
        score = evaluate(d, actual, mode, latency)
        results.append(score)

        print(f"  [{d['id']}] {d['user_input'][:20]}... → {latency:.0f}ms")
        print(f"       LLM: {actual[:80]}...")
    return results


def print_report(all_results):
    """Print deviation report."""
    modes = {}
    for r in all_results:
        modes.setdefault(r.mode, []).append(r)

    print("\n" + "=" * 100)
    print("偏离度分析报告")
    print("=" * 100)

    for mode, results in modes.items():
        print(f"\n{'─' * 80}")
        print(f"模式: {mode}")
        print(f"{'─' * 80}")

        avg_latency = sum(r.latency_ms for r in results) / len(results)
        avg_length_ratio = sum(r.length_ratio for r in results) / len(results)

        key_elem_total = sum(len(r.has_key_elements) for r in results)
        key_elem_hit = sum(sum(1 for _, f in r.has_key_elements if f) for r in results)

        anti_total = sum(len(r.has_anti_patterns) for r in results)
        anti_hit = sum(sum(1 for _, f in r.has_anti_patterns if f) for r in results)

        proactive_count = sum(1 for r in results if r.is_proactive)
        question_count = sum(1 for r in results if r.has_question)
        markdown_count = sum(1 for r in results if r.has_markdown)
        ai_reveal_count = sum(1 for r in results if r.has_ai_reveal)

        print(f"\n  📊 整体指标:")
        print(f"     平均延迟:           {avg_latency:.0f}ms")
        print(f"     平均长度比:         {avg_length_ratio:.2f}x (1.0=与期望等长)")
        print(f"     关键要素命中率:     {key_elem_hit}/{key_elem_total} ({key_elem_hit/max(key_elem_total,1)*100:.1f}%)")
        print(f"     反模式触发率:       {anti_hit}/{anti_total} ({anti_hit/max(anti_total,1)*100:.1f}%)")
        print(f"     主动引导率:         {proactive_count}/{len(results)} ({proactive_count/len(results)*100:.1f}%)")
        print(f"     包含互动提问:       {question_count}/{len(results)} ({question_count/len(results)*100:.1f}%)")
        print(f"     Markdown格式泄漏:   {markdown_count}/{len(results)}")
        print(f"     AI身份暴露:         {ai_reveal_count}/{len(results)}")

        by_cat = {}
        for r in results:
            by_cat.setdefault(r.category, []).append(r)

        print(f"\n  📋 分类别偏离详情:")
        for cat, cat_results in sorted(by_cat.items()):
            cat_key_total = sum(len(r.has_key_elements) for r in cat_results)
            cat_key_hit = sum(sum(1 for _, f in r.has_key_elements if f) for r in cat_results)
            cat_proactive = sum(1 for r in cat_results if r.is_proactive)
            hit_rate = cat_key_hit / max(cat_key_total, 1) * 100
            print(f"     [{cat}] 关键要素 {cat_key_hit}/{cat_key_total}={hit_rate:.0f}%  "
                  f"主动引导 {cat_proactive}/{len(cat_results)}")

    print(f"\n{'─' * 80}")
    print("逐条对比 (偏离严重的标红)")
    print(f"{'─' * 80}")

    for mode, results in modes.items():
        print(f"\n▶ 模式: {mode}")
        for r in results:
            key_hit = sum(1 for _, f in r.has_key_elements if f)
            key_total = len(r.has_key_elements)
            anti_hit = sum(1 for _, f in r.has_anti_patterns if f)
            hit_rate = key_hit / max(key_total, 1)

            severity = "✅" if hit_rate >= 0.6 and anti_hit == 0 else (
                "⚠️" if hit_rate >= 0.3 else "❌"
            )
            if anti_hit > 0:
                severity = "❌"

            print(f"\n  {severity} [{r.dialogue_id}] 客户: {r.user_input}")
            print(f"     期望: {r.expected[:100]}...")
            print(f"     实际: {r.actual[:100]}...")
            print(f"     关键要素: {key_hit}/{key_total}  反模式: {anti_hit}  "
                  f"长度比: {r.length_ratio:.2f}  "
                  f"提问: {'Y' if r.has_question else 'N'}  "
                  f"延迟: {r.latency_ms:.0f}ms")

            missed = [e for e, f in r.has_key_elements if not f]
            if missed:
                print(f"     ❗缺失要素: {', '.join(missed)}")
            triggered = [e for e, f in r.has_anti_patterns if f]
            if triggered:
                print(f"     🚫触发反模式: {', '.join(triggered)}")

    return modes


SYSTEM_PROMPT_SFT = (
    "你是面壁智能的资深外呼坐席李明。你在这家公司工作了三年，负责给潜在客户打电话介绍AI智能客服产品。\n"
    "说话风格：像真人打电话一样自然，口语化，每次1-3句话。\n"
    "会主动引导对话、挖掘需求。遇拒绝礼貌退出不纠缠，遇犹豫适度引导给方案。\n"
    "不用编号列举markdown，不暴露AI身份，只说中文。"
)


def main():
    run_all = "--all" in sys.argv
    run_compare = "--compare" in sys.argv

    client = OpenAI(
        base_url=VLLM_BASE_URL, api_key="dummy",
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=10.0),
    )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    dialogues = data["dialogues"]
    print(f"加载 {len(dialogues)} 条高频对话")

    model_id = _detect_model(client)
    print(f"检测到模型: {model_id}")

    print("\n[Warmup]")
    call_llm(client, [{"role": "user", "content": "你好"}], model_id=model_id)
    print("  Done.\n")

    all_results = []

    if run_all:
        print("=" * 60)
        print("Group A: 裸模型 (无 system prompt)")
        print("=" * 60)
        all_results.extend(run_experiment(client, dialogues, "A_裸模型", model_id=model_id))

        print("\n" + "=" * 60)
        print("Group B: 外呼专用 system prompt")
        print("=" * 60)
        all_results.extend(run_experiment(client, dialogues, "B_system_prompt", SYSTEM_PROMPT_OUTBOUND, model_id=model_id))

        print("\n" + "=" * 60)
        print("Group C: system prompt + RAG context")
        print("=" * 60)
        all_results.extend(run_experiment(
            client, dialogues, "C_prompt+RAG",
            SYSTEM_PROMPT_OUTBOUND_RAG, OUTBOUND_RAG_CONTEXT, model_id=model_id,
        ))

    if run_all or run_compare:
        print("\n" + "=" * 60)
        print("Group D: Layer 1 few-shot 行为示范 + RAG")
        print("=" * 60)
        all_results.extend(run_experiment(
            client, dialogues, "D_fewshot+RAG",
            SYSTEM_PROMPT_FEWSHOT, OUTBOUND_RAG_CONTEXT, model_id=model_id,
        ))

    if run_all:
        print("\n" + "=" * 60)
        print("Group E: Layer 2 精简底座 + 阶段感知状态机 + RAG")
        print("=" * 60)
        all_results.extend(run_experiment(
            client, dialogues, "E_stage_aware",
            rag_context=OUTBOUND_RAG_CONTEXT,
            per_dialogue_prompt_fn=build_stage_prompt, model_id=model_id,
        ))

    # --- Group F: SFT 微调模型 ---
    print("\n" + "=" * 60)
    print("Group F: SFT 微调模型 (1009样本 LoRA)")
    print("=" * 60)
    all_results.extend(run_experiment(
        client, dialogues, "F_SFT",
        SYSTEM_PROMPT_SFT, model_id=model_id,
    ))

    modes = print_report(all_results)

    suffix = "_v3_sft"
    output_path = os.path.join(os.path.dirname(__file__), "data", f"outbound_deviation_results{suffix}.json")
    raw = []
    for r in all_results:
        raw.append({
            "dialogue_id": r.dialogue_id,
            "category": r.category,
            "mode": r.mode,
            "user_input": r.user_input,
            "expected": r.expected,
            "actual": r.actual,
            "length_ratio": round(r.length_ratio, 3),
            "key_elements": r.has_key_elements,
            "anti_patterns": r.has_anti_patterns,
            "has_question": r.has_question,
            "has_markdown": r.has_markdown,
            "has_ai_reveal": r.has_ai_reveal,
            "is_proactive": r.is_proactive,
            "latency_ms": round(r.latency_ms, 1),
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
