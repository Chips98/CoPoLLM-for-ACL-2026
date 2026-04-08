# -*- coding: utf-8 -*-
"""
多智能体系统实现 (Specialist CBT Version)
包含四个核心智能体：医生、患者、分析师、评估者
实现基于 CBT 临床技术的强化学习训练闭环
[最新更新]：采用 10 种专病专治的 CBT 技术，配合 1:1 黄金策略矩阵
[消融实验支持]：支持通过YAML配置控制各种奖励组件的启用/禁用
"""
import asyncio
import json
import re
from typing import Dict, List, Any, Tuple
import numpy as np

# --- 配置加载器支持 ---
try:
    from config_loader import get_config
    # 加载配置文件
    cp_config = get_config()
    reward_config = cp_config.get_reward_config()
    api_config = cp_config.llm_api_config

    print("✅ multi_agents.py 成功加载YAML配置文件")
    print(f"🔬 奖励消融设置:")
    print(f"  - 安全奖励: {'❌ 禁用' if cp_config.is_safety_reward_disabled else '✅ 启用'}")
    print(f"  - 策略匹配奖励: {'❌ 禁用' if cp_config.is_strategy_match_reward_disabled else '✅ 启用'}")
    print(f"  - 症状改善奖励: {'❌ 禁用' if cp_config.is_symptom_improvement_reward_disabled else '✅ 启用'}")

except ImportError as e:
    print(f"⚠️  无法加载配置文件，使用默认参数: {e}")
    # 使用原有的默认参数（向后兼容）
    cp_config = None
    reward_config = None
    api_config = None

# --- 1. 全新定义的动作空间：CBT 手术刀式技术 ---

ACTION_SPACE_MAP = {
    # --- 基础支持层 ---
    0: {
        "strategy_name": "情感验证与共情 (Validation)",
        "desc": (
            "【核心定义】在不带有评判的前提下，识别、命名并接纳用户当前的情绪痛苦。"
            "【执行指令】不要急于解决问题或提供建议。使用温暖、接纳的语气。告诉患者他们的感受是合理的、可以被理解的。"
            "【关键话术】'听起来你现在真的很难过。' '面对这样的情况，感到愤怒是很正常的。' '我能感受到这件事对你的打击有多大。'"
        )
    },
    
    # --- 针对：非黑即白 ---
    1: {
        "strategy_name": "寻找灰色地带 (Finding the Gray)",
        "desc": (
            "【核心定义】引导用户打破‘全有或全无’（完美或失败）的二元对立，在连续谱上寻找中间状态。"
            "【执行指令】指出这种极端思维。询问用户：'如果100分是完美，0分是绝对灾难，你觉得目前的情况实际处于多少分？' 引导用户看到虽不完美但仍有价值的部分。"
            "【关键话术】'难道只有100分才算成功吗？80分是否也包含了一些努力？' '这真的是彻底的失败吗？还是说只是部分不如意？'"
        )
    },
    
    # --- 针对：过度概括 / 灾难化 ---
    2: {
        "strategy_name": "寻找例外证据 (Examine the Evidence)",
        "desc": (
            "【核心定义】引导用户像侦探一样，寻找与当前消极结论相反的客观证据（例外情况）。"
            "【执行指令】当用户使用'总是'、'从不'、'全都不好'等词汇时，温和地挑战其结论。询问过去是否发生过积极的例外，或者当前情境中被忽略的正面细节。"
            "【关键话术】'你说你总是搞砸，但有没有哪一次是做得还不错的？' '让我们看看证据，支持这个想法的证据有哪些？不支持的有哪些？'"
        )
    },
    
    # --- 针对：读心术 / 先入为主 ---
    3: {
        "strategy_name": "事实核查 (Reality Testing)",
        "desc": (
            "【核心定义】区分‘想象（推测）’与‘事实’。鼓励用户寻找确凿证据，而非假设他人的意图。"
            "【执行指令】询问用户支持其猜想（如‘他讨厌我’）的客观证据是什么。提出替代解释（Alternative Explanations）。鼓励沟通核实。"
            "【关键话术】'你怎么确信他在想什么？有实质性的证据吗？' '除了他讨厌你，有没有可能他只是太忙了没看到消息？' '我们要不要去问问他，而不是自己猜？'"
        )
    },
    
    # --- 针对：灾难化 ---
    4: {
        "strategy_name": "去灾难化 (De-catastrophizing)",
        "desc": (
            "【核心定义】不否认风险，而是通过具体化‘最坏结果’并制定应对计划，来降低对未知的恐惧。"
            "【执行指令】不要只说'没事的'。请问：'如果最坏的情况真的发生了，具体会怎么样？你会立刻完蛋吗？' 然后引导：'如果是那样，我们能做些什么来应对？' 帮助患者找回掌控感。"
            "【关键话术】'让我们假设最坏的情况发生了，你会怎么处理？' '这件事在这一生中真的有毁灭性的影响吗？' '哪怕发生了，你也有办法活下去，对吗？'"
        )
    },
    
    # --- 针对：应该句式 ---
    5: {
        "strategy_name": "利弊分析 (Cost-Benefit Analysis)",
        "desc": (
            "【核心定义】引导用户评估死守某种僵化规则（‘我必须...’）的实用性，对比其带来的好处与心理代价。"
            "【执行指令】针对用户的‘应该’或‘必须’，询问：'坚持这个高标准给你带来了什么好处？又让你付出了什么代价（如焦虑、拖延）？' 引导用户建立更灵活的标准。"
            "【关键话术】'对自己要求这么严格，虽然让你很上进，但似乎也让你非常疲惫，值得吗？' '如果把标准稍微降低一点，会发生什么可怕的事吗？'"
        )
    },
    
    # --- 针对：个人化 ---
    6: {
        "strategy_name": "责任饼图 (Reattribution)",
        "desc": (
            "【核心定义】帮助用户列出导致结果的所有潜在因素，重新分配责任比例，减轻过度的内疚感。"
            "【执行指令】画一个虚拟的‘责任饼图’。引导用户列出除了自己以外的其他影响因素（如运气、他人、环境）。询问：'这件事真的是你一个人的错吗？还有谁/什么在其中起了作用？'"
            "【关键话术】'让我们画个饼图，这件事里有多少比例是你的责任，多少是环境或他人的责任？' '你是不是把不该你背的锅也背在身上了？'"
        )
    },
    
    # --- 针对：标签化 ---
    7: {
        "strategy_name": "区分行为与人 (Behavior vs Identity)",
        "desc": (
            "【核心定义】引导用户将‘特定的失败行为’与‘整体的人格评价’剥离开来，反对给自己贴标签。"
            "【执行指令】明确指出：做错一件事不等于你就是一个失败的人。引导用户用具体的行为描述代替抽象的负面标签。"
            "【关键话术】'你只是这次考试没考好（行为），这并不代表你就是个笨蛋（标签）。' '人是复杂的，一个标签能定义全部的你吗？'"
        )
    },
    
    # --- 针对：情感推理 ---
    8: {
        "strategy_name": "情感分离 (Feelings vs Facts)",
        "desc": (
            "【核心定义】明确区分‘主观感受’与‘客观事实’。引导用户像法官一样审视证据，而不是被情绪牵着走。"
            "【执行指令】告诉用户：感觉强烈不代表它是真的。引导用户跳出情绪，用理性的眼光看事实。询问：'如果你的朋友遇到这事，你会觉得他没救了吗？还是这只是你现在的感觉？'"
            "【关键话术】'你感觉自己很蠢，但这不代表事实就是这样。' '这是一种强烈的感觉，还是一个被证明的事实？' '让我们把情绪放一边，只看证据。'"
        )
    },
    
    # --- 安全底线 ---
    9: {
        "strategy_name": "危机干预 (Crisis Intervention)",
        "desc": (
            "【核心定义】检测到高危风险（自伤、自杀、伤人）时的紧急阻断策略。"
            "【执行指令】语气必须严肃、直接且关切。**立刻停止**对认知的讨论。直接询问自杀意念，表达对生命安全的担忧，提供紧急求助渠道（如热线、医院）。"
            "【关键话术】'我听到了你的痛苦，但我非常担心你的安全。你现在有伤害自己的念头吗？' '请务必先保证自己的安全，这是最重要的。请拨打...' "
        )
    }
}

# --- 2. 专病专治策略矩阵 (1:1 Mapping) ---
# 确保每个偏差都有唯一的 Gold 策略，彻底解决模式坍塌
STRATEGY_MATRIX = {
    "非黑即白": {
        "gold": [1],         # 唯一解：寻找灰色地带
        "silver": [8, 2],    # 替补：共情, 找证据
    },
    "过度概括": {
        "gold": [2],         # 唯一解：寻找例外证据
        "silver": [0, 1],    # 替补：共情, 找灰色
    },
    "灾难化": {
        "gold": [4],         # 唯一解：去灾难化（最坏结果+应对计划）
        "silver": [0, 2],    # 替补：共情, 找证据
    },
    "读心术": {
        "gold": [3],         # 唯一解：事实核查
        "silver": [0, 8],    # 替补：共情, 情感分离
    },
    "情感推理": {
        "gold": [8],         # 唯一解：情感分离
        "silver": [0, 3],    # 替补：共情, 事实核查
    },
    "应该句式": {
        "gold": [5],         # 唯一解：利弊分析（规则的代价）
        "silver": [0, 7],    # 替补：共情, 区分行为与人
    },
    "个人化": {
        "gold": [6],         # 唯一解：责任饼图
        "silver": [0, 3],    # 替补：共情, 事实核查
    },
    "标签化": {
        "gold": [7],         # 唯一解：区分行为与人
        "silver": [0, 5],    # 替补：共情, 利弊分析
    }
}


# 解析状态文本，返回所有匹配到的认知偏差类型列表
def parse_bias_list(state_text: str) -> List[str]:
    """
    解析状态文本，返回所有匹配到的认知偏差类型列表。
    训练数据标签已经标准化，直接提取即可。
    """
    found_biases = set()

    # 直接检查标准偏差类型（训练数据已标准化）
    for bias in STRATEGY_MATRIX.keys():
        if bias in state_text:
            found_biases.add(bias)

    return list(found_biases)

# 保留原函数作为兼容
def parse_bias_type(state_text: str) -> str:
    """兼容性函数：返回第一个匹配的偏差"""
    biases = parse_bias_list(state_text)
    return biases[0] if biases else "未知"

# [辅助] 解析偏差强度
def parse_intensity(state_text: str) -> str:
    if "严重" in state_text: return "严重"
    if "轻微" in state_text: return "轻微"
    return "中等" 

class MultiAgentSystem:
    """多智能体系统：协调医生、患者、分析师、评估者"""

    def __init__(self, llm_api_func):
        self.call_llm = llm_api_func

    async def doctor_agent(self, context: str, state_text: str, action_index: int, patient_current_utterance: str = None) -> str:
        """医生智能体：基于具体CBT技术生成回应"""
        action = ACTION_SPACE_MAP[action_index]

        if not patient_current_utterance and context:
            lines = context.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('患者:') or line.startswith('患者：'):
                    patient_current_utterance = line.split(':', 1)[1].strip()
                    break

        patient_utterance_section = f"\n{patient_current_utterance}" if patient_current_utterance else ""

        prompt = f"""[SYSTEM]
你是一位专业的CBT心理治疗师。你正在使用特定的认知干预技术来帮助患者。

[USER]
# 对话上下文
{context}

# 患者当前发言
{patient_utterance_section}

# 患者状态
{state_text}

# 你必须使用的具体策略 (Action {action_index})
- 策略名称: {action['strategy_name']}
- 策略定义: {action['desc']}

[指令]
请严格应用上述“技术定义”来生成回复。
1. 如果是“寻找灰色地带”，请引导患者打破二元对立。
2. 如果是“责任饼图”，请具体讨论责任分配。
3. 语气要专业、温暖且具有引导性。

no think
请生成医生的回应：

[ASSISTANT]"""

        messages = [{"role": "user", "content": prompt}]
        response = await self.call_llm(messages)
        response = self.extract_real_answer(response)
        return response.strip()

    def extract_real_answer(self, response_text: str) -> str:
        if "</think>" in response_text:
            parts = response_text.split("</think>")
            if len(parts) > 1:
                real_content = parts[-1].strip()
                real_content = real_content.lstrip('\n')
                return real_content
        return response_text

    async def patient_agent(self, context: str, doctor_response: str, patient_previous_utterance: str = None, patient_previous_state: str = None) -> str:
        """患者智能体：生成自然回复"""
        if not patient_previous_utterance and context:
            lines = context.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('患者:') or line.startswith('患者：'):
                    patient_previous_utterance = line.split(':', 1)[1].strip()
                    break

        previous_utterance_section = f"\n{patient_previous_utterance}" if patient_previous_utterance else ""
        previous_state_section = f"\n{patient_previous_state}" if patient_previous_state else ""

        prompt = f"""[SYSTEM]
你是一个患者模拟器。请基于上下文生成一句自然的回复。

[USER]
# 对话上下文
{context}

# 你的上一轮发言
{previous_utterance_section}
# 你的心理状态
{previous_state_section}

# 医生最新的回应
"{doctor_response}"

no think
请生成一句患者的回复（1-2句话，体现真实的情绪反应）：

[ASSISTANT]"""

        messages = [{"role": "user", "content": prompt}]
        response = await self.call_llm(messages)
        response = self.extract_real_answer(response)
        return response.strip()

    # --- 奖励机制核心 ---
    
    def _check_safety_mechanism(self, state_text: str, action_index: int, patient_reply: str) -> float:
        """安全熔断：针对 Action 9 (危机干预) - 支持消融实验"""
        # 消融实验检查：如果禁用安全奖励，则跳过安全检查
        if cp_config and cp_config.is_safety_reward_disabled:
            return None

        HIGH_RISK_KEYWORDS = ["高危", "自杀", "自伤", "危机", "生命安全", "想死", "结束生命", "伤害自己"]
        is_high_risk = any(keyword in state_text or keyword in patient_reply for keyword in HIGH_RISK_KEYWORDS)

        if is_high_risk:
            if action_index == 9:  # [修改] 危机干预现在是 Action 9
                # 使用配置中的安全奖励值，或默认值
                safety_reward = reward_config.get('base_reward_mismatch', -1.0) if reward_config else 4.0
                return max(safety_reward, 4.0)  # 确保安全奖励足够高
            else:
                return -1.0  # 漏报惩罚

        if not is_high_risk and action_index == 9:
            return -2.0  # 误报惩罚

        return None

    # [修改] 适配多标签的评估函数
    async def evaluator_agent(self, state_text: str, action_index: int, doctor_response: str,
                            patient_reply: str, patient_original_utterance: str = None) -> Tuple[float, Dict[str, Any]]:

        # 1. 安全熔断 (保持不变)
        safety_reward = self._check_safety_mechanism(state_text, action_index, patient_reply)
        if safety_reward is not None:
            # print(f"[DEBUG] 动作:{action_index} | 安全检查 | Reward:{safety_reward:.2f}")
            return safety_reward, {"source": "safety_mechanism", "details": "安全检查"}

        # 2. 多标签规则对齐 (Multi-label Alignment) - 支持策略匹配奖励消融
        detected_biases = parse_bias_list(state_text)
        current_intensity = parse_intensity(state_text)

        # 如果没解析出任何偏差，尝试回退到 Unknown 处理
        if not detected_biases:
            if action_index == 0: return 0.0, {"source": "unknown_fallback"}
            else: return -0.1, {"source": "unknown_penalty"}

        # 消融实验检查：如果禁用策略匹配奖励，跳过策略匹配逻辑
        if cp_config and cp_config.is_strategy_match_reward_disabled:
            # 给予基础奖励，跳过策略匹配
            base_reward = 0.0
            rule_reward = base_reward
            strategy_type = "baseline"
        else:
            # 正常的策略匹配逻辑
            # 构建 Gold 和 Silver 的并集
            union_gold = set()
            union_silver = set()

            for bias in detected_biases:
                config = STRATEGY_MATRIX.get(bias)
                if config:
                    union_gold.update(config['gold'])
                    union_silver.update(config['silver'])

            # 使用配置中的基础奖励值或默认值
            default_gold_reward = reward_config.get('base_reward_gold', 1.0) if reward_config else 1.8
            default_silver_reward = reward_config.get('base_reward_silver', 0.5) if reward_config else 0.2
            default_neutral_reward = reward_config.get('base_reward_neutral', 0.0) if reward_config else 0.0
            default_mismatch_reward = reward_config.get('base_reward_mismatch', -1.0) if reward_config else -0.5

            # 判定策略类型
            base_reward = default_mismatch_reward
            strategy_type = "mismatch"

            if action_index in union_gold:
                base_reward = default_gold_reward
                strategy_type = "gold"

            elif action_index in union_silver:
                base_reward = default_silver_reward
                strategy_type = "silver"
                # [补丁] 如果包含"应该句式"或"非黑即白"，且选了Action 0，降级
                if action_index == 0 and any(b in ["非黑即白", "应该句式"] for b in detected_biases):
                    base_reward = default_silver_reward * 0.5  # 降低到一半
                    strategy_type = "weak_silver"

            elif action_index == 0: # Action 0 作为最后的通用保底
                base_reward = default_neutral_reward
                strategy_type = "neutral"

            # 3. 强度修正 (使用配置中的参数或默认值)
            intensity_modifier = 0.0
            if current_intensity == "严重":
                if strategy_type == "gold":
                    intensity_modifier = reward_config.get('intensity_severe_gold_bonus', 1.2) if reward_config else 1.2
                elif strategy_type in ["silver", "weak_silver", "neutral"]:
                    intensity_modifier = reward_config.get('intensity_severe_non_gold_penalty', -0.5) if reward_config else -0.5
            elif current_intensity == "轻微":
                if strategy_type == "gold":
                    intensity_modifier = reward_config.get('intensity_mild_gold_penalty', -0.8) if reward_config else -0.8
                elif strategy_type == "silver":
                    intensity_modifier = reward_config.get('intensity_mild_silver_bonus', 0.6) if reward_config else 0.6

            rule_reward = base_reward + intensity_modifier

            # Mismatch 直接返回
            if strategy_type == "mismatch":
                bias_str = ",".join(detected_biases)
                # print(f"[DEBUG] 动作:{action_index} | 偏差:[{bias_str}] | 结果:Mismatch | Reward:{rule_reward:.2f}")
                return rule_reward, {"source": "rule_mismatch", "bias": bias_str, "reward": rule_reward}

        # 4. LLM 质量评估 (症状改善奖励) - 支持消融实验
        if cp_config and cp_config.is_symptom_improvement_reward_disabled:
            # 消融实验：禁用症状改善奖励
            quality_bonus = 0.0
            bonus_source = "disabled_symptom_improvement"
        else:
            # 正常的质量评估
            action_name = ACTION_SPACE_MAP[action_index]['strategy_name']
            action_desc = ACTION_SPACE_MAP[action_index]['desc']

            prompt = f"""[SYSTEM]
你是一位CBT督导。请评估医生回复是否合格地运用了指定技术。

[USER]
技术要求: {action_name} ({action_desc})
医生回复: "{doctor_response}"

【评分 (0-5分)】
- 技巧(skill): 医生是否真的使用了该技术的核心逻辑？(例如：选了'责任饼图'是否真的在讨论责任分配？)
- 连贯(coherence): 回复是否自然流畅？

【输出JSON】
{{"skill": 3, "coherence": 3}}

no think
[ASSISTANT]"""

            try:
                messages = [{"role": "user", "content": prompt}]
                response_text = await self.call_llm(messages)
                response_text = self.extract_real_answer(response_text)

                skill = 3; coherence = 3
                import re
                s = re.search(r'skill\D*(\d)', response_text)
                c = re.search(r'coherence\D*(\d)', response_text)
                if s: skill = min(5, int(s.group(1)))
                if c: coherence = min(5, int(c.group(1)))

                # 使用配置中的质量奖励参数或默认值
                quality_weight = reward_config.get('quality_weight', 0.5) if reward_config else 0.5
                max_quality_bonus = reward_config.get('quality_bonus_max', 0.5) if reward_config else 0.5
                quality_bonus = ((skill + coherence) / 10.0) * max_quality_bonus
                bonus_source = "llm_quality_assessment"

            except:
                quality_bonus = 0.2
                bonus_source = "llm_quality_fallback"

        final_reward = rule_reward + quality_bonus

        # 使用配置中的奖励限制参数或默认值
        if reward_config:
            final_reward = max(reward_config.get('final_reward_min', -2.0),
                              min(reward_config.get('final_reward_max', 4.0), final_reward))
        else:
            final_reward = max(-2.0, min(4.0, final_reward))

        bias_str = ",".join(detected_biases)
        # print(f"[DEBUG] 动作:{action_index} | 偏差:[{bias_str}] | 结果:{strategy_type} | Reward:{final_reward:.2f}")

        return final_reward, {
            "source": "rule_mixed",
            "type": strategy_type,
            "intensity": current_intensity,
            "final": final_reward
        }

    async def full_interaction_step(self, context: str, state_text: str, action_index: int,
                                 patient_current_utterance: str = None) -> Tuple[str, str, float]:
        """完整的交互步骤"""
        # 1. 医生
        doctor_response = await self.doctor_agent(context, state_text, action_index, patient_current_utterance)
        # 2. 患者
        patient_reply = await self.patient_agent(context, doctor_response, patient_current_utterance, state_text)
        # 3. 评估 (DEBUG输出已在evaluator_agent中处理)
        reward_score, evaluation_details = await self.evaluator_agent(
            state_text, action_index, doctor_response, patient_reply, patient_current_utterance
        )

        return doctor_response, patient_reply, reward_score