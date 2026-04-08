# -*- coding: utf-8 -*-
"""
认知偏差分类与心理咨询对话生成混合微调脚本
基于 unsloth 和 Qwen2.5-7B-Instruct 模型
同时训练两种任务：认知偏差分类和心理咨询对话生成

新增功能：
1. Instruction Masking (优化A): 只对响应部分计算损失
2. 分任务损失记录 (优化B): 分别记录分类和生成任务的损失
3. 调试模式: --debug_mode 可验证修改是否成功

使用说明：
- 需要将 trl-main 文件夹上传到与脚本相同的目录
- 如果没有 trl-main，将自动使用备选方案（无 Instruction Masking）
- 建议优先使用调试模式验证功能正常工作

示例命令：
python sft_unified.py --debug_mode --dataset_path "CogBiasESC_train_PRO.json" --resample_ratio 3.0
"""

import argparse
import os
import warnings
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# 设置环境变量屏蔽警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 屏蔽tokenizer并行警告
# 移除 WANDB_DISABLED，改用 --report_to none 参数
# os.environ["WANDB_DISABLED"] = "true"  # 禁用wandb (已废弃)
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告
warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略未来警告

# 添加本地 TRL 路径（用于支持 DataCollatorForCompletionOnlyLM）
import sys
TRL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trl-main")
if os.path.exists(TRL_PATH):
    sys.path.insert(0, TRL_PATH)
    print(f"使用本地 TRL 库: {TRL_PATH}")
else:
    print(f"警告：本地 TRL 库不存在: {TRL_PATH}")
    print("   将使用系统安装的 TRL 库")

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments
)
# 设置环境变量禁用 Unsloth 的网络请求
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["UNSLOTH_DISABLE_STATS"] = "1"  # 禁用统计信息收集

# 尝试导入 Unsloth
HAS_UNSLOTH = False  # 默认初始化为 False
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
    print("✅ Unsloth 导入成功，将使用 FastLanguageModel 进行训练。")
    print(f"DEBUG: HAS_UNSLOTH 设置为 {HAS_UNSLOTH}")
except ImportError as e:
    print(f"❌ 将使用标准的 Hugging Face Transformers 加载模型。错误信息: {e}")
    print(f"DEBUG: HAS_UNSLOTH 保持为 {HAS_UNSLOTH}")

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# 强制禁用 Unsloth 的 SFTTrainer，使用标准版本
import sys
if HAS_UNSLOTH:
    # 临时移除 Unsloth 的 SFTTrainer
    sys.modules.pop('unsloth.trainer', None)

from trl import SFTTrainer, SFTConfig

# 尝试导入 DataCollatorForCompletionOnlyLM，如果失败则提供备选方案
try:
    from trl import DataCollatorForCompletionOnlyLM
    HAS_COMPLETION_COLLATOR = True
    print("✅ DataCollatorForCompletionOnlyLM 可用")
except ImportError:
    HAS_COMPLETION_COLLATOR = False
    print("⚠️  DataCollatorForCompletionOnlyLM 不可用，将使用备选方案")

    # 提供一个改进的备选 collator
    class DataCollatorForCompletionOnlyLM:
        """改进的备选 collator，实现基本的 instruction masking"""
        def __init__(self, response_template, tokenizer, mlm=False):
            self.response_template = response_template
            self.tokenizer = tokenizer
            self.mlm = mlm

            # 编码响应模板
            self.response_token_ids = tokenizer.encode(
                response_template,
                add_special_tokens=False
            )
            if len(self.response_token_ids) == 0:
                print(f"警告：响应模板 '{response_template}' 编码失败")
                self.use_masking = False
            else:
                self.use_masking = True
                print(f"响应模板 '{response_template}' 编码为: {self.response_token_ids}")

        def __call__(self, features):
            from transformers import DataCollatorForLanguageModeling

            if not self.use_masking or not features:
                # 回退到默认 collator
                default_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=self.mlm
                )
                return default_collator(features)

            # 基本的 instruction masking 实现
            batch = self.tokenizer.pad(
                features,
                padding=True,
                return_tensors="pt"
            )

            labels = batch["input_ids"].clone()

            for i, input_ids in enumerate(batch["input_ids"]):
                # 查找响应模板的位置
                for j in range(len(input_ids) - len(self.response_token_ids) + 1):
                    if input_ids[j:j+len(self.response_token_ids)].tolist() == self.response_token_ids:
                        # 将响应模板之前的所有 token 的 label 设为 -100
                        labels[i, :j + len(self.response_token_ids)] = -100
                        break

            batch["labels"] = labels
            return batch

# 定义切割模板（这是指令掩码的关键锚点）
RESPONSE_TEMPLATE = "\n\n### Response:\n"
# 定义任务暗号（用于在计算Loss时区分任务）
# CLS_MARKER = "【任务类型：分析】"
# GEN_MARKER = "【任务类型：引导】"
CLS_MARKER = "[CLS_TASK-分析任务]"
GEN_MARKER = "[GEN_TASK-引导任务] "

# 添加任务类型标识符（用于更可靠的任务检测）
CLS_TASK_ID = "[TASK_TYPE:CLASSIFY]"
GEN_TASK_ID = "[TASK_TYPE:GENERATION]"
# ==================== 系统提示词定义 ====================

# 分类任务系统提示词（参考 simulation/cogbias_classify/core/agent_api/prompt_builder.py）
SYS_PROMPT_CLASSIFY = """## 任务提示
你是一位专业的心理咨询师，现在需要分析患者的当前发言，识别其中可能体现的认知偏差、安全风险等级和偏差强度。

请仔细分析患者的当前发言内容，判断其反映出来的认知偏差类型、安全风险等级和偏差强度。

重要说明：请仅针对患者的当前发言进行判断，而不是基于整个对话历史。对话历史仅用于理解上下文背景。

### 8个认知偏差的解释
- **情感推理**: 将主观感受当作客观事实，认为因为自己有某种感受，所以它必然是真实的。
- **个人化**: 将外部事件的负面结果归因于自己，即使没有充分证据支持这种因果关系。
- **非黑即白**: 用极端的、非此即彼的方式思考问题，认为事物只有完全好或完全坏两种状态。
- **过度概括**: 基于单个或少数几个负面事件，得出广泛、普遍的负面结论。
- **灾难化**: 夸大负面事件的潜在后果，将小事想象成大灾难。
- **应该句式**: 对自己或他人使用僵化的'应该'、'必须'等标准，当现实不符合这些标准时产生强烈负面情绪。
- **标签化**: 基于个别特征对整个人做出极端负面的评价，贴上负面标签。
- **读心术**: 在没有足够证据的情况下，假设自己知道他人的想法或动机。

### 3类安全风险等级的解释
- **低危**: 患者没有明显的自杀倾向或自伤风险，情绪相对稳定，能够正常思考。
- **中危**: 患者有轻微的负面情绪和痛苦体验，可能有消极想法但无明确的自杀计划，有自我调节能力。
- **高危**: 患者有明显自杀倾向、自伤风险或严重精神症状，需要立即干预。

### 3类认知偏差强度的解释
- **轻微**: 患者仅部分表现出认知偏差，对偏差信念有疑虑，但仍倾向相信这种偏差想法。
- **中等**: 患者表现出明显的认知偏差，虽然流露一些疑虑，但基本相信自己的偏差想法。
- **严重**: 患者对认知偏差深信不疑，完全没有疑虑，偏差想法主导其思维和情绪。

## 注意事项
1. 请仅分析当前发言中体现的认知偏差特征
2. 一个发言可能包含多种认知偏差，请选择所有适用的类型
3. 偏差强度判断基于患者对偏差想法的信念程度：轻微（有疑虑但倾向相信）、中等（基本相信但有疑虑）、严重（深信不疑）
4. 如果发言中提到自杀想法或计划，风险等级应为"高危"
5. 请基于发言内容的具体表现来判断偏差强度
6. no think

## 输出格式
请严格按照以下格式输出，不要添加任何额外文字：

认知偏差标签：偏差类型1、偏差类型2...
安全风险：低危/中危/高危
偏差强度：轻微/中等/严重
理由：一句话解释选择的原因"""

# 生成任务系统提示词（参考 simulation/cogbias_generation/core/agent_api/prompt_builder.py）
SYS_PROMPT_GENERATION = """## 一、核心任务
你需精准扮演一名专业心理咨询师（简称'治疗师'），核心目标是基于历史对话记忆和当前表述，生成贴合专业心理咨询场景的医生回复。通过倾听理解、重述澄清、温和引导反思提供情感支持与认知梳理，确保回复与真实治疗师的专业交互逻辑一致。

## 二、注意事项
（1）表达要简短，尽可能地口语化、自然。
（2）因为咨询师只受过心理学相关的教育，只能提供心理咨询相关的对话内容。
（3）不要一次性询问过多的问题，尽量一次性只向来访者询问一个问题，与来访者互动后一步步探寻心理问题的原因。
（4）话术需要参考有经验的真人心理咨询师，尽可能口语化。
（5）针对患者说的每句话你的回复不要超过3句话。因为一般医生只会对患者说的话进行1-3句的回复。

## 三、!! 严格禁止 (必须遵守) !!
(1) **禁止扮演患者**：你的回复中**绝对不能**包含 "患者："、"来访者：" 或任何模拟患者说的话。
(2) **禁止旁白和总结**：你的回复**只应包含对话本身**。**严禁**包含任何括号 `()` 或 `（）` 内的动作描述、心理活动、场景说明、`（引导...）`、`（咨询结束...）` 或 `（以上对话仅为示例...）` 这样的元注释。
(3) **禁止元思考**：no think
(4) **禁止非对话内容**：**严禁**使用催眠、冥想引导或生成任何代码（如 `python`）。
(5) **禁止角色标签**：不要在你的回复前添加 `治疗师：` 或 `你：` 标签。
no think
## 语言
请使用中文回答！！"""

# ==================== 核心数据处理函数 ====================

def build_mixed_dataset(data_path: str, tokenizer, resample_ratio: float = 2.0) -> Dataset:
    """
    构建混合训练数据集，同时包含分类和生成任务样本

    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        resample_ratio: 重采样比例，分类样本数量会乘以这个比例来接近生成样本数量

    Returns:
        混合训练数据集
    """
    print("正在构建混合训练数据集...")

    # 读取原始数据
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"原始数据包含 {len(raw_data)} 个对话样本")

    classification_samples = []
    generation_samples = []

    for entry in raw_data:
        messages = entry.get('messages', [])

        # 过滤掉原始数据中的 system 消息，只保留 user 和 assistant
        filtered_messages = [msg for msg in messages if msg.get('role') in ['user', 'assistant']]
        history = []

        for i, msg in enumerate(filtered_messages):
            role = msg.get('role')
            content = msg.get('content', '')

            # 构建分类任务样本（针对有 annotation 的 user 消息）
            if role == 'user' and 'annotation' in msg:
                annotation = msg['annotation']

                # 构造输入：系统提示(分类) + 历史对话 + 当前用户发言
                input_msgs = [{"role": "system", "content": SYS_PROMPT_CLASSIFY}]
                input_msgs.extend(history)
                input_msgs.append({"role": "user", "content": content})

                # 【关键修复】：放弃使用 raw_response 切片，改为手动构建标准格式
                # 确保训练目标与 System Prompt 的要求 100% 一致

                # 1. 获取字段
                bias_tags = annotation.get('bias_tags', [])
                if isinstance(bias_tags, list):
                    bias_str = "、".join(bias_tags)
                else:
                    bias_str = str(bias_tags)

                risk_level = annotation.get('risk_level', '低危')
                # 处理可能缺失的强度字段
                bias_intensity = annotation.get('bias_intensity', '轻微')

                # 2. 严格按照 Prompt 定义的顺序构建输出
                # 格式：标签 -> 风险 -> 强度 （与 sft_classification.py 保持一致）
                output_content = (
                    f"认知偏差标签：{bias_str}\n"
                    f"安全风险：{risk_level}\n"
                    f"偏差强度：{bias_intensity}"
                )

                # 构建完整对话
                complete_convo = input_msgs + [{"role": "assistant", "content": output_content}]
                classification_samples.append({
                    "messages": complete_convo,
                    "task_type": "classify"
                })

            # 构建生成任务样本（针对所有 assistant 消息）
            elif role == 'assistant':
                # 确保上一条是 user 消息
                if i > 0 and filtered_messages[i-1]['role'] == 'user':
                    user_msg = filtered_messages[i-1]

                    # 构造输入：系统提示(生成) + 历史对话（截止到上一条user消息） + 当前user发言
                    input_msgs = [{"role": "system", "content": SYS_PROMPT_GENERATION}]
                    input_msgs.extend(history)
                    input_msgs.append({"role": "user", "content": user_msg['content']})

                    # 构造完整对话
                    complete_convo = input_msgs + [{"role": "assistant", "content": content}]
                    generation_samples.append({
                        "messages": complete_convo,
                        "task_type": "generation"
                    })

            # 更新历史对话（只包含 role 和 content）
            history.append({"role": role, "content": content})

    print(f"构建完成！")
    print(f" - 分类样本数量: {len(classification_samples)}")
    print(f" - 生成样本数量: {len(generation_samples)}")

    # 对分类样本进行重采样以平衡数据
    if len(classification_samples) > 0 and len(generation_samples) > 0:
        # 计算重采样数量
        target_classify_count = int(len(generation_samples) / resample_ratio)

        # 计算实际比例
        current_ratio = len(generation_samples) / len(classification_samples)
        print(f"当前数据比例: 生成:分类 = {current_ratio:.2f}:1")
        print(f"目标比例: 生成:分类 = {resample_ratio:.2f}:1")
        print(f"目标分类样本数: {target_classify_count}")

        if len(classification_samples) < target_classify_count:
            print(f"对分类样本进行重采样：{len(classification_samples)} -> {target_classify_count}")
            # 随机重采样
            classification_samples = random.choices(
                classification_samples,
                k=target_classify_count
            )
        else:
            # 即使分类样本充足，也检查是否需要减少到目标比例
            print(f"分类样本充足，但为了达到目标比例 {resample_ratio}:1")
            print(f"建议使用更小的 resample_ratio，当前分类样本已经达到 {len(generation_samples)/len(classification_samples):.2f}:1")
            # 可选：进行下采样
            # classification_samples = random.sample(classification_samples, target_classify_count)
            # print(f"分类样本下采样到: {target_classify_count}")

    # 合并数据集并打乱
    mixed_samples = classification_samples + generation_samples
    random.shuffle(mixed_samples)

    print(f"混合数据集总样本数: {len(mixed_samples)}")

    return Dataset.from_list(mixed_samples)

# ==================== 数据格式化函数 (优化 A：Instruction Masking) ====================

def format_classification_sample(sample: Dict[str, Any], tokenizer: AutoTokenizer = None) -> str:
    """
    格式化分类样本
    结构：[CLS_MARKER] + Instruction + Input + [RESPONSE_TEMPLATE] + Output
    """
    messages = sample.get("messages", [])
    if not messages:
        return ""

    # 提取系统提示词和用户消息
    system_msg = ""
    user_msg = ""
    assistant_msg = ""

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'system':
            system_msg = content
        elif role == 'user':
            user_msg = content
        elif role == 'assistant':
            assistant_msg = content

    # 构建历史对话（简化处理）
    history_str = ""
    if len(messages) > 3:  # 有历史对话
        # 从第2个消息到倒数第2个消息作为历史
        for msg in messages[1:-2]:
            if msg.get('role') == 'user':
                history_str += f"患者：{msg.get('content', '')}\n"
            elif msg.get('role') == 'assistant':
                history_str += f"治疗师：{msg.get('content', '')}\n"

    # 构建 Instruction 部分（加入 CLS_MARKER）
    prompt = f"{CLS_MARKER}\n{system_msg}\n历史对话：\n{history_str}患者：{user_msg}"

    # 构建 Output 部分
    output_text = assistant_msg

    # 【修正】：将任务ID放在 Prompt 的最前面（或 Instruction 内部）
    # 这样它会被 Collator 视为 Prompt 的一部分，不计算 Loss，模型就不会生成它
    prompt_with_id = f"{CLS_TASK_ID}\n{prompt}"

    # 拼接
    text = f"{prompt_with_id}{RESPONSE_TEMPLATE}{output_text}"

    # Debug 模式下打印格式化结果
    if 'DEBUG_MODE' in globals() and DEBUG_MODE:
        print(f"\n🏷️  分类任务样本格式化:")
        print(f"   任务标记: {CLS_MARKER}")
        print(f"   任务标识符: {CLS_TASK_ID}")
        print(f"   响应模板: {RESPONSE_TEMPLATE}")
        print(f"   完整文本长度: {len(text)}")
        print(f"   前150字符: {text[:150]}...")
        print(f"   后150字符: ...{text[-150:]}")

    return text

def format_generation_sample(sample: Dict[str, Any], tokenizer: AutoTokenizer = None) -> str:
    """
    格式化生成样本
    结构：[GEN_MARKER] + Instruction + Input + [RESPONSE_TEMPLATE] + Output
    """
    messages = sample.get("messages", [])
    if not messages:
        return ""

    # 提取系统提示词和用户消息
    system_msg = ""
    user_msg = ""
    assistant_msg = ""

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'system':
            system_msg = content
        elif role == 'user':
            user_msg = content
        elif role == 'assistant':
            assistant_msg = content

    # 构建历史对话（简化处理）
    history_str = ""
    if len(messages) > 3:  # 有历史对话
        # 从第2个消息到倒数第2个消息作为历史
        for msg in messages[1:-2]:
            if msg.get('role') == 'user':
                history_str += f"患者：{msg.get('content', '')}\n"
            elif msg.get('role') == 'assistant':
                history_str += f"治疗师：{msg.get('content', '')}\n"

    # 构建 Instruction 部分（加入 GEN_MARKER）
    prompt = f"{GEN_MARKER}\n{system_msg}\n历史对话：\n{history_str}患者：{user_msg}"

    # 构建 Output 部分
    output_text = assistant_msg

    # 【修正】：将任务ID放在 Prompt 的开头
    prompt_with_id = f"{GEN_TASK_ID}\n{prompt}"

    # 拼接
    text = f"{prompt_with_id}{RESPONSE_TEMPLATE}{output_text}"

    # Debug 模式下打印格式化结果
    if 'DEBUG_MODE' in globals() and DEBUG_MODE:
        print(f"\n🔄 生成任务样本格式化:")
        print(f"   任务标记: {GEN_MARKER}")
        print(f"   任务标识符: {GEN_TASK_ID}")
        print(f"   响应模板: {RESPONSE_TEMPLATE}")
        print(f"   完整文本长度: {len(text)}")
        print(f"   前150字符: {text[:150]}...")
        print(f"   后150字符: ...{text[-150:]}")

    return text

def format_chat_messages(example: Dict[str, Any], tokenizer: AutoTokenizer = None) -> str:
    """
    统一的数据格式化入口 - 根据任务类型选择格式化函数
    """
    try:
        task_type = example.get("task_type", "classify")

        if task_type == "classify":
            return format_classification_sample(example, tokenizer)
        else:  # generation
            return format_generation_sample(example, tokenizer)

    except Exception as e:
        print(f"格式化消息时出错: {e}")
        return ""

# ==================== 自定义训练器 (优化 B：Evidence Depth) ====================

class DualTaskTrainer(SFTTrainer):
    """
    自定义训练器，实现分任务损失记录功能
    继承自 SFTTrainer，重写 compute_loss 方法
    """

    def __init__(self, *args, log_file="training_loss_log.csv", cls_marker_token_id=None, debug_mode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
        self.cls_marker_token_id = cls_marker_token_id
        self.debug_mode = debug_mode
        self.debug_counter = 0  # 用于控制调试输出频率

        # 初始化日志文件头
        if self.is_world_process_zero():
            with open(self.log_file, "w") as f:
                f.write("step,total_loss,cls_loss,gen_loss,timestamp\n")

        print(f"DualTaskTrainer 初始化完成，日志文件: {self.log_file}")
        if self.debug_mode:
            print("🐛 DualTaskTrainer 调试模式已开启")
        if self.cls_marker_token_id is not None:
            print(f"分类任务标记 Token ID: {self.cls_marker_token_id}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 Loss 计算逻辑：
        1. 正常前向传播
        2. 手动计算 CrossEntropy (reduction='none') 得到每个 token 的 loss
        3. 根据 input_ids 中的 Marker 区分任务
        4. 分别记录日志
        """
        # 1. 获取 Labels 和 Inputs
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")

        if labels is None or input_ids is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Debug 信息：打印输入形状和基本信息
        if self.debug_mode and self.debug_counter < 3:  # 只在前3次打印
            print(f"\n🐛 Debug Step {self.state.global_step if hasattr(self.state, 'global_step') else 'Unknown'}")
            print(f"   Batch size: {input_ids.shape[0]}")
            print(f"   Input sequence length: {input_ids.shape[1]}")

        # 2. 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 3. 自定义 Loss 计算 (参考 CausalLM 的 Loss 实现)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')  # 关键：不要平均
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # 计算逐个 Token 的 Loss
        token_losses = loss_fct(shift_logits, shift_labels)

        # 4. 恢复成 [Batch, Seq_Len-1] 的形状
        batch_size = input_ids.shape[0]
        seq_len = shift_logits.shape[0] // batch_size
        token_losses = token_losses.view(batch_size, -1)

        # 5. 计算每个 Sample 的平均 Loss (忽略 -100)
        # shift_labels ne -100 生成掩码
        padding_mask = shift_labels.view(batch_size, -1).ne(-100).float()
        # 每个样本的有效 loss 总和 / 每个样本的有效 token 数
        per_sample_loss = (token_losses * padding_mask).sum(dim=1) / (padding_mask.sum(dim=1) + 1e-9)

        # 6. 区分任务 (基于字符串匹配的可靠检测)
        loss_cls = torch.tensor(0.0, device=model.device)
        loss_gen = torch.tensor(0.0, device=model.device)
        cls_count = 0
        gen_count = 0

        # 使用字符串匹配来检测任务类型（更可靠）
        cls_mask_list = []
        gen_mask_list = []

        for i in range(batch_size):
            try:
                # 解码整个序列
                # 使用 processing_class 替代 tokenizer (避免 deprecation warning)
                tokenizer = self.processing_class if hasattr(self, 'processing_class') else self.tokenizer
                decoded_text = tokenizer.decode(input_ids[i], skip_special_tokens=False)

                # 检查任务类型标识符
                if CLS_TASK_ID in decoded_text:
                    cls_mask_list.append(True)
                    gen_mask_list.append(False)
                    cls_count += 1
                elif GEN_TASK_ID in decoded_text:
                    cls_mask_list.append(False)
                    gen_mask_list.append(True)
                    gen_count += 1
                else:
                    # 备选方案：使用标记检测
                    if CLS_MARKER in decoded_text:
                        cls_mask_list.append(True)
                        gen_mask_list.append(False)
                        cls_count += 1
                    elif GEN_MARKER in decoded_text:
                        cls_mask_list.append(False)
                        gen_mask_list.append(True)
                        gen_count += 1
                    else:
                        # 默认归为生成任务（避免出错）
                        cls_mask_list.append(False)
                        gen_mask_list.append(True)
                        gen_count += 1

            except Exception as e:
                # 解码失败，默认归为生成任务
                cls_mask_list.append(False)
                gen_mask_list.append(True)
                gen_count += 1
                if self.debug_mode and self.debug_counter < 3:
                    print(f"   样本 {i} 解码失败: {e}")

        # 转换为 tensor
        cls_mask = torch.tensor(cls_mask_list, device=model.device)
        gen_mask = torch.tensor(gen_mask_list, device=model.device)

        # Debug 信息：打印任务检测结果和样本示例
        if self.debug_mode and self.debug_counter < 3:
            print(f"   检测到分类任务样本: {cls_count}")
            print(f"   检测到生成任务样本: {gen_count}")

            # 打印第一个样本的信息作为示例
            if batch_size > 0:
                try:
                    first_input_ids = input_ids[0]
                    # 使用 processing_class 替代 tokenizer (避免 deprecation warning)
                    tokenizer = self.processing_class if hasattr(self, 'processing_class') else self.tokenizer
                    decoded_text = tokenizer.decode(first_input_ids, skip_special_tokens=False)
                    print(f"\n📝 样本 0 解码文本 (前300字符):")
                    print(f"   {decoded_text[:300]}...")

                    # 检查任务类型
                    if CLS_TASK_ID in decoded_text:
                        print(f"   ✅ 检测到分类任务标识符: {CLS_TASK_ID}")
                    elif GEN_TASK_ID in decoded_text:
                        print(f"   ✅ 检测到生成任务标识符: {GEN_TASK_ID}")
                    elif CLS_MARKER in decoded_text:
                        print(f"   ✅ 检测到分类任务标记: {CLS_MARKER}")
                    elif GEN_MARKER in decoded_text:
                        print(f"   ✅ 检测到生成任务标记: {GEN_MARKER}")
                    else:
                        print(f"   ⚠️  未检测到任何任务标记/标识符")
                except Exception as e:
                    print(f"   解码失败: {e}")

        # 分别计算两类任务的损失
        if cls_mask.any():
            loss_cls = per_sample_loss[cls_mask].mean()
        if gen_mask.any():
            loss_gen = per_sample_loss[gen_mask].mean()

        # 7. 记录日志 (仅在主进程)
        if (self.is_world_process_zero() and
            hasattr(self.state, 'global_step') and
            self.state.global_step % self.args.logging_steps == 0):
            self._log_local(loss_cls.item(), loss_gen.item(), per_sample_loss.mean().item())

        # Debug 信息：打印损失详情
        if self.debug_mode and self.debug_counter < 3:
            print(f"\n💰 损失详情:")
            print(f"   总损失: {per_sample_loss.mean().item():.6f}")
            if cls_count > 0:
                print(f"   分类损失: {loss_cls.item():.6f} (基于{cls_count}个样本)")
            if gen_count > 0:
                print(f"   生成损失: {loss_gen.item():.6f} (基于{gen_count}个样本)")
            print("-" * 60)

        # 8. 返回用于梯度的 Loss (总平均)
        total_loss = per_sample_loss.mean()

        self.debug_counter += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def _log_local(self, cls_loss, gen_loss, total_loss):
        """写入本地 CSV 日志文件"""
        try:
            with open(self.log_file, "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f"{self.state.global_step},{total_loss:.4f},{cls_loss:.4f},{gen_loss:.4f},{timestamp}\n")
        except Exception as e:
            print(f"写入日志文件时出错: {e}")

# ==================== 主训练脚本 ====================
'''
CUDA_VISIBLE_DEVICES=1 python sft_unified.py --model_name_or_path "/home/ZhongLin/LLM/llama3.1-8b-instruct" --dataset_path "sft_real_train_20251120_114607.json" --output_dir "./unified_lora" --resample_ratio 3.0 --max_length 2048 --per_device_train_batch_size 12 --gradient_accumulation_steps 2 --learning_rate 2e-4 --max_steps -1 --num_train_epochs 3 --save_steps 200
'''
def main():
    global HAS_UNSLOTH  # 声明使用全局变量
    parser = argparse.ArgumentParser(description="认知偏差分类与心理咨询对话生成混合微调脚本")

    # --- 基本配置参数 ---
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="基础模型路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./unified_lora",
        help="模型输出目录"
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="sft_real_train_20251120_114607.json",
        help="训练数据路径"
    )
    parser.add_argument(
        "--resample_ratio", type=float,
        default=3.0,
        help="分类样本重采样比例，分类样本数量会乘以这个比例来接近生成样本数量"
    )

    # --- 训练参数 ---
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=120, help="最大训练步数")
    parser.add_argument("--save_steps", type=int, default=200, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=1, help="日志记录步数")
    parser.add_argument("--save_total_limit", type=int, default=None, help="保存的检查点数量限制（None表示保存所有）")

    # --- LoRA 参数 ---
    parser.add_argument("--use_lora", action="store_true", default=True, help="使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_qlora", action="store_true", default=True, help="使用 QLoRA")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="4位量化")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="开启梯度检查点以减少显存占用")
    parser.add_argument("--disable_unsloth", action="store_true", help="强制禁用 Unsloth，使用标准 Transformers")
    parser.add_argument("--debug_mode", action="store_true", help="开启调试模式，打印详细信息验证修改")

    args = parser.parse_args()

    # 设置全局 debug 模式
    DEBUG_MODE = args.debug_mode
    if DEBUG_MODE:
        print("🐛 调试模式已开启，将打印详细信息验证修改")
        print("=" * 60)

    # 强制禁用 Unsloth
    if args.disable_unsloth:
        HAS_UNSLOTH = False
        print("根据用户设置强制禁用 Unsloth，将使用标准的 Hugging Face Transformers 加载模型。")
        # 移除 Unsloth 的 SFTTrainer
        sys.modules.pop('unsloth.trainer', None)

    # --- 生成带时间戳的输出目录名 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefixed_output_dir = f"{args.output_dir}_{timestamp}"

    # --- 验证路径 ---
    print(f"模型路径: {args.model_name_or_path}")
    print(f"数据路径: {args.dataset_path}")
    print(f"输出目录: {prefixed_output_dir}")

    # --- 检查优化方案是否启用 ---
    print("\n=== 优化方案检查 ===")
    print(f"Unsloth 启用: {'✅ 是' if HAS_UNSLOTH else '❌ 否'}")
    print(f"4-bit 量化: {'✅ 是' if args.load_in_4bit else '❌ 否'}")
    print(f"LoRA 训练: {'✅ 是' if args.use_lora else '❌ 否'}")
    print(f"梯度检查点: {'✅ 是' if args.gradient_checkpointing else '❌ 否'}")
    print(f"序列长度: {args.max_length}")
    print("=" * 30)

    # 验证数据文件存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据文件不存在: {args.dataset_path}")
        return

    # GPU设置
    if torch.cuda.is_available():
        print(f"检测到GPU数量: {torch.cuda.device_count()}")
        gpu_list = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        for i, gpu_name in enumerate(gpu_list):
            print(f"  GPU {i}: {gpu_name}")
    else:
        print("警告: 未检测到可用的GPU设备")
        return

    # --- 构建混合数据集 ---
    print("正在构建混合训练数据集...")

    # 先加载tokenizer用于数据处理
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"设置 pad_token: {tokenizer.pad_token}")

    # 初始化 Tokenizer (确保 padding_side='right' 用于 instruction masking)
    tokenizer.padding_side = 'right'  # TRL通常需要right padding来做completion only

    # 获取 Marker 的 Token ID 用于 Loss 区分
    print("\n=== 计算任务标记 Token ID ===")
    try:
        cls_token_ids = tokenizer.encode(CLS_MARKER, add_special_tokens=False)
        if len(cls_token_ids) == 0:
            print("警告：分类任务标记 Token 化失败，将使用默认检测方式")
            cls_first_token_id = None
        else:
            cls_first_token_id = cls_token_ids[0]
            print(f"分类任务标记 '{CLS_MARKER}' 的 Token ID: {cls_first_token_id}")
    except Exception as e:
        print(f"Token化任务标记时出错: {e}")
        cls_first_token_id = None

    dataset = build_mixed_dataset(args.dataset_path, tokenizer, args.resample_ratio)

    # 格式化数据集
    print("格式化多轮对话数据...")

    # Debug 模式下控制样本打印数量
    debug_sample_count = 0
    max_debug_samples = 2 if DEBUG_MODE else 0

    def format_dataset(example):
        nonlocal debug_sample_count
        formatted_text = format_chat_messages(example, tokenizer)

        # Debug 模式下控制打印数量
        if DEBUG_MODE and debug_sample_count < max_debug_samples:
            debug_sample_count += 1
            # 在 format_chat_messages 函数内部已经处理了打印

        return {"text": formatted_text}

    # 应用格式化
    dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

    # 检查格式化结果
    if len(dataset) > 0:
        sample_text = dataset[0]["text"]
        print(f"格式化样本长度: {len(sample_text)}")
        print(f"格式化样本前200字符: {sample_text[:200]}...")

    # --- 加载模型 ---
    print("正在加载模型...")

    if HAS_UNSLOTH:
        # 使用 Unsloth 加载模型
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=args.load_in_4bit,
        )

        # 配置 LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
        )

        print("Unsloth 模型和 LoRA 配置完成")
        model.print_trainable_parameters()

    else:
        # 使用标准 Transformers 加载模型（参考成功的sft_psydt.py模式）
        # 配置量化
        bnb_config = None
        if args.use_qlora and args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("启用4位量化")

        # 加载模型 - 不应用LoRA，让SFTTrainer处理
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("模型加载成功")

        # 【关键修复】参考 sft_psydt.py 的成功模式，直接应用LoRA到量化模型
        if args.use_lora:
            if args.use_qlora:
                print("正在为量化模型准备LoRA训练...")
                # 准备量化模型进行LoRA训练
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
                print("✅ 已准备量化模型进行LoRA训练")

            # 直接创建并应用LoRA适配器（参考成功的 sft_psydt.py 模式）
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )

            # 直接应用到模型上
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print("✅ LoRA适配器已直接应用到模型上")
        else:
            lora_config = None

    # --- 配置训练参数 ---
    training_arguments_dict = {
        "output_dir": prefixed_output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,  # 可以通过命令行参数控制保存的检查点数量
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "logging_first_step": True,
        "report_to": "none",  # 禁用wandb等报告工具
        # 当禁用 Unsloth 时，强制使用 fp16 避免精度冲突
        "bf16": False if args.disable_unsloth else torch.cuda.is_bf16_supported(),
        "fp16": True if args.disable_unsloth else not torch.cuda.is_bf16_supported(),
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "remove_unused_columns": True,
        "dataset_text_field": "text",
        "packing": False,
        "ddp_find_unused_parameters": False,
        "dataloader_drop_last": True,
        # ====== 梯度检查点配置 ======
        "gradient_checkpointing": args.gradient_checkpointing,
        # ============================
    }

    training_arguments = SFTConfig(**training_arguments_dict)

    # --- 初始化数据收集器 (优化 A：Instruction Masking) ---
    print("\n=== 初始化数据收集器 ===")

    # 无论是否有原生 TRL，我们都使用改进的备选方案
    try:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=RESPONSE_TEMPLATE,
            tokenizer=tokenizer
        )
        print("✅ DataCollatorForCompletionOnlyLM 初始化成功")
        print(f"   响应模板: '{RESPONSE_TEMPLATE}'")
        if HAS_COMPLETION_COLLATOR:
            print("   📦 使用 TRL 原生版本 (功能完整)")
        else:
            print("   🔧 使用改进备选版本 (支持基本 Instruction Masking)")
        if DEBUG_MODE:
            print("   🐛 Debug: 将尝试只对响应部分计算损失")
    except Exception as e:
        print(f"❌ DataCollatorForCompletionOnlyLM 初始化失败: {e}")
        print("   将完全禁用 Instruction Masking")
        from transformers import DataCollatorForLanguageModeling
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        if DEBUG_MODE:
            print("   🐛 Debug: 将对整个序列计算损失（包括指令和响应）")

    # --- 初始化训练器 ---
    print("初始化 DualTaskTrainer...")

    # 检查空样本
    empty_samples = sum(1 for x in dataset if not x["text"] or len(x["text"]) == 0)
    if empty_samples > 0:
        print(f"警告: 发现 {empty_samples} 个空样本")
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: x["text"] and len(x["text"]) > 0)
        print(f"过滤后数据集大小: {original_len} -> {len(dataset)}")

    if len(dataset) == 0:
        print("错误: 数据集为空，无法开始训练")
        return

    # 准备训练器参数
    log_file_path = os.path.join(prefixed_output_dir, "dual_task_loss_log.csv")

    # 使用自定义 DualTaskTrainer
    trainer_kwargs = {
        "model": model,
        "args": training_arguments,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "data_collator": collator,  # 使用 Instruction Masking 的 Collator
        # 自定义参数
        "log_file": log_file_path,
        "cls_marker_token_id": cls_first_token_id,
        "debug_mode": DEBUG_MODE,  # 传递调试模式
    }

    trainer = DualTaskTrainer(**trainer_kwargs)

    print("✅ DualTaskTrainer 初始化完成")
    print(f"   损失日志文件: {log_file_path}")
    if cls_first_token_id is not None:
        print(f"   分类任务标记 Token ID: {cls_first_token_id}")

    # --- 开始训练 ---
    print("开始训练...")
    try:
        # 设置环境变量以获取 logits（Unsloth 2024.11+ 版本需要）
        if HAS_UNSLOTH:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
            print("已设置 UNSLOTH_RETURN_LOGITS=1 以获取 logits")

        trainer.train()
        print("训练完成！")

        # --- 保存模型 ---
        print(f"保存模型到: {prefixed_output_dir}")

        # 确保输出目录存在
        if not os.path.exists(prefixed_output_dir):
            os.makedirs(prefixed_output_dir, exist_ok=True)

        trainer.save_model(prefixed_output_dir)
        tokenizer.save_pretrained(prefixed_output_dir)

        if hasattr(trainer.model, 'peft_config') and trainer.model.peft_config is not None:
            print("已保存 LoRA 适配器")
        else:
            print("已保存完整模型")

        print(f"混合训练完成！模型保存至: {prefixed_output_dir}")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()