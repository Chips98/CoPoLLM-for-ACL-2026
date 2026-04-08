# -*- coding: utf-8 -*-
"""
认知偏差分类任务微调脚本（单轮Chat形式）
基于 unsloth 和 Qwen2.5-7B-Instruct 模型
专注于训练模型的认知偏差分类能力：偏差类型、强度和危险等级识别
使用单轮Chat形式：将历史对话作为上下文输入，分类结果作为助手输出
"""

import argparse
import os
import warnings
import json
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# 设置环境变量屏蔽警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 屏蔽tokenizer并行警告
os.environ["WANDB_DISABLED"] = "true"  # 禁用wandb
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告
warnings.filterwarnings("ignore", category=FutureWarning)  # 忽略未来警告

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
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
    print("Unsloth 可用，将使用 FastLanguageModel 进行训练。")
except ImportError:
    HAS_UNSLOTH = False
    print("将使用标准的 Hugging Face Transformers 加载模型。")

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# 强制禁用 Unsloth 的 SFTTrainer，使用标准版本
import sys
if HAS_UNSLOTH:
    # 临时移除 Unsloth 的 SFTTrainer
    sys.modules.pop('unsloth.trainer', None)

from trl import SFTTrainer, SFTConfig

# ==================== 系统提示词定义 ====================

# 分类任务系统提示词（单轮Chat形式）
SYS_PROMPT_CLASSIFY = """你是一位专业的心理咨询督导，专门分析患者言语中的认知偏差。

请根据提供的对话历史和患者当前发言，识别其中可能体现的认知偏差类型、安全风险等级和偏差强度。

## 8个认知偏差的解释
- **情感推理**: 将主观感受当作客观事实，认为因为自己有某种感受，所以它必然是真实的。
- **个人化**: 将外部事件的负面结果归因于自己，即使没有充分证据支持这种因果关系。
- **非黑即白**: 用极端的、非此即彼的方式思考问题，认为事物只有完全好或完全坏两种状态。
- **过度概括**: 基于单个或少数几个负面事件，得出广泛、普遍的负面结论。
- **灾难化**: 夸大负面事件的潜在后果，将小事想象成大灾难。
- **应该句式**: 对自己或他人使用僵化的'应该'、'必须'等标准，当现实不符合这些标准时产生强烈负面情绪。
- **标签化**: 基于个别特征对整个人做出极端负面的评价，贴上负面标签。
- **读心术**: 在没有足够证据的情况下，假设自己知道他人的想法或动机。

## 风险等级和强度说明
- **安全风险**: 低危/中危/高危（自杀倾向或严重症状需立即干预为高危）
- **偏差强度**: 轻微（有疑虑但倾向相信）/中等（基本相信但有疑虑）/严重（深信不疑）

请严格按照以下格式输出，不要添加任何额外文字：

认知偏差标签：偏差类型1,偏差类型2...
安全风险：低危/中危/高危
偏差强度：轻微/中等/严重"""

# ==================== 核心数据处理函数 ====================

def build_classification_dataset(data_path: str, tokenizer=None, debug: bool = False) -> Dataset:
    """
    构建分类任务数据集（单轮Chat形式）
    将历史对话作为上下文，构造成单轮Chat样本

    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        debug: 是否打印调试信息

    Returns:
        分类训练数据集（Chat形式）
    """
    print("正在构建分类任务数据集（单轮Chat形式）...")

    # 读取原始数据
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"原始数据包含 {len(raw_data)} 个对话样本")

    classification_samples = []
    processed_count = 0
    skipped_count = 0

    for entry_idx, entry in enumerate(raw_data):
        messages = entry.get('messages', [])

        # 过滤掉原始数据中的 system 消息，只保留 user 和 assistant
        filtered_messages = [msg for msg in messages if msg.get('role') in ['user', 'assistant']]

        # 构建对话上下文
        context_lines = []

        for i, msg in enumerate(filtered_messages):
            role = msg.get('role')
            content = msg.get('content', '')

            # 构建分类任务样本（针对有 annotation 的 user 消息）
            if role == 'user' and 'annotation' in msg:
                annotation = msg['annotation']

                # 构建对话历史文本（上下文）
                context_text = ""
                if context_lines:
                    context_text = "对话历史：\n" + "".join(context_lines) + "\n"
                else:
                    context_text = "对话历史：无\n"

                # 构造用户输入（当前发言 + 分析要求）
                user_input = f"{context_text}患者当前发言：\"{content}\"\n\n请分析上述患者当前发言中的认知偏差。"

                # 构造助手输出（分类结果）【错误！】
                # if 'raw_response' in annotation:
                #     # 提取 raw_response 中的前三行，忽略理由部分
                #     raw_lines = annotation['raw_response'].strip().split('\n')
                #     if len(raw_lines) >= 3:
                #         assistant_output = '\n'.join(raw_lines[:3])
                #     else:
                #         assistant_output = annotation['raw_response']
                # else:
                #     # 动态拼接标注信息（不包含理由）
                #     bias_tags = "、".join(annotation.get('bias_tags', []))
                #     risk_level = annotation.get('risk_level', '')
                #     bias_intensity = annotation.get('bias_intensity', '')

                #     assistant_output = f"认知偏差标签：{bias_tags}\n安全风险：{risk_level}\n偏差强度：{bias_intensity}"
                # print(f"assistant_output-mode-1:{assistant_output}")
                # 【关键修改】：放弃使用 raw_response 切片，改为手动构建标准格式
                # 确保训练目标与 System Prompt 的要求 100% 一致
                
                # 1. 获取字段
                bias_tags = annotation.get('bias_tags', [])
                if isinstance(bias_tags, list):
                    bias_str = ",".join(bias_tags)
                else:
                    bias_str = str(bias_tags)
                
                risk_level = annotation.get('risk_level', '低危')
                # 处理可能缺失的强度字段
                bias_intensity = annotation.get('bias_intensity', '轻微')

                # 2. 严格按照 Prompt 定义的顺序构建输出
                # 格式：标签 -> 风险 -> 强度
                assistant_output = (
                    f"认知偏差标签：{bias_str}\n"
                    f"安全风险：{risk_level}\n"
                    f"偏差强度：{bias_intensity}"
                )
                # print(f"assistant_output-mode-2:{assistant_output}")
                # breakpoint()

                # 构建单轮Chat样本
                single_turn_chat = [
                    {"role": "system", "content": SYS_PROMPT_CLASSIFY},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_output}
                ]

                classification_samples.append({
                    "messages": single_turn_chat,
                    "task_type": "classify",
                    "sample_id": f"{entry_idx}_{i}"
                })

                processed_count += 1

                # 调试打印
                if debug and processed_count <= 3:
                    print(f"\n=== 样本 {processed_count} ===")
                    print(f"数据来源: entry[{entry_idx}], message[{i}]")
                    print(f"对话长度: {len(context_lines)} 轮")
                    print(f"用户输入长度: {len(user_input)} 字符")
                    print(f"助手输出长度: {len(assistant_output)} 字符")
                    print(f"用户输入: {user_input[:200]}...")
                    print(f"助手输出: {assistant_output}")
                    print("="*50)

            # 更新上下文（用于后续样本）
            role_name = "患者" if role == "user" else "医生"
            context_lines.append(f"{role_name}：{content}\n")

        skipped_count += len(filtered_messages) - len([m for m in filtered_messages if m.get('role') == 'user' and 'annotation' in m])

    print(f"分类数据集构建完成！")
    print(f" - 处理的分类样本数量: {processed_count}")
    print(f" - 跳过的消息数量: {skipped_count}")

    if processed_count == 0:
        print("⚠️ 警告: 没有找到任何带有annotation的样本！")
        if debug:
            print("调试信息: 检查前3条数据的结构...")
            for i, entry in enumerate(raw_data[:3]):
                print(f"\nEntry {i}:")
                messages = entry.get('messages', [])
                for j, msg in enumerate(messages):
                    role = msg.get('role', 'unknown')
                    has_annotation = 'annotation' in msg
                    print(f"  Message {j}: role={role}, has_annotation={has_annotation}")
                    if has_annotation:
                        print(f"    Annotation keys: {list(msg['annotation'].keys())}")

    return Dataset.from_list(classification_samples)

# ==================== 多轮对话数据格式化函数 ====================

def format_chat_messages(example: Dict[str, Any], tokenizer: AutoTokenizer = None) -> str:
    """
    将单轮Chat的 messages 列表转换为训练文本
    使用聊天模板自动处理角色和内容
    """
    try:
        messages = example.get("messages", [])
        if not messages:
            return ""

        # 使用聊天模板格式化
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # 训练数据包含完整对话
        )

        return formatted_text

    except Exception as e:
        print(f"格式化消息时出错: {e}")
        return ""

# ==================== 主训练脚本 ====================
'''
CUDA_VISIBLE_DEVICES=0 python sft_classification.py --model_name_or_path "/home/ZhongLin/LLM/llama3.1-8b-instruct" --dataset_path "sft_real_train_20251120_114607.json" --output_dir "./classification_lora" --num_train_epochs 1 --max_length 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --max_steps -1 --learning_rate 2e-4

'''
def main():
    parser = argparse.ArgumentParser(description="认知偏差分类任务微调脚本（单轮Chat形式）")

    # --- 基本配置参数 ---
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="基础模型路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./classification_lora",
        help="模型输出目录"
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="sft_real_train_20251120_114607.json",
        help="训练数据路径"
    )

    # --- 调试参数 ---
    parser.add_argument("--debug", action="store_true", help="启用调试模式，打印数据示例")
    parser.add_argument("--debug_samples", type=int, default=5, help="调试模式下打印的样本数量")

    # --- 训练参数 ---
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度（默认2048避免显存过大）")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="批次大小（开启梯度检查点后可以使用更大的batch_size）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数")
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

    # --- 从检查点恢复训练参数 ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从检查点目录恢复训练（指定检查点目录路径）")

    args = parser.parse_args()

    # 验证从检查点恢复训练参数
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            print(f"错误: 检查点路径不存在: {args.resume_from_checkpoint}")
            return
        print(f"将从检查点恢复训练: {args.resume_from_checkpoint}")

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
    print(f"训练形式: 单轮Chat（Context-as-Input）")
    print(f"调试模式: {'开启' if args.debug else '关闭'}")

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

    # --- 构建分类数据集 ---
    print("正在构建分类任务数据集...")

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

    # 构建数据集
    dataset = build_classification_dataset(args.dataset_path, tokenizer, debug=args.debug)

    if len(dataset) == 0:
        print("错误: 数据集为空，无法开始训练")
        return

    # 格式化数据集
    print("格式化Chat数据...")

    def format_dataset(example):
        formatted_text = format_chat_messages(example, tokenizer)
        return {"text": formatted_text}

    # 应用格式化
    dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

    # 检查格式化结果
    if len(dataset) > 0:
        sample_text = dataset[0]["text"]
        print(f"格式化样本长度: {len(sample_text)}")
        print(f"格式化样本前200字符: {sample_text[:200]}...")

        # 调试模式下打印更多样本
        if args.debug:
            print(f"\n=== 调试：格式化后样本示例 ===")
            for i in range(min(args.debug_samples, len(dataset))):
                sample = dataset[i]["text"]
                print(f"\n样本 {i+1}:")
                print(f"长度: {len(sample)} 字符")
                print(f"内容: {sample[:500]}...")
                print("-" * 30)

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
        "report_to": "none",
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
        # 从检查点恢复训练配置
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }

    training_arguments = SFTConfig(**training_arguments_dict)

    # --- 初始化训练器 ---
    print("初始化 SFTTrainer（单轮Chat形式）...")

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

    # 【关键修复】参考成功脚本模式，不再传递peft_config（已经直接应用到模型）
    trainer_kwargs = {
        "model": model,
        "args": training_arguments,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        # 不再传递 peft_config，因为已经直接应用到模型上了
    }

    trainer = SFTTrainer(**trainer_kwargs)

    print("SFTTrainer 初始化完成")

    # --- 保存推理示例 ---
    inference_example_path = os.path.join(prefixed_output_dir, "inference_example.txt")
    with open(inference_example_path, 'w', encoding='utf-8') as f:
        f.write("# 分类任务推理示例（单轮Chat形式）\n\n")
        f.write("## 推理格式:\n")
        f.write("- System: 定义分类任务\n")
        f.write("- User: 对话历史 + 当前发言 + 分析要求\n")
        f.write("- Assistant: 分类结果（认知偏差标签、安全风险、偏差强度）\n\n")

        f.write("## 推理代码示例:\n")
        f.write("```python\n")
        f.write("from peft import PeftModel\n")
        f.write("from transformers import AutoTokenizer, AutoModelForCausalLM\n\n")
        f.write("# 加载模型\n")
        f.write("base_model = 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'\n")
        f.write("tokenizer = AutoTokenizer.from_pretrained(base_model)\n")
        f.write("model = AutoModelForCausalLM.from_pretrained(base_model)\n")
        f.write("cls_model = PeftModel.from_pretrained(model, './classification_lora_20251121_xxxxxx')\n\n")
        f.write("# 构建单轮Chat输入\n")
        f.write("messages = [\n")
        f.write("    {\"role\": \"system\", \"content\": SYS_PROMPT_CLASSIFY},\n")
        f.write("    {\"role\": \"user\", \"content\": \"对话历史：...\\n患者当前发言：\\\"我觉得这次项目肯定会失败，以前也发生过类似的事情\\\"\\n\\n请分析上述患者当前发言中的认知偏差。\"}\n")
        f.write("]\n\n")
        f.write("# 生成回复\n")
        f.write("inputs = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True)\n")
        f.write("outputs = cls_model.generate(**inputs, max_new_tokens=100)\n")
        f.write("result = tokenizer.decode(outputs[0], skip_special_tokens=True)\n")
        f.write("print(result)\n")
        f.write("```\n\n")

        f.write("## 预期输出:\n")
        f.write("认知偏差标签：过度概括、灾难化\n")
        f.write("安全风险：低危\n")
        f.write("偏差强度：中等\n")

    # --- 开始训练 ---
    print("开始分类任务训练（单轮Chat形式）...")
    try:
        trainer.train()
        print("分类任务训练完成！")

        # --- 保存模型 ---
        print(f"保存模型到: {prefixed_output_dir}")

        # 确保输出目录存在
        if not os.path.exists(prefixed_output_dir):
            os.makedirs(prefixed_output_dir, exist_ok=True)

        trainer.save_model(prefixed_output_dir)
        tokenizer.save_pretrained(prefixed_output_dir)

        if hasattr(trainer.model, 'peft_config') and trainer.model.peft_config is not None:
            print("已保存分类任务 LoRA 适配器")
        else:
            print("已保存完整模型")

        print(f"分类任务微调完成！模型保存至: {prefixed_output_dir}")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()