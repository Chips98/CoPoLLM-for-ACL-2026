# -*- coding: utf-8 -*-
"""
心理咨询对话生成任务微调脚本
基于 unsloth 和 Qwen2.5-7B-Instruct 模型
专注于训练模型的心理咨询对话生成能力，不包含认知偏差分类
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

# 生成任务系统提示词（参考 sft_unified.py 中的生成提示词）
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

def build_generation_dataset(data_path: str, tokenizer) -> Dataset:
    """
    构建生成任务数据集，只包含心理咨询对话样本，去除所有分类标注

    Args:
        data_path: 数据文件路径
        tokenizer: 分词器

    Returns:
        生成训练数据集
    """
    print("正在构建生成任务数据集...")

    # 读取原始数据
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"原始数据包含 {len(raw_data)} 个对话样本")

    generation_samples = []

    for entry in raw_data:
        messages = entry.get('messages', [])

        # 过滤掉原始数据中的 system 消息，只保留 user 和 assistant
        filtered_messages = [msg for msg in messages if msg.get('role') in ['user', 'assistant']]

        # 构建新的对话：强制使用生成任务的系统提示词，去除所有annotation
        new_messages = [{"role": "system", "content": SYS_PROMPT_GENERATION}]

        for msg in filtered_messages:
            role = msg.get('role')
            content = msg.get('content', '')

            # 只保留 user 和 assistant 消息，完全去除 annotation 信息
            if role in ['user', 'assistant']:
                new_messages.append({
                    "role": role,
                    "content": content
                })

        # 只保留包含完整对话的样本（至少有一轮对话）
        if len(new_messages) > 2:  # system + 至少一对对话
            generation_samples.append({
                "messages": new_messages,
                "task_type": "generation"
            })

    print(f"生成数据集构建完成！")
    print(f" - 生成样本数量: {len(generation_samples)}")

    return Dataset.from_list(generation_samples)

# ==================== 多轮对话数据格式化函数 ====================

def format_chat_messages(example: Dict[str, Any], tokenizer: AutoTokenizer = None) -> str:
    """
    将多轮对话的 messages 列表转换为训练文本
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

'''
CUDA_VISIBLE_DEVICES=2 python sft_generation.py --model_name_or_path "/home/ZhongLin/LLM/llama3.1-8b-instruct" --dataset_path "sft_real_train_20251120_114607.json" --output_dir "./generation_lora" --num_train_epochs 1 --max_length 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 --max_steps -1 --learning_rate 2e-4
'''
# ==================== 主训练脚本 ====================

def main():
    parser = argparse.ArgumentParser(description="心理咨询对话生成任务微调脚本")

    # --- 基本配置参数 ---
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="基础模型路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./generation_lora",
        help="模型输出目录"
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="sft_real_train_20251120_114607.json",
        help="训练数据路径"
    )

    # --- 训练参数 ---
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数")
    parser.add_argument("--save_steps", type=int, default=200, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_total_limit", type=int, default=None, help="保存的检查点数量限制（None表示保存所有）")

    # --- LoRA 参数 ---
    parser.add_argument("--use_lora", action="store_true", default=True, help="使用 LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_qlora", action="store_true", default=True, help="使用 QLoRA")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="4位量化")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="开启梯度检查点以减少显存占用")

    # --- 继续训练参数（用于链式训练：先训练生成任务，再训练分类任务） ---
    parser.add_argument("--continue_from_lora", action="store_true", default=False,
                       help="是否从已有LoRA适配器继续训练（第二阶段训练）")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="已有LoRA适配器的路径，当continue_from_lora为True时使用")

    # --- 从检查点恢复训练参数 ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从检查点目录恢复训练（指定检查点目录路径）")

    parser.add_argument("--disable_unsloth", action="store_true", help="强制禁用 Unsloth，使用标准 Transformers")

    args = parser.parse_args()

    # 验证继续训练参数
    if args.continue_from_lora:
        if not args.lora_path:
            print("错误: 启用继续训练时必须提供 --lora_path 参数")
            return
        if not os.path.exists(args.lora_path):
            print(f"错误: LoRA路径不存在: {args.lora_path}")
            return
        print(f"将从已有LoRA适配器继续训练: {args.lora_path}")

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

    # --- 构建生成数据集 ---
    print("正在构建生成任务数据集...")

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

    dataset = build_generation_dataset(args.dataset_path, tokenizer)

    # 格式化数据集
    print("格式化多轮对话数据...")

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

    # --- 加载模型 ---
    print("正在加载模型...")

    if HAS_UNSLOTH:
        # 使用 Unsloth 加载模型
        if args.continue_from_lora:
            # 从已有LoRA加载
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.lora_path,  # 直接从LoRA目录加载
                max_seq_length=args.max_length,
                dtype=None,
                load_in_4bit=args.load_in_4bit,
            )
            print(f"从LoRA适配器加载: {args.lora_path}")
        else:
            # 从基础模型加载
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
            print("配置新的LoRA适配器")

        print("Unsloth 模型和 LoRA 配置完成")
        model.print_trainable_parameters()

    else:
        # 使用标准 Transformers 加载模型
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

        if args.continue_from_lora:
            # 从LoRA适配器加载
            print("从已有LoRA适配器加载模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                base_model,
                args.lora_path,
                torch_dtype=torch.float16,
            )
            print(f"成功加载LoRA适配器: {args.lora_path}")
        else:
            # 从基础模型加载（参考成功的sft_psydt.py模式）
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
    print("初始化 SFTTrainer...")

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

    # --- 开始训练 ---
    print("开始生成任务训练...")
    try:
        trainer.train()
        print("生成任务训练完成！")

        # --- 保存模型 ---
        print(f"保存模型到: {prefixed_output_dir}")

        # 确保输出目录存在
        if not os.path.exists(prefixed_output_dir):
            os.makedirs(prefixed_output_dir, exist_ok=True)

        trainer.save_model(prefixed_output_dir)
        tokenizer.save_pretrained(prefixed_output_dir)

        if hasattr(trainer.model, 'peft_config') and trainer.model.peft_config is not None:
            print("已保存生成任务 LoRA 适配器")
        else:
            print("已保存完整模型")

        print(f"生成任务微调完成！模型保存至: {prefixed_output_dir}")

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()