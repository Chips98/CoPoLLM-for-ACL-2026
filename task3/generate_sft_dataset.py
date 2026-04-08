"""
SFT数据集生成脚本 - 基于真实患者发言数据
1. 从task1/outputs中获取真实患者发言和assistant发言
2. 对于无认知偏差的数据，直接保存原始对话
3. 对于有认知偏差的数据，使用DQN+LLM生成改进的回复
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
import aiohttp

# 导入共用工具
from utils import (
    DQNInference, save_json,
    create_cumulative_test_format, extract_real_answer,
    EMBEDDING_API_CONFIG, generate_improved_response_with_dqn,
    cleanup_llm_session, polish_normal_response,
    build_state_embedding, ACTION_SPACE_MAP, call_llm_api
)



# 配置参数 - 默认值，可通过命令行参数覆盖
TRAINED_MODEL_PATH = "/Users/zl_24/Documents/Codes/CogEmo-Agent/task2/results/10w/dqn_checkpoints_20251119_130151/policy_net_final.pth"
OUTPUT_DIR = "output_real_data_11-28-ablation"

# 输入数据文件路径（使用balanced数据集）
# TRAIN_INPUT_FILE = "/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs_11-19/balanced_D4_train_samples.json"
# TEST_INPUT_FILE = "/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs_11-19/balanced_D4_test_samples.json"

TRAIN_INPUT_FILE = "/Users/zl_24/Documents/Codes/CogEmo-Agent/data/CogBiasESC/CogBiasESC_train.json"
TEST_INPUT_FILE = "/Users/zl_24/Documents/Codes/CogEmo-Agent/data/CogBiasESC/CogBiasESC_test.json"

# 控制参数
POLISH_NORMAL_TURNS = True  # 是否对非偏差样本进行润色和重写（全局默认值，可通过命令行修改）


def load_annotated_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """加载标注数据"""
    print(f"正在读取标注文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"错误：输入文件 {file_path} 未找到。")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果指定了最大样本数，则限制读取数量
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
        print(f"成功读取 {len(data)} 个对话样本（限制前{max_samples}个）")
    else:
        print(f"成功读取 {len(data)} 个对话样本")

    return data

def has_cognitive_bias(annotation: Dict) -> bool:
    """检查是否有认知偏差（适应balanced数据集格式）"""
    if not annotation:
        return False

    # 在balanced数据集中，有认知偏差的样本才有annotation字段
    bias_tags = annotation.get("bias_tags", ["无"])
    # 如果bias_tags包含"无"或者为空，则认为无认知偏差
    if "无" in bias_tags or not bias_tags or bias_tags == ["无"]:
        return False

    return True

async def process_conversation_data(
    conversations: List[Dict],
    dqn_inference: DQNInference,
    session: aiohttp.ClientSession,
    is_train: bool = True,
    api_mode: str = "local",
    batch_size: int = 16  # 减少并发数，避免服务器过载
) -> List[Dict]:
    """处理对话数据，生成改进的SFT数据集 - 支持并发处理"""
    processed_conversations = []
    total_llm_calls = 0
    total_dqn_calls = 0
    start_time = time.time()

    # 创建并发限制器，避免过载
    semaphore = asyncio.Semaphore(16)  # 最多8个并发API调用

    # 分批并发处理（参考task2的模式）
    with tqdm(total=len(conversations), desc="批量处理对话") as pbar:
        for batch_start in range(0, len(conversations), batch_size):
            batch_end = min(batch_start + batch_size, len(conversations))
            batch_conversations = conversations[batch_start:batch_end]

            # 创建并发任务
            conversation_tasks = []
            for conv in batch_conversations:
                task = process_single_conversation(
                    conv, dqn_inference, session, semaphore, is_train, api_mode, POLISH_NORMAL_TURNS
                )
                conversation_tasks.append(task)

            # 并发执行当前批次的所有对话处理
            batch_results = await asyncio.gather(*conversation_tasks, return_exceptions=True)

            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"[错误] 对话处理失败: {result}")
                    continue

                processed_conv, llm_calls, dqn_calls = result
                if processed_conv:  # 确保有效对话
                    processed_conversations.append(processed_conv)
                    total_llm_calls += llm_calls
                    total_dqn_calls += dqn_calls

            # 更新进度条（显示实际处理的样本数）
            pbar.update(len(batch_conversations))

    # 性能统计
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== 性能统计 ===")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"处理对话数: {len(processed_conversations)}")
    print(f"平均每对话: {total_time/len(processed_conversations):.2f}秒")
    print(f"总LLM调用: {total_llm_calls}次")
    print(f"总DQN调用: {total_dqn_calls}次")
    print(f"平均LLM响应时间: {total_time/total_llm_calls:.2f}秒" if total_llm_calls > 0 else "N/A")

    # 统计认知偏差处理情况
    bias_samples = 0
    normal_samples = 0
    for conv in processed_conversations:
        for msg in conv.get("messages", []):
            if msg.get("role") == "user" and "annotation" in msg:
                annotation = msg.get("annotation", {})
                if has_cognitive_bias(annotation):
                    bias_samples += 1
                else:
                    normal_samples += 1

    print(f"\n=== 样本处理统计 ===")
    print(f"有认知偏差样本: {bias_samples}个 (使用DQN+LLM生成)")
    print(f"无认知偏差样本: {normal_samples}个 ({'润色优化' if POLISH_NORMAL_TURNS else '保留原始'})")
    print(f"润色功能状态: {'开启' if POLISH_NORMAL_TURNS else '关闭'}")

    return processed_conversations

async def process_single_conversation(
    conv: Dict,
    dqn_inference: DQNInference,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    is_train: bool = True,
    api_mode: str = "local",
    polish_normal: bool = True
) -> tuple:
    """处理单个对话 - 返回处理结果和调用统计"""
    async with semaphore:  # 限制并发数
        llm_calls = 0
        dqn_calls = 0

        # 处理训练集：保留原有结构
        if is_train:
            processed_conv = {
                "id": conv.get("id", 0),
                "normalizedTag": conv.get("normalizedTag", "心理咨询"),
                "messages": []
            }
        else:
            # 处理测试集：保留所有原始信息
            processed_conv = {
                "id": conv.get("id", 0),
                "normalizedTag": conv.get("normalizedTag", "心理咨询"),
            }
            # 保留portrait和record（如果存在）
            if "portrait" in conv:
                processed_conv["portrait"] = conv["portrait"]
            if "record" in conv:
                processed_conv["record"] = conv["record"]
            processed_conv["messages"] = []

        # 复制system消息
        for msg in conv.get("messages", []):
            if msg.get("role") == "system":
                processed_conv["messages"].append(msg)
                break

        # 简化的顺序处理逻辑，确保user-assistant配对正确
        messages = conv.get("messages", [])

        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                # 添加用户消息
                user_msg = {
                    "role": "user",
                    "content": msg.get("content", "")
                }
                # 保留annotation信息
                if "annotation" in msg:
                    user_msg["annotation"] = msg["annotation"]
                processed_conv["messages"].append(user_msg)

                # 查找对应的assistant回复
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get("role") == "assistant":
                        annotation = msg.get("annotation", {})

                        try:
                            if has_cognitive_bias(annotation):
                                # 有认知偏差：使用DQN+LLM生成改进回复
                                bias_tags = annotation.get("bias_tags", [])
                                bias_intensity = annotation.get("bias_intensity", "无")
                                risk_level = annotation.get("risk_level", "无")
                                patient_content = msg.get("content", "")

                                # 生成改进回复，同时获取策略名称
                                improved_response, strategy_name = await generate_improved_response_with_dqn(
                                    patient_content=patient_content,
                                    bias_tags=bias_tags,
                                    bias_intensity=bias_intensity,
                                    risk_level=risk_level,
                                    dqn_inference=dqn_inference,
                                    session=session,
                                    api_mode=api_mode
                                )
                                llm_calls += 1
                                dqn_calls += 1

                                processed_conv["messages"].append({
                                    "role": "assistant",
                                    "strategy_choce": strategy_name,
                                    "content": improved_response
                                })
                            else:
                                # 无认知偏差：根据参数决定是否润色
                                if polish_normal:
                                    # 使用通用共情策略润色
                                    # 构建简短的上下文用于Prompt
                                    history_msgs = processed_conv["messages"][-3:]
                                    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs])

                                    # 获取对话阶段
                                    stage = annotation.get("dialogue_stage", "一般咨询/信息收集")

                                    # 润色原始回复
                                    improved_response = await polish_normal_response(
                                        patient_content=msg.get("content", ""),
                                        original_doctor_reply=next_msg.get("content", ""),
                                        dialogue_stage=stage,
                                        session=session,
                                        api_mode=api_mode,
                                        history_context=history_context
                                    )
                                    llm_calls += 1

                                    processed_conv["messages"].append({
                                        "role": "assistant",
                                        "content": improved_response
                                    })
                                else:
                                    # 不润色，直接使用原始回复
                                    processed_conv["messages"].append({
                                        "role": "assistant",
                                        "content": next_msg.get("content", "")
                                    })

                        except Exception as e:
                            print(f"[错误] 生成回复失败: {e}")
                            # 降级方案：使用原始回复
                            processed_conv["messages"].append({
                                "role": "assistant",
                                "content": next_msg.get("content", "")
                            })

        # 确保对话至少有一轮完整交互
        user_messages = [msg for msg in processed_conv["messages"] if msg["role"] == "user"]
        assistant_messages = [msg for msg in processed_conv["messages"] if msg["role"] == "assistant"]

        if len(user_messages) > 0 and len(assistant_messages) > 0:
            return processed_conv, llm_calls, dqn_calls
        else:
            return None, llm_calls, dqn_calls

async def process_datasets(max_samples: int = None, api_mode: str = "local"):
    """处理训练集和测试集"""
    print("=== 开始处理基于真实患者发言的SFT数据集 ===")
    print(f"当前API模式: {api_mode}")

    if max_samples:
        print(f"样本数量限制: {max_samples}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化DQN推理器
    dqn_inference = DQNInference(TRAINED_MODEL_PATH)

    # 创建aiohttp会话
    connector = aiohttp.TCPConnector(limit=EMBEDDING_API_CONFIG['max_concurrent'])
    timeout = aiohttp.ClientTimeout(total=EMBEDDING_API_CONFIG['timeout'])

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 处理训练集（整合式格式）
        print("\n--- 处理训练集（整合式） ---")
        train_conversations = load_annotated_data(TRAIN_INPUT_FILE, max_samples)
        if train_conversations:
            processed_train = await process_conversation_data(
                train_conversations, dqn_inference, session, is_train=True, api_mode=api_mode
            )
            print(f"训练集处理完成: {len(processed_train)} 个有效对话")
        else:
            processed_train = []

        # 处理测试集（整合式格式）
        print("\n--- 处理测试集（整合式） ---")
        test_conversations = load_annotated_data(TEST_INPUT_FILE, max_samples)
        if test_conversations:
            processed_test = await process_conversation_data(
                test_conversations, dqn_inference, session, is_train=False, api_mode=api_mode
            )
            print(f"测试集处理完成: {len(processed_test)} 个有效对话")
        else:
            processed_test = []

    await cleanup_llm_session()

    if not processed_train and not processed_test:
        print("[警告] 没有生成任何有效样本")
        return

    # 创建测试集累积式格式（基于原始数据的sample_id）
    cumulative_test = create_cumulative_test_format_with_sample_id(test_conversations, processed_test)

    # 保存数据集
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_file = os.path.join(OUTPUT_DIR, f"sft_real_train_{timestamp}.json")
    test_integral_file = os.path.join(OUTPUT_DIR, f"sft_real_test_integral_{timestamp}.json")  # 整合式
    test_cumulative_file = os.path.join(OUTPUT_DIR, f"sft_real_test_cumulative_{timestamp}.json")  # 累积式

    save_json(processed_train, train_file)
    save_json(processed_test, test_integral_file)  # 保存整合式测试集
    save_json(cumulative_test, test_cumulative_file)  # 保存累积式测试集

    # 统计信息
    print(f"\n=== SFT数据集生成完成 ===")
    print(f"训练集: {len(processed_train)} 个完整对话样本 -> {train_file}")
    print(f"测试集(整合式): {len(processed_test)} 个完整对话样本 -> {test_integral_file}")
    print(f"测试集(累积式): {len(cumulative_test)} 个累积式样本 -> {test_cumulative_file}")

    # 统计咨询类型分布
    if processed_train:
        consultation_types = {}
        for conv in processed_train:
            tag = conv.get("normalizedTag", "未知")
            consultation_types[tag] = consultation_types.get(tag, 0) + 1

        print(f"\n训练集咨询类型分布:")
        for tag, count in sorted(consultation_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tag}: {count}个")

    if cumulative_test:
        test_consultation_types = {}
        for conv in cumulative_test:
            tag = conv.get("normalizedTag", "未知")
            test_consultation_types[tag] = test_consultation_types.get(tag, 0) + 1

        print(f"\n测试集咨询类型分布:")
        for tag, count in sorted(test_consultation_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tag}: {count}个")

    # 统计策略分布（仅统计有认知偏差的样本）
    print(f"\n=== 策略分布统计 ===")

    def analyze_strategy_distribution(conversations: List[Dict], dataset_name: str):
        """分析策略分布"""
        strategy_count = {}
        total_bias_samples = 0

        for conv in conversations:
            for msg in conv.get("messages", []):
                if msg.get("role") == "assistant" and "strategy_choce" in msg:
                    strategy = msg["strategy_choce"]
                    strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
                    total_bias_samples += 1

        if total_bias_samples > 0:
            print(f"\n{dataset_name}策略分布 (总计{total_bias_samples}个有认知偏差样本):")
            for strategy, count in sorted(strategy_count.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_bias_samples) * 100
                print(f"  {strategy}: {count}个 ({percentage:.1f}%)")
        else:
            print(f"\n{dataset_name}无使用策略的样本")

        return strategy_count, total_bias_samples

    # 分析训练集策略分布
    train_strategy_count, train_bias_total = analyze_strategy_distribution(processed_train, "训练集")

    # 分析测试集策略分布
    test_strategy_count, test_bias_total = analyze_strategy_distribution(processed_test, "测试集")

    # 总体策略分布
    total_strategy_count = {}
    total_bias_samples_all = train_bias_total + test_bias_total

    for strategy_dict in [train_strategy_count, test_strategy_count]:
        for strategy, count in strategy_dict.items():
            total_strategy_count[strategy] = total_strategy_count.get(strategy, 0) + count

    if total_bias_samples_all > 0:
        print(f"\n总体策略分布 (总计{total_bias_samples_all}个有认知偏差样本):")
        for strategy, count in sorted(total_strategy_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_bias_samples_all) * 100
            print(f"  {strategy}: {count}个 ({percentage:.1f}%)")

def create_cumulative_test_format_with_sample_id(
    original_test_conversations: List[Dict],
    processed_test_conversations: List[Dict]
) -> List[Dict]:
    """
    创建累积式测试集格式 - 基于原始数据的sample_id
    正确处理sample_id的对应关系，每个对话的sample_id都从0开始
    """
    cumulative_samples = []

    # 处理每个对话
    for i, (orig_conv, proc_conv) in enumerate(zip(original_test_conversations, processed_test_conversations)):
        dialogue_id = orig_conv.get("id", i)

        # 获取处理后的消息
        messages = proc_conv.get("messages", [])
        if len(messages) < 3:  # 至少需要system + user + assistant
            continue

        # 为每个对话创建累积式样本，sample_id从0开始
        current_sample_id = 0

        # 第一轮：system + user + assistant
        if len(messages) >= 3:
            cumulative_sample = {
                "id": dialogue_id,
                "sample_id": current_sample_id,
                "normalizedTag": proc_conv.get("normalizedTag", "心理咨询"),
            }
            # 保留portrait和record
            if "portrait" in proc_conv:
                cumulative_sample["portrait"] = proc_conv["portrait"]
            if "record" in proc_conv:
                cumulative_sample["record"] = proc_conv["record"]
            cumulative_sample["messages"] = messages[:3]  # system + user + assistant
            cumulative_samples.append(cumulative_sample)
            current_sample_id += 1

        # 后续轮：每一轮增加user + assistant
        for i in range(3, len(messages), 2):  # 从第2个user消息开始
            if i + 1 < len(messages):  # 确保有完整的user + assistant轮次
                # 截取从开始到当前assistant回复的所有消息
                cumulative_messages = messages[:i+2]

                cumulative_sample = {
                    "id": dialogue_id,
                    "sample_id": current_sample_id,
                    "normalizedTag": proc_conv.get("normalizedTag", "心理咨询"),
                }
                # 保留portrait和record
                if "portrait" in proc_conv:
                    cumulative_sample["portrait"] = proc_conv["portrait"]
                if "record" in proc_conv:
                    cumulative_sample["record"] = proc_conv["record"]
                cumulative_sample["messages"] = cumulative_messages

                cumulative_samples.append(cumulative_sample)
                current_sample_id += 1

    print(f"累积式测试集生成完成: {len(cumulative_samples)} 个样本")
    return cumulative_samples


'''

'''
if __name__ == "__main__":
    import argparse
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description="生成基于真实患者发言的SFT数据集")
    parser.add_argument(
        "--max_samples", "-n",
        type=int,
        default=None,
        help="要处理的样本数量；不指定则处理全部样本"
    )
    parser.add_argument(
        "--api-mode",
        type=str,
        choices=["local", "dashscope", "openrouter"],
        default="local",
        help="API模式选择：local=本地VLLM，dashscope=阿里云通义千问，openrouter=OpenRouter"
    )
    parser.add_argument(
        "--no-polish-normal",
        action="store_true",
        help="关闭非偏差样本的润色功能，仅对有认知偏差的样本进行DQN生成"
    )
    # 添加四个新的命令行参数
    parser.add_argument(
        "--model-path",
        type=str,
        default=TRAINED_MODEL_PATH,
        help=f"训练好的DQN模型路径 (默认: {TRAINED_MODEL_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"输出目录路径 (默认: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=TRAIN_INPUT_FILE,
        help=f"训练集输入文件路径 (默认: {TRAIN_INPUT_FILE})"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=TEST_INPUT_FILE,
        help=f"测试集输入文件路径 (默认: {TEST_INPUT_FILE})"
    )
    args = parser.parse_args()

    # 更新全局参数 - 使用命令行参数覆盖默认值
    POLISH_NORMAL_TURNS = not args.no_polish_normal
    TRAINED_MODEL_PATH = args.model_path
    OUTPUT_DIR = args.output_dir
    TRAIN_INPUT_FILE = args.train_file
    TEST_INPUT_FILE = args.test_file

    print(f"\n=== 配置信息 ===")
    print(f"DQN模型路径: {TRAINED_MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"训练集文件: {TRAIN_INPUT_FILE}")
    print(f"测试集文件: {TEST_INPUT_FILE}")
    print(f"非偏差样本润色: {'开启' if POLISH_NORMAL_TURNS else '关闭'}")
    print(f"API模式: {args.api_mode}")
    print(f"最大样本数: {args.max_samples if args.max_samples else '全部'}")
    print(f"================")

    asyncio.run(process_datasets(max_samples=args.max_samples, api_mode=args.api_mode))