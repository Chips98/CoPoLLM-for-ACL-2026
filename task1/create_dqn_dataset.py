"""
脚本2：DQN 数据集转换脚本
功能：读取 脚本1 (annotate_bias_chain.py) 输出的 D4_annotated.jsonl，
      使用 嵌入API 进行编码，并创建 dqn.py 所需的 S_pool.jsonl。
"""
import json
import os
import numpy as np
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
from collections import defaultdict
import time
from typing import Dict, List, Any

# --- 1. 配置 ---

# (来自标注脚本的输出)
INPUT_FILE = "/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs/D4_test_annotated-qwen3-8b.json"

#我们想要将三个数据集的json文件一并处理INPUT_FILE_LIST,最后生成一个训练集
INPUT_FILE_LIST = ["/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs_11-19/CPsyCounD_train_annotated-local.json","/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs_11-19/D4_train_annotated-local.json","/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs_11-19/PsyDTCorpus_train_annotated-local.json"]


# (将要喂给 dqn.py 的输入)
OUTPUT_FILE = "/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs/Total_S_train_pool.jsonl"

# 嵌入API配置
EMBEDDING_API_CONFIG = {
    'api_base': 'http://localhost:6862/v1',
    'model_name': 'Qwen3-Embedding-0.6B',
    'timeout': 30,
    'max_concurrent': 32,  # 异步并发数
    'api_key': 'dummy-key'  # VLLM通常不需要真实API key
}

# 固定的偏差列表 (用于t向量)
# 必须与 脚本1 的 get_prompt 严格对应
BIAS_LIST = ["非黑即白", "过度概括", "灾难化", "读心术", "情感推理", "应该句式", "标签化", "个人化"]

# --- 2. 文本到数值的映射 (用于 s_text) ---

# (这些映射用于构建 s_text，SBERT 将从这些词汇中理解语义)
RISK_MAP = {"高危": "高危", "中危": "中危", "低危": "低危", "无": "无风险", "未知": "未知风险"}
STAGE_MAP = {
    "1. 开场": "开场",
    "2. 问题澄清": "问题澄清",
    "3. 情绪疏导": "情绪疏导",
    "4. 策略制定": "策略制定",
    "5. 收尾": "收尾",
    "未知": "未知阶段"
}
INTENSITY_MAP = {"严重": "严重", "中等": "中等", "轻微": "轻微", "无": "无", "未知": "未知强度"}


# --- 3. 核心功能 ---

async def get_embedding_from_text(text: str, session: aiohttp.ClientSession) -> np.ndarray:
    """
    异步获取文本嵌入向量
    """
    data = {
        "model": EMBEDDING_API_CONFIG["model_name"],
        "input": text,  # 嵌入API通常使用input字段
        "encoding_format": "float"
    }

    try:
        headers = {"Authorization": f"Bearer {EMBEDDING_API_CONFIG['api_key']}"}
        async with session.post(
            f"{EMBEDDING_API_CONFIG['api_base']}/embeddings",
            json=data,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=EMBEDDING_API_CONFIG['timeout'])
        ) as response:
            if response.status == 200:
                result = await response.json()
                if 'data' in result and len(result['data']) > 0 and 'embedding' in result['data'][0]:
                    embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)
                    return embedding
                else:
                    print(f"❌ 嵌入API返回格式错误: {result}")
                    return None
            else:
                error_text = await response.text()
                print(f"❌ 嵌入API请求失败，状态码: {response.status}, 错误: {error_text}")
                return None
    except Exception as e:
        print(f"❌ 嵌入API请求异常: {e}")
        return None

def load_annotated_data(path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    加载单个标注数据文件，按 dialogue_id 重新组织，并过滤无认知偏差的样本
    """
    if not os.path.exists(path):
        print(f"错误：输入文件 {path} 未找到。")
        print("请先运行 标注脚本 来生成此文件。")
        exit(1)

    dialogues = defaultdict(list)
    total_samples = 0
    skipped_samples = 0

    print(f"正在读取已标注文件: {path}")

    # 读取JSON数组格式的文件
    with open(path, 'r', encoding='utf-8') as f:
        dialogues_data = json.load(f)

    for dialogue in dialogues_data:
        dialogue_id = dialogue.get("id", 0)

        for turn_id, message in enumerate(dialogue.get("messages", [])):
            # 只处理用户发言进行统计
            if message.get("role") != "user":
                continue

            total_samples += 1

            # 检查是否有认知偏差标注
            annotation = message.get("annotation")
            if not annotation:
                continue

            bias_tags = annotation.get("bias_tags", ["无"])
            bias_intensity = annotation.get("bias_intensity", "无")

            # 使用更宽松的过滤条件：只过滤标签为["无"]且强度也为"无"的样本
            # 这与标注脚本的统计逻辑保持一致
            if (bias_tags == ["无"] or not bias_tags) and bias_intensity == "无":
                skipped_samples += 1
                continue

            # 构造数据结构
            data = {
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "role": message["role"],
                "content": message["content"],
                "annotation": annotation
            }

            dialogues[dialogue_id].append(data)

    # 按 turn_id 排序，确保顺序正确
    sorted_dialogues = {}
    for dia_id, msgs in dialogues.items():
        sorted_dialogues[dia_id] = sorted(msgs, key=lambda x: x["turn_id"])

    print(f"加载并分组了 {len(sorted_dialogues)} 场对话。")
    print(f"总样本数: {total_samples}, 跳过无认知偏差样本: {skipped_samples}, 保留样本数: {total_samples - skipped_samples}")
    return sorted_dialogues

def load_multiple_annotated_files(file_paths: List[str]) -> Dict[int, List[Dict[str, Any]]]:
    """
    加载多个标注数据文件并合并，按 dialogue_id 重新组织，并过滤无认知偏差的样本
    """
    all_dialogues = defaultdict(list)
    total_files = len(file_paths)
    total_samples = 0
    total_skipped = 0

    print(f"开始加载 {total_files} 个文件...")

    for file_idx, file_path in enumerate(file_paths, 1):
        print(f"\n--- 处理第 {file_idx}/{total_files} 个文件 ---")

        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，跳过")
            continue

        try:
            # 读取单个文件
            with open(file_path, 'r', encoding='utf-8') as f:
                dialogues_data = json.load(f)

            file_samples = 0
            file_skipped = 0

            for dialogue in dialogues_data:
                # 使用文件名和索引来确保dialogue_id不重复
                dialogue_id = int(f"{file_idx}{dialogue.get('id', 0):04d}")

                for turn_id, message in enumerate(dialogue.get("messages", [])):
                    # 只处理用户发言
                    if message.get("role") != "user":
                        continue

                    file_samples += 1
                    total_samples += 1

                    # 检查是否有认知偏差标注
                    annotation = message.get("annotation")
                    if not annotation:
                        continue

                    bias_tags = annotation.get("bias_tags", ["无"])
                    bias_intensity = annotation.get("bias_intensity", "无")

                    # 过滤条件：只过滤标签为["无"]且强度也为"无"的样本
                    if (bias_tags == ["无"] or not bias_tags) and bias_intensity == "无":
                        file_skipped += 1
                        total_skipped += 1
                        continue

                    # 构造数据结构
                    data = {
                        "dialogue_id": dialogue_id,
                        "turn_id": turn_id,
                        "role": message["role"],
                        "content": message["content"],
                        "annotation": annotation
                    }

                    all_dialogues[dialogue_id].append(data)

            print(f"文件 {os.path.basename(file_path)} 完成: 样本 {file_samples}, 跳过 {file_skipped}, 保留 {file_samples - file_skipped}")

        except Exception as e:
            print(f"错误：处理文件 {file_path} 时出错: {e}")
            continue

    # 按 turn_id 排序所有对话
    sorted_dialogues = {}
    for dia_id, msgs in all_dialogues.items():
        sorted_dialogues[dia_id] = sorted(msgs, key=lambda x: x["turn_id"])

    print(f"\n--- 所有文件加载完成 ---")
    print(f"总共处理 {len(sorted_dialogues)} 场对话")
    print(f"总样本数: {total_samples}, 总跳过: {total_skipped}, 总保留: {total_samples - total_skipped}")

    return sorted_dialogues

async def process_batch(batch_data: List[Dict], session: aiohttp.ClientSession) -> List[Dict]:
    """
    批量处理状态，获取嵌入向量
    """
    tasks = []
    for data in batch_data:
        task = get_embedding_from_text(data['s_text'], session)
        tasks.append((task, data))

    results = []
    completed_tasks = await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)

    for (task, data), embedding in zip(tasks, completed_tasks):
        if isinstance(embedding, Exception):
            print(f"❌ 嵌入失败: {embedding}")
            continue

        if embedding is not None:
            output_data = {
                "embedding": embedding.tolist(),
                "context": data['context'],
                "s_text": data['s_text']  # 修复：添加缺失的s_text字段
            }
            results.append(output_data)

    return results

async def process_dialogues_async(dialogues: Dict[int, List[Dict[str, Any]]], output_path: str):
    """
    异步处理所有对话，计算偏差链，生成嵌入，并写入 S_pool.jsonl
    """
    total_states_saved = 0

    # 收集所有需要编码的状态
    states_to_encode = []

    print(f"开始处理 {len(dialogues)} 场对话...")
    for dia_id, messages in tqdm(dialogues.items(), desc="准备数据"):
        current_chain_length = 0
        history_context_list = []

        for msg in messages:
            # 只处理患者('user')的发言
            if msg["role"] != "user":
                history_context_list.append(f"医生: {msg['content']}")
                continue

            # --- 1. 计算偏差链 l ---
            annotation = msg["annotation"]
            bias_tags = annotation.get("bias_tags", ["无"])

            if "无" in bias_tags or not bias_tags:
                current_chain_length = 0  # 偏差链中断
            else:
                current_chain_length += 1  # 偏差链延续

            # --- 2. 构建状态描述句 (s_text) ---
            # 获取各个字段
            patient_content = msg["content"]
            bias_content = ','.join(bias_tags)
            intensity_content = annotation.get("bias_intensity", "无")
            risk_content = annotation.get("risk_level", "无")
            reason_content = annotation.get("reason", "")

            s_text = f"患者当前发言：{patient_content}\n" \
                     f"认知偏差：{bias_content}\n" \
                     f"偏差强度：{intensity_content}\n" \
                     f"风险等级：{risk_content}\n" \
                     f"理由：{reason_content}"

            # --- 3. 构建历史上下文 (context) ---
            current_context_str = "\n".join(history_context_list)
            if not current_context_str:
                current_context_str = "无对话历史。"

            # --- 4. 保存需要编码的状态 ---
            states_to_encode.append({
                's_text': s_text,
                'context': current_context_str,
                'chain_length': current_chain_length
            })

            # (重要) 将当前发言加入历史，供下一轮使用
            history_context_list.append(f"患者: {msg['content']}")

    print(f"总共收集了 {len(states_to_encode)} 个状态需要编码")

    # --- 5. 异步并发编码 ---
    connector = aiohttp.TCPConnector(limit=EMBEDDING_API_CONFIG['max_concurrent'])
    timeout = aiohttp.ClientTimeout(total=EMBEDDING_API_CONFIG['timeout'])

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 分批处理，避免内存过大
        batch_size = EMBEDDING_API_CONFIG['max_concurrent']
        batches = [states_to_encode[i:i+batch_size] for i in range(0, len(states_to_encode), batch_size)]

        print(f"开始异步编码，共 {len(batches)} 批次，每批最多 {batch_size} 个状态")

        # 准备写入 S_pool.jsonl
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for batch_idx, batch in enumerate(tqdm(batches, desc="编码批次")):
                results = await process_batch(batch, session)

                # 写入结果
                for result in results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    total_states_saved += 1

                print(f"批次 {batch_idx + 1}/{len(batches)} 完成，保存 {len(results)} 个状态")

    print("\n--- 数据集转换完成 ---")
    print(f"总共保存了 {total_states_saved} 个有效状态到 {output_path}")

# --- 4. 主程序入口 ---

if __name__ == "__main__":
    print("=== DQN 数据集转换脚本 (嵌入API编码) ===")

    # 1. 加载多个标注数据文件
    # 使用INPUT_FILE_LIST中的所有文件路径
    dialogues_data = load_multiple_annotated_files(INPUT_FILE_LIST)

    # 2. 异步处理并保存
    if dialogues_data:
        asyncio.run(process_dialogues_async(dialogues_data, OUTPUT_FILE))
    else:
        print("没有加载到数据，程序退出。")