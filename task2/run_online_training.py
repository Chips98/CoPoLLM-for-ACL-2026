"""
主训练脚本：DQN 在线训练循环 (BG-DSS 3.3.2节)
[优化] 版本：DDQN + KL散度约束 + 详细损失日志
"""
import torch
import torch.optim as optim
import random
import json
import numpy as np
import asyncio
import re
import os
from openai import AsyncOpenAI
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import csv

# 实验数据记录辅助函数
def parse_state_text(text):
    """从文本提取特征用于实验分析"""
    # 假设文本格式: "认知偏差：灾难化; 危险等级：低危; 偏差强度：严重..."
    bias = re.search(r"认知偏差[：:]\s*([^;]+)", text)
    risk = re.search(r"危险等级[：:]\s*([^;]+)", text)
    intensity = re.search(r"偏差强度[：:]\s*([^;]+)", text)

    return (
        bias.group(1).strip() if bias else "未知",
        risk.group(1).strip() if risk else "未知",
        intensity.group(1).strip() if intensity else "未知"
    )

def get_q_values_for_logging(state_embedding, policy_net):
    """获取当前状态的所有Q值用于记录"""
    with torch.no_grad():
        state_tensor = torch.tensor(state_embedding, dtype=torch.float).unsqueeze(0).to(device)
        q_values = policy_net(state_tensor)
        return q_values.cpu().numpy()[0].tolist()

def log_experiment_data(log_file, episode, state_text, action_index, reward, q_values):
    """记录实验数据到CSV文件"""
    bias_type, risk_level, intensity = parse_state_text(state_text)

    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([episode, bias_type, intensity, risk_level, action_index, reward, json.dumps(q_values)])

# --- 1. 导入DQN核心组件 (从 dqn.py) ---
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dqn import (
        QNetwork, ReplayBuffer, Experience,
        optimize_model, select_action,
        EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, ACTION_DIM,
        BATCH_SIZE, GAMMA, LEARNING_RATE,
        TARGET_UPDATE_INTERVAL, REPLAY_BUFFER_CAPACITY,NUM_PARALLEL_ENVIRONMENTS,CHECKPOINT_NUM,
        K_EPISODES, device,
        KL_BETA, KL_TEMP,  # [新增] 导入KL参数
        # [断点训练配置]
        RESUME_TRAINING, CHECKPOINT_MODEL_PATH, CHECKPOINT_LOSS_LOG_PATH,
        CHECKPOINT_EXP_LOG_PATH, RESUME_EPISODE_COUNT,
        load_checkpoint, append_to_loss_log, append_to_exp_log
    )
    print("成功从 dqn.py 导入DQN组件。")
    print(f"[配置] DDQN + KL (Beta={KL_BETA}, Temp={KL_TEMP})")
except ImportError as e:
    print(f"错误：无法从 dqn.py 导入。导入异常: {e}")
    print("请确保 dqn.py 在同一目录或Python路径中。")
    exit(1)


# --- 2. 配置 (嵌入API, LLM, 动作空间) ---

# 配置加载器支持
try:
    from config_loader import get_config
    # 加载配置文件
    cp_config = get_config()

    # 使用YAML配置，如果加载失败则使用原有默认值
    embedding_config = cp_config.embedding_api_config
    llm_config = cp_config.llm_api_config

    print("✅ run_online_training.py 成功加载YAML配置文件")
    print(f"🔗 嵌入API: {embedding_config['model_name']} @ {embedding_config['api_base']}")
    print(f"🤖 LLM API: {llm_config['model_name']} @ {llm_config['api_base']}")

except ImportError as e:
    print(f"⚠️  无法加载配置文件，使用默认API配置: {e}")
    # 使用原有的默认API配置（向后兼容）
    embedding_config = {
        "api_base": "http://localhost:6862/v1",
        "model_name": "Qwen3-Embedding-0.6B",
        "api_key": "dummy-key",
        "timeout": 30,
        "max_concurrent": 32,
    }
    llm_config = {
        "model_name": "Qwen3-8B",
        "base_url": "http://localhost:7862",
        "api_key": "dummy-key",
        "temperature": 0.7,
        "max_tokens": 512,
        "timeout": 30,
        "max_concurrent": 32
    }

# 标准化配置格式以保持兼容性
EMBEDDING_API_CONFIG = {
    "api_base": embedding_config["api_base"],
    "model_name": embedding_config["model_name"],
    "api_key": embedding_config["api_key"],
    "timeout": embedding_config.get("timeout", 30),
    "max_concurrent": embedding_config.get("max_concurrent", 32),
    "vector_dimension": 1024  # 嵌入向量维度
}

LOCAL_API_CONFIG = {
    "model": llm_config["model_name"],
    "base_url": llm_config["api_base"],  # 修正字段名匹配
    "api_key": llm_config["api_key"],
    "temperature": llm_config.get("temperature", 0.7),
    "max_tokens": llm_config.get("max_tokens", 512),
    "timeout": llm_config.get("timeout", 30),
    "max_concurrent": llm_config.get("max_concurrent", 32)
}


# --- 3. 多智能体系统和辅助函数 ---
# 导入多智能体系统
from multi_agents import MultiAgentSystem

import aiohttp
async_session = None

async def get_llm_session():
    global async_session
    if async_session is None:
        async_session = aiohttp.ClientSession()
    return async_session

async def cleanup_llm_session():
    global async_session
    if async_session:
        await async_session.close()
        async_session = None

async def call_llm_api(messages: list) -> str:
    """LLM API调用函数"""
    try:
        base_url = LOCAL_API_CONFIG["base_url"].rstrip('/')
        if base_url.endswith('/v1'):
            # 如果base_url以/v1结尾，添加/chat/completions
            url = f"{base_url}/chat/completions"
        elif not base_url.endswith('/v1/chat/completions'):
            # 如果base_url不以/v1/chat/completions结尾，添加完整路径
            url = f"{base_url}/v1/chat/completions"
        else:
            # 如果已经以/v1/chat/completions结尾，直接使用
            url = base_url
        data = { "model": LOCAL_API_CONFIG["model"], "messages": messages, "temperature": LOCAL_API_CONFIG["temperature"], "max_tokens": LOCAL_API_CONFIG["max_tokens"] }
        headers = {"Content-Type": "application/json"}

        # 添加Authorization头部（如果有的话）
        if LOCAL_API_CONFIG.get("api_key") and LOCAL_API_CONFIG["api_key"] != "None":
            headers["Authorization"] = f"Bearer {LOCAL_API_CONFIG['api_key']}"
        session = await get_llm_session()
        async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=LOCAL_API_CONFIG["timeout"])) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"VLLM API请求失败，状态码: {response.status}, 错误信息: {error_text}")
            response_data = await response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                choice = response_data['choices'][0]; result = choice['message'].get('content', '').strip()
            else: result = ""
            return result
    except Exception as e: return f"LLM_ERROR: {e}"

def load_embedding_api_client():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter); session.mount("https://", adapter)
    session.headers.update({"Authorization": f"Bearer {EMBEDDING_API_CONFIG['api_key']}", "Content-Type": "application/json"})
    return session

def load_initial_states(path):
    print(f"Loading initial states from {path}...")
    s_pool = []
    try:
        # 直接使用传入的唯一路径
        if not os.path.exists(path):
            print(f"[错误] 数据文件未找到: {path}")
            raise FileNotFoundError("数据文件未找到")

        print(f"使用数据文件: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # 检查必需字段
                    if "embedding" not in data:
                        print(f"[警告] 第{line_num}行缺少embedding字段，跳过")
                        continue

                    if "context" not in data:
                        print(f"[警告] 第{line_num}行缺少context字段，使用默认值")
                        context = "无对话历史。"
                    else:
                        context = data["context"]

                    if "s_text" not in data:
                        print(f"[警告] 第{line_num}行缺少s_text字段，使用默认值")
                        s_text = "认知偏差：无"
                    else:
                        s_text = data["s_text"]

                    s_pool.append({
                        "embedding": data["embedding"],
                        "context": context,
                        "s_text": s_text
                    })

                    if len(s_pool) >= 1000:  # 限制加载数量以提高启动速度
                        break

                except json.JSONDecodeError as e:
                    print(f"[警告] 第{line_num}行JSON解析错误，跳过: {e}")
                    continue
                except Exception as e:
                    print(f"[警告] 第{line_num}行处理错误，跳过: {e}")
                    continue

        print(f"Loaded {len(s_pool)} initial states.")
        if not s_pool:
            raise Exception("S_pool is empty!")
        return s_pool
    except Exception as e:
        print(f"[错误] 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        print("使用默认模拟数据...")
        return [{"embedding": np.random.rand(EMBEDDING_DIM).tolist(), "context": "你好", "s_text": "认知偏差：无"}]

# --- 4. 辅助函数 ---
def get_embedding_from_text(text: str, embedding_client) -> np.ndarray:
    """从文本获取embedding向量"""
    try:
        data = {"model": EMBEDDING_API_CONFIG["model_name"], "messages": [{"role": "user", "content": text}], "encoding_format": "float"}
        response = embedding_client.post(f"{EMBEDDING_API_CONFIG['api_base']}/embeddings", json=data, timeout=EMBEDDING_API_CONFIG["timeout"])
        if response.status_code == 200:
            embedding = response.json()["data"][0]["embedding"]
            return np.array(embedding)
        else:
            return np.zeros(EMBEDDING_API_CONFIG["vector_dimension"])
    except Exception as e:
        return np.zeros(EMBEDDING_API_CONFIG["vector_dimension"])

async def get_embedding_from_text_async(text: str, embedding_client) -> np.ndarray:
    """异步获取embedding向量"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embedding_from_text, text, embedding_client)


# --- 5. 主训练循环 (异步) ---

async def vectorized_environment_step(env_states, policy_net, embedding_client, S_pool, multi_agent_system, experiment_log_file=None, episode_counter=0):
    """
    重构的环境交互函数：使用多智能体系统完成完整的训练闭环

    Args:
        env_states: 环境状态列表
        policy_net: DQN策略网络
        embedding_client: embedding API客户端
        S_pool: 初始状态池
        multi_agent_system: 多智能体系统实例
        experiment_log_file: 实验数据记录文件路径
        episode_counter: 当前训练迭代计数器

    Returns:
        批量经验列表
    """
    batch_states = []
    batch_contexts = []
    batch_texts = []
    batch_actions = []

    # 选择初始状态和动作
    for env_id in range(len(env_states)):
        state_data = random.choice(S_pool)
        s_embedding = state_data["embedding"]
        s_context = state_data["context"]
        s_text = state_data["s_text"]

        # DQN选择动作
        action_tensor = select_action(s_embedding, policy_net)
        action_index = action_tensor.item()

        # 记录实验数据（获取所有Q值）
        if experiment_log_file is not None:
            q_values = get_q_values_for_logging(s_embedding, policy_net)
            # 暂时记录奖励为0，后续在获得实际奖励后更新
            log_experiment_data(experiment_log_file, episode_counter + env_id, s_text, action_index, 0.0, q_values)

        batch_states.append(s_embedding)
        batch_contexts.append(s_context)
        batch_texts.append(s_text)
        batch_actions.append(action_index)

    # 提取患者当前发言（从S_pool中获取）
    patient_current_utterances = []
    for i in range(len(batch_states)):
        # 从上下文中提取患者发言
        utterance = "你好"  # 默认值
        if batch_contexts[i]:
            lines = batch_contexts[i].strip().split('\n')
            for line in reversed(lines):
                if line.startswith('患者:') or line.startswith('患者：'):
                    utterance = line.split(':', 1)[1].strip()
                    break
        patient_current_utterances.append(utterance)

    # 使用多智能体系统进行完整的交互
    interaction_tasks = []
    for i in range(len(batch_states)):
        task = multi_agent_system.full_interaction_step(
            batch_contexts[i], batch_texts[i], batch_actions[i], patient_current_utterances[i]
        )
        interaction_tasks.append(task)

    # 并行执行所有交互任务
    interaction_results = await asyncio.gather(*interaction_tasks, return_exceptions=True)

    # 处理交互结果，准备embedding
    embedding_tasks = []
    batch_experiences = []

    for i in range(len(batch_states)):
        if isinstance(interaction_results[i], Exception):
            # 错误处理：使用默认值
            batch_rewards = 0.0
            s_prime_text = "认知偏差：无; 危险等级：无; 对话阶段：未知; 偏差强度：无; 偏差链长：0"
        else:
            # 解析正常的交互结果
            doctor_response, patient_reply, reward_score = interaction_results[i]
            batch_rewards = reward_score
            # 简化状态文本生成（不再需要分析智能体）
            s_prime_text = f"认知偏差：未评估; 危险等级：低危; 对话阶段：策略制定; 偏差强度：轻微; 偏差链长：1"

        # 获取新状态的embedding
        task = get_embedding_from_text_async(s_prime_text, embedding_client)
        embedding_tasks.append(task)

        # 暂时保存信息，等embedding计算完成后再构建完整经验
        batch_experiences.append({
            's_embedding': batch_states[i],
            'action_index': batch_actions[i],
            'R': batch_rewards,
            's_prime_text': s_prime_text,
            'env_id': i,
            'interaction_result': interaction_results[i] if not isinstance(interaction_results[i], Exception) else None
        })

    # 等待所有embedding计算完成
    prime_embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)

    # 构建最终的经验列表
    final_experiences = []
    for i, experience in enumerate(batch_experiences):
        if isinstance(prime_embeddings[i], Exception):
            prime_embedding = np.zeros(EMBEDDING_API_CONFIG["vector_dimension"])
        else:
            prime_embedding = prime_embeddings[i]

        final_experience = {
            's_embedding': experience['s_embedding'],
            'action_index': experience['action_index'],
            'R': experience['R'],
            's_prime_embedding': prime_embedding,
            'env_id': experience['env_id'],
            'interaction_result': experience['interaction_result']
        }
        final_experiences.append(final_experience)

    return final_experiences

# 验证集相关功能已删除，专注于训练本身

# --- [重大修改] 主训练循环 (Vectorized + 详细日志) ---

async def main_vectorized_training_loop():
    """
    真正的并行环境训练循环
    [修改] 增加了对 (total_loss, loss_q, loss_kl) 的详细日志记录
    [新增] 支持断点训练功能
    """
    print("[INFO] 开始并行环境训练主循环 (DDQN + KL + 验证)...")

    # 0. 断点训练检查
    start_experience = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if RESUME_TRAINING and CHECKPOINT_MODEL_PATH:
        # 如果是断点训练，使用指定的日志文件
        loss_log_file = CHECKPOINT_LOSS_LOG_PATH if CHECKPOINT_LOSS_LOG_PATH else f"training_loss_vectorized_{timestamp}.txt"
        experiment_log_file = CHECKPOINT_EXP_LOG_PATH if CHECKPOINT_EXP_LOG_PATH else f"experiment_data_{timestamp}.csv"

        # 加载checkpoint获取起始点
        checkpoint = load_checkpoint(CHECKPOINT_MODEL_PATH)
        if checkpoint:
            start_experience = checkpoint.get('experiences', 0)
            print(f"[INFO] 断点训练模式：从experience {start_experience}开始续训练")
            # 如果日志文件不存在，创建新文件并写入头部
            if not os.path.exists(loss_log_file):
                with open(loss_log_file, 'w', encoding='utf-8') as f:
                    f.write("# DQN并行环境训练损失记录 (DDQN + KL)\n")
                    f.write(f"# 断点训练续写: 从experience {start_experience} 开始\n")
                    f.write("# 格式: 映射Episode编号, 总损失, Q损失, KL损失, 平均奖励, 时间戳\n")
                    f.write("episode_approx,total_loss,loss_q,loss_kl,avg_reward,timestamp\n")
            # 如果实验日志文件不存在，创建新文件并写入头部
            if not os.path.exists(experiment_log_file):
                with open(experiment_log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['episode', 'bias_type', 'intensity', 'risk_level', 'selected_action', 'reward', 'q_values'])
        else:
            print("[警告] Checkpoint加载失败，使用新的日志文件")
            loss_log_file = f"training_loss_vectorized_{timestamp}.txt"
            experiment_log_file = f"experiment_data_{timestamp}.csv"
    else:
        # 正常训练模式，创建新的日志文件
        loss_log_file = f"training_loss_vectorized_{timestamp}.txt"
        experiment_log_file = f"experiment_data_{timestamp}.csv"

        # 写入CSV头
        with open(experiment_log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'bias_type', 'intensity', 'risk_level', 'selected_action', 'reward', 'q_values'])

        # 写入训练日志文件头
        with open(loss_log_file, 'w', encoding='utf-8') as f:
            f.write("# DQN并行环境训练损失记录 (DDQN + KL)\n")
            f.write(f"# 总训练迭代: {K_EPISODES}, 并行环境数: {NUM_PARALLEL_ENVIRONMENTS}\n")
            f.write("# 格式: 映射Episode编号, 总损失, Q损失, KL损失, 平均奖励, 时间戳\n")
            f.write("# 注意: Episode编号 = batch * NUM_PARALLEL_ENVIRONMENTS，用于反映真实训练进度\n")
            f.write("episode_approx,total_loss,loss_q,loss_kl,avg_reward,timestamp\n")

    # [修改] 记录所有损失
    loss_total_history = []
    loss_q_history = []
    loss_kl_history = []

    print(f"[INFO] 损失记录文件: {loss_log_file}")
    print(f"[INFO] 实验数据记录文件: {experiment_log_file}")

    # 2. 初始化模型（支持断点训练）
    print("[INFO] 初始化DQN模型...")
    policy_net = QNetwork(EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, ACTION_DIM).to(device)
    target_net = QNetwork(EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, ACTION_DIM).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    # [断点训练] 加载checkpoint
    if RESUME_TRAINING and CHECKPOINT_MODEL_PATH:
        checkpoint = load_checkpoint(CHECKPOINT_MODEL_PATH)
        if checkpoint:
            # 加载模型和优化器状态
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            target_net.load_state_dict(checkpoint.get('target_net_state_dict', policy_net.state_dict()))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_experience = checkpoint.get('experiences', 0)
            print(f"[INFO] 成功加载checkpoint: 已训练 {start_experience} experiences")
        else:
            print("[警告] Checkpoint加载失败，从头开始训练")
            start_experience = 0

    model_save_dir = f"dqn_checkpoints_{timestamp}"
    os.makedirs(model_save_dir, exist_ok=True)
    initial_model_path = os.path.join(model_save_dir, "policy_net_initial.pth")

    # 保存初始模型状态（如果是断点训练，也保存一个备份）
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'experiences': start_experience,
        'batch': 0,
        'timestamp': timestamp
    }, initial_model_path)


    # 3. 初始化多智能体系统
    print("[INFO] 初始化多智能体系统...")
    multi_agent_system = MultiAgentSystem(call_llm_api)

    # 4. 加载客户端和数据
    embedding_client = load_embedding_api_client()
    test_embedding = get_embedding_from_text("测试文本", embedding_client)
    # 使用统一的数据路径（从dqn.py导入）
    from dqn import INPUT_PATH
    S_pool = load_initial_states(INPUT_PATH)

    # 测试LLM和多智能体系统
    test_messages = [{"role": "user", "content": "你好，连接测试"}]
    test_response = await call_llm_api(test_messages)
    if "LLM_ERROR" in test_response:
        print(f"[ERROR] LLM连接测试失败: {test_response}")
        return

    print("[INFO] 多智能体系统初始化成功")

    # 5. 并行环境配置
    UPDATE_TARGET_EVERY_N_BATCHES = TARGET_UPDATE_INTERVAL

    print(f"[INFO] 并行环境配置: 环境数={NUM_PARALLEL_ENVIRONMENTS}, 批量大小={BATCH_SIZE}")
    print(f"[INFO] 目标网络更新: 每 {UPDATE_TARGET_EVERY_N_BATCHES} 批次更新一次")
    print(f"[INFO] 模型检查点保存: 每 {CHECKPOINT_NUM} 批次保存一次")
    print(f"[INFO] 使用新的多智能体架构 (医生+患者+评估者)")
    print(f"--- 开始并行环境训练 {K_EPISODES} 迭代 ---")

    # 6. 并行环境训练循环（支持断点训练）
    if RESUME_TRAINING:
        # 断点训练：从checkpoint的experiences开始计算目标
        target_experiences = start_experience + K_EPISODES
        current_experiences = start_experience
        pbar = tqdm(total=target_experiences, initial=start_experience, desc=f"DQN 多智能体并行训练 (续训)")
    else:
        # 正常训练：从0开始
        target_experiences = K_EPISODES
        current_experiences = 0
        pbar = tqdm(total=K_EPISODES, desc="DQN 多智能体并行训练")

    total_experiences = current_experiences
    batch_count = 0

    print(f"[INFO] 训练目标: 从 {current_experiences} 到 {target_experiences}")

    try:
        while total_experiences < target_experiences:
            env_states = [None] * NUM_PARALLEL_ENVIRONMENTS
            batch_experiences = await vectorized_environment_step(
                env_states, policy_net, embedding_client, S_pool, multi_agent_system, experiment_log_file, total_experiences
            )

            for exp in batch_experiences:
                replay_buffer.push(exp['s_embedding'], exp['action_index'], exp['R'], exp['s_prime_embedding'])
                total_experiences += 1


            # 🔥 批量训练
            if len(replay_buffer) >= BATCH_SIZE:
                # 每个环境步骤触发一次或多次训练
                num_training_steps = min(4, len(batch_experiences) // 8) # 适度训练
                if num_training_steps == 0: num_training_steps = 1
                
                # [修改] 记录详细损失
                batch_losses_total = []
                batch_losses_q = []
                batch_losses_kl = []
                
                for _ in range(num_training_steps):
                    loss_tuple = optimize_model(policy_net, target_net, optimizer, replay_buffer)
                    if loss_tuple is not None:
                        total_loss, loss_q, loss_kl = loss_tuple
                        batch_losses_total.append(total_loss)
                        batch_losses_q.append(loss_q)
                        batch_losses_kl.append(loss_kl)

                # 记录损失
                if batch_losses_total:
                    avg_loss_total = np.mean(batch_losses_total)
                    avg_loss_q = np.mean(batch_losses_q)
                    avg_loss_kl = np.mean(batch_losses_kl)
                    avg_reward = np.mean([exp['R'] for exp in batch_experiences])
                    
                    # 记录每批的平均损失
                    loss_total_history.append(avg_loss_total)
                    loss_q_history.append(avg_loss_q)
                    loss_kl_history.append(avg_loss_kl)

                    current_time = datetime.now().strftime("%H:%M:%S")
                    batch_count += 1

                    # --- [修改] 目标网络更新 (按批次) ---
                    if batch_count % UPDATE_TARGET_EVERY_N_BATCHES == 0:
                        target_net.load_state_dict(policy_net.state_dict())
                        print(f"--- [UPDATE] 批次 {batch_count}: 目标网络已更新 ---")

                    # [修改] 记录到文件 - 将batch映射到episode范围以便绘图时反映真实训练进度
                    # 计算当前batch对应的近似episode编号
                    actual_episode = total_experiences # 使用实际的episode编号（支持断点训练）
                    with open(loss_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{actual_episode},{avg_loss_total:.6f},{avg_loss_q:.6f},{avg_loss_kl:.6f},{avg_reward:.3f},{current_time}\n")

                    # [修改] 中间模型保存策略 (简化，删除最佳准确率保存)
                    save_checkpoint = False
                    checkpoint_name = ""
                    if batch_count % CHECKPOINT_NUM == 0: # 每CHECKPOINT_NUM批次保存
                        save_checkpoint = True; checkpoint_name = f"policy_net_batch_{batch_count:04d}.pth"

                    if save_checkpoint:
                        checkpoint_path = os.path.join(model_save_dir, checkpoint_name)
                        torch.save({
                            'policy_net_state_dict': policy_net.state_dict(),
                            'target_net_state_dict': target_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'experiences': total_experiences,
                            'batch': batch_count,
                            'loss_total': avg_loss_total,
                            'loss_q': avg_loss_q,
                            'loss_kl': avg_loss_kl,
                            'avg_reward': avg_reward,
                            'timestamp': current_time,
                            'total_loss_history': loss_total_history[-100:] if len(loss_total_history) > 100 else loss_total_history
                        }, checkpoint_path)
                        # print(f"\n[CHECKPOINT] 模型已保存: {checkpoint_path}")

                    # [修改] 更新进度条 (简化，删除验证准确率)
                    pbar.update(len(batch_experiences))
                    pbar.set_postfix({
                        "L_Total": f"{avg_loss_total:.4f}",
                        "L_Q": f"{avg_loss_q:.4f}",
                        "L_KL": f"{avg_loss_kl:.4f}",
                        "Reward": f"{avg_reward:.3f}",
                        "Buffer": len(replay_buffer),
                    })

                    # [修改] 定期打印
                    # if batch_count % 10 == 0:
                        # print(f"[TRAIN] 批次{batch_count}: L_Total={avg_loss_total:.4f}, L_Q={avg_loss_q:.4f}, L_KL={avg_loss_kl:.4f}, R={avg_reward:.3f}, Val={current_val_acc:.2f}%, Exps={total_experiences}")

            await asyncio.sleep(0.001) # 防止CPU占用过高

    except Exception as e:
        print(f"[ERROR] 并行环境训练异常: {e}")
        import traceback; traceback.print_exc()

    finally:
        # [修改] 训练结束统计 (简化，删除验证准确率)
        print("\n--- 并行环境训练完成 ---")
        if loss_total_history:
            print(f"[STATS] 总经验数: {total_experiences}")
            print(f"[STATS] 总批次: {batch_count}")
            print(f"[STATS] 最终总损失: {loss_total_history[-1]:.4f}")
            print(f"[STATS] 平均总损失: {np.mean(loss_total_history):.4f}")

        # [修改] 保存最终完整模型 (简化，删除验证准确率)
        final_model_path = os.path.join(model_save_dir, "policy_net_final.pth")
        torch.save({
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'experiences': total_experiences,
            'batch': batch_count,
            'final_loss_total': loss_total_history[-1] if loss_total_history else None,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'total_loss_history': loss_total_history,
            'total_loss_q_history': loss_q_history,
            'total_loss_kl_history': loss_kl_history
        }, final_model_path)
        
        # 保存一个易于加载的纯 state_dict
        torch.save(policy_net.state_dict(), "dqn_policy_net_vectorized_final.pth")
        print(f"🎉 并行环境训练模型已保存。")
        
        # [修改] 生成训练报告 (简化，删除验证准确率)
        report_path = os.path.join(model_save_dir, "training_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"DQN并行环境训练报告 (DDQN + KL)\n==================\n\n")
            f.write(f"训练时间戳: {timestamp}\n")
            f.write(f"总经验数: {total_experiences}\n")
            f.write(f"总批次: {batch_count}\n")
            if loss_total_history:
                f.write(f"最终总损失: {loss_total_history[-1]:.6f}\n")
                f.write(f"平均总损失: {np.mean(loss_total_history):.6f}\n")
            if loss_q_history:
                f.write(f"平均Q损失: {np.mean(loss_q_history):.6f}\n")
            if loss_kl_history:
                f.write(f"平均KL损失: {np.mean(loss_kl_history):.6f}\n")
            f.write("\n--- 超参数 ---\n")
            f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
            f.write(f"KL_BETA: {KL_BETA}\n")
            f.write(f"GAMMA: {GAMMA}\n")
            f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"TARGET_UPDATE_INTERVAL (Batches): {UPDATE_TARGET_EVERY_N_BATCHES}\n")
        print(f"📊 训练报告已保存: {report_path}")

# --- 6. 运行 ---

if __name__ == "__main__":
    print("=== DQN 在线训练系统 (三智能体架构 + DDQN + 混合奖励) ===")
    print("架构: 医生智能体 + 患者智能体 + 评估智能体")
    print("策略: 9类CBT策略 (包含危机干预)")
    print("奖励: 安全熔断机制 + 多维度质量评估")
    print("特色: 明确的患者发言上下文，简化评估流程")
    
    training_mode = "vectorized"
    
    if len(sys.argv) > 1:
        print("[ERROR] 串行或并发模式在此优化版中已弃用。")
        print("[INFO] 🚀 强制使用并行环境训练模式 (vectorized)")

    print()
    print("正在启动DQN在线训练主循环...")
    try:
        asyncio.run(main_vectorized_training_loop())
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断训练")
    except Exception as e:
        print(f"[ERROR] 训练过程中发生严重错误: {e}")
        import traceback; traceback.print_exc()
    finally:
        print("[INFO] 清理资源...")
        asyncio.run(cleanup_llm_session())
        print("[INFO] 程序退出")