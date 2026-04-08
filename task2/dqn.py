import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import json
import os
from collections import deque, namedtuple
import numpy as np

# --- 配置加载器支持 ---
try:
    from config_loader import get_config
    # 加载配置文件
    cp_config = get_config()
    config = cp_config.config

    # 验证配置
    if not cp_config.validate_config():
        print("❌ 配置验证失败，使用默认参数")
        raise ImportError("配置验证失败")

    print("✅ 成功加载YAML配置文件")

except ImportError as e:
    print(f"⚠️  无法加载配置文件，使用默认参数: {e}")
    # 使用原有的默认参数（向后兼容）
    config = None

# --- 1. 超参数定义 (Hyperparameters) ---
# 支持YAML配置，如果配置加载失败则使用原有默认值

# 数据路径
INPUT_PATH = cp_config.input_path if config else "/home/admin/CogEmo/task2/data/Total_S_train_pool.jsonl"

# 断点训练配置
RESUME_TRAINING = cp_config.resume_training if config else False
CHECKPOINT_MODEL_PATH = cp_config.checkpoint_model_path if config else "/home/admin/CogEmo/task2/results/dqn_checkpoints_20251119_130151/policy_net_final.pth"
CHECKPOINT_LOSS_LOG_PATH = cp_config.checkpoint_loss_log_path if config else "/home/admin/CogEmo/task2/results/training_loss_vectorized_20251119_130151.txt"
CHECKPOINT_EXP_LOG_PATH = cp_config.checkpoint_exp_log_path if config else "/home/admin/CogEmo/task2/results/experiment_data_20251119_130151.csv"
RESUME_EPISODE_COUNT = cp_config.resume_episode_count if config else 100000

# 训练参数 - 确保数值类型正确转换
K_EPISODES = int(cp_config.k_episodes) if config else 100000
GAMMA = float(cp_config.gamma) if config else 0.8
BATCH_SIZE = int(cp_config.batch_size) if config else 32
LEARNING_RATE = float(cp_config.learning_rate) if config else 1e-4

# 消融实验支持的KL参数 - 确保数值类型正确转换
if config:
    # 根据消融设置调整KL参数
    KL_BETA = 0.0 if cp_config.is_kl_disabled else float(cp_config.kl_beta)
    KL_TEMP = float(cp_config.kl_temp)
    print(f"🔬 KL约束: {'禁用' if cp_config.is_kl_disabled else f'启用(beta={KL_BETA})'}")
else:
    # 原有默认值
    KL_BETA = 0.1
    KL_TEMP = 1.0

# DQN 架构参数 - 支持YAML配置，确保数值类型正确转换
EMBEDDING_DIM = int(cp_config.embedding_dim) if config else 1024
HIDDEN_DIM_1 = int(cp_config.hidden_dim_1) if config else 256
HIDDEN_DIM_2 = int(cp_config.hidden_dim_2) if config else 128
ACTION_DIM = int(cp_config.action_dim) if config else 10

# Epsilon-Greedy 策略参数 - 支持YAML配置，确保数值类型正确转换
EPS_START = float(cp_config.eps_start) if config else 0.9
EPS_END = float(cp_config.eps_end) if config else 0.1
EPS_DECAY_STEPS = int(cp_config.eps_decay_steps) if config else 50000

# 更新参数 - 支持YAML配置，确保数值类型正确转换
CHECKPOINT_NUM = int(cp_config.checkpoint_num) if config else 50
TARGET_UPDATE_INTERVAL = int(cp_config.target_update_interval) if config else 10
REPLAY_BUFFER_CAPACITY = int(cp_config.replay_buffer_capacity) if config else 100000

NUM_PARALLEL_ENVIRONMENTS = int(cp_config.num_parallel_environments) if config else 32
VALIDATE_EVERY_N_BATCHES = 10 # 每10个训练批次验证一次

# 打印配置信息
if config:
    print(f"🔧 DQN配置: 嵌入维度={EMBEDDING_DIM}, 隐藏层=[{HIDDEN_DIM_1}, {HIDDEN_DIM_2}], 动作空间={ACTION_DIM}")
    print(f"🎯 训练参数: 学习率={LEARNING_RATE}, 批次大小={BATCH_SIZE}, 折扣因子={GAMMA}")
    print(f"🔄 更新参数: 目标网络更新间隔={TARGET_UPDATE_INTERVAL}, 经验回放容量={REPLAY_BUFFER_CAPACITY}")


# 设备检查和选择
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 检测到CUDA设备，使用GPU训练")
    print(f"   GPU型号: {torch.cuda.get_device_name()}")
    print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    print(f"💻 未检测到CUDA设备，使用CPU训练")

print(f"📱 使用设备: {device}")
print(f"🔥 PyTorch版本: {torch.__version__}")

# 定义经验元组格式
Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state'))

# --- 2. DQN 模型构建 (QNetwork) ---

class QNetwork(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, h1_dim)
        self.dropout1 = nn.Dropout(p=0.1) 
        self.layer2 = nn.Linear(h1_dim, h2_dim)
        self.dropout2 = nn.Dropout(p=0.1)
        self.layer3 = nn.Linear(h2_dim, output_dim) # 输出8个动作的Q值

    def forward(self, state_embedding):
        x = F.relu(self.layer1(state_embedding))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.layer3(x)

# --- 3. 经验回放缓冲区 (ReplayBuffer) ---

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        """
        [优化] 将经验 (s, a, R, s') 存入缓冲区 (存储Numpy/Python类型, 非Tensor)
        """
        # 存储原始数据，而不是Tensor，以节省显存并加快push速度
        self.buffer.append(Experience(state, action, reward, next_state))

    def sample(self, batch_size):
        """从缓冲区中随机采样 B 个经验"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- 4. 辅助函数与模拟器占位符 ---

steps_done = 0

def select_action(state_embedding, policy_net):
    """
    步骤 2.1 (部分): 动作选择 (ε-贪心策略)
    """
    global steps_done
    
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        (1. - min(1.0, steps_done / EPS_DECAY_STEPS))
    
    steps_done += 1

    if random.random() < eps_threshold:
        return torch.tensor([[random.randrange(ACTION_DIM)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            # state_embedding 是 np.array/list
            state_tensor = torch.tensor(state_embedding, dtype=torch.float).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].view(1, 1)

def load_checkpoint(model_path):
    """加载checkpoint模型和训练状态"""
    if not os.path.exists(model_path):
        print(f"[警告] Checkpoint文件不存在: {model_path}")
        return None

    try:
        print(f"[INFO] 加载checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 检查checkpoint是否包含必要信息
        required_keys = ['policy_net_state_dict', 'optimizer_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                print(f"[错误] Checkpoint缺少必要信息: {key}")
                return None

        print(f"[INFO] Checkpoint加载成功")
        print(f"       - 模型状态: 已保存")
        print(f"       - 优化器状态: 已保存")
        print(f"       - 训练episodes: {checkpoint.get('experiences', 0)}")
        print(f"       - 训练批次: {checkpoint.get('batch', 0)}")

        return checkpoint

    except Exception as e:
        print(f"[错误] 加载checkpoint失败: {e}")
        return None

def append_to_loss_log(log_file, episode, total_loss, loss_q, loss_kl, avg_reward):
    """续写损失日志文件"""
    try:
        current_time = __import__('datetime').datetime.now().strftime("%H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{episode},{total_loss:.6f},{loss_q:.6f},{loss_kl:.6f},{avg_reward:.3f},{current_time}\n")
        return True
    except Exception as e:
        print(f"[警告] 续写损失日志失败: {e}")
        return False

def append_to_exp_log(exp_file, episode, bias_type, intensity, risk_level, action_index, reward, q_values):
    """续写实验数据文件"""
    try:
        import csv
        with open(exp_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([episode, bias_type, intensity, risk_level, action_index, reward, json.dumps(q_values)])
        return True
    except Exception as e:
        print(f"[警告] 续写实验数据失败: {e}")
        return False

def load_initial_states(path=INPUT_PATH):
    """(模拟) 步骤 1: 加载初始状态池 - 使用统一数据路径"""
    print(f"加载输入数据：{path}...")
    s_pool = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                s_pool.append({
                    "embedding": data["embedding"], # 1024维向量
                    "context": data["context"]    
                })
        print(f"Loaded {len(s_pool)} initial states.")
        if not s_pool: raise Exception("S_pool is empty!")
        return s_pool
    except FileNotFoundError:
        print("[Error] S_pool.jsonl not found. Using dummy data.")
        return [{"embedding": np.random.rand(EMBEDDING_DIM).tolist(), "context": "你好"}]

# --- 模拟器占位符 (Placeholder Functions) ---
# 在真实流程中, 您需要用 SBERT 和 LLM API 调用替换它们

def get_embedding_from_text(text):
    """
    (占位符) 方案B的核心: 将文本状态转换为嵌入
    在步骤 2.3 中, '患者LLM' 输出新状态的文本描述后，调用此函数。
    """
    # 真实实现: return sentence_bert_model.encode(text)
    # 模拟实现:
    # print(f"[SBERT Sim] Encoding: '{text[:20]}...'")
    return np.random.rand(EMBEDDING_DIM).tolist() # 返回一个假的768维向量

def simulate_llm_interaction(s_context, s_embedding, action_index):
    """
    (占位符) 模拟 步骤 2.2 和 2.3
    - 步骤 2.2: 医生LLM生成回应 (内部模拟)
    - 步骤 2.3: 患者LLM生成反馈, 并输出 (R, s')
    """
    # 1. (模拟 2.2) 医生LLM生成回应
    # doctor_response = ... (调用提示词1)
    
    # 2. (模拟 2.3) 患者LLM模拟与评估
    #    (调用提示词2, 它返回JSON)
    # 真实实现: 
    # response_json = patient_llm(s_context, doctor_response)
    # R = (response_json["feedback_score"]["..."] + ...) / 12.0
    # s_prime_text = f"认知偏差：{...}; 危险等级：{...}"
    # s_prime_embedding = get_embedding_from_text(s_prime_text)
    
    # 模拟实现 (返回假数据):
    # 模拟一个随机的收益 R
    R = random.uniform(0.1, 1.0) 
    
    # 模拟一个新的状态 s' (注意：在真实场景中，这个嵌入必须通过 SBERT 获得)
    s_prime_embedding = np.random.rand(EMBEDDING_DIM).tolist()

    return R, s_prime_embedding

# --- 5. DQN 训练逻辑 (Training Logic) ---

def optimize_model(policy_net, target_net, optimizer, replay_buffer):
    """
    [重大修改] 步骤 2.5: Q网络批量更新 (DDQN + KL约束 + 消融实验支持)
    """
    if len(replay_buffer) < BATCH_SIZE:
        return None # 缓冲区未满，不训练
    
    # 1. 采样
    experiences = replay_buffer.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))

    # 2. [优化] 批量转换数据 (从CPU/Numpy到GPU/Tensor)
    # batch.state 是一个元组，包含 B 个 np.array
    state_batch = torch.tensor(np.vstack(batch.state), dtype=torch.float32).to(device)
    # batch.action 是一个元组，包含 B 个 int
    action_batch = torch.tensor(batch.action, dtype=torch.long).to(device).unsqueeze(1) # [B] -> [B, 1]
    # batch.reward 是一个元组，包含 B 个 float
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device) # [B]
    
    # 处理 next_state (s')
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states_list = [s for s in batch.next_state if s is not None]

    if non_final_next_states_list:
        non_final_next_states = torch.tensor(np.vstack(non_final_next_states_list), dtype=torch.float32).to(device)
    else:
        # 如果所有next_state都是None，创建一个零张量作为占位符
        non_final_next_states = torch.zeros(1, EMBEDDING_DIM, device=device)
        non_final_mask = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=device)

    # 3. 计算 Q_predicted (Q_θ(s_i, a_i))
    q_predicted_all = policy_net(state_batch)
    q_predicted = q_predicted_all.gather(1, action_batch)

    # 4. [修改] 计算 Q_target (y_i) - 支持DDQN消融实验
    # y_i = R_i + γ * Q_θ⁻(s'_i, argmax_a' Q_θ(s'_i, a'))  (DDQN)
    # y_i = R_i + γ * max_a' Q_θ⁻(s'_i, a')              (标准DQN)
    q_target_next = torch.zeros(BATCH_SIZE, device=device)
    if non_final_mask.any():
        if config and cp_config.is_ddqn_disabled:
            # 标准DQN逻辑：直接用target_net找到最大Q值动作
            with torch.no_grad():
                s_prime_q_values = target_net(non_final_next_states).max(dim=1).values # [N]
        else:
            # DDQN逻辑：用policy_net选动作，用target_net评估
            # 1. 用 policy_net 找出 s' 的最佳动作 (Action Selection)
            s_prime_actions = policy_net(non_final_next_states).argmax(dim=1).unsqueeze(1) # [N, 1]

            # 2. 用 target_net 计算这些动作的 Q 值 (Value Evaluation)
            with torch.no_grad():
                s_prime_q_values = target_net(non_final_next_states).gather(1, s_prime_actions).squeeze(1) # [N]

        q_target_next[non_final_mask] = s_prime_q_values
    
    q_target = reward_batch + (GAMMA * q_target_next)

    # 5. 计算损失 (Loss)
    
    # 5.1. Q值损失 (DDQN)
    loss_q = F.smooth_l1_loss(q_predicted, q_target.unsqueeze(1))

    # 5.2. KL散度约束 (根据消融实验设置决定是否启用)
    if config and cp_config.is_kl_disabled:
        # 消融实验：禁用KL约束
        loss_kl = torch.tensor(0.0, device=device)
        total_loss = loss_q
        algorithm_used = "DQN"
    else:
        # 正常训练：启用KL约束
        with torch.no_grad():
            logits_target = target_net(state_batch).detach() # [B, ACTION_DIM]

        # q_predicted_all 就是 logits_policy
        log_p_policy = F.log_softmax(q_predicted_all / KL_TEMP, dim=1)
        p_target = F.softmax(logits_target / KL_TEMP, dim=1)

        # F.kl_div(input, target) -> KL(target || input)
        # reduction='batchmean' = sum(kl) / batch_size
        loss_kl = F.kl_div(log_p_policy, p_target, reduction='batchmean') * (KL_TEMP * KL_TEMP)

        total_loss = loss_q + (KL_BETA * loss_kl)
        algorithm_used = "DDQN+KL" if not (config and cp_config.is_ddqn_disabled) else "DQN+KL"

    # 记录使用的算法类型
    if config:
        if not hasattr(optimize_model, '_algorithm_printed'):
            print(f"🧠 使用算法: {algorithm_used}")
            optimize_model._algorithm_printed = True

    # 6. 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1.0) # 梯度裁剪
    optimizer.step()
    
    # [修改] 返回三种损失，用于详细日志
    return total_loss.item(), loss_q.item(), loss_kl.item()

# --- 6. DQN 推理逻辑 (Inference Logic) ---

def get_optimal_action(s_embedding_tensor, policy_net):
    """步骤 3: 最终推理"""
    policy_net.eval() 
    with torch.no_grad():
        q_values = policy_net(s_embedding_tensor.to(device))
        action_index = q_values.argmax().item()
    policy_net.train() 
    return action_index

# --- 7. 主训练循环 (Main Training Loop) ---

def main_training_logic():

    # 1. 初始化模型
    policy_net = QNetwork(EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, ACTION_DIM).to(device)
    target_net = QNetwork(EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, ACTION_DIM).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 2. 初始化优化器和缓冲区
    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    # 3. 断点训练检查
    start_episode = 0
    if RESUME_TRAINING and CHECKPOINT_MODEL_PATH:
        checkpoint = load_checkpoint(CHECKPOINT_MODEL_PATH)
        if checkpoint:
            # 加载模型状态
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            target_net.load_state_dict(checkpoint.get('target_net_state_dict', policy_net.state_dict()))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 获取训练起始点
            start_episode = checkpoint.get('experiences', 0)
            print(f"[INFO] 从episode {start_episode}开始续训练")
            print(f"[INFO] 续训练目标: K_EPISODES = {K_EPISODES}, 总计将达到 {start_episode + K_EPISODES}")
        else:
            print("[警告] Checkpoint加载失败，从头开始训练")

    # 4. 加载初始状态池 - 使用统一数据路径
    S_pool = load_initial_states(INPUT_PATH)

    print(f"--- Starting Training for {K_EPISODES} iterations (DDQN + KL) ---")

    # 4. 开始 K 次迭代
    for k in range(K_EPISODES):
        
        initial_state_data = random.choice(S_pool)
        s_embedding = initial_state_data["embedding"] # np.array/list
        s_context = initial_state_data["context"]     
        
        action_tensor = select_action(s_embedding, policy_net)
        action_index = action_tensor.item() 

        R, s_prime_embedding = simulate_llm_interaction(s_context, s_embedding, action_index)
        
        # [优化] push 原始类型
        replay_buffer.push(s_embedding, action_index, R, s_prime_embedding)
        
        # 步骤 2.5: Q网络参数更新
        loss_tuple = optimize_model(policy_net, target_net, optimizer, replay_buffer)

        # 步骤 2.6: 目标网络更新 (按迭代次数)
        # [注意] 在 run_online_training.py 中, 我们是按批次更新
        if k % (TARGET_UPDATE_INTERVAL * 10) == 0: # 假设
            target_net.load_state_dict(policy_net.state_dict())
            
        # 增强的训练日志
        if k % 10 == 0: 
            eps_threshold = EPS_END + (EPS_START - EPS_END) * (1. - min(1.0, steps_done / EPS_DECAY_STEPS))
            if loss_tuple is not None:
                total_loss, loss_q, loss_kl = loss_tuple
                print(f"迭代 [{k:4d}/{K_EPISODES}] | L_Total: {total_loss:.4f} (L_Q: {loss_q:.4f}, L_KL: {loss_kl:.4f}) | "
                      f"Eps: {eps_threshold:.3f} | 动作: {action_index} | 奖励: {R:.3f} | 缓冲区: {len(replay_buffer)}")
            else:
                print(f"迭代 [{k:4d}/{K_EPISODES}] | Eps: {eps_threshold:.3f} | "
                      f"动作: {action_index} | 奖励: {R:.3f} | 缓冲区: {len(replay_buffer)} (缓冲区未满)")

    print("--- Training Finished ---")
    
    torch.save(policy_net.state_dict(), "dqn_policy_net_final.pth")
    print("Model saved to dqn_policy_net_final.pth")


# --- 如何运行 ---
if __name__ == "__main__":
    
    # === 1. 运行主训练循环 ===
    # main_training_logic()
    
    # === 2. (示例) 如何使用训练好的模型进行推理 ===
    print("\n--- Inference Example ---")
    
    inference_net = QNetwork(EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, ACTION_DIM).to(device)
    try:
        inference_net.load_state_dict(torch.load("dqn_policy_net_final.pth"))
        print("Loaded trained model for inference.")
    except FileNotFoundError:
        print("No trained model found. Using initialized model.")

    s_text = "认知偏差：灾难化; 危险等级：高危; ..."
    s_embedding_example = get_embedding_from_text(s_text) 
    s_tensor = torch.tensor(s_embedding_example, dtype=torch.float).unsqueeze(0).to(device)
    
    optimal_action = get_optimal_action(s_tensor, inference_net)
    print(f"State Text (Simulated): '{s_text}'")
    print(f"Inferred Optimal Action Index (a_m^*): {optimal_action}")