"""
共用工具模块
包含DQN推理、LLM调用、数据处理等共享功能
"""
import torch
import json
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional
import sys
import os

# DQN模型导入
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'task2'))
from dqn import QNetwork, EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, ACTION_DIM, device

# 策略映射配置 - 更新为10个CBT专病专治策略（与task2保持一致）
ACTION_SPACE_MAP = {
    0: {
        "strategy_name": "情感验证与共情",
        "desc": "【核心定义】在不带有评判的前提下，识别、命名并接纳用户当前的情绪痛苦。【执行指令】不要急于解决问题或提供建议。使用温暖、接纳的语气。告诉患者他们的感受是合理的、可以被理解的。【关键话术】'听起来你现在真的很难过。' '面对这样的情况，感到愤怒是很正常的。' '我能感受到这件事对你的打击有多大。'"
    },
    1: {
        "strategy_name": "寻找灰色地带",
        "desc": "【核心定义】引导用户打破'全有或全无'（完美或失败）的二元对立，在连续谱上寻找中间状态。【执行指令】指出这种极端思维。询问用户：'如果100分是完美，0分是绝对灾难，你觉得目前的情况实际处于多少分？' 引导用户看到虽不完美但仍有价值的部分。【关键话术】'难道只有100分才算成功吗？80分是否也包含了一些努力？' '这真的是彻底的失败吗？还是说只是部分不如意？'"
    },
    2: {
        "strategy_name": "寻找例外证据",
        "desc": "【核心定义】引导用户像侦探一样，寻找与当前消极结论相反的客观证据（例外情况）。【执行指令】当用户使用'总是'、'从不'、'全都不好'等词汇时，温和地挑战其结论。询问过去是否发生过积极的例外，或者当前情境中被忽略的正面细节。【关键话术】'你说你总是搞砸，但有没有哪一次是做得还不错的？' '让我们看看证据，支持这个想法的证据有哪些？不支持的有哪些？'"
    },
    3: {
        "strategy_name": "事实核查",
        "desc": "【核心定义】区分'想象（推测）'与'事实'。鼓励用户寻找确凿证据，而非假设他人的意图。【执行指令】询问用户支持其猜想（如'他讨厌我'）的客观证据是什么。提出替代解释（Alternative Explanations）。鼓励沟通核实。【关键话术】'你怎么确信他在想什么？有实质性的证据吗？' '除了他讨厌你，有没有可能他只是太忙了没看到消息？' '我们要不要去问问他，而不是自己猜？'"
    },
    4: {
        "strategy_name": "去灾难化",
        "desc": "【核心定义】不否认风险，而是通过具体化'最坏结果'并制定应对计划，来降低对未知的恐惧。【执行指令】不要只说'没事的'。请问：'如果最坏的情况真的发生了，具体会怎么样？你会立刻完蛋吗？' 然后引导：'如果是那样，我们能做些什么来应对？' 帮助患者找回掌控感。【关键话术】'让我们假设最坏的情况发生了，你会怎么处理？' '这件事在这一生中真的有毁灭性的影响吗？' '哪怕发生了，你也有办法活下去，对吗？'"
    },
    5: {
        "strategy_name": "利弊分析",
        "desc": "【核心定义】引导用户评估死守某种僵化规则（'我必须...'）的实用性，对比其带来的好处与心理代价。【执行指令】针对用户的'应该'或'必须'，询问：'坚持这个高标准给你带来了什么好处？又让你付出了什么代价（如焦虑、拖延）？' 引导用户建立更灵活的标准。【关键话术】'对自己要求这么严格，虽然让你很上进，但似乎也让你非常疲惫，值得吗？' '如果把标准稍微降低一点，会发生什么可怕的事吗？'"
    },
    6: {
        "strategy_name": "责任饼图",
        "desc": "【核心定义】帮助用户列出导致结果的所有潜在因素，重新分配责任比例，减轻过度的内疚感。【执行指令】画一个虚拟的'责任饼图'。引导用户列出除了自己以外的其他影响因素（如运气、他人、环境）。询问：'这件事真的是你一个人的错吗？还有谁/什么在其中起了作用？'【关键话术】'让我们画个饼图，这件事里有多少比例是你的责任，多少是环境或他人的责任？' '你是不是把不该你背的锅也背在身上了？'"
    },
    7: {
        "strategy_name": "区分行为与人",
        "desc": "【核心定义】引导用户将'特定的失败行为'与'整体的人格评价'剥离开来，反对给自己贴标签。【执行指令】明确指出：做错一件事不等于你就是一个失败的人。引导用户用具体的行为描述代替抽象的负面标签。【关键话术】'你只是这次考试没考好（行为），这并不代表你就是个笨蛋（标签）。' '人是复杂的，一个标签能定义全部的你吗？'"
    },
    8: {
        "strategy_name": "情感分离",
        "desc": "【核心定义】明确区分'主观感受'与'客观事实'。引导用户像法官一样审视证据，而不是被情绪牵着走。【执行指令】告诉用户：感觉强烈不代表它是真的。引导用户跳出情绪，用理性的眼光看事实。询问：'如果你的朋友遇到这事，你会觉得他没救了吗？还是这只是你现在的感觉？'【关键话术】'你感觉自己很蠢，但这不代表事实就是这样。' '这是一种强烈的感觉，还是一个被证明的事实？' '让我们把情绪放一边，只看证据。'"
    },
    9: {
        "strategy_name": "危机干预",
        "desc": "【核心定义】检测到高危风险（自伤、自杀、伤人）时的紧急阻断策略。【执行指令】语气必须严肃、直接且关切。**立刻停止**对认知的讨论。直接询问自杀意念，表达对生命安全的担忧，提供紧急求助渠道（如热线、医院）。【关键话术】'我听到了你的痛苦，但我非常担心你的安全。你现在有伤害自己的念头吗？' '请务必先保证自己的安全，这是最重要的。请拨打...' "
    }
}

# 本地VLLM API配置
LOCAL_API_CONFIG = {
    "model": "",
    "base_url": "",
    "api_key": "",
    "temperature": 0.7,
    "max_tokens": 256,
    "timeout": 30
}

# DASHSCOPE API配置
DASHSCOPE_API_CONFIG = {
    "model": "",
    "base_url": "",
    "api_key": "",
    "temperature": 0.7,
    "max_tokens": 512,
    "timeout": 30
}

# OPENROUTER API配置
OPENROUTER_API_CONFIG = {
    "model": "",
    "base_url": "",
    "api_key": "",
    "temperature": 0.7,
    "max_tokens": 512,
    "timeout": 30
}

# 默认LLM API配置（保持向后兼容）
LLM_API_CONFIG = LOCAL_API_CONFIG

# 嵌入API配置 - 支持三类LLM的嵌入API
EMBEDDING_API_CONFIG = {
    'api_base': 'http://localhost:6862/v1',
    'model_name': 'Qwen3-Embedding-0.6B',
    'timeout': 30,
    'max_concurrent': 32,
    'api_key': 'dummy-key'
}

class DQNInference:
    """DQN推理器"""

    def __init__(self, model_path: str):
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> QNetwork:
        """加载DQN模型"""
        print(f"正在加载DQN模型: {model_path}")
        policy_net = QNetwork(EMBEDDING_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, ACTION_DIM).to(self.device)

        # 加载检查点文件（PyTorch 2.6+需要weights_only=False来处理numpy对象）
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 如果检查点包含policy_net_state_dict，则提取它
        if 'policy_net_state_dict' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            print(f"从检查点加载模型权重，训练最终奖励: {checkpoint.get('final_avg_reward', 'N/A')}")
        else:
            # 否则直接加载（可能是直接保存的权重）
            policy_net.load_state_dict(checkpoint)

        policy_net.eval()
        print("DQN模型加载成功")
        return policy_net

    def get_best_worst_actions(self, embedding: List[float]) -> tuple[int, int]:
        """获取最优和最差策略索引"""
        with torch.no_grad():
            s_tensor = torch.tensor(embedding, dtype=torch.float).unsqueeze(0).to(self.device)
            q_values = self.model(s_tensor).flatten()
            sorted_indices = torch.argsort(q_values, descending=True).cpu().numpy()
            return int(sorted_indices[0]), int(sorted_indices[-1])

    def get_best_action(self, embedding: List[float]) -> int:
        """获取最优策略索引"""
        best_action, _ = self.get_best_worst_actions(embedding)
        return best_action

async def build_state_embedding(patient_content: str, bias_tags: list, bias_intensity: str, risk_level: str, session: aiohttp.ClientSession) -> np.ndarray:
    """
    构建状态嵌入向量 - 按照task1/create_dqn_dataset.py的逻辑
    """
    # 构建状态描述文本
    bias_content = ','.join(bias_tags) if bias_tags else "无"

    s_text = f"患者当前发言：{patient_content}\n" \
             f"认知偏差：{bias_content}\n" \
             f"偏差强度：{bias_intensity}\n" \
             f"风险等级：{risk_level}\n"

    # 获取嵌入向量
    embedding = await get_embedding_from_text(s_text, session)
    return embedding

async def generate_improved_response_with_dqn(
    patient_content: str,
    bias_tags: list,
    bias_intensity: str,
    risk_level: str,
    dqn_inference: DQNInference,
    session: aiohttp.ClientSession,
    api_mode: str = "local"
) -> tuple[str, str]:
    """
    使用DQN推理最佳策略，然后生成高质量回复
    使用从task1/bias_strategy_label.json加载的策略信息
    """
    # 构建状态嵌入
    embedding = await build_state_embedding(patient_content, bias_tags, bias_intensity, risk_level, session)

    if embedding is None:
        print("❌ 嵌入向量生成失败，使用默认策略")
        return "我理解你的感受。能告诉我更多细节吗？", "情感验证与共情"

    # DQN推理最优策略
    best_action_idx = dqn_inference.get_best_action(embedding.tolist())
    best_strategy = ACTION_SPACE_MAP.get(best_action_idx, ACTION_SPACE_MAP[0])

    # 构建包含策略信息的提示词
    bias_content = ','.join(bias_tags) if bias_tags else "无"

    prompt = f"""## 任务
你是一名专业的心理咨询师。根据患者发言和认知偏差分析，基于指定的引导策略生成高质量的心理咨询师回复。

## 患者当前发言
"{patient_content}"

## 认知偏差分析
- 认知偏差类型：{bias_content}
- 偏差强度：{bias_intensity}
- 风险等级：{risk_level}

## 指定的引导策略
策略名称：{best_strategy['strategy_name']}
策略详细说明：
{best_strategy['desc']}

## 严格要求
1. 严格按照上述策略的定义和执行指令进行回复
2. 遵循策略中的关键话术指导
3. 回复要体现该策略的核心定义和目标
4. 参考策略中的执行指令来把握回复风格
5. 回复要专业、温暖、有共情心
6. 长度控制在30-150字
7. 语言要自然口语化，符合真实咨询场景
8. 避免使用过于学术化的术语
10.不要总是说一些看起来比较死板的句式："听到你..."、"看起来..."、"我听到..."、"我感受到..."。总之回答要多样。

## 请基于上述策略指导生成专业的心理咨询师回复："""

    # 调用LLM生成回复
    response = await call_llm_api([{"role": "user", "content": prompt}], api_mode)
    # print(f"response:{response}")
    # 清理回复内容
    cleaned_response = extract_real_answer(response).strip()

    # 返回回复内容和策略名称
    return cleaned_response, best_strategy['strategy_name']

# 全局aiohttp会话 (直接学习task2模式)
async_session = None

async def get_llm_session():
    """获取或创建aiohttp会话"""
    global async_session
    if async_session is None:
        async_session = aiohttp.ClientSession()
    return async_session

async def cleanup_llm_session():
    """清理LLM会话"""
    global async_session
    if async_session:
        await async_session.close()
        async_session = None

async def get_embedding_from_text(text: str, session: aiohttp.ClientSession) -> np.ndarray:
    """
    异步获取文本嵌入向量
    """
    data = {
        "model": EMBEDDING_API_CONFIG["model_name"],
        "input": text,
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

def get_api_config(api_mode: str) -> Dict[str, Any]:
    """
    根据API模式获取对应的配置

    Args:
        api_mode: API模式 ("local", "dashscope", "openrouter")

    Returns:
        API配置字典
    """
    if api_mode == "local":
        return LOCAL_API_CONFIG
    elif api_mode == "dashscope":
        return DASHSCOPE_API_CONFIG
    elif api_mode == "openrouter":
        return OPENROUTER_API_CONFIG
    else:
        raise ValueError(f"不支持的API模式: {api_mode}")

async def call_llm_api(messages: list, api_mode: str = "local", max_retries: int = 3) -> str:
    """
    调用LLM API，支持多种API模式和重试机制

    Args:
        messages: 消息列表
        api_mode: API模式 ("local", "dashscope", "openrouter")
        max_retries: 最大重试次数

    Returns:
        LLM响应文本
    """
    for attempt in range(max_retries + 1):
        try:
            # 获取对应API模式的配置
            config = get_api_config(api_mode)

            # 检测API类型并决定调用方式
            api_type = None
            if api_mode == "local":
                api_type = "vllm"
            elif api_mode == "dashscope":
                api_type = "dashscope"
            elif api_mode == "openrouter":
                api_type = "openrouter"

            # 根据API类型选择调用方式
            if api_type == "vllm":
                return await _call_vllm_api(messages, config)
            else:
                return await _call_openai_compatible_api(messages, config, api_type)

        except Exception as e:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2  # 指数退避：2秒, 4秒, 6秒
                print(f"[LLM API调用错误] ({api_mode}) 尝试 {attempt + 1}/{max_retries + 1}: {e}")
                print(f"等待 {wait_time}秒后重试...")
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"[LLM API调用失败] ({api_mode}) 所有重试均失败: {e}")
                # 返回一个安全的默认回复，而不是错误信息
                return "我理解你的感受。能告诉我更多关于这件事的细节吗？"

async def _call_vllm_api_with_retry(messages: List[Dict[str, str]], config: Dict[str, Any], max_retries: int = 3) -> str:
    """
    调用VLLM API（本地部署）- 带重试机制
    """
    for attempt in range(max_retries + 1):
        try:
            return await _call_vllm_api(messages, config)
        except (aiohttp.ClientError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as e:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2
                print(f"[VLLM连接错误] 尝试 {attempt + 1}/{max_retries + 1}: {type(e).__name__}")
                await asyncio.sleep(wait_time)

                # 重新创建会话连接
                await cleanup_llm_session()
                continue
            else:
                raise e
        except Exception as e:
            if attempt < max_retries:
                wait_time = (attempt + 1) * 2
                print(f"[VLLM错误] 尝试 {attempt + 1}/{max_retries + 1}: {e}")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise e

async def _call_vllm_api(messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    """
    调用VLLM API（本地部署）
    """
    # 构建API端点URL
    base_url = config["base_url"].rstrip('/')
    if not base_url.endswith('/v1/chat/completions'):
        url = f"{base_url}/v1/chat/completions"
    else:
        url = base_url

    # 准备请求数据
    data = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"]
    }

    # 准备请求头
    headers = {"Content-Type": "application/json"}

    # 获取会话
    session = await get_llm_session()

    # 发送请求
    async with session.post(
        url,
        headers=headers,
        json=data,
        timeout=aiohttp.ClientTimeout(total=config["timeout"])
    ) as response:

        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"VLLM API请求失败，状态码: {response.status}, 错误信息: {error_text}")

        response_data = await response.json()
        # print(f"response_data:{response_data}")
        # 提取响应内容
        if 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
            else:
                raise Exception(f"API响应格式异常: {response_data}")
        else:
            raise Exception(f"API响应缺少choices字段: {response_data}")

async def _call_openai_compatible_api(messages: List[Dict[str, str]], config: Dict[str, Any], api_type: str) -> str:
    """
    调用OpenAI兼容API（Dashscope和OpenRouter）
    """
    # 构建API端点URL - 根据不同的API类型正确构建URL
    base_url = config["base_url"].rstrip('/')

    if base_url.endswith('/v1/chat/completions'):
        # 如果已经是完整的endpoint，直接使用
        url = base_url
    elif api_type == "openrouter" and base_url.endswith('/v1'):
        # OpenRouter: base_url + /chat/completions
        url = f"{base_url}/chat/completions"
    elif api_type == "dashscope" and base_url.endswith('/compatible-mode/v1'):
        # Dashscope: base_url + /chat/completions
        url = f"{base_url}/chat/completions"
    else:
        # 其他情况：添加标准OpenAI路径
        url = f"{base_url}/v1/chat/completions"

    # 准备请求数据
    data = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"]
    }

    # 准备请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }

    # OpenRouter需要特殊的headers（参考task1/get_bias_label.py）
    if api_type == "openrouter":
        headers.update({
            "HTTP-Referer": "https://openrouter.ai",
            "X-Title": "CogEmo-Agent"
        })

    # 获取会话
    session = await get_llm_session()

    # 发送请求
    async with session.post(
        url,
        headers=headers,
        json=data,
        timeout=aiohttp.ClientTimeout(total=config["timeout"])
    ) as response:

        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"{api_type.upper()} API请求失败，状态码: {response.status}, 错误信息: {error_text}")

        response_data = await response.json()

        # 提取响应内容
        if 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content']
            else:
                raise Exception(f"API响应格式异常: {response_data}")
        else:
            raise Exception(f"API响应缺少choices字段: {response_data}")


def extract_real_answer(response_text: str) -> str:
    if "</think>" in response_text:
        parts = response_text.split("</think>"); real_content = parts[-1].strip().lstrip('\n'); return real_content
    return response_text

async def polish_normal_response(
    patient_content: str,
    original_doctor_reply: str,
    dialogue_stage: str,
    session: aiohttp.ClientSession,
    api_mode: str = "local",
    history_context: str=""
) -> str:
    """
    基于原始医生回复进行润色，提升共情和专业度，但保持原意。
    针对无明显认知偏差的对话，生成高质量的通用咨询回复。
    侧重：共情、积极倾听、信息收集、建立关系
    """
    # 基于原始回复进行润色（所有非偏差样本都进行润色）
    prompt = f"""## 任务
你是一名专业的心理咨询师。请对以下"原始医生回复"进行**润色和优化**。

## 上下文
{history_context}
## 患者当前的发言
"{patient_content}"
## 原始医生回复
"{original_doctor_reply}"

## 当前对话阶段
{dialogue_stage}

## 优化要求
1. **保持原意**：必须保留原始回复中的核心问题或建议（例如，如果原回复问了睡眠，你也要问睡眠，不要改问饮食）。
2. **提升共情**：在原始回复的基础上，增加情感接纳和温暖的语气（例如将"你睡得好吗"改为"听起来最近压力很大，那你的睡眠质量怎么样呢？"）。
3. **风格统一**：保持专业、包容、耐心的咨询师形象。
4. **长度适中**：不要过度扩写，保持口语化，50-150字。
5. **自然交流**：避免AI味，像真实的咨询师一样交流。

## 请输出润色后的回复（仅输出回复内容）："""

    # 调用LLM生成回复
    response = await call_llm_api([{"role": "user", "content": prompt}], api_mode)
    # 清理回复内容
    cleaned_response = extract_real_answer(response).strip()
    return cleaned_response


# 对话阶段定义
CONVERSATION_STAGES = {
    0: {"name": "开始阶段", "desc": "建立关系，问候和了解基本情况", "keywords": ["你好", "开始", "第一次来", "介绍一下"]},
    1: {"name": "症状询问", "desc": "了解具体困扰和问题细节", "keywords": ["具体", "情况", "症状", "什么时候", "怎么想的"]},
    2: {"name": "分析引导", "desc": "帮助认识思维模式，提供分析", "keywords": ["想法", "思维", "角度", "认识", "理解"]},
    3: {"name": "给出帮助", "desc": "提供具体建议和解决方法", "keywords": ["建议", "方法", "策略", "尝试", "练习"]},
    4: {"name": "结束阶段", "desc": "总结讨论，制定后续计划", "keywords": ["总结", "计划", "下次", "继续", "希望"]}
}

# ===== 患者仿真相关代码 (仅用于DPO数据集生成，SFT数据集不使用) =====
# 注意：以下代码仅用于生成仿真对话数据，当前SFT数据集使用真实患者数据

# 患者画像生成配置
PATIENT_PROFILES = {
    "age_groups": ["18-25岁", "26-35岁", "36-45岁", "46-55岁", "56-65岁"],
    "genders": ["男", "女"],
    "education": ["高中及以下", "大专", "本科", "研究生及以上"],
    "occupations": ["学生", "职场新人", "专业技术人员", "管理人员", "自由职业", "退休", "待业"],
    "personalities": ["内向敏感", "外向开朗", "理性分析", "情感丰富", "谨慎多虑", "乐观积极"],
    "topics": ["婚恋", "情绪", "人际", "家庭", "治疗", "成长", "自我", "行为", "职场", "社会", "性心理", "心理学知识"]
}

# 话题标签映射到咨询类型
TOPIC_TO_CONSULTATION_TYPE = {
    "婚恋": "婚恋咨询",
    "情绪": "情绪管理",
    "人际": "人际关系",
    "家庭": "家庭关系",
    "治疗": "心理治疗",
    "成长": "个人成长",
    "自我": "自我认知",
    "行为": "行为调整",
    "职场": "职场心理",
    "社会": "社会适应",
    "性心理": "性心理",
    "心理学知识": "心理教育"
}

def generate_patient_profile(patient_id: int) -> dict:
    """生成患者画像（仅用于DPO数据集仿真）"""
    """
    生成患者画像 - 添加逻辑规则
    """
    import random

    # 首先选择年龄，然后根据年龄确定合适的职业
    age = random.choice(PATIENT_PROFILES["age_groups"])

    # 根据年龄选择合适的职业
    if age in ["18-25岁"]:
        # 年轻人主要在学习和职场初期
        occupation_pool = ["学生", "职场新人", "自由职业", "待业"]
    elif age in ["26-35岁"]:
        # 青年主要在职场发展期
        occupation_pool = ["职场新人", "专业技术人员", "管理人员", "自由职业", "待业"]
    elif age in ["36-45岁"]:
        # 中年主要是职业成熟期
        occupation_pool = ["专业技术人员", "管理人员", "自由职业", "待业"]
    elif age in ["46-55岁"]:
        # 中老年职业稳定期
        occupation_pool = ["专业技术人员", "管理人员", "自由职业", "退休", "待业"]
    else:  # 56-65岁
        # 老年主要在退休期
        occupation_pool = ["管理人员", "自由职业", "退休", "待业"]

    # 根据年龄调整教育背景的概率
    if age in ["18-25岁"]:
        education_pool = ["高中及以下", "大专", "本科", "研究生及以上"]
        # 年轻人更可能有高学历
        education_weights = [0.1, 0.2, 0.5, 0.2]
    elif age in ["56-65岁"]:
        # 老年人更可能有较低的学历
        education_pool = ["高中及以下", "大专", "本科", "研究生及以上"]
        education_weights = [0.4, 0.3, 0.2, 0.1]
    else:
        # 其他年龄段均匀分布
        education_pool = PATIENT_PROFILES["education"]
        education_weights = [0.25, 0.25, 0.25, 0.25]

    profile = {
        "patient_id": patient_id,
        "age": age,
        "gender": random.choice(PATIENT_PROFILES["genders"]),
        "education": random.choices(education_pool, weights=education_weights)[0],
        "occupation": random.choice(occupation_pool),
        "personality": random.choice(PATIENT_PROFILES["personalities"]),
        "primary_topic": random.choice(PATIENT_PROFILES["topics"]),
        "secondary_topics": random.sample(PATIENT_PROFILES["topics"], min(2, random.randint(0, 2)))
    }

    return profile

async def generate_patient_response(
    conversation_history: str,
    patient_profile: dict,
    last_doctor_message: str,
    conversation_stage: int,
    topic_context: str = "",
    api_mode: str = "local"
) -> str:
    """
    患者智能体生成回复
    """
    stage_info = CONVERSATION_STAGES.get(conversation_stage, CONVERSATION_STAGES[0])

    # 构建患者智能体的提示词
    prompt = f"""## 角色定义
你是一名寻求心理咨询的患者。请根据你的个人情况和当前对话阶段，生成真实、自然的回复。

## 个人画像
- 年龄：{patient_profile['age']}
- 性别：{patient_profile['gender']}
- 教育背景：{patient_profile['education']}
- 职业：{patient_profile['occupation']}
- 性格特点：{patient_profile['personality']}
- 主要困扰话题：{patient_profile['primary_topic']}
- 相关话题：{', '.join(patient_profile['secondary_topics'])}

## 当前对话阶段
{conversation_stage}. {stage_info['name']}：{stage_info['desc']}

## 医生刚刚说的话
"{last_doctor_message}"

## 对话历史
{conversation_history}

## 话题背景
{topic_context}

## 回复要求
1. 根据你的个人画像特点，生成符合身份的回复
2. 回复要体现你的性格特征（内向、外向、理性、情感等）
3. 根据当前对话阶段调整回复内容：
   - 开始阶段：表达困扰，寻求帮助
   - 症状询问：详细描述问题和感受
   - 分析引导：对医生的分析做出回应
   - 给出帮助：对建议的反应和疑问
   - 结束阶段：表达感受和期望
4. 语言要自然口语化，像真实患者
5. 回复长度控制在30-150字
6. 不要提及"我是患者"等元认知
7. 不要过于积极配合，要体现真实的犹豫和思考


## 示例

User:  医生您好。 
Doctor:  你好 我们开始吧 请问你最近遇到了什么问题呢？ 

User:  我最近在备考，但是学着学着就会犯困。 
Doctor:  明白，会每天感觉很疲惫，没有精力吗？ 

User:  我最近在备考，但是学着学着就会犯困。没有很疲惫 如果不学习，就很有精神 一看书看一会儿就觉得困了，尤其是早起的上午。 
Doctor:  懂的，会觉得对学习很没有兴趣吗？
...
User: 暂时没有其他困扰。
Doctor:  好的，明白了，最近情绪还好吗？

User: 还可以。
Doctor: 明白了，听上去最近因为备考感觉学习时容易犯困，让你感到有些困扰，不用太担心，可以在日常学习生活中通过运动或其他娱乐活动劳逸结合，注意放松 好的，那我们的问诊就到这里了。

##　我们这是为了模拟真实患者（USER）回复构造的数据样本，你给的回复。要尽可能的像真人。

现在请以患者身份回复："""

    response = await call_llm_api([{"role": "user", "content": prompt}], api_mode)

    # 清理患者回复
    cleaned_response = extract_real_answer(response).strip()

    return cleaned_response

async def generate_agent_conversation(
    patient_profile: dict,
    dqn_inference: DQNInference,
    max_turns: int = 6
) -> dict:
    """
    使用智能体交互生成完整对话
    """
    # 根据患者主要话题动态设置normalizedTag
    primary_topic = patient_profile["primary_topic"]
    normalized_tag = TOPIC_TO_CONSULTATION_TYPE.get(primary_topic, "心理咨询")

    conversation_data = {
        "id": patient_profile["patient_id"],
        "normalizedTag": normalized_tag,
        "patient_profile": patient_profile,
        "conversation_topic": patient_profile["primary_topic"],
        "messages": []
    }

    # 添加system消息
    conversation_data["messages"].append({
        "role": "system",
        "content": "你是一位专业的心理咨询师，能够为来访者提供专业的心理支持和指导，帮助来访者缓解负面情绪，实现心理健康成长。"
    })

    # 对话历史构建
    conversation_history = ""
    current_stage = 0
    turn_count = 0

    # 第一轮：患者开始对话
    initial_patient_message = await generate_patient_response(
        conversation_history="",
        patient_profile=patient_profile,
        last_doctor_message="",
        conversation_stage=0,
        topic_context=f"患者的主要困扰是关于{patient_profile['primary_topic']}方面的问题"
    )

    conversation_data["messages"].append({
        "role": "user",
        "content": initial_patient_message
    })
    conversation_history += f"患者: {initial_patient_message}\n"

    # 多轮智能体对话
    while turn_count < max_turns:
        try:
            # 获取当前对话阶段
            current_stage = get_conversation_stage(turn_count, max_turns + 1)

            # 为DQN分析构建当前状态
            current_context = conversation_history.strip()

            # 使用患者画像信息生成状态分析
            topic_analysis = f"患者画像：{patient_profile['age']}{patient_profile['gender']}，{patient_profile['education']}，{patient_profile['occupation']}，{patient_profile['personality']}性格，主要困扰：{patient_profile['primary_topic']}"

            # 生成1024维嵌入向量（匹配DQN模型期望）
            import hashlib
            state_string = f"{current_context}_{topic_analysis}_{current_stage}"
            # 生成足够的hash值来创建1024维向量
            hash_str = hashlib.md5(state_string.encode()).hexdigest()
            # 重复hash值并转换为1024维向量
            embedding = []
            for i in range(1024):
                char_idx = i % len(hash_str)
                embedding.append(float(ord(hash_str[char_idx])) / 255.0)

            # DQN选择最优策略
            best_action_idx, _ = dqn_inference.get_best_worst_actions(embedding)

            # 医生智能体回复
            doctor_response = await get_doctor_response_async(
                current_context,
                topic_analysis,
                best_action_idx,
                current_stage
            )

            conversation_data["messages"].append({
                "role": "assistant",
                "content": doctor_response
            })
            conversation_history += f"医生: {doctor_response}\n"

            turn_count += 1

            # 检查是否需要结束对话
            if turn_count >= max_turns:
                break

            # 患者智能体回复
            patient_response = await generate_patient_response(
                conversation_history=conversation_history,
                patient_profile=patient_profile,
                last_doctor_message=doctor_response,
                conversation_stage=current_stage + 1,
                topic_context=f"当前讨论{patient_profile['primary_topic']}话题，处于{CONVERSATION_STAGES[current_stage]['name']}"
            )

            conversation_data["messages"].append({
                "role": "user",
                "content": patient_response
            })
            conversation_history += f"患者: {patient_response}\n"

        except Exception as e:
            print(f"[智能体对话生成失败 第{turn_count}轮] {e}")
            break

    return conversation_data

def get_conversation_stage(conversation_turn: int, total_turns: int) -> int:
    """
    根据对话轮次确定当前阶段 - 支持1-20轮对话
    """
    if total_turns <= 2:
        # 极短对话：开始 -> 结束
        return 0 if conversation_turn == 0 else 4
    elif total_turns <= 4:
        # 短对话：开始 -> 询问 -> 结束
        if conversation_turn == 0:
            return 0  # 开始阶段
        elif conversation_turn == total_turns - 1:
            return 4  # 结束阶段
        else:
            return 1  # 症状询问
    elif total_turns <= 8:
        # 中等对话：开始 -> 询问 -> 分析 -> 结束
        stage_progress = conversation_turn / (total_turns - 1)
        if stage_progress < 0.25:
            return 0  # 开始阶段
        elif stage_progress < 0.5:
            return 1  # 症状询问
        elif stage_progress < 0.75:
            return 2  # 分析引导
        else:
            return 4  # 结束阶段
    else:
        # 长对话（9-20轮）：完整流程
        stage_progress = conversation_turn / (total_turns - 1)
        if stage_progress < 0.15:
            return 0  # 开始阶段 (15%)
        elif stage_progress < 0.35:
            return 1  # 症状询问 (20%)
        elif stage_progress < 0.55:
            return 2  # 分析引导 (20%)
        elif stage_progress < 0.80:
            return 3  # 给出帮助 (25%)
        else:
            return 4  # 结束阶段 (20%)

async def get_doctor_response_async(context: str, state_text: str, action_index: int, conversation_stage: int = None, api_mode: str = "local") -> str:
    """
    医生LLM：生成回应 (使用改进的医生角色提示词)
    """
    action = ACTION_SPACE_MAP.get(action_index, {"desc": "未知", "weights": "未知"})

    # 解析对话上下文，提取完整的对话历史
    conversation_history = context.strip()

    # 提取当前患者的最后一句话（要回复的内容）
    lines = conversation_history.split('\n')
    current_patient_message = ""
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("患者: "):
            current_patient_message = line.replace("患者: ", "", 1)
            break

    # 确定对话阶段
    if conversation_stage is None:
        conversation_stage = 0  # 默认开始阶段

    stage_info = CONVERSATION_STAGES.get(conversation_stage, CONVERSATION_STAGES[0])

    # 构建包含阶段指导的专业心理咨询师提示词
    prompt = f"""## 核心任务
你是一名专业心理咨询师。根据患者的表述和当前咨询阶段，生成一句简短、自然、口语化的医生回复。

## 患者当前表述
"{current_patient_message}"

## 当前咨询阶段
{conversation_stage}. {stage_info['name']}：{stage_info['desc']}

## 策略指导
策略类型：{action['desc']}

## 阶段重点
{stage_info['name']}的关键词汇：{', '.join(stage_info['keywords'])}

## 严格要求
1. 回复必须符合当前咨询阶段的重点和风格
2. 回复必须像真实医生说话，简洁自然，1-3句话
3. 严禁任何思考过程、分析、说明、总结
4. 严禁提及策略、阶段、权重、方法等任何技术内容
5. 严禁使用括号、星号等格式标记
6. 直接生成医生会说的话即可

## 示例

User:  医生您好。 
Doctor:  你好 我们开始吧 请问你最近遇到了什么问题呢？ 

User:  我最近在备考，但是学着学着就会犯困。 
Doctor:  明白，会每天感觉很疲惫，没有精力吗？ 

User:  我最近在备考，但是学着学着就会犯困。没有很疲惫 如果不学习，就很有精神 一看书看一会儿就觉得困了，尤其是早起的上午。 
Doctor:  懂的，会觉得对学习很没有兴趣吗？
...
User: 暂时没有其他困扰。
Doctor:  好的，明白了，最近情绪还好吗？

User: 还可以。
Doctor: 明白了，听上去最近因为备考感觉学习时容易犯困，让你感到有些困扰，不用太担心，可以在日常学习生活中通过运动或其他娱乐活动劳逸结合，注意放松 好的，那我们的问诊就到这里了。

##　我们这是为了模拟真实咨询师回复构造的数据样本，你给的回复。要尽可能的像真人。
请基于"## 患者当前表述"给出你的回复："""

    # 调用LLM API
    raw_response = await call_llm_api([{"role": "user", "content": prompt}], api_mode)
    # print(f"raw_response:{raw_response}")
    # 清理响应内容，如果清理失败会抛出异常
    try:
        cleaned_response = extract_real_answer(raw_response)
        # print(f"cleaned_response:{cleaned_response}")
        return cleaned_response
    except ValueError as e:
        print(f"[清理响应失败] {e}")
        raise Exception(f"医生回复生成失败: {e}")

class LLMClient:
    """简单的LLM客户端包装器 (为了向后兼容)"""
    def __init__(self, max_concurrent: int = 32):
        self.max_concurrent = max_concurrent

    async def close(self):
        """关闭会话"""
        await cleanup_llm_session()

    async def generate_response(self, context: str, state_text: str, action_index: int, api_mode: str = "local") -> str:
        """生成单个医生回复"""
        return await get_doctor_response_async(context, state_text, action_index, api_mode=api_mode)

    async def generate_responses_batch(self, requests: list) -> list:
        """批量生成医生回复 (使用task2的直接并发模式)"""
        # 🔥 关键优化：真正的并发LLM调用 (直接学习task2)
        # 创建所有LLM任务
        doctor_tasks = []
        for context, state_text, action_index in requests:
            task = get_doctor_response_async(context, state_text, action_index)
            doctor_tasks.append(task)

        # 并发执行所有医生LLM调用
        results = await asyncio.gather(*doctor_tasks, return_exceptions=True)
        return results

def load_initial_states(path: str) -> List[Dict]:
    """加载初始状态数据"""
    print(f"正在加载状态数据: {path}")
    s_pool = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if "embedding" in data and "context" in data and "s_text" in data:
                    s_pool.append(data)
        print(f"成功加载 {len(s_pool)} 个状态")
        return s_pool
    except Exception as e:
        print(f"[错误] 状态数据加载失败: {e}")
        return []

def convert_context_to_messages(context: str) -> List[Dict[str, str]]:
    """将上下文转换为ChatML格式的消息列表"""
    messages = []
    history_lines = context.strip().split('\n')
    for line in history_lines:
        line = line.strip()
        if line.startswith("患者: "):
            messages.append({"role": "user", "content": line.replace("患者: ", "", 1)})
        elif line.startswith("医生: "):
            messages.append({"role": "assistant", "content": line.replace("医生: ", "", 1)})
    return messages

def save_jsonl(data: List[Dict], filepath: str):
    """保存JSONL格式文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存到: {filepath}")

def save_json(data: List[Dict], filepath: str):
    """保存JSON格式文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {filepath}")

def build_conversation_history(initial_states: List[Dict], max_turns: int = 6) -> List[Dict]:
    """
    构建完整的多轮对话历史
    Args:
        initial_states: 初始状态数据列表
        max_turns: 最大对话轮数
    Returns:
        完整对话样本列表
    """
    conversations = []

    # 从每个初始状态开始构建对话
    for i, initial_state in enumerate(initial_states):
        # 随机决定对话长度（3-6轮）
        conversation_length = min(max_turns, 3 + (i % 4))  # 3-6轮

        conversation_data = {
            "id": i,
            "normalizedTag": "心理咨询",
            "messages": []
        }

        # 添加system消息
        conversation_data["messages"].append({
            "role": "system",
            "content": "你是一位专业的心理咨询师，能够为来访者提供专业的心理支持和指导，帮助来访者缓解负面情绪，实现心理健康成长。"
        })

        # 构建对话历史
        current_context = ""
        current_turn = 0

        # 第一轮：用户发起对话
        user_messages = [
            "医生，请问我做选择特别困难，该怎么办？",
            "医生你好。",
            "请开始心理咨询对话",
            "我感觉最近情绪不太好，想找您聊聊。"
        ]

        user_msg = user_messages[i % len(user_messages)]
        conversation_data["messages"].append({
            "role": "user",
            "content": user_msg
        })
        current_context += f"患者: {user_msg}\n"

        while current_turn < conversation_length - 1:
            # 医生回复
            doctor_response = f"这是第{current_turn + 1}轮医生的回复"  # 这里会被后续LLM调用替换
            conversation_data["messages"].append({
                "role": "assistant",
                "content": doctor_response
            })
            current_context += f"医生: {doctor_response}\n"

            # 用户下一轮消息
            if current_turn < conversation_length - 2:
                # 模拟用户回复（使用初始状态中的实际数据）
                if current_turn < len(initial_states[i].get("conversation_history", [])):
                    next_user_msg = initial_states[i]["conversation_history"][current_turn]
                else:
                    # 使用初始的s_text作为用户消息
                    next_user_msg = initial_state["s_text"]

                conversation_data["messages"].append({
                    "role": "user",
                    "content": next_user_msg
                })
                current_context += f"患者: {next_user_msg}\n"

            current_turn += 1

        conversations.append(conversation_data)

    return conversations

def create_cumulative_test_format(conversations: List[Dict]) -> List[Dict]:
    """
    创建累积式测试集格式 - 匹配D4_test.json格式
    Args:
        conversations: 完整对话列表
    Returns:
        累积式测试样本列表
    """
    cumulative_samples = []
    sample_id = 0

    for conversation in conversations:
        messages = conversation["messages"]
        base_id = conversation["id"]

        # 第一轮：system + user + assistant（完整的第一轮对话）
        if len(messages) >= 3:
            cumulative_sample = {
                "id": base_id,
                "sample_id": sample_id,
                "normalizedTag": conversation["normalizedTag"],
                "messages": messages[:3]  # system + user + assistant
            }
            cumulative_samples.append(cumulative_sample)
            sample_id += 1

        # 后续轮：每一轮增加user + assistant
        for i in range(3, len(messages), 2):  # 从第2个user消息开始
            if i + 1 < len(messages):  # 确保有完整的user + assistant轮次
                # 截取从开始到当前assistant回复的所有消息
                cumulative_messages = messages[:i+2]

                cumulative_sample = {
                    "id": base_id,
                    "sample_id": sample_id,
                    "normalizedTag": conversation["normalizedTag"],
                    "messages": cumulative_messages
                }

                cumulative_samples.append(cumulative_sample)
                sample_id += 1

    return cumulative_samples