"""
脚本1：偏差链标注脚本 (三合一合并版)
功能：读取原始对话JSON (如 D4_train.json / D4_test.json)，
      异步并发调用LLM（已配置为本地Qwen3-8b），
      解析CBT状态，并将结果保存为JSON文件（含原结构与逐条消息的annotation）。
"""
import json
import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
import traceback
import re
import argparse

# --- 1. LLM 客户端模块 (修复版 - 采用VLLM调用方式) ---

import aiohttp
from typing import Dict

# 本地VLLM API配置
LOCAL_API_CONFIG = {
    "model": "Qwen3-30B-A3B", # Qwen3-30B-A3B、qwen3-8b
    "base_url": "http://localhost:7862", # VLLM使用基础URL
    "api_key": "None", # 本地API key
    "temperature": 0.0,
    "max_tokens": 512,
    "timeout": 30
}

# DASHSCOPE API配置
DASHSCOPE_API_CONFIG = {
    "model": "qwen-turbo",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-bdf466c95c6140b38a3d0766ae0765bb",
    "temperature": 0.0,
    "max_tokens": 512,
    "timeout": 30
}

# OPENROUTER API配置
OPENROUTER_API_CONFIG = {
    "model": "google/gemini-2.5-flash-lite-preview-09-2025",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "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",
    "temperature": 0.0,
    "max_tokens": 512,
    "timeout": 30
}

# 全局aiohttp会话
async_session = None

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

async def get_llm_session():
    """获取或创建aiohttp会话"""
    global async_session
    if async_session is None:
        async_session = aiohttp.ClientSession()
    return async_session

async def get_llm_response(messages: List[Dict[str, str]], api_mode: str = "local") -> str:
    """
    异步调用LLM API，支持多种API模式

    Args:
        messages: 消息列表
        api_mode: API模式 ("local", "dashscope", "openrouter")

    Returns:
        LLM响应文本
    """
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
        print(f"LLM 调用出错 ({api_mode}): {e}")
        return f"LLM_ERROR: {e}"

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
    headers = {
        "Content-Type": "application/json"
    }

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

        # 提取响应内容
        if 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'message' in choice:
                result = choice['message'].get('content', '')
                if result is None:
                    result = ""
                else:
                    result = result.strip()
            else:
                result = ""
        else:
            result = ""

        return result

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

    # OpenRouter需要特殊的headers（参考llm_client.py）
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
            if 'message' in choice:
                result = choice['message'].get('content', '')
                if result is None:
                    result = ""
                else:
                    result = result.strip()
            else:
                result = ""
        else:
            result = ""

        return result

def extract_real_answer(response_text: str) -> str:
    """
    提取LLM输出中`</think>`后面的真实内容

    Args:
        response_text: LLM原始响应文本

    Returns:
        str: 处理后的真实内容
    """
    # 查找最后一个`</think>`后的内容
    if "</think>" in response_text:
        # 分割字符串，取最后一部分
        parts = response_text.split("</think>")
        if len(parts) > 1:
            # 获取最后一个`</think>`后面的内容
            real_content = parts[-1].strip()
            # 清理开头的换行符
            real_content = real_content.lstrip('\n')
            return real_content

    # 如果没有`</think>`，返回原文本
    return response_text

async def cleanup_llm_session():
    """清理LLM会话"""
    global async_session
    if async_session:
        await async_session.close()
        async_session = None

# --- 2. 提示词与解析模块 (来自 prompts_preprocessing.py) ---

def get_prompt(patient_utterance: str, history: str) -> str:
    """
    生成CBT（认知行为疗法）分析提示词
    """
    return f"""[SYSTEM]
你是一个专业的CBT（认知行为疗法）分析专家。你的任务是：只对“当前患者发言”进行标注。
请特别注意各个标签使用的信息范围：
- 认知偏差标签 (t)：只能使用当前发言的内容，禁止使用对话上下文。
- 偏差强度 (i)：只能使用当前发言的内容，禁止使用对话上下文。
- 对话阶段 (p)：可以结合上下文和当前发言判断。
- 安全风险 (r)：可以结合上下文和当前发言判断。

如果当前发言本身没有明显的认知偏差特征，则必须将“认知偏差标签”标为“无”，
不能因为历史对话曾经出现偏差就延续打标签。

[USER]
# 定义列表
## 1. 对话阶段 (p)
- 开场（建立关系，初步问候。）
- 问题澄清（识别和定义患者的核心问题。）
- 情绪疏导（处理当下的强烈情绪，提供共情。）
- 策略制定（教授CBT技能，纠正偏差，制定行动计划。）
- 收尾（总结，布置任务，结束对话。）

## 2. 认知偏差标签 (t)
[非黑即白, 过度概括, 灾难化, 读心术, 情感推理, 应该句式, 标签化, 个人化, 无]
- 非黑即白：用“要么…要么…”、“不是100分就是0分”、“不是成功就是失败”等极端两分法看问题。
  关键词例：完全、根本、一点都不、彻底、没用、全都、全完了。

- 过度概括：从一次或少数几次经历就推断“总是这样”、“每次都这样”、“以后都不会变”。
  关键词例：总是、每次、从来、永远、一直、所有人都。

- 灾难化：把事情往最坏、最极端的方向想，预言“肯定完了”、“后果不堪设想”。
  关键词例：完了、崩溃了、毁了、受不了、绝对不行、一点希望都没有。

- 读心术：在没有明确证据的情况下，臆测别人怎么想自己。
  关键词例：他们一定觉得…、他肯定看不起我、大家都在背后笑我。

- 情感推理：因为“我现在这样感觉”，就认定“现实就是这样”。
  典型模式：“我觉得自己很差/没用，所以我肯定很差/没用”“我很害怕，所以事情肯定会出问题”。
  ⚠ 仅仅表达情绪（如“我很难过”“最近很烦”），如果没有把情绪当成证据去下结论，不算情感推理，应标为“无”。

- 应该句式：用很多“应该/必须/一定要”来苛求自己或他人。
  关键词例：我应该…、我必须…、别人就应该…、绝对不能…。

- 标签化：给自己或他人贴上固定的、否定性的标签。
  关键词例：我是废物、我是失败者、他就是渣男、我就是一无是处。

- 个人化：把并非完全由自己决定的结果，全都怪在自己身上。
  关键词例：都是我的错、如果不是我…就不会这样、出了问题一定是我没做好。

- 无：当前发言只是描述事实、表达感受或提问，没有明显的认知偏差模式。

说明：
- “无”：当前发言只是描述事实、简单表达感受或打招呼，没有出现明显的认知偏差模式。
- 禁止因为上一轮有偏差，就在这一轮“顺手”加同样标签；每一轮都要重新判断。

## 3. 安全风险 (r)
[高危 (如：自残、自杀念头), 中危 (如：绝望、无助), 低危 (如：焦虑、抑郁), 无 (如：一般烦恼)]

## 4. 偏差强度 (i)
[严重 (患者对此深信不疑), 中等 (患者流露疑虑，但仍倾向相信), 轻微 (患者仅部分表现出偏差), 无 (未识别到偏差)]

# 重要规则（请严格遵守）
1. 认知偏差标签 (t) 和偏差强度 (i) 只根据【当前患者发言】判断：
   - 不得引用对话上下文中的内容作为打标签的依据。
   - 如果当前发言里没有明显的偏差特征，必须标注为“无”。
2. 对话阶段 (p) 和安全风险 (r) 可以综合考虑【对话上下文】和【当前发言】。
3. 在“理由”中：
   - 解释认知偏差标签和偏差强度时，只能引用【当前患者发言】中的原话或含义；
   - 不得引用历史轮次的具体内容（例如“上一句你说…”、“之前你提到…”等）。
   - 一旦理由中出现历史轮次内容，说明你违反了规则。
4. 如果你不确定当前发言是否包含认知偏差，请优先选择“无”，而不是随意选择“情感推理”。

# 示例（反例纠正）
【历史对话片段】
患者：两周多了，感觉自己陷进那种自卑的状态了，人也比较丧，特别嗜睡。
（这一句可以标注：情感推理 / 低自尊等）

助手：最近两周心情怎么样呀？有没有情绪很低落？

患者：除了没自信，其他都还好。

【正确标注示范】
对于“除了没自信，其他都还好。”这一句：
- 认知偏差标签：无
- 偏差强度：无
理由：这句话只是描述当前状态（有点没自信，但其他还好），没有体现出“绝对化”“灾难化”“读心”“情绪当事实”等偏差模式。

---
no think
# 对话上下文（仅可用于判断对话阶段和安全风险，不能用于认知偏差标签）
{history}

# 【当前患者发言】（认知偏差标签和强度只能基于下面这一句）
{patient_utterance}

# 输出格式（请严格按照以下格式输出）：
认知偏差标签：(从“定义列表2”中选择，可多选，或填“无”)
安全风险：(从“定义列表3”中选择)
对话阶段：(从“定义列表1”中选择)
偏差强度：(从“定义列表4”中选择)
理由：(用1-2句话解释你为什么这么标注，特别是偏差标签的理由。解释偏差时只引用当前发言的内容。)

[ASSISTANT]
"""

def parse_cbt_response(response_text: str) -> Dict[str, Any]:
    """
    解析LLM输出的CBT分析文本。
    (这是您要求的新增功能，用于解析文本)
    """
    # 提取真实回答内容
    real_answer = extract_real_answer(response_text)

    parsed = {
        "bias_tags": [],
        "risk_level": "未知",
        "dialogue_stage": "未知",
        "bias_intensity": "未知",
        "reason": "",
        "raw_response": real_answer # 保存处理后的回复
    }

    try:
        # 使用处理后的真实回答进行解析
        # 提取 认知偏差标签
        tags_match = re.search(r"认知偏差标签：(.*?)(?:\n|$)", real_answer)
        if tags_match:
            tags_str = tags_match.group(1).strip()
            if tags_str == "无":
                parsed["bias_tags"] = ["无"]
            else:
                # 从列表中匹配
                all_biases = ["非黑即白", "过度概括", "灾难化", "读心术", "情感推理", "应该句式", "标签化", "个人化"]
                found_tags = [tag for tag in all_biases if tag in tags_str]
                parsed["bias_tags"] = found_tags if found_tags else ["无"]

        # 提取 安全风险
        risk_match = re.search(r"安全风险：(.*?)(?:\n|$)", real_answer)
        if risk_match:
            parsed["risk_level"] = risk_match.group(1).strip()
            # 规范化（例如 "高危 (...)" -> "高危"）
            if "高危" in parsed["risk_level"]: parsed["risk_level"] = "高危"
            elif "中危" in parsed["risk_level"]: parsed["risk_level"] = "中危"
            elif "低危" in parsed["risk_level"]: parsed["risk_level"] = "低危"
            elif "无" in parsed["risk_level"]: parsed["risk_level"] = "无"


        # 提取 对话阶段
        stage_match = re.search(r"对话阶段：(.*?)(?:\n|$)", real_answer)
        if stage_match:
            parsed["dialogue_stage"] = stage_match.group(1).strip()
            # 规范化 (例如 "1. 开场 (...)" -> "1. 开场")
            if "1." in parsed["dialogue_stage"]: parsed["dialogue_stage"] = "1. 开场"
            elif "2." in parsed["dialogue_stage"]: parsed["dialogue_stage"] = "2. 问题澄清"
            elif "3." in parsed["dialogue_stage"]: parsed["dialogue_stage"] = "3. 情绪疏导"
            elif "4." in parsed["dialogue_stage"]: parsed["dialogue_stage"] = "4. 策略制定"
            elif "5." in parsed["dialogue_stage"]: parsed["dialogue_stage"] = "5. 收尾"


        # 提取 偏差强度
        intensity_match = re.search(r"偏差强度：(.*?)(?:\n|$)", real_answer)
        if intensity_match:
            parsed["bias_intensity"] = intensity_match.group(1).strip()
            if "严重" in parsed["bias_intensity"]: parsed["bias_intensity"] = "严重"
            elif "中等" in parsed["bias_intensity"]: parsed["bias_intensity"] = "中等"
            elif "轻微" in parsed["bias_intensity"]: parsed["bias_intensity"] = "轻微"
            elif "无" in parsed["bias_intensity"]: parsed["bias_intensity"] = "无"

        # 提取 理由
        reason_match = re.search(r"理由：(.*?)(?:\n|$)", real_answer, re.DOTALL)
        if reason_match:
            parsed["reason"] = reason_match.group(1).strip()
            
        return parsed

    except Exception as e:
        print(f"解析文本失败: {e}\n原文: {response_text[:100]}...")
        parsed["reason"] = f"PARSE_ERROR: {e}"
        return parsed


# --- 3. 偏差链解析器 (来自 get_bias_chain.py) ---

class BiasChainParser:
    """
    偏差链解析器 - 异步并发处理对话
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        max_workers: int = 5,
        delay_between_calls: float = 0.5,
        dataset_type: str = "test",  # "train" 或 "test"
        roles_to_label: str = "both",  # "user" | "assistant" | "both"
        api_mode: str = "local"  # API模式选择
    ):
        self.input_path = input_path
        self.output_path = output_path
        # 确保输出目录存在 - 修复相对路径问题
        output_dir = os.path.dirname(output_path)
        if output_dir:  # 只有当目录名不为空时才创建
            os.makedirs(output_dir, exist_ok=True)

        self.max_workers = max_workers
        self.delay_between_calls = delay_between_calls
        self.parsed_count = 0
        self.error_count = 0
        self.output_lock = Lock()
        self.dataset_type = dataset_type
        self.roles_to_label = roles_to_label
        self.api_mode = api_mode

        # 内存收集：{ dialogue_id: { turn_id: {role, content, annotation} } }
        self.collected: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def __del__(self):
        # 无需关闭输出文件（改为一次性JSON保存）
        pass

    def _build_history_str(self, messages: List[Dict[str, str]], current_turn: int) -> str:
        """构建字符串格式的对话历史"""
        history_list = []
        for i in range(current_turn): # 只包含当前轮次之前的
            msg = messages[i]
            role = "患者" if msg["role"] == "user" else "医生"
            history_list.append(f"{role}: {msg['content']}")
        
        if not history_list:
            return "无对话历史。"
        return "\n".join(history_list)

    async def _process_message(self, dialogue_id: int, turn_id: int, message: Dict[str, str], history_str: str, pbar: tqdm):
        """处理单条消息：调用LLM、解析、保存"""
        try:
            # 1. 获取提示词
            prompt = get_prompt(message["content"], history_str)
            # print(f"prompt:{prompt}")
            # 2. 调用LLM (异步)
            # system prompt 通常是 "You are a helpful assistant."
            # 我们的 get_prompt 已经包含了所有指令，所以用 user role
            llm_messages = [{"role": "user", "content": prompt}]
            response_text = await get_llm_response(llm_messages, self.api_mode)
            
            await asyncio.sleep(self.delay_between_calls) # 控制速率

            if "LLM_ERROR:" in response_text:
                raise Exception(response_text)

            # 3. 解析回复
            parsed_result = parse_cbt_response(response_text)

            # 4. 收集到内存结构
            with self.output_lock:
                if dialogue_id not in self.collected:
                    self.collected[dialogue_id] = {}
                self.collected[dialogue_id][turn_id] = {
                    "role": message["role"],
                    "content": message["content"],
                    "annotation": parsed_result
                }
                self.parsed_count += 1
                
            pbar.update(1) # 更新总进度条

        except Exception as e:
            with self.output_lock:
                self.error_count += 1
            print(f"[错误] Dialogue {dialogue_id}, Turn {turn_id}: {e}\n{traceback.format_exc()}")
            pbar.update(1) # 即使失败也要更新进度条

    async def _process_dialogue_async(self, dialogue: Dict[str, Any], pbar: tqdm, semaphore: asyncio.Semaphore):
        """
        (子任务) 异步处理单个对话中的所有消息
        """
        # 使用id作为对话分组标识；若无则退化到sample_id/id
        dialogue_id = dialogue.get("id", dialogue.get("sample_id", -1))
        messages = dialogue["messages"]

        tasks = []
        for turn_id, message in enumerate(messages):
            # 训练/测试集标注角色控制
            if self.roles_to_label == "user" and message["role"] != "user":
                continue
            if self.roles_to_label == "assistant" and message["role"] != "assistant":
                continue
            # system通常不参与标注
            if message["role"] == "system":
                continue

  
            # 前缀历史（不含当前）
            try:
                history_str = self._build_history_str(messages, turn_id)
            except Exception:
                history_str = "无对话历史。"

            # 使用信号量控制并发
            await semaphore.acquire()
            task = asyncio.create_task(
                self._process_message_wrapper(
                    dialogue_id, turn_id, message, history_str, pbar, semaphore
                )
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def _process_message_wrapper(self, *args):
        """包装 _process_message 以便在 finally 中释放信号量"""
        semaphore = args[-1]
        try:
            await self._process_message(*args[:-1])
        finally:
            semaphore.release()

    def load_dialogues(self) -> List[Dict[str, Any]]:
        """从JSON文件加载对话
        - 训练集：直接返回列表
        - 测试集：每个id仅保留最后一个sample_id对应的样本（累加式对话的最终版本）
        """
        print(f"正在从 {self.input_path} 加载数据...")
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载 {len(data)} 条记录。")

            if self.dataset_type == "test":
                # 分组：id -> max(sample_id) 保留
                latest_by_id: Dict[Any, Dict[str, Any]] = {}
                for item in data:
                    did = item.get("id")
                    sid = item.get("sample_id", -1)
                    if did is None:
                        # 回退到原行为
                        continue
                    if did not in latest_by_id or sid > latest_by_id[did].get("sample_id", -1):
                        latest_by_id[did] = item
                reduced = list(latest_by_id.values())
                print(f"测试集模式：按id聚合后剩余 {len(reduced)} 条（取每个id最后一个sample_id）。")
                return reduced
            else:
                # 训练集直接返回
                return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            return []

    async def run_parser(self, limit: Optional[int] = None):
        """
        异步运行整个解析流程
        """
        dialogues = self.load_dialogues()
        if not dialogues:
            return

        if limit:
            print(f"--- 测试模式：仅处理 {limit} 条对话 ---")
            dialogues = dialogues[:limit]

        # 计算总共需要标注的消息数（按角色过滤）
        def need_label(role: str) -> bool:
            if role == "system":
                return False
            if self.roles_to_label == "both":
                return True
            return role == self.roles_to_label

        total_messages = sum(1 for d in dialogues for m in d["messages"] if need_label(m["role"]))

        if total_messages == 0:
            print("没有找到需要标注的消息。")
            return

        print(f"总对话数: {len(dialogues)}")
        print(f"总待标注消息数: {total_messages}（角色={self.roles_to_label}）")
        print(f"并发数 (max_workers): {self.max_workers}")

        # 创建信号量以控制并发
        semaphore = asyncio.Semaphore(self.max_workers)

        try:
            with tqdm(total=total_messages, desc="标注进度") as pbar:
                dialogue_tasks = [
                    self._process_dialogue_async(dialogue, pbar, semaphore)
                    for dialogue in dialogues
                ]
                await asyncio.gather(*dialogue_tasks)

            print("\n--- 标注完成 ---")
            print(f"成功标注: {self.parsed_count}")
            print(f"失败/错误: {self.error_count}")
            # 将标注结果合并回原始结构并保存为JSON
            self._save_as_json(dialogues)
            print(f"标注结果已保存至(JSON): {self.output_path}")

        finally:
            # 确保清理LLM会话
            await cleanup_llm_session()

        # 质量检查和统计
        self.run_quality_check()

    def _save_as_json(self, dialogues: List[Dict[str, Any]]):
        """
        将内存中的collected标注合并到对话结构中，并保存为单个JSON文件：
        - 对每条消息（user/assistant，排除system）附加一个annotation字段
        - 测试集：已在load阶段聚合为每个id的最后一个sample样本
        """
        output_dialogues = []
        for dia in dialogues:
            # 深拷贝（避免原对象被修改）
            d = json.loads(json.dumps(dia, ensure_ascii=False))
            did = d.get("id", d.get("sample_id", -1))
            if did in self.collected:
                for turn_id, msg in enumerate(d.get("messages", [])):
                    if msg.get("role") == "system":
                        continue
                    ann = self.collected.get(did, {}).get(turn_id)
                    if ann:
                        # 仅写解析出的annotation，避免重复保存role/content
                        msg["annotation"] = ann["annotation"]
            output_dialogues.append(d)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dialogues, f, ensure_ascii=False, indent=2)

    def run_quality_check(self):
        """
        运行质量检查和统计分析
        """
        print("\n" + "="*60)
        print("📊 质量检查和统计分析")
        print("="*60)

        try:
            # 读取输出文件进行统计分析（从JSON内展开annotation）
            if not os.path.exists(self.output_path):
                print(f"❌ 输出文件 {self.output_path} 不存在，无法进行质量检查")
                return

            annotations = []
            with open(self.output_path, 'r', encoding='utf-8') as f:
                out_data = json.load(f)
                for dia in out_data:
                    for msg in dia.get("messages", []):
                        ann = msg.get("annotation")
                        if ann:
                            annotations.append(ann)

            total_annotations = len(annotations)
            if total_annotations == 0:
                print("❌ 没有找到有效的标注数据")
                return

            print(f"📋 总标注数量: {total_annotations}")

            # 1. 基础统计
            success_rate = (self.parsed_count / (self.parsed_count + self.error_count)) * 100 if (self.parsed_count + self.error_count) > 0 else 0
            print(f"✅ 成功率: {success_rate:.1f}% ({self.parsed_count}成功 / {self.parsed_count + self.error_count}总计)")
            print(f"❌ 失败数量: {self.error_count}")

            # 2. 认知偏差标签统计
            bias_stats = self._analyze_bias_tags(annotations)
            print(f"\n🏷️  认知偏差标签分布:")
            print("-" * 40)
            for bias, count in sorted(bias_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_annotations) * 100
                print(f"  {bias:8s}: {count:4d} ({percentage:5.1f}%)")

            # 3. 安全风险统计
            risk_stats = self._analyze_risk_levels(annotations)
            print(f"\n⚠️  安全风险分布:")
            print("-" * 40)
            for risk, count in sorted(risk_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_annotations) * 100
                print(f"  {risk:6s}: {count:4d} ({percentage:5.1f}%)")

            # 4. 对话阶段统计
            stage_stats = self._analyze_dialogue_stages(annotations)
            print(f"\n💬 对话阶段分布:")
            print("-" * 40)
            for stage, count in sorted(stage_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_annotations) * 100
                print(f"  {stage:12s}: {count:4d} ({percentage:5.1f}%)")

            # 5. 偏差强度统计
            intensity_stats = self._analyze_bias_intensity(annotations)
            print(f"\n📈 偏差强度分布:")
            print("-" * 40)
            for intensity, count in sorted(intensity_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_annotations) * 100
                print(f"  {intensity:6s}: {count:4d} ({percentage:5.1f}%)")

            # 6. 质量评估
            print(f"\n🎯 质量评估:")
            print("-" * 40)

            # 计算偏差覆盖情况
            bias_coverage = len([b for b in bias_stats.keys() if b != "无"]) / 8 * 100
            print(f"  偏差类型覆盖: {bias_coverage:.1f}% (8种偏差类型中有{len([b for b in bias_stats.keys() if b != '无'])}种被识别)")

            # 有偏差的样本比例
            biased_samples = sum(count for bias, count in bias_stats.items() if bias != "无")
            bias_ratio = (biased_samples / total_annotations) * 100
            print(f"  有偏差样本比例: {bias_ratio:.1f}% ({biased_samples}/{total_annotations})")

            # 平均每个样本的偏差数量
            avg_biases_per_sample = biased_samples / total_annotations
            print(f"  平均每样本偏差数: {avg_biases_per_sample:.2f}")

            # 解析质量检查
            parse_errors = sum(1 for ann in annotations if "PARSE_ERROR" in ann.get("reason", ""))
            if parse_errors > 0:
                print(f"  ⚠️  解析错误: {parse_errors} 个样本")
            else:
                print(f"  ✅ 解析质量: 无解析错误")

            print("\n" + "="*60)
            print("🎉 质量检查完成！")
            print("="*60)

        except Exception as e:
            print(f"❌ 质量检查过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_bias_tags(self, annotations: List[Dict]) -> Dict[str, int]:
        """分析认知偏差标签分布"""
        bias_counts = {}

        # 定义8种标准偏差
        standard_biases = ["非黑即白", "过度概括", "灾难化", "读心术", "情感推理", "应该句式", "标签化", "个人化"]

        # 初始化计数器
        for bias in standard_biases + ["无"]:
            bias_counts[bias] = 0

        for ann in annotations:
            bias_tags = ann.get("bias_tags", [])
            if "无" in bias_tags:
                bias_counts["无"] += 1
            else:
                for bias in bias_tags:
                    if bias in bias_counts:
                        bias_counts[bias] += 1
                    else:
                        bias_counts[bias] = 1  # 处理未预见的偏差标签

        return bias_counts

    def _analyze_risk_levels(self, annotations: List[Dict]) -> Dict[str, int]:
        """分析安全风险分布"""
        risk_counts = {}
        for ann in annotations:
            risk = ann.get("risk_level", "未知")
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        return risk_counts

    def _analyze_dialogue_stages(self, annotations: List[Dict]) -> Dict[str, int]:
        """分析对话阶段分布"""
        stage_counts = {}
        for ann in annotations:
            stage = ann.get("dialogue_stage", "未知")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        return stage_counts

    def _analyze_bias_intensity(self, annotations: List[Dict]) -> Dict[str, int]:
        """分析偏差强度分布"""
        intensity_counts = {}
        for ann in annotations:
            intensity = ann.get("bias_intensity", "未知")
            intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
        return intensity_counts


# --- 4. 主程序入口 ---

def parse_args():
    parser = argparse.ArgumentParser(description="偏差链标注脚本：读取对话JSON，调用LLM进行标注，并输出为JSON文件。")
    parser.add_argument(
        "--input-file",
        type=str,
        default="/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/D4_test.json",
        help="输入JSON路径（测试集或训练集）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs/D4_test_annotated.json",
        help="输出JSON路径（写入带annotation的对话结构）"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="并发数量（异步请求的最大并发）"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="两次请求之间的延迟（秒）"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["test", "train"],
        default="test",
        help="数据集类型：test=仅取每个id的最后一个sample；train=逐条样本"
    )
    parser.add_argument(
        "--roles-to-label",
        type=str,
        choices=["user", "assistant", "both"],
        default="user",
        help="需要标注的角色"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="测试模式：仅处理少量对话（由 --test-limit 控制）"
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=1,
        help="测试模式下处理的对话数量"
    )
    parser.add_argument(
        "--api-mode",
        type=str,
        choices=["local", "dashscope", "openrouter"],
        default="local",
        help="API模式选择：local=本地VLLM，dashscope=阿里云通义千问，openrouter=OpenRouter"
    )
    return parser.parse_args()
'''
python get_bias_label.py \
--input-file /Users/zl_24/Documents/Codes/CogEmo-Agent/data/train/CPsyCounD_train.json \
--output-file /Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs/CPsyCounD_train_annotated.json \
--dataset-type train \




python get_bias_label.py \
--api-mode local \
--input-file /Users/zl_24/Documents/Codes/CogEmo-Agent/data/train/PsyDTCorpus_train.json \
--output-file /Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs/PsyDTCorpus_train_annotated-local.json \
--dataset-type train \
--test-mode \
--test-limit 1000
&& \
python get_bias_label.py \
--api-mode local \
--input-file /Users/zl_24/Documents/Codes/CogEmo-Agent/data/test/PsyDTCorpus_test.json \
--output-file /Users/zl_24/Documents/Codes/CogEmo-Agent/task1/outputs/PsyDTCorpus_test_annotated-local.json \
--dataset-type test
'''
if __name__ == "__main__":
    # --- 解析命令行参数 ---
    args = parse_args()
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    MAX_WORKERS = args.max_workers
    DELAY = args.delay
    DATASET_TYPE = args.dataset_type
    ROLES_TO_LABEL = args.roles_to_label
    TEST_MODE = args.test_mode
    TEST_LIMIT = args.test_limit
    API_MODE = args.api_mode
    
    # --- 执行 ---
    print("=== 偏差链标注脚本 (合并版) ===")
    print(f"当前API模式: {API_MODE}")

    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件 {INPUT_FILE} 未找到。")
    else:
        parser = BiasChainParser(
            input_path=INPUT_FILE,
            output_path=OUTPUT_FILE,
            max_workers=MAX_WORKERS,
            delay_between_calls=DELAY,
            dataset_type=DATASET_TYPE,
            roles_to_label=ROLES_TO_LABEL,
            api_mode=API_MODE
        )
        
        start_time = time.time()
        
        try:
            asyncio.run(parser.run_parser(limit=TEST_LIMIT if TEST_MODE else None))
        except KeyboardInterrupt:
            print("\n用户中断。")
        
        end_time = time.time()
        
        print(f"总耗时: {end_time - start_time:.2f} 秒")
