"""
CPRL配置加载器
统一管理YAML配置文件的加载和消融实验参数的解析
"""
import yaml
import os
from typing import Dict, Any, List


class CPRLConfig:
    """CPRL配置管理类"""

    def __init__(self, config_path: str = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，默认使用当前目录下的config.yaml
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        self.config_path = config_path
        self.config = self._load_config()
        self._setup_ablation_settings()

    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 成功加载配置文件: {self.config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 配置文件未找到: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"❌ 配置文件格式错误: {e}")

    def _setup_ablation_settings(self):
        """直接使用配置文件中的消融开关设置"""
        # 不再需要mode参数，直接使用各个disable_*开关
        self._print_ablation_settings()

    def _print_ablation_settings(self):
        """打印当前消融设置"""
        ablation_config = self.config['ablation']
        print("\n📋 消融实验设置:")
        print(f"  - KL约束: {'❌ 禁用' if ablation_config['disable_kl_constraint'] else '✅ 启用'}")
        print(f"  - DDQN: {'❌ 禁用' if ablation_config['disable_ddqn'] else '✅ 启用'}")
        print(f"  - 安全奖励: {'❌ 禁用' if ablation_config['disable_safety_reward'] else '✅ 启用'}")
        print(f"  - 策略匹配奖励: {'❌ 禁用' if ablation_config['disable_strategy_match_reward'] else '✅ 启用'}")
        print(f"  - 症状改善奖励: {'❌ 禁用' if ablation_config['disable_symptom_improvement_reward'] else '✅ 启用'}")
        print()

    # 便捷的属性访问方法
    @property
    def is_kl_disabled(self) -> bool:
        """KL约束是否禁用"""
        return self.config['ablation']['disable_kl_constraint']

    @property
    def is_ddqn_disabled(self) -> bool:
        """DDQN是否禁用"""
        return self.config['ablation']['disable_ddqn']

    @property
    def is_safety_reward_disabled(self) -> bool:
        """安全奖励是否禁用"""
        return self.config['ablation']['disable_safety_reward']

    @property
    def is_strategy_match_reward_disabled(self) -> bool:
        """策略匹配奖励是否禁用"""
        return self.config['ablation']['disable_strategy_match_reward']

    @property
    def is_symptom_improvement_reward_disabled(self) -> bool:
        """症状改善奖励是否禁用"""
        return self.config['ablation']['disable_symptom_improvement_reward']

    # 数据路径访问方法
    @property
    def input_path(self) -> str:
        """输入数据路径"""
        return self.config['data']['input_path']

    # 训练参数访问方法
    @property
    def k_episodes(self) -> int:
        return self.config['training']['k_episodes']

    @property
    def gamma(self) -> float:
        return self.config['training']['gamma']

    @property
    def batch_size(self) -> int:
        return self.config['training']['batch_size']

    @property
    def learning_rate(self) -> float:
        value = self.config['training']['learning_rate']
        # 处理科学计数法字符串
        if isinstance(value, str):
            return float(value)
        return float(value)

    @property
    def kl_beta(self) -> float:
        return self.config['training']['kl_beta']

    @property
    def kl_temp(self) -> float:
        return self.config['training']['kl_temp']

    @property
    def num_parallel_environments(self) -> int:
        return self.config['training']['num_parallel_environments']

    # DQN架构参数访问方法
    @property
    def embedding_dim(self) -> int:
        return self.config['dqn_architecture']['embedding_dim']

    @property
    def hidden_dim_1(self) -> int:
        return self.config['dqn_architecture']['hidden_dim_1']

    @property
    def hidden_dim_2(self) -> int:
        return self.config['dqn_architecture']['hidden_dim_2']

    @property
    def action_dim(self) -> int:
        return self.config['dqn_architecture']['action_dim']

    # Epsilon-Greedy参数访问方法
    @property
    def eps_start(self) -> float:
        return self.config['epsilon_greedy']['eps_start']

    @property
    def eps_end(self) -> float:
        return self.config['epsilon_greedy']['eps_end']

    @property
    def eps_decay_steps(self) -> int:
        return self.config['epsilon_greedy']['eps_decay_steps']

    # 更新参数访问方法
    @property
    def target_update_interval(self) -> int:
        return self.config['update']['target_update_interval']

    @property
    def replay_buffer_capacity(self) -> int:
        return self.config['update']['replay_buffer_capacity']

    @property
    def checkpoint_num(self) -> int:
        return self.config['update']['checkpoint_num']

    # 断点训练配置访问方法
    @property
    def resume_training(self) -> bool:
        return self.config['resume']['enabled']

    @property
    def checkpoint_model_path(self) -> str:
        return self.config['resume']['checkpoint_model_path']

    @property
    def checkpoint_loss_log_path(self) -> str:
        return self.config['resume']['checkpoint_loss_log_path']

    @property
    def checkpoint_exp_log_path(self) -> str:
        return self.config['resume']['checkpoint_exp_log_path']

    @property
    def resume_episode_count(self) -> int:
        return self.config['resume']['resume_episode_count']

    # API配置访问方法
    @property
    def embedding_api_config(self) -> Dict[str, Any]:
        return self.config['api']['embedding']

    @property
    def llm_api_config(self) -> Dict[str, Any]:
        return self.config['api']['llm']

    # 奖励系统配置访问方法
    def get_reward_config(self) -> Dict[str, Any]:
        """获取奖励系统配置"""
        return self.config['reward_system']

    # 其他便捷方法
    def get_experiment_info(self) -> Dict[str, str]:
        """获取实验信息"""
        return {
            'name': self.config['experiment']['name'],
            'version': self.config['experiment']['version'],
            'description': self.config['experiment']['description'],
            'ablation_mode': self.config['ablation']['mode']
        }

    def get_output_paths(self) -> Dict[str, str]:
        """获取输出路径配置"""
        return self.config['output']

    def validate_config(self) -> bool:
        """验证配置文件的完整性和正确性"""
        required_sections = ['experiment', 'ablation', 'data', 'training', 'dqn_architecture',
                           'epsilon_greedy', 'update', 'reward_system', 'api', 'output']

        for section in required_sections:
            if section not in self.config:
                print(f"❌ 配置文件缺少必要部分: {section}")
                return False

        # 验证消融开关 - 确保所有开关都是布尔值
        ablation_switches = [
            'disable_kl_constraint',
            'disable_ddqn',
            'disable_safety_reward',
            'disable_strategy_match_reward',
            'disable_symptom_improvement_reward'
        ]

        for switch in ablation_switches:
            if not isinstance(self.config['ablation'][switch], bool):
                print(f"❌ 无效的消融开关 {switch}: 必须是true或false")
                return False

        print("✅ 配置文件验证通过")
        return True


# 全局配置实例
_config_instance = None

def get_config(config_path: str = None) -> CPRLConfig:
    """获取全局配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = CPRLConfig(config_path)
    return _config_instance

def reload_config(config_path: str = None) -> CPRLConfig:
    """重新加载配置"""
    global _config_instance
    _config_instance = CPRLConfig(config_path)
    return _config_instance