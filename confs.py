from dataclasses import dataclass

@dataclass
class Config:
    """環境とエージェント共通設定"""
    # TAR2の効果を確認するため、スパース報酬設定に変更（途中報酬を0にする）
    delivery_reward: float = 400.0
    pickup_reward: float = 50.0       # 元に戻す
    collision_penalty: float = -10.0  # 元に戻す
    step_cost: float = -0.1           # 元に戻す
    wait_penalty: float = -0.5        # 元に戻す
    
    coop_bonus_threshold: float = 20.0
    max_steps: int = 250
    grid_size: int = 15
    local_obs_size: int = 5
    coop_factor: float = 0.0 # 報酬の分配度合い

@dataclass
class AgentConfig:
    """エージェント学習設定"""
    lr: float = 0.0001
    epsilon: float = 1.0
    epsilon_decay: float = 0.9997
    epsilon_min: float = 0.05
    gamma: float = 0.95
    batch_size: int = 128
    buffer_capacity: int = 50000
    use_vdn: bool = False  # VDN を使用するか
    use_tar2: bool = False # TAR2 (報酬再分配) を使用するか <--- 追加

@dataclass
class TrainingConfig:
    """学習ループ設定"""
    num_episodes: int = 20000
    use_shared_replay: bool = True
    target_update_interval: int = 10
    log_interval: int = 100
    agent_config: AgentConfig = None
    
    def __post_init__(self):
        if self.agent_config is None:
            self.agent_config = AgentConfig()