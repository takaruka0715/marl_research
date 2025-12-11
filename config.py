from dataclasses import dataclass

@dataclass
class Config:
    """環境とエージェント共通設定"""
    delivery_reward: float = 100.0
    pickup_reward: float = 50.0
    collision_penalty: float = -10.0
    step_cost: float = -0.1
    wait_penalty: float = -0.5
    coop_bonus_threshold: float = 20.0
    max_steps: int = 500
    grid_size: int = 15
    local_obs_size: int = 5
    coop_factor: float = 0.5

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

@dataclass
class TrainingConfig:
    """学習ループ設定"""
    num_episodes: int = 30000
    use_shared_replay: bool = True
    target_update_interval: int = 10
    log_interval: int = 100
