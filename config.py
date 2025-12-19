from dataclasses import dataclass

@dataclass
class Config:
    """環境とエージェント共通設定"""
    delivery_reward: float = 200.0
    pickup_reward: float = 50.0
    collision_penalty: float = -10.0
    step_cost: float = -0.05
    wait_penalty: float = 0.0          # 環境クラスとの互換性のために必須
    
    # --- 協調・環境設定 ---
    coop_bonus_threshold: float = 20.0 # 今回のエラー原因：追加
    coop_factor: float = 0.5           # 環境が参照する可能性があるため追加
    grid_size: int = 15
    local_obs_size: int = 5
    max_steps: int = 600
    
    # --- 最適化用設定（サボり・放置防止） ---
    customer_patience_limit: int = 200
    penalty_customer_left: float = -50.0 
    holding_item_step_cost: float = -0.3  # 料理を持ったまま移動・待機する追加コスト
    idle_penalty: float = -0.2            # その場に留まることへのペナルティ
    # --------------------------------------

@dataclass
class AgentConfig:
    lr: float = 0.0001
    epsilon: float = 1.0
    epsilon_decay: float = 0.9997
    epsilon_min: float = 0.05
    gamma: float = 0.95
    batch_size: int = 128
    buffer_capacity: int = 50000
    use_vdn: bool = True
    use_tar2: bool = False

@dataclass
class TrainingConfig:
    num_episodes: int = 20000
    use_shared_replay: bool = True
    agent_config: AgentConfig = None
    def __post_init__(self):
        if self.agent_config is None: self.agent_config = AgentConfig()