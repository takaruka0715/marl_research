from dataclasses import dataclass

@dataclass
class Config:
    """環境とエージェント共通設定"""
    delivery_reward: float = 100.0
    pickup_reward: float = 10.0
    collision_penalty: float = -10.0
    step_cost: float = -0.01
    wait_penalty: float = 0.0
    
    coop_bonus_threshold: float = 20.0
    max_steps: int = 500
    grid_size: int = 15
    local_obs_size: int = 5
    coop_factor: float = 0.5

@dataclass
class AttentionConfig:
    """Urgency-Aware Attention 設定"""
    use_attention: bool = False   # 提案手法の有効化フラグ
    urgency_alpha: float = 2.0    # 時間経過の重み (wait_time ^ alpha)
    urgency_max: float = 100.0    # クリップ上限 (E_max)
    task_embed_dim: int = 64      # タスク特徴量の埋め込み次元
    attn_head_dim: int = 64       # Attentionのヘッド次元
    max_orders: int = 10          # ネットワークに入力する最大注文数（パディング用）

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
    use_vdn: bool = False
    use_tar2: bool = False
    # アテンション設定を追加
    attn_config: AttentionConfig = None 

    def __post_init__(self):
        if self.attn_config is None:
            self.attn_config = AttentionConfig()

@dataclass
class TrainingConfig:
    """学習ループ設定"""
    num_episodes: int = 15000
    use_shared_replay: bool = True
    target_update_interval: int = 10
    log_interval: int = 100
    agent_config: AgentConfig = None
    
    def __post_init__(self):
        if self.agent_config is None:
            self.agent_config = AgentConfig()