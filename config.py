from dataclasses import dataclass

@dataclass
class Config:
    """環境とエージェント共通設定"""
    delivery_reward: float = 500.0
    pickup_reward: float = 100.0
    # 緩和：衝突の痛みを減らし、移動を促す
    collision_penalty: float = -2.0  
    step_cost: float = -0.01
    wait_penalty: float = 0.0          
    
    # --- 協調・環境設定 ---
    coop_bonus_threshold: float = 20.0 
    coop_factor: float = 0.5         
    grid_size: int = 15
    local_obs_size: int = 5
    max_steps: int = 600
    
    # --- 最適化用設定（初期学習のために大幅に緩和） ---
    customer_patience_limit: int = 200
    #penalty_customer_left: float = -50.0 
    penalty_customer_left: float = -50.0
    holding_item_step_cost: float = -0.05  # 緩和
    idle_penalty: float = -0.05            # 緩和

@dataclass
class AgentConfig:
    lr: float = 0.0001
    epsilon: float = 1.0
    epsilon_decay: float = 0.9995
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