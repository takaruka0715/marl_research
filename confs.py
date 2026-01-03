from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    """環境（RestaurantEnv）の基本設定"""
    # 基本の報酬設定（Dense Reward: 密な報酬）として定義
    # ※ TAR2使用時の「スパース報酬」への書き換えは main.py 側で行う運用にする
    """
    delivery_reward: float = 100.0   # 普段の配膳報酬（TAR2時は400.0等へ上書きされる想定）
    pickup_reward: float = 10.0      # ピックアップ報酬
    collision_penalty: float = -10.0 # 衝突ペナルティ
    step_cost: float = -0.01         # ステップコスト
    wait_penalty: float = -0.5       # 待機ペナルティ
    
    coop_bonus_threshold: float = 20.0
    """

    delivery_reward: float = 10.0     # 400.0 は大きすぎるので 10.0 に
    pickup_reward: float = 1.0        # 配膳の 1/10 程度が目安
    
    # ペナルティ系
    collision_penalty: float = -0.5   # 衝突は「痛い」と感じる程度
    step_cost: float = -0.01          # 移動コストは小さく（-0.1だと遠くのゴールを諦めがち）
    wait_penalty: float = -0.05       # 顧客を待たせるペナルティ
    
    # 協力ボーナスなど
    coop_bonus_threshold: float = 2.0 # delivery_reward に合わせて調整
    coop_factor: float = 0.0         # 協力報酬の分配度合い
    
    # 環境の構造設定
    grid_size: int = 15
    max_steps: int = 250
    layout_type: str = "basic"       # 'empty', 'basic', 'complex' をここで管理
    local_obs_size: int = 5
    enable_customers: bool = True    # 顧客の有無も設定へ

@dataclass
class AgentConfig:
    """エージェント・学習アルゴリズム設定"""
    # ネットワーク・学習パラメータ
    lr: float = 0.0001
    gamma: float = 0.99              # 一般的に0.95-0.99。長期タスクなら0.99推奨
    batch_size: int = 128
    buffer_capacity: int = 50000
    
    # Epsilon-Greedy
    epsilon: float = 1.0
    epsilon_decay: float = 0.9997
    epsilon_min: float = 0.05

    # アルゴリズム選択フラグ
    use_vdn: bool = False
    use_qmix: bool = False           # <--- 【追加】QMIXフラグ
    use_tar2: bool = False           # TAR2を使用するか
    
    # モデル構造パラメータ（必要に応じて追加）
    hidden_dim: int = 64             # ネットワークの隠れ層サイズなど

@dataclass
class TrainingConfig:
    """学習ループ実行設定"""
    num_episodes: int = 20000
    use_shared_replay: bool = True
    target_update_interval: int = 10
    log_interval: int = 100
    save_interval: int = 1000        # <--- 【追加】モデル保存頻度
    
    # 乱数シード（再現性のため）
    seed: int = 42                   # <--- 【追加】シード値
    
    agent_config: Optional[AgentConfig] = None
    
    def __post_init__(self):
        if self.agent_config is None:
            self.agent_config = AgentConfig()