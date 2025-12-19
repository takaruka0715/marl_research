import torch
from config import Config, AgentConfig, TrainingConfig
from training import Trainer
from visualization import plot_learning_curves, create_restaurant_gif

def main(use_vdn=False, use_tar2=False):
    """
    Args:
        use_vdn (bool): VDNを使用するか (FalseならIndependent DQN)
        use_tar2 (bool): TAR2による報酬再分配を使用するか
    """
    print("="*70)
    print(f"Multi-Agent Restaurant RL System")
    print(f"Algorithm: {'VDN' if use_vdn else 'Independent DQN'}")
    print(f"Credit Assignment: {'TAR2 (Redistribution)' if use_tar2 else 'Standard'}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*70)
    
    # 1. 基本設定の読み込み
    config = Config()
    
    # 2. TAR2を使用する場合の設定上書き
    # 途中報酬をカットして、スパース報酬（結果重視）の設定にする
    if use_tar2:
        print(">> TAR2 mode detected: Overwriting intermediate rewards to 0.0 (Sparse Reward Setting)")
        config.pickup_reward = 10.0
        config.collision_penalty = -10.0 
        config.step_cost = -0.01        
        config.wait_penalty = 0.0
    else:
        print(">> Standard mode: Using dense rewards defined in config.py")

    # 3. エージェントと学習設定の初期化（ここが抜けていました）
    agent_config = AgentConfig(use_vdn=use_vdn, use_tar2=use_tar2)
    training_config = TrainingConfig(agent_config=agent_config)
    
    # 4. トレーナーの初期化
    trainer = Trainer(
        num_episodes=training_config.num_episodes,
        use_shared_replay=training_config.use_shared_replay,
        use_vdn=use_vdn,
        use_tar2=use_tar2,
        config=config
    )
    
    # 5. 学習実行
    agents, episode_rewards, avg_rewards, served_stats, final_env = trainer.train()
    
    # 6. 出力ファイル名の決定ロジック
    if use_vdn:
        if use_tar2:
            mode_suffix = "vdn_tar2"
        else:
            mode_suffix = "vdn"
    else:
        # VDNを使っていない (Independent DQN)
        mode_suffix = "dqn"

    plot_filename = f"learning_curves_{mode_suffix}.png"
    gif_filename = f"restaurant_{mode_suffix}.gif"

    # 7. 結果の可視化と保存
    print(f"\nPlotting learning curves to {plot_filename}...")
    plot_learning_curves(episode_rewards, avg_rewards, served_stats, filename=plot_filename)
    
    print(f"\nGenerating animation to {gif_filename}...")
    create_restaurant_gif(final_env, agents, filename=gif_filename)
    
    print("\nAll Done.")

if __name__ == "__main__":
    # 実験設定: 必要に応じてTrue/Falseを切り替えてください
    main(use_vdn=True, use_tar2=False)