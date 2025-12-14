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
    
    config = Config()
    
    # ---------------------------------------------------------
    # 【重要】TAR2を使用する場合のみ、環境からの途中報酬をカットする
    # ---------------------------------------------------------
    if use_tar2:
        print(">> TAR2 mode detected: Overwriting intermediate rewards to 0.0 (Sparse Reward Setting)")
        config.pickup_reward = 10.0
        config.collision_penalty = -10.0 
        config.step_cost = -0.01        
        config.wait_penalty = 0.0       # 待機は戦略的に重要かもしれないので0のまま
    else:
        print(">> Standard mode: Using dense rewards defined in config.py")

    agent_config = AgentConfig(use_vdn=use_vdn, use_tar2=use_tar2)
    training_config = TrainingConfig(agent_config=agent_config)
    
    trainer = Trainer(
        num_episodes=training_config.num_episodes,
        use_shared_replay=training_config.use_shared_replay,
        use_vdn=use_vdn,
        use_tar2=use_tar2,  # 修正: ここで引数を渡す
        config=config
    )
    
    agents, episode_rewards, avg_rewards, served_stats, final_env = trainer.train()
    
    print("\nPlotting learning curves...")
    plot_learning_curves(episode_rewards, avg_rewards, served_stats)
    
    print("\nGenerating animation...")
    create_restaurant_gif(final_env, agents)
    
    print("\nAll Done.")

if __name__ == "__main__":
    # 実験設定: VDN + TAR2 を有効化
    main(use_vdn=False, use_tar2=False)