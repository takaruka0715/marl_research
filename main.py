import torch
from config import Config, AgentConfig, TrainingConfig
from training import Trainer
from visualization import plot_learning_curves, create_restaurant_gif

def main(use_vdn=False):
    print("="*70)
    print(f"Multi-Agent Restaurant RL System — {'VDN' if use_vdn else 'Independent DQN'}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*70)
    
    config = Config()
    agent_config = AgentConfig(use_vdn=use_vdn)
    training_config = TrainingConfig(agent_config=agent_config)
    
    trainer = Trainer(
        num_episodes=training_config.num_episodes,
        use_shared_replay=training_config.use_shared_replay,
        use_vdn=use_vdn,
        config=config
    )
    
    agents, episode_rewards, avg_rewards, served_stats, final_env = trainer.train()
    
    print("\nPlotting learning curves...")
    plot_learning_curves(episode_rewards, avg_rewards, served_stats)
    
    print("\nGenerating animation...")
    create_restaurant_gif(final_env, agents)
    
    print("\nAll Done.")

if __name__ == "__main__":
    # VDN を使用する場合は True に設定
    main(use_vdn=False)  # False: Independent DQN, True: VDN