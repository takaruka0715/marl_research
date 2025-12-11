import torch
from config import Config, AgentConfig, TrainingConfig
from training import Trainer
from visualization import plot_learning_curves, create_restaurant_gif

def main():
    print("="*70)
    print("Multi-Agent Restaurant RL System — Refactored")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*70)
    
    config = Config()
    training_config = TrainingConfig()
    
    trainer = Trainer(
        num_episodes=training_config.num_episodes,
        use_shared_replay=training_config.use_shared_replay,
        config=config
    )
    
    agents, episode_rewards, avg_rewards, served_stats, final_env = trainer.train()
    
    print("\nPlotting learning curves...")
    plot_learning_curves(episode_rewards, avg_rewards, served_stats)
    
    print("\nGenerating animation...")
    create_restaurant_gif(final_env, agents)
    
    print("\nAll Done.")

if __name__ == "__main__":
    main()