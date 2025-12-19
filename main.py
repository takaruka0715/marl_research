import torch
import os
from config import Config, AgentConfig, TrainingConfig
from training import Trainer
from visualization import plot_learning_curves, create_restaurant_gif, plot_performance_metrics

def main(use_vdn=True, use_tar2=False):
    os.makedirs("results", exist_ok=True)
    print("="*70)
    print(f"Multi-Agent Restaurant RL System")
    print(f"Algorithm: {'VDN' if use_vdn else 'Independent DQN'}")
    print("="*70)
    
    config = Config()
    agent_config = AgentConfig(use_vdn=use_vdn, use_tar2=use_tar2)
    training_config = TrainingConfig(agent_config=agent_config)
    
    trainer = Trainer(
        num_episodes=training_config.num_episodes,
        use_shared_replay=training_config.use_shared_replay,
        use_vdn=use_vdn,
        use_tar2=use_tar2,
        config=config
    )
    
    # 戻り値の数（7つ）を trainer.py の実装に合わせる
    agents, ep_rewards, avg_rewards, served_stats, stats_log, transitions, final_env = trainer.train()
    
    suffix = "vdn" if use_vdn else "dqn"
    
    # グラフとGIFの出力
    plot_learning_curves(ep_rewards, avg_rewards, served_stats, filename=f"results/curves_{suffix}.png")
    plot_performance_metrics(stats_log, stage_transitions=transitions, filename=f"results/metrics_{suffix}.png")
    create_restaurant_gif(final_env, agents, filename=f"results/restaurant_{suffix}.gif")
    
    print(f"\nCompleted. Results in 'results/' folder.")

if __name__ == "__main__":
    main(use_vdn=True, use_tar2=False)