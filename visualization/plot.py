import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(episode_rewards, avg_rewards, served_stats):
    """学習曲線を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    stage_boundaries = [1500, 3000, 5500, 7500, 8750]
    
    for i, agent in enumerate(['agent_0', 'agent_1']):
        axes[0, i].plot(episode_rewards[agent], alpha=0.2, color='gray')
        axes[0, i].plot(avg_rewards[agent], linewidth=2, color='blue', label='Avg Reward')
        for b in stage_boundaries:
            axes[0, i].axvline(x=b, color='red', linestyle='--', alpha=0.3)
        axes[0, i].set_title(f'{agent} Rewards')
        axes[0, i].set_xlabel('Episode')
        axes[0, i].set_ylabel('Reward')
        axes[0, i].legend()
        
        axes[1, i].plot(served_stats[agent], alpha=0.3, color='green')
        window = 50
        if len(served_stats[agent]) >= window:
            served_ma = [np.mean(served_stats[agent][max(0, j-window):j+1]) 
                        for j in range(len(served_stats[agent]))]
            axes[1, i].plot(served_ma, linewidth=2, color='darkgreen', label='MA')
        for b in stage_boundaries:
            axes[1, i].axvline(x=b, color='red', linestyle='--', alpha=0.3)
        axes[1, i].set_title(f'{agent} Served Count')
        axes[1, i].set_xlabel('Episode')
        axes[1, i].set_ylabel('Count')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig('restaurant_learning_curves_cooking.png')
    print("Saved learning curves to restaurant_learning_curves_cooking.png")
    plt.close()
