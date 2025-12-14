import numpy as np
import matplotlib.pyplot as plt

# 引数に filename を追加し、デフォルト値を設定
def plot_learning_curves(episode_rewards, avg_rewards, served_stats, filename='restaurant_learning_curves.png'):
    """学習曲線を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # ※カリキュラムの境界線などの設定はそのまま維持
    stage_boundaries = [1500, 3000, 5500, 7500, 8750] # 必要に応じて調整
    
    for i, agent in enumerate(['agent_0', 'agent_1']):
        axes[0, i].plot(episode_rewards[agent], alpha=0.2, color='gray')
        axes[0, i].plot(avg_rewards[agent], linewidth=2, color='blue', label='Avg Reward')
        # (中略: グラフ描画処理はそのまま)
        
        # ...
        
        axes[1, i].set_xlabel('Episode')
        axes[1, i].set_ylabel('Count')
        axes[1, i].legend()
    
    plt.tight_layout()
    
    # 引数で受け取ったファイル名で保存
    plt.savefig(filename)
    print(f"Saved learning curves to {filename}")
    plt.close()