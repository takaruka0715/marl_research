import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(episode_rewards, avg_rewards, served_stats, collision_rates, avg_wait_times, filename='learning_curves.png'):
    """
    学習曲線と評価指標を可視化 (2x2 レイアウト)
    1. Average Reward (Team)
    2. Total Served (Team)
    3. Collision Rate (Team)
    4. Average Wait Time
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # データ整形の準備
    agents = list(episode_rewards.keys())
    episodes = range(len(episode_rewards[agents[0]]))
    
    # 移動平均のウィンドウサイズ
    window = 50

    def moving_average(data, w):
        return np.convolve(data, np.ones(w), 'valid') / w

    # --- 1. Team Average Reward (左上) ---
    ax = axes[0, 0]
    # チーム合計報酬を計算
    team_rewards = np.zeros(len(episodes))
    for agent in agents:
        team_rewards += np.array(episode_rewards[agent])
    
    if len(team_rewards) >= window:
        ma_reward = moving_average(team_rewards, window)
        ax.plot(range(window-1, len(team_rewards)), ma_reward, color='blue', label='Team Reward (MA)')
    
    ax.set_title('Team Total Reward (Moving Avg)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- 2. Total Served Dishes (右上) ---
    ax = axes[0, 1]
    # チーム合計配膳数を計算
    team_served = np.zeros(len(episodes))
    for agent in agents:
        team_served += np.array(served_stats[agent])
    
    if len(team_served) >= window:
        ma_served = moving_average(team_served, window)
        ax.plot(range(window-1, len(team_served)), ma_served, color='green', label='Total Served (MA)')
    
    ax.set_title('Total Dishes Served')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- 3. Collision Rate (左下) ---
    ax = axes[1, 0]
    if len(collision_rates) >= window:
        ma_collision = moving_average(collision_rates, window)
        ax.plot(range(window-1, len(collision_rates)), ma_collision, color='red', label='Collision Rate (MA)')
    
    ax.set_title('Collision Rate (collisions / total_steps)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- 4. Average Customer Wait Time (右下) ---
    ax = axes[1, 1]
    if len(avg_wait_times) >= window:
        ma_wait = moving_average(avg_wait_times, window)
        ax.plot(range(window-1, len(avg_wait_times)), ma_wait, color='orange', label='Avg Wait Time (MA)')

    ax.set_title('Average Customer Wait Time')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved extended learning curves to {filename}")
    plt.close()