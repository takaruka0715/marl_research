import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(episode_rewards, avg_rewards, served_stats, filename='results/learning_curves.png'):
    """報酬曲線のプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for i, agent in enumerate(['agent_0', 'agent_1']):
        axes[0, i].plot(episode_rewards[agent], alpha=0.2, color='gray')
        axes[0, i].plot(avg_rewards[agent], linewidth=2, color='blue', label='Avg Reward')
        axes[0, i].set_title(f'Learning Curve - {agent}')
        axes[1, i].plot(served_stats[agent], color='green', alpha=0.5)
        axes[1, i].set_title(f'Served Count - {agent}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_performance_metrics(stats_log, filename='results/performance_metrics.png'):
    """成功率・待ち時間・衝突数の推移グラフ（PNG）"""
    if not stats_log: return

    # 確実に数値リストとして抽出（TypeError: unhashable type: 'dict' 回避策）
    episodes = [int(s['episode']) for s in stats_log]
    success_rates = [float(s['success_rate']) * 100 for s in stats_log]
    wait_times = [float(s['avg_wait_time']) for s in stats_log]
    collisions = [float(s['collisions']) for s in stats_log]

    def moving_avg(data, window=50):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # 配膳成功率
    axes[0].plot(episodes, success_rates, alpha=0.3, color='green')
    axes[0].plot(episodes[min(49, len(episodes)-1):], moving_avg(success_rates), color='darkgreen', label='MA(50)')
    axes[0].set_title('Order Completion Rate (%)')
    axes[0].set_ylabel('%')
    axes[0].grid(True)

    # 平均待ち時間
    axes[1].plot(episodes, wait_times, alpha=0.3, color='blue')
    axes[1].plot(episodes[min(49, len(episodes)-1):], moving_avg(wait_times), color='darkblue')
    axes[1].set_title('Average Customer Waiting Time')
    axes[1].set_ylabel('Steps')
    axes[1].grid(True)

    # 衝突数
    axes[2].plot(episodes, collisions, alpha=0.3, color='red')
    axes[2].plot(episodes[min(49, len(episodes)-1):], moving_avg(collisions), color='darkred')
    axes[2].set_title('Collision Count per Episode')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Count')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()