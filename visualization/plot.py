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

def plot_performance_metrics(stats_log, stage_transitions=None, filename='results/performance_metrics.png'):
    """成功率・待ち時間・衝突数の推移グラフ（PNG）"""
    if not stats_log: return

    # データの抽出
    episodes = [int(s['episode']) for s in stats_log]
    success_rates = [float(s['success_rate']) * 100 for s in stats_log]
    wait_times = [float(s['avg_wait_time']) for s in stats_log]
    collisions = [float(s['collisions']) for s in stats_log]

    def moving_avg(data, window=50):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    titles = ['Order Completion Rate (%)', 'Average Customer Waiting Time', 'Collision Count per Episode']
    data_list = [success_rates, wait_times, collisions]
    colors = ['green', 'blue', 'red']
    dark_colors = ['darkgreen', 'darkblue', 'darkred']
    y_labels = ['%', 'Steps', 'Count']

    for i in range(3):
        axes[i].plot(episodes, data_list[i], alpha=0.3, color=colors[i])
        axes[i].plot(episodes[min(49, len(episodes)-1):], moving_avg(data_list[i]), color=dark_colors[i], label='MA(50)')
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(y_labels[i])
        axes[i].grid(True)
        
        # 【追加】ステージの切り替わりを垂直線で描画
        if stage_transitions:
            for ep, desc in stage_transitions:
                if ep > 0: # 初期ステージ(Ep 0)以外
                    axes[i].axvline(x=ep, color='purple', linestyle='--', alpha=0.7)
                    if i == 0: # 一番上のグラフだけにテキストを表示
                        axes[i].text(ep, axes[i].get_ylim()[1], desc, rotation=45, color='purple')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()