import numpy as np
import argparse
from envs import RestaurantEnv
from agents.rule_based_agent import RuleBasedAgent
from confs import Config

def run_benchmark(num_episodes=100, layout='complex'):
    print(f"=== Starting Rule-Based Benchmark ===")
    print(f"Episodes: {num_episodes}")
    print(f"Layout: {layout}")
    
    config = Config()
    
    env = RestaurantEnv(
        layout_type=layout,
        enable_customers=True,
        customer_spawn_interval=20, # 忙しい状況を再現
        local_obs_size=5,
        config=config
    )
    
    # 【修正】エージェント作成前に一度環境をリセットして、
    # counter_pos などのレイアウト情報を生成させる
    env.reset()
    
    # エージェント作成 (これで env.counter_pos が参照可能になります)
    agents = {
        'agent_0': RuleBasedAgent('agent_0', env),
        'agent_1': RuleBasedAgent('agent_1', env)
    }
    
    # 統計用
    total_rewards = []
    served_counts = []
    collision_rates = []
    wait_times = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        episode_reward = 0
        done = False
        
        while env.agents:
            actions = {}
            
            # 各エージェントの行動決定
            # 相互参照のため、相手エージェントオブジェクトを渡す
            act0 = agents['agent_0'].get_action(obs, other_agent=agents['agent_1'])
            act1 = agents['agent_1'].get_action(obs, other_agent=agents['agent_0'])
            
            actions['agent_0'] = act0
            actions['agent_1'] = act1
            
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            
            sum_r = sum(rewards.values())
            episode_reward += sum_r
            
            obs = next_obs
        
        # エピソード完了後の集計
        total_rewards.append(episode_reward)
        served = sum(env.served_count.values())
        served_counts.append(served)
        
        total_steps = env.num_moves * 2 # 2 agents
        col_rate = sum(env.collision_count.values()) / total_steps if total_steps > 0 else 0
        collision_rates.append(col_rate)
        
        avg_wait = np.mean(env.completed_wait_times) if env.completed_wait_times else 0
        wait_times.append(avg_wait)
        
        print(f"Ep {ep+1}/{num_episodes} | Reward: {episode_reward:.1f} | Served: {served} | Col: {col_rate:.3f}", end='\r')

    # 結果出力
    print(f"\n\n{'='*60}")
    print(f"[SUMMARY] RULE-BASED AGENT RESULTS (Average over {num_episodes} eps)")
    print(f"{'='*60}")
    
    print(f"  - Average Reward:       {np.mean(total_rewards):8.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  - Average Served:       {np.mean(served_counts):8.2f} +/- {np.std(served_counts):.2f} dishes")
    print(f"  - Collision Rate:       {np.mean(collision_rates):8.4f} +/- {np.std(collision_rates):.4f}")
    print(f"  - Avg Wait Time:        {np.mean(wait_times):8.2f} +/- {np.std(wait_times):.2f} steps")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--layout', type=str, default='complex')
    args = parser.parse_args()
    
    run_benchmark(args.episodes, args.layout)
