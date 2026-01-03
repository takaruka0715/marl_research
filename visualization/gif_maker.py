import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import numpy as np

# 注意: ここに 'from .customers import ...' などは不要です

def create_restaurant_gif(env, agents, filename='restaurant_service_parallel.gif'):
    """環境の遷移を GIF で保存 (ParallelEnv対応版)"""
    
    # 環境リセット (ParallelEnvは (obs, info) を返す)
    observations, infos = env.reset(seed=42)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def draw_frame(frame_data):
        ax.clear()
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#f5f5dc')
        
        # 障害物
        for x, y in env.obstacles:
            if (x, y) in env.tables:
                color = '#8B4513'
            elif env.counter_pos and abs(x - env.counter_pos[0]) <= 1:
                color = 'gray'
            else:
                color = 'black'
            ax.add_patch(Rectangle((y-0.45, x-0.45), 0.9, 0.9, facecolor=color))
        
        # 座席
        for sx, sy in env.seats:
            ax.add_patch(Circle((sy, sx), 0.15, facecolor='lightblue'))
        
        # 顧客
        for c in frame_data['customers']:
            if c['state'] in ['seated', 'ordered', 'served']:
                color = 'orange' if c['state'] == 'ordered' else 'lightgreen'
                ax.add_patch(Circle((c['position'][1], c['position'][0]), 0.3, 
                                   facecolor=color, ec='black'))
        
        # 注文
        for ox, oy in frame_data['active_orders']:
            ax.add_patch(Circle((oy, ox), 0.4, facecolor='yellow', alpha=0.5, ec='red'))
        
        # エージェント
        agent_colors = ['red', 'blue']
        dir_angles = [270, 0, 90, 180] # 0:Up, 1:Right, 2:Down, 3:Left
        
        for idx, agent in enumerate(env.possible_agents):
            if agent not in frame_data['agent_positions']:
                continue

            pos = frame_data['agent_positions'][agent]
            d = frame_data['agent_directions'][agent]
            inv = frame_data['agent_inventory'][agent]
            
            # エージェント本体
            ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx]))
            
            # 在庫数の表示
            if inv > 0:
                ax.text(pos[1], pos[0], str(inv), color='white', ha='center', 
                       va='center', fontweight='bold')
            
            # 向きの表示
            angle = dir_angles[d]
            ax.add_patch(Wedge((pos[1], pos[0]), 0.5, angle-30, angle+30, 
                               alpha=0.4, color='black'))
        
        ax.invert_yaxis()
        # タイトルに情報表示
        food_count = frame_data.get("ready_dishes", 0)
        step_count = len(env.history)
        ax.set_title(f'Step: {step_count} | Food: {food_count}')
    
    # --- シミュレーション実行 (ParallelEnv) ---
    max_steps = 400
    for step in range(max_steps):
        if not env.agents:
            break
            
        actions = {}
        
        # アルゴリズムに応じた行動選択
        with torch.no_grad():
            if 'qmix' in agents:
                qmix_agent = agents['qmix']
                actions = qmix_agent.select_actions(observations)
                
            elif 'vdn' in agents:
                vdn_agent = agents['vdn']
                actions = vdn_agent.select_actions(observations)
                
            else:
                # Independent DQN
                for agent_id in env.agents:
                    if agent_id in observations:
                        agent_obs = observations[agent_id]
                        dqn_agent = agents[agent_id]
                        action = dqn_agent.select_action(agent_obs)
                        actions[agent_id] = action
                        
        # 環境を1ステップ進める
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if not env.agents:
            break
            
    # アニメーション生成
    if len(env.history) > 0:
        ani = animation.FuncAnimation(fig, draw_frame, frames=env.history[::2], interval=150)
        ani.save(filename, writer='pillow', fps=6)
        print(f"Saved GIF to {filename}")
    else:
        print("Warning: No history found to generate GIF.")
    
    plt.close()