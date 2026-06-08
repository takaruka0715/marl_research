# visualization/gif_maker.py

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import numpy as np

def create_restaurant_gif(env, agents, filename='restaurant_service_parallel.gif'):
    """環境の遷移を GIF で保存 (ParallelEnv対応版)"""
    
    # 環境リセット (ParallelEnvは (obs, info) を返す)
    observations, infos = env.reset()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def draw_frame(frame_data):
        ax.clear()
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#f5f5dc')
        
        # --- 障害物 ---
        for x, y in env.obstacles:
            if (x, y) in env.tables:
                color = '#8B4513'
            elif env.counter_pos and abs(x - env.counter_pos[0]) <= 1:
                color = 'gray'
            else:
                color = 'black'
            ax.add_patch(Rectangle((y-0.45, x-0.45), 0.9, 0.9, facecolor=color))
        
        # --- 座席 ---
        for sx, sy in env.seats:
            ax.add_patch(Circle((sy, sx), 0.15, facecolor='lightblue'))
        
        # --- 顧客（修正箇所） ---
        for c in frame_data['customers']:
            if c['state'] in ['seated', 'ordered', 'served']:
                # 色分け
                if c['state'] == 'ordered':
                    color = 'orange'
                elif c['state'] == 'served':
                    color = 'lightgreen' # 食事中
                else:
                    color = 'lightgreen' # 注文待ち

                # 顧客の円を描画
                # position[1]がx座標(col), position[0]がy座標(row)
                cx, cy = c['position'][1], c['position'][0]
                
                ax.add_patch(Circle((cx, cy), 0.3, facecolor=color, ec='black'))
                
                # ★追加: 待ち時間を表示
                # 待ち時間を取得（辞書から取得）
                wait_time = c.get('wait_time', 0)
                
                # 文字色: 視認性を上げるため、待ち時間が長い場合は赤にする等の調整も可能
                text_color = 'red' if wait_time > 30 else 'black'
                
                ax.text(cx, cy, str(wait_time), 
                        color=text_color, ha='center', va='center', 
                        fontsize=8, fontweight='bold')

        # --- 注文 ---
        for ox, oy in frame_data['active_orders']:
            ax.add_patch(Circle((oy, ox), 0.4, facecolor='yellow', alpha=0.5, ec='red'))
        
        # --- エージェント ---
        agent_colors = ['red', 'blue']
        dir_angles = [270, 0, 90, 180] # 0:Up, 1:Right, 2:Down, 3:Left
        
        for idx, agent in enumerate(env.possible_agents):
            if agent not in frame_data['agent_positions']:
                continue

            pos = frame_data['agent_positions'][agent]
            d = frame_data['agent_directions'][agent]
            
            # インベントリ表示の型対応
            raw_inv = frame_data['agent_inventory'][agent]
            if isinstance(raw_inv, list):
                inv_count = len(raw_inv)
            else:
                inv_count = int(raw_inv)
            
            # エージェント本体
            ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx]))
            
            # 所持皿数
            if inv_count > 0:
                ax.text(pos[1], pos[0], str(inv_count), color='white', ha='center', 
                       va='center', fontweight='bold')
            
            # 向き表示
            angle = dir_angles[d]
            ax.add_patch(Wedge((pos[1], pos[0]), 0.5, angle-30, angle+30, 
                      alpha=0.4, color='black'))
        
        ax.invert_yaxis()
        
        # カウンター上の料理数
        ready_dishes_data = frame_data.get("ready_dishes", [])
        if isinstance(ready_dishes_data, list):
            ready_count = len(ready_dishes_data)
        else:
            ready_count = int(ready_dishes_data)

        ax.set_title(f'Step: {len(env.history)} | Food: {ready_count}')
    
    # --- シミュレーション実行 (GIF生成用推論) ---
    limit_steps = getattr(env, 'max_steps', 400) 
    
    for step in range(limit_steps):
        if not env.agents:
            break
            
        actions = {}
        avail_actions = env.get_avail_actions()
        
        with torch.no_grad():
            if 'qmix' in agents:
                qmix_agent = agents['qmix']
                actions = qmix_agent.select_actions(observations, avail_actions=avail_actions)
            elif 'vdn' in agents:
                vdn_agent = agents['vdn']
                actions = vdn_agent.select_actions(observations, avail_actions=avail_actions)
            else:
                for agent_id in env.agents:
                    if agent_id in observations:
                        agent_obs = observations[agent_id]
                        dqn_agent = agents[agent_id]
                        action = dqn_agent.select_action(agent_obs, avail_actions=avail_actions[agent_id])
                        actions[agent_id] = action
                        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if not env.agents:
            break
            
    # アニメーション生成
    ani = animation.FuncAnimation(fig, draw_frame, frames=env.history, interval=100)
    ani.save(filename, writer='pillow', fps=10)
    print(f"Saved GIF to {filename}")
    plt.close()
