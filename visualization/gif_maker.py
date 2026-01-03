import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import numpy as np

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
        dir_angles = [270, 0, 90, 180] # 0:Up, 1:Right, 2:Down, 3:Left (Assuming logic)
        
        for idx, agent in enumerate(env.possible_agents):
            if agent not in frame_data['agent_positions']:
                continue

            pos = frame_data['agent_positions'][agent]
            d = frame_data['agent_directions'][agent]
            
            # --- 【修正】インベントリ表示の型対応 ---
            # 環境側の変更で inventory が list になった場合と int の場合の両方に対応
            raw_inv = frame_data['agent_inventory'][agent]
            if isinstance(raw_inv, list):
                inv_count = len(raw_inv)
            else:
                inv_count = int(raw_inv)
            # ----------------------------------------
            
            ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx]))
            
            if inv_count > 0:
                ax.text(pos[1], pos[0], str(inv_count), color='white', ha='center', 
                       va='center', fontweight='bold')
            
            angle = dir_angles[d]
            ax.add_patch(Wedge((pos[1], pos[0]), 0.5, angle-30, angle+30, 
                alpha=0.4, color='black'))
        
        ax.invert_yaxis()
        
        # カウンター上の料理数を取得（リストか数値か判定）
        ready_dishes_data = frame_data.get("ready_dishes", [])
        if isinstance(ready_dishes_data, list):
            ready_count = len(ready_dishes_data)
        else:
            ready_count = int(ready_dishes_data)

        ax.set_title(f'Step: {len(env.history)} | Food: {ready_count}')
    
    # --- シミュレーション実行 (ParallelEnv) ---
    max_steps = 400
    for step in range(max_steps):
        # PettingZoo ParallelEnv: エージェントがいなくなったら終了
        if not env.agents:
            break
            
        actions = {}
        
        # アルゴリズムに応じた行動選択
        with torch.no_grad():
            if 'qmix' in agents:
                # QMIX: エージェント管理クラスが select_actions を持つ想定
                qmix_agent = agents['qmix']
                # observations は {agent_id: obs} なのでそのまま渡す
                actions = qmix_agent.select_actions(observations)
                
            elif 'vdn' in agents:
                # VDN: エージェント管理クラスが select_actions を持つ想定
                vdn_agent = agents['vdn']
                actions = vdn_agent.select_actions(observations)
                 
            else:
                # Independent DQN: 各エージェントごとに個別に select_action
                for agent_id in env.agents:
                    if agent_id in observations:
                        agent_obs = observations[agent_id]
                        dqn_agent = agents[agent_id]
                        
                        # DQNエージェントは単体の観測を受け取る
                        action = dqn_agent.select_action(agent_obs)
                        actions[agent_id] = action
                        
        # 環境を1ステップ進める (同時更新)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # 全員終了したらループを抜ける
        if not env.agents:
            break
            
    # アニメーション生成
    # env.history にはステップごとのスナップショットが保存されている
    ani = animation.FuncAnimation(fig, draw_frame, frames=env.history[::2], interval=150)
    ani.save(filename, writer='pillow', fps=6)
    print(f"Saved GIF to {filename}")
    plt.close()