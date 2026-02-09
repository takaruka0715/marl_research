import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge
import torch
import numpy as np

from envs import RestaurantEnv
from agents.rule_based_agent import RuleBasedAgent
from confs import Config

def run_debug_episode(output_filename="debug_rule_based.gif"):
    print("=== Generating Debug GIF for Rule-Based Agent ===")
    
    # 1. 環境設定
    config = Config()
    env = RestaurantEnv(
        layout_type='complex',
        enable_customers=True,
        customer_spawn_interval=20,
        local_obs_size=5,
        config=config
    )
    
    # エージェント生成前の初期化
    env.reset()
    
    # エージェントの準備
    agents = {
        'agent_0': RuleBasedAgent('agent_0', env),
        'agent_1': RuleBasedAgent('agent_1', env)
    }
    
    # 2. エピソード実行 (1エピソード)
    obs, info = env.reset()
    done = False
    
    print("Running simulation steps...")
    
    # 最大ステップ数まで実行
    max_steps = env.max_steps
    for step in range(max_steps):
        if not env.agents:
            break
            
        actions = {}
        
        # 行動決定 (相互参照あり)
        act0 = agents['agent_0'].get_action(obs, other_agent=agents['agent_1'])
        act1 = agents['agent_1'].get_action(obs, other_agent=agents['agent_0'])
        
        actions['agent_0'] = act0
        actions['agent_1'] = act1
        
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        obs = next_obs
        
        # ログ出力 (進捗確認用)
        if step % 50 == 0:
            served = sum(env.served_count.values())
            print(f"Step {step}: Served={served}, Agents={env.agent_positions}")

    print(f"Simulation finished. Total steps: {len(env.history)}")
    print("Creating GIF animation...")

    # 3. GIF生成 (visualization/gif_maker.py のロジックを流用)
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
                color = '#8B4513' # Table
            elif env.counter_pos and abs(x - env.counter_pos[0]) <= 1:
                color = 'gray'    # Counter area
            else:
                color = 'black'   # Wall
            ax.add_patch(Rectangle((y-0.45, x-0.45), 0.9, 0.9, facecolor=color))
        
        # --- 座席 ---
        for sx, sy in env.seats:
            ax.add_patch(Circle((sy, sx), 0.15, facecolor='lightblue'))
        
        # --- 顧客 ---
        for c in frame_data['customers']:
            if c['state'] in ['seated', 'ordered', 'served']:
                if c['state'] == 'ordered':
                    color = 'orange'
                elif c['state'] == 'served':
                    color = 'lightgreen'
                else:
                    color = 'lightgreen' # seated

                cx, cy = c['position'][1], c['position'][0]
                ax.add_patch(Circle((cx, cy), 0.3, facecolor=color, ec='black'))
                
                # 待ち時間表示
                wait_time = c.get('wait_time', 0)
                text_color = 'red' if wait_time > 50 else 'black'
                ax.text(cx, cy, str(wait_time), color=text_color, ha='center', va='center', fontsize=8, fontweight='bold')

        # --- 注文 (Active Orders) ---
        for ox, oy in frame_data['active_orders']:
            ax.add_patch(Circle((oy, ox), 0.4, facecolor='yellow', alpha=0.5, ec='red'))
        
        # --- エージェント ---
        agent_colors = ['red', 'blue']
        dir_angles = [270, 0, 90, 180] # 0:Up, 1:Right, 2:Down, 3:Left
        
        possible_agents = ['agent_0', 'agent_1']
        
        for idx, agent in enumerate(possible_agents):
            if agent not in frame_data['agent_positions']:
                continue

            pos = frame_data['agent_positions'][agent]
            d = frame_data['agent_directions'][agent]
            
            # インベントリ
            raw_inv = frame_data['agent_inventory'][agent]
            inv_count = len(raw_inv) if isinstance(raw_inv, list) else int(raw_inv)
            
            # 本体
            ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx]))
            
            # 所持数
            if inv_count > 0:
                ax.text(pos[1], pos[0], str(inv_count), color='white', ha='center', va='center', fontweight='bold')
            
            # 向き
            angle = dir_angles[d]
            ax.add_patch(Wedge((pos[1], pos[0]), 0.5, angle-30, angle+30, alpha=0.4, color='black'))
        
        ax.invert_yaxis()
        
        # カウンター料理数
        ready_dishes_data = frame_data.get("ready_dishes", [])
        ready_count = len(ready_dishes_data) if isinstance(ready_dishes_data, list) else int(ready_dishes_data)
        
        ax.set_title(f'Rule-Based Debug | Step: {len(env.history)} | Food Ready: {ready_count}')

    # アニメーション作成
    ani = animation.FuncAnimation(fig, draw_frame, frames=env.history, interval=100)
    ani.save(output_filename, writer='pillow', fps=10)
    print(f"Saved GIF to {output_filename}")
    plt.close()

if __name__ == "__main__":
    run_debug_episode()