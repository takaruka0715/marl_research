import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge

def create_restaurant_gif(env, agents, filename='restaurant_service_cooking.gif'):
    """環境の遷移を GIF で保存"""
    # 【修正点】学習速度向上のためにオフにしていた記録機能を、GIF作成時のみオンにする
    env.record_enabled = True 
    env.reset(seed=42)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def draw_frame(frame_data):
        ax.clear()
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#f5f5dc')
        
        # 障害物（テーブルなど）
        for x, y in env.obstacles:
            if (x, y) in env.tables:
                color = '#8B4513' # 茶色（テーブル）
            elif env.counter_pos and abs(x - env.counter_pos[0]) <= 0 and abs(y - env.counter_pos[1]) <= 0:
                color = 'gray' # 灰色（カウンター本体）
            else:
                color = 'black' # 黒色（壁・椅子）
            ax.add_patch(Rectangle((y-0.45, x-0.45), 0.9, 0.9, facecolor=color))
        
        # 座席の表示（水色の円）
        for sx, sy in env.seats:
            ax.add_patch(Circle((sy, sx), 0.15, facecolor='lightblue'))
        
        # 顧客の表示
        for c in frame_data['customers']:
            if c['state'] in ['seated', 'ordered', 'served']:
                color = 'orange' if c['state'] == 'ordered' else 'lightgreen'
                ax.add_patch(Circle((c['position'][1], c['position'][0]), 0.3, 
                                facecolor=color, ec='black'))
        
        # 注文の強調表示
        for ox, oy in frame_data['active_orders']:
            ax.add_patch(Circle((oy, ox), 0.4, facecolor='yellow', alpha=0.5, ec='red'))
        
        # エージェントの表示
        agent_colors = ['red', 'blue']
        dir_angles = [270, 0, 90, 180] # 上, 右, 下, 左
        
        for idx, agent in enumerate(env.possible_agents):
            pos = frame_data['agent_positions'][agent]
            d = frame_data['agent_directions'][agent]
            inv = frame_data['agent_inventory'][agent]
            
            ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx]))
            
            # 所持品（料理）がある場合
            if inv > 0:
                ax.text(pos[1], pos[0], "F", color='white', ha='center', 
                       va='center', fontweight='bold')
            
            # 視界方向のウェッジ
            angle = dir_angles[d]
            ax.add_patch(Wedge((pos[1], pos[0]), 0.5, angle-30, angle+30, 
                              alpha=0.4, color='black'))
        
        ax.invert_yaxis()
        ax.set_title(f'Step: {len(env.history)} | Food Ready: {frame_data["ready_dishes"]}')

    # シミュレーションの実行（400ステップ分）
    for step in range(400):
        agent_name = env.agent_selection
        if env.truncations.get(agent_name, False):
            env.step(None)
            continue
        
        state = env.observe(agent_name)

        with torch.no_grad():
            if 'vdn' in agents:
                vdn_agent = agents['vdn']
                agent_idx = int(agent_name.split('_')[1])
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(vdn_agent.device)
                q_values = vdn_agent.q_network.local_q_networks[agent_idx](state_tensor)
                action = q_values.argmax().item()
            else:
                current_agent = agents[agent_name]
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(current_agent.device)
                action = current_agent.q_network(state_tensor).argmax().item()

        env.step(action)
        
        if all(env.truncations.get(a, False) for a in env.possible_agents):
            break
    
    # 【修正点】履歴が空でないことを確認してからアニメーション作成
    if not env.history:
        print("警告: 履歴が空のため GIF を作成できません。record_enabled が正しく機能しているか確認してください。")
        return

    ani = animation.FuncAnimation(fig, draw_frame, frames=env.history[::2], interval=150)
    ani.save(filename, writer='pillow', fps=6)
    
    # 次の学習に影響を与えないようフラグを戻す
    env.record_enabled = False 
    print(f"Saved GIF to {filename}")
    plt.close()