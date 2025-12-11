import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Wedge

def create_restaurant_gif(env, agents, filename='restaurant_service_cooking.gif'):
    """環境の遷移を GIF で保存"""
    env.reset(seed=42)
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
        dir_angles = [270, 0, 90, 180]
        
        for idx, agent in enumerate(env.possible_agents):
            pos = frame_data['agent_positions'][agent]
            d = frame_data['agent_directions'][agent]
            inv = frame_data['agent_inventory'][agent]
            
            ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx]))
            
            if inv > 0:
                ax.text(pos[1], pos[0], str(inv), color='white', ha='center', 
                       va='center', fontweight='bold')
            
            angle = dir_angles[d]
            ax.add_patch(Wedge((pos[1], pos[0]), 0.5, angle-30, angle+30, 
                              alpha=0.4, color='black'))
        
        ax.invert_yaxis()
        ax.set_title(f'Step: {len(env.history)} | Food: {frame_data["ready_dishes"]}')
    
    # エージェント行動
    for step in range(400):
        agent_name = env.agent_selection
        if env.truncations.get(agent_name, False):
            env.step(None)
            continue
        
        state = env.observe(agent_name)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agents[agent_name].device)
            action = agents[agent_name].q_network(state_tensor).argmax().item()
        env.step(action)
        
        if all(env.truncations.get(a, False) for a in env.possible_agents):
            break
    
    ani = animation.FuncAnimation(fig, draw_frame, frames=env.history[::2], interval=150)
    ani.save(filename, writer='pillow', fps=6)
    print(f"Saved GIF to {filename}")
    plt.close()
