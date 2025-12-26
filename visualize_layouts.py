import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
from envs.restaurant_env import RestaurantEnv
from config import Config

def save_layout_image(layout_type, grid_size=15, filename=None):
    """指定されたレイアウトの初期状態を画像として保存する"""
    if filename is None:
        filename = f"layout_{layout_type}.png"
    
    # 環境の初期化
    config = Config()
    env = RestaurantEnv(grid_size=grid_size, layout_type=layout_type, config=config)
    env.reset()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 描画設定 (gif_maker.py のロジックを流用) 
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#f5f5dc') # 背景色 
    
    # 1. 障害物・テーブル・カウンターの描画 [cite: 175, 176]
    for x, y in env.obstacles:
        if (x, y) in env.tables:
            color = '#8B4513' # テーブル（茶色） 
        elif env.counter_pos and abs(x - env.counter_pos[0]) <= 1 and abs(y - env.counter_pos[1]) <= 1:
            color = 'gray'    # カウンター（灰色） 
        else:
            color = 'black'   # 外壁など 
        ax.add_patch(Rectangle((y-0.45, x-0.45), 0.9, 0.9, facecolor=color))
    
    # 2. 座席の描画 
    for sx, sy in env.seats:
        ax.add_patch(Circle((sy, sx), 0.15, facecolor='lightblue'))
    
    # 3. 入口の描画 (デバッグ用に追加)
    if env.entrance_pos:
        ex, ey = env.entrance_pos
        ax.add_patch(Rectangle((ey-0.5, ex-0.5), 1.0, 1.0, facecolor='green', alpha=0.3))
        ax.text(ey, ex, "Entrance", ha='center', va='center', fontsize=8)

    # 4. エージェントの初期位置 
    agent_colors = ['red', 'blue']
    for idx, agent in enumerate(env.possible_agents):
        pos = env.agent_positions[agent]
        ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx], ec='black'))
        ax.text(pos[1], pos[0], f"A{idx}", color='white', ha='center', va='center', fontweight='bold')

    ax.invert_yaxis() # 行列の座標系に合わせる [cite: 182]
    ax.set_title(f"Restaurant Layout: {layout_type} (Size: {grid_size}x{grid_size})")
    
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    # 出力用ディレクトリ作成
    os.makedirs("layout_images", exist_ok=True)
    
    # 各レイアウトを出力 [cite: 78, 79, 80]
    layouts = ['empty', 'basic', 'complex']
    
    for l_type in layouts:
        save_layout_image(l_type, filename=f"layout_images/layout_{l_type}.png")