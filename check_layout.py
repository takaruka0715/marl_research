import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
from env import RestaurantEnv
from training import Curriculum
from config import Config

def check_and_save_layouts():
    # 設定の読み込み
    config = Config()
    curriculum = Curriculum()
    
    # 保存用フォルダの作成
    os.makedirs("results", exist_ok=True)
    
    # 各ステージのレイアウトを確認
    for i, stage in enumerate(curriculum.stages):
        layout_name = stage['layout']
        print(f"Generating preview for Stage {i+1}: {layout_name}")
        
        # 環境の初期化
        env = RestaurantEnv(
            layout_type=layout_name,
            enable_customers=stage['customers'],
            customer_spawn_interval=stage['spawn_interval'],
            config=config
        )
        env.reset()
        
        # 描画処理
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # グリッドのサイズ設定
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(-0.5, env.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('#f5f5dc') # 背景色
        
        # 障害物とテーブルの描画 [cite: 101, 102]
        for x, y in env.obstacles:
            # テーブル（茶色）か壁（黒）かを判定
            color = '#8B4513' if (x, y) in env.tables else 'black'
            ax.add_patch(Rectangle((y-0.45, x-0.45), 0.9, 0.9, facecolor=color))
        
        # 座席の描画（水色） [cite: 102]
        for sx, sy in env.seats:
            ax.add_patch(Circle((sy, sx), 0.15, facecolor='lightblue'))
            
        # 調理カウンターの描画（灰色） [cite: 101]
        if hasattr(env, 'counter_pos') and env.counter_pos:
            cx, cy = env.counter_pos
            ax.add_patch(Rectangle((cy-0.45, cx-0.45), 0.9, 0.9, facecolor='gray'))

        # エージェントの初期位置を描画 [cite: 105]
        agent_colors = ['red', 'blue']
        for idx, agent in enumerate(env.possible_agents):
            # env.get_agent_pos の代わりに env.positions を参照 
            if hasattr(env, 'positions'):
                pos = env.positions[agent]
            else:
                # 万が一 positions がない場合のフォールバック
                pos = (0, 0)
            
            ax.add_patch(Circle((pos[1], pos[0]), 0.35, facecolor=agent_colors[idx]))

        # Y軸を反転させてグリッド座標に合わせる [cite: 107]
        ax.invert_yaxis()
        plt.title(f"Stage {i+1}: {stage['description']} ({layout_name})")
        
        # 画像の保存
        save_path = f"results/layout_stage_{i+1}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    check_and_save_layouts()