import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Patch
from envs.layout import LayoutBuilder

def visualize_curriculum_seats():
    # 設定（環境設定に合わせる）
    grid_size = 15
    layout_type = 'complex'
    
    # レイアウト情報の取得 [cite: 503]
    obstacles, tables, seats, counter_pos, entrance_pos = LayoutBuilder.create_layout(layout_type, grid_size)
    
    # カリキュラムのステージ定義（training/curriculum.py の内容に基づく）
    # training/curriculum.py の新しい設定に合わせる
    stages = [
        {'name': 'Stage 1: Very Long', 'min_dist': 12, 'desc': 'Dist >= 12'},
        {'name': 'Stage 2: Long',      'min_dist': 8,  'desc': 'Dist >= 9'},
        {'name': 'Stage 3: Medium',    'min_dist': 6,  'desc': 'Dist >= 6'},
        {'name': 'Stage 4: Full',      'min_dist': 0,  'desc': 'Dist >= 0'},
    ]
    
    # グラフの枠数を増やす
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"Curriculum Learning Target Seats (Layout: {layout_type})", fontsize=16)

    cx, cy = counter_pos

    for idx, stage in enumerate(stages):
        ax = axes[idx]
        min_dist = stage['min_dist']
        
        # 1. グリッドと背景
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.invert_yaxis() # 行列座標系に合わせる（上が0）
        ax.set_facecolor('#f5f5dc') # ベージュ背景
        ax.set_title(f"{stage['name']}\n({stage['desc']})", fontsize=12, fontweight='bold')

        # 2. 障害物（壁・テーブル）の描画
        for x, y in obstacles:
            color = 'black' # 壁
            if (x, y) in tables:
                color = '#8B4513' # テーブル（茶色）
            elif abs(x - cx) + abs(y - cy) <= 1: # カウンター周辺
                color = 'gray'
            
            # Matplotlibは (x, y) = (col, row) なので注意。
            # 環境の座標系は (row, col) なので、Rectangle((y, x), ...) となる
            ax.add_patch(Rectangle((y - 0.45, x - 0.45), 0.9, 0.9, facecolor=color))

        # 3. カウンター位置の強調
        ax.text(cy, cx, "Kitchen", color='white', ha='center', va='center', fontsize=8, fontweight='bold')

        # 4. 座席の描画と判定 logic 
        valid_count = 0
        total_count = 0
        
        for sx, sy in seats:
            total_count += 1
            # マンハッタン距離の計算
            dist = abs(sx - cx) + abs(sy - cy)
            
            is_valid = dist >= min_dist
            
            if is_valid:
                face_color = '#32CD32' # LimeGreen (有効)
                edge_color = 'darkgreen'
                valid_count += 1
                alpha = 1.0
            else:
                face_color = '#D3D3D3' # LightGray (無効)
                edge_color = 'gray'
                alpha = 0.5

            # 座席の描画
            ax.add_patch(Circle((sy, sx), 0.3, facecolor=face_color, edgecolor=edge_color, alpha=alpha))
            
            # 距離を表示
            text_color = 'black' if is_valid else 'gray'
            ax.text(sy, sx, str(dist), color=text_color, ha='center', va='center', fontsize=9, fontweight='bold')

        # 統計情報の表示
        ax.text(0.5, -0.05, f"Valid Seats: {valid_count} / {total_count}", 
                transform=ax.transAxes, ha='center', fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # 凡例の作成
    legend_elements = [
        Patch(facecolor='#32CD32', edgecolor='darkgreen', label='Target (Valid)'),
        Patch(facecolor='#D3D3D3', edgecolor='gray', label='Ignored (Too Close)'),
        Patch(facecolor='#8B4513', label='Table'),
        Patch(facecolor='gray', label='Kitchen/Counter'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # タイトルと凡例用のスペース確保
    
    output_file = "curriculum_visualization.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    visualize_curriculum_seats()