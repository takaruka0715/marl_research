# env/layout.py

class LayoutBuilder:
    @staticmethod
    def create_layout(layout_type, grid_size):
        obstacles, tables, seats = [], [], []
        counter_pos = (1, 1) # 全レイアウト固定
        obstacles.append(counter_pos)
        entrance_pos = (1, grid_size // 2)
        
        if layout_type != 'empty':
            for i in range(grid_size): # 外周壁
                for pos in [(0, i), (grid_size-1, i), (i, 0), (i, grid_size-1)]:
                    if pos not in obstacles: obstacles.append(pos)

        # ステージごとのテーブル配置（通路確保のため間隔を広めに設定）
        if layout_type == 'empty':
            table_pos = [(7, 7)]
        elif layout_type == 'basic':
            table_pos = [(4, 4), (10, 10)]
        else: # complex
            table_pos = [(3, 4), (3, 10), (10, 4), (10, 10)]
            for i in range(6, 9): obstacles.append((i, 7)) # 中央の小壁

        for tx, ty in table_pos:
            LayoutBuilder._add_table(obstacles, tables, seats, tx, ty, grid_size)
            
        return obstacles, tables, seats, counter_pos, entrance_pos

    @staticmethod
    def _add_table(obstacles, tables, seats, x, y, grid_size):
        tables.append((x, y))
        for dx in [0, 1]:
            for dy in [0, 1]:
                if (x+dx, y+dy) not in obstacles: obstacles.append((x+dx, y+dy))
        
        # 4方向の椅子候補
        candidates = [(x-1,y), (x-1,y+1), (x+2,y), (x+2,y+1), (x,y-1), (x+1,y-1), (x,y+2), (x+1,y+2)]
        for sx, sy in candidates:
            if 1 < sx < grid_size-2 and 1 < sy < grid_size-2: # 壁やカウンター付近を避ける
                if (sx, sy) not in obstacles:
                    seats.append((sx, sy))
                    obstacles.append((sx, sy)) # 椅子を障害物化