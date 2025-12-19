# env/layout.py

class LayoutBuilder:
    """レストランレイアウト生成"""
    
    @staticmethod
    def create_layout(layout_type, grid_size):
        """
        レイアウト生成（empty/basic/complex）
        returns: (obstacles, tables, seats, counter_pos, entrance_pos)
        """
        obstacles = []
        tables = []
        seats = []
        
        # 【修正点：カウンター位置の固定】
        # 全レイアウトにおいて共通の位置（左上の角付近）に配置します
        counter_pos = (1, 1)
        obstacles.append(counter_pos)
        
        # お客さんの入り口（共通）
        entrance_pos = (1, grid_size // 2)
        
        if layout_type == 'empty':
            # Stage 1: 中央にテーブルを1つ配置
            LayoutBuilder._add_table(obstacles, tables, seats, 7, 7, grid_size)
        
        elif layout_type == 'basic':
            # 外周の壁
            obstacles = LayoutBuilder._add_walls(obstacles, grid_size)
            
            # 【修正点：通路の確保】
            # 椅子を障害物にするため、壁から離して配置し、ロボットが通れる隙間を作ります
            table_positions = [(4, 4), (4, 10), (10, 4), (10, 10)]
            for tx, ty in table_positions:
                LayoutBuilder._add_table(obstacles, tables, seats, tx, ty, grid_size)
        
        elif layout_type == 'complex':
            # 外周の壁
            obstacles = LayoutBuilder._add_walls(obstacles, grid_size)
            
            # 中央の仕切り壁（通り抜け用の隙間を空ける）
            for i in range(2, 13):
                if i not in [7, 8]: # 中央に通路を確保
                    obstacles.append((i, 7))
            
            # テーブル配置（椅子を含めても通路が残るように調整）
            table_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 3), (7, 11)]
            for tx, ty in table_positions:
                LayoutBuilder._add_table(obstacles, tables, seats, tx, ty, grid_size)
        
        return obstacles, tables, seats, counter_pos, entrance_pos
    
    @staticmethod
    def _add_walls(obstacles, grid_size):
        for i in range(grid_size):
            obstacles.append((0, i))
            obstacles.append((grid_size - 1, i))
            obstacles.append((i, 0))
            obstacles.append((i, grid_size - 1))
        return obstacles
    
    @staticmethod
    def _add_table(obstacles, tables, seats, x, y, grid_size):
        """
        テーブル(2x2)と、その上下左右に椅子を配置
        """
        # テーブル本体(2x2)を障害物として追加
        tables.append((x, y))
        for dx in [0, 1]:
            for dy in [0, 1]:
                if (x + dx, y + dy) not in obstacles:
                    obstacles.append((x + dx, y + dy))
        
        # 【修正点：上下左右に椅子を配置し、障害物として登録】
        # 椅子（座席）の相対座標
        seat_candidates = [
            (x - 1, y), (x - 1, y + 1), # 上側
            (x + 2, y), (x + 2, y + 1), # 下側
            (x, y - 1), (x + 1, y - 1), # 左側
            (x, y + 2), (x + 1, y + 2)  # 右側
        ]
        
        for sx, sy in seat_candidates:
            # グリッド内に収まり、かつ既存の壁・テーブルと重ならない場合のみ配置
            if 0 < sx < grid_size - 1 and 0 < sy < grid_size - 1:
                if (sx, sy) not in obstacles:
                    seats.append((sx, sy))
                    # 【重要】椅子を障害物リストに追加することでロボットが進入不可になる
                    obstacles.append((sx, sy))