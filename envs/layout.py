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
        counter_pos = None
        entrance_pos = None
        
        if layout_type == 'empty':
            obstacles = LayoutBuilder._add_walls(obstacles, grid_size)
            counter_pos = (7, 1)
            LayoutBuilder._add_counter(obstacles, 7, 1, length=3, horizontal=False)
            entrance_pos = (1, 7)
            LayoutBuilder._add_table(obstacles, tables, seats, 3, 3, grid_size)
        
        elif layout_type == 'basic':
            obstacles = LayoutBuilder._add_walls(obstacles, grid_size)
            counter_pos = (7, 1)
            LayoutBuilder._add_counter(obstacles, 7, 1, length=3, horizontal=False)
            
            for tx, ty in [(3, 3), (3, 8), (8, 3), (8, 8), (6, 11)]:
                LayoutBuilder._add_table(obstacles, tables, seats, tx, ty, grid_size)
            entrance_pos = (1, 7)
        
        elif layout_type == 'complex':
            obstacles = LayoutBuilder._add_walls(obstacles, grid_size)
            counter_pos = (7, 1)
            LayoutBuilder._add_counter(obstacles, 7, 1, length=5, horizontal=False)
            
            for i in range(3):
                obstacles.append((12, 5 + i))
            
            for tx in [2, 6, 10]:
                for ty in [2, 6, 10]:
                    LayoutBuilder._add_table(obstacles, tables, seats, tx, ty, grid_size)
            LayoutBuilder._add_table(obstacles, tables, seats, 4, 12, grid_size)
            entrance_pos = (1, 7)
        
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
    def _add_counter(obstacles, x, y, length=3, horizontal=True):
        for i in range(length):
            if horizontal:
                obstacles.append((x, y + i))
            else:
                obstacles.append((x + i, y))
    
    @staticmethod
    def _add_table(obstacles, tables, seats, x, y, grid_size):
        if x < grid_size - 1 and y < grid_size - 1:
            tables.append((x, y))
            for dx in [0, 1]:
                for dy in [0, 1]:
                    obstacles.append((x + dx, y + dy))
            
            seat_positions = [(x - 1, y), (x + 2, y), (x, y - 1), (x, y + 2)]
            for sx, sy in seat_positions:
                if (0 < sx < grid_size - 1 and 0 < sy < grid_size - 1 and
                    (sx, sy) not in obstacles):
                    seats.append((sx, sy))
