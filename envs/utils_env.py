def check_collision(new_pos, obstacles, customer_positions, other_positions):
    """衝突判定"""
    x, y = new_pos
    if (x, y) in obstacles or (x, y) in customer_positions or (x, y) in other_positions:
        return True
    return False

def get_adjacent_positions(pos):
    """隣接座標を返す"""
    ox, oy = pos
    return [(ox - 1, oy), (ox + 1, oy), (ox, oy - 1), (ox, oy + 1)]

def normalize_position(pos, grid_size):
    """座標を正規化 [0, 1]"""
    x, y = pos
    return (x / grid_size, y / grid_size)
