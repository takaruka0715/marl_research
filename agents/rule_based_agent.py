import numpy as np
import heapq
import random

class RuleBasedAgent:
    """
    A*探索とヒューリスティックを用いたルールベースエージェント (修正版v4)
    - 環境の操作仕様(0=Forward, 1=TurnRight, 2=TurnLeft)に対応
    - 待機位置(Home)への移動ロジック修正
    - 衝突回避ロジックの方向転換対応
    """
    def __init__(self, agent_id, env):
        self.id = agent_id
        self.env = env
        self.grid_size = env.grid_size
        self.agent_idx = int(agent_id.split('_')[1]) # 0 or 1
        
        # カウンターの基準位置 (7, 1)
        cx, cy = env.counter_pos if env.counter_pos else (7, 1)
        
        # --- 待機位置(Home)の決定 ---
        if self.agent_idx == 0:
            candidate_home = (cx - 1, cy) # (6, 1) カウンターの上
        else:
            candidate_home = (cx, cy + 1) # (7, 2) カウンターの右
            
        all_obstacles = set(env.obstacles) | set(env.seats)
        if env.counter_pos:
            all_obstacles.add(env.counter_pos)
            
        if candidate_home in all_obstacles:
            found = False
            adj_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
            for dr, dc in adj_offsets:
                nx, ny = cx + dr, cy + dc
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    (nx, ny) not in all_obstacles):
                    self.home_pos = (nx, ny)
                    found = True
                    break
            if not found:
                self.home_pos = candidate_home
        else:
            self.home_pos = candidate_home

    def get_action(self, obs, other_agent=None):
        my_pos = self.env.agent_positions[self.id]
        my_dir = self.env.agent_directions[self.id] # 0:Up, 1:Right, 2:Down, 3:Left
        
        inventory = self.env.agent_inventory[self.id]
        has_dish = len(inventory) > 0
        
        target_pos = None
        target_type = None 

        # --- Step 1. ターゲット決定 ---
        if has_dish:
            dish = inventory[0]
            target_pos = dish['target_seat']
            target_type = 'delivery'
        else:
            best_dish = None
            max_wait = -1
            customer_waits = {c.seat_position: c.wait_time 
                              for c in self.env.customer_manager.customers 
                              if c.state == 'ordered'}
            
            for dish in self.env.ready_dishes:
                seat = dish['target_seat']
                if seat in customer_waits:
                    wait_time = customer_waits[seat]
                    if wait_time > max_wait:
                        max_wait = wait_time
                        best_dish = dish
            
            if best_dish:
                target_pos = self.env.counter_pos # (7, 1)
                target_type = 'pickup'
            else:
                target_pos = self.home_pos
                target_type = 'home'

        # --- Step 2. インタラクション判定 ---
        if target_type == 'pickup':
            cx, cy = self.env.counter_pos
            if abs(my_pos[0] - cx) + abs(my_pos[1] - cy) <= 1:
                if len(self.env.ready_dishes) > 0:
                    return 4 # Pickup
        
        elif target_type == 'delivery':
            sx, sy = target_pos
            if abs(my_pos[0] - sx) + abs(my_pos[1] - sy) <= 1:
                 return 4 # Serve

        # --- Step 3. 経路探索 (A*) ---
        static_obstacles = set(self.env.obstacles) | set(self.env.seats)
        if self.env.counter_pos:
            static_obstacles.add(self.env.counter_pos)

        valid_goals = []
        if target_type == 'home':
            if target_pos not in static_obstacles:
                valid_goals = [target_pos]
            else:
                goal_candidates = self._get_adjacent_cells(target_pos)
                valid_goals = [g for g in goal_candidates if g not in static_obstacles]
        else:
            goal_candidates = self._get_adjacent_cells(target_pos)
            valid_goals = [g for g in goal_candidates if g not in static_obstacles]
        
        if not valid_goals:
            return 4 # Wait
        
        # 既にゴールにいるなら向きを整えるか待機
        if my_pos in valid_goals:
            return 4 # Wait

        next_pos = self._find_next_step_astar(my_pos, valid_goals, static_obstacles)
        
        if next_pos is None:
             return 4 # Wait

        # --- Step 4. 衝突回避 ---
        final_next_pos = next_pos
        collision_risk = False
        
        if other_agent:
            other_pos = self.env.agent_positions[other_agent.id]
            if next_pos == other_pos:
                collision_risk = True
            dist_to_other = abs(next_pos[0] - other_pos[0]) + abs(next_pos[1] - other_pos[1])
            if dist_to_other <= 1:
                collision_risk = True

        if collision_risk:
            is_priority = self._check_priority(self, other_agent)
            if not is_priority or next_pos == other_pos:
                # 譲る or 相手がいる -> 回避先を探す
                escape_pos = self._find_escape_pos(my_pos, next_pos, other_pos, static_obstacles)
                if escape_pos:
                    final_next_pos = escape_pos
                else:
                    return 4 # 動けない

        # --- Step 5. アクションへの変換 (Direction Control) ---
        return self._get_direction_action(my_pos, my_dir, final_next_pos)

    def _find_escape_pos(self, curr, intended, other, obstacles):
        candidates = self._get_adjacent_cells(curr)
        valid = [p for p in candidates if p not in obstacles and p != other and p != intended]
        if valid:
            return random.choice(valid)
        return None

    def _get_direction_action(self, curr, curr_dir, next_pos):
        """
        現在地と次の位置から、必要なアクション(0:Fwd, 1:Right, 2:Left)を決定する
        """
        dr = next_pos[0] - curr[0]
        dc = next_pos[1] - curr[1]
        
        # 目標とする向き
        desired_dir = -1
        if dr == -1: desired_dir = 0 # Up
        elif dc == 1: desired_dir = 1 # Right
        elif dr == 1: desired_dir = 2 # Down
        elif dc == -1: desired_dir = 3 # Left
        
        if desired_dir == -1:
            return 4 # Wait (Stay)

        # 現在の向きと比較
        if curr_dir == desired_dir:
            return 0 # Move Forward
        
        # 回転が必要
        # 右回転で合うか？
        if (curr_dir + 1) % 4 == desired_dir:
            return 1 # Turn Right
        # 左回転で合うか？
        elif (curr_dir - 1) % 4 == desired_dir:
            return 2 # Turn Left
        else:
            # 180度反対の場合 -> どちらかに回る
            return 1 # Turn Right (2回で合う)

    # --- Helper Methods ---
    def _get_adjacent_cells(self, pos):
        r, c = pos
        return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

    def _check_priority(self, me, other):
        my_inv = len(self.env.agent_inventory[me.id])
        other_inv = len(self.env.agent_inventory[other.id])
        if my_inv > 0 and other_inv == 0: return True
        if my_inv == 0 and other_inv > 0: return False
        return me.id < other.id

    def _find_next_step_astar(self, start, goals, obstacles):
        if start in goals: return start
        queue = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        closest_goal = None
        count = 0
        max_nodes = 500 

        while queue and count < max_nodes:
            _, current = heapq.heappop(queue)
            count += 1
            if current in goals:
                closest_goal = current
                break
            for next_node in self._get_adjacent_cells(current):
                if not (0 <= next_node[0] < self.grid_size and 0 <= next_node[1] < self.grid_size):
                    continue
                if next_node in obstacles:
                    continue
                new_cost = cost_so_far[current] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    min_dist = min(abs(next_node[0]-g[0]) + abs(next_node[1]-g[1]) for g in goals)
                    priority = new_cost + min_dist
                    heapq.heappush(queue, (priority, next_node))
                    came_from[next_node] = current
        
        if closest_goal:
            curr = closest_goal
            path_len = 0
            while came_from[curr] != start:
                curr = came_from[curr]
                path_len += 1
                if curr is None or path_len > 100: return None
            return curr
        return None