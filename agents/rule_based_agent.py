import numpy as np
import heapq
import random
import itertools

class SpaceTimePlanner:
    """
    時空間A* (Prioritized Planning) を用いた中央集権型経路プランナー
    """
    def __init__(self, env):
        self.env = env
        self.paths = {'agent_0': [], 'agent_1': []}
        self.targets = {'agent_0': None, 'agent_1': None}
        self.reservations = {}
        self.last_update_step = -1
        self.home_positions = {}
        self._init_home_positions()

    def _init_home_positions(self):
        """各エージェントの待機場所（ホーム）を初期化"""
        cx, cy = self.env.counter_pos if self.env.counter_pos else (7, 1)
        homes = {'agent_0': (cx - 1, cy), 'agent_1': (cx, cy + 1)}
        static_obs = set(self.env.obstacles) | set(self.env.seats)
        if self.env.counter_pos:
            static_obs.add(self.env.counter_pos)
        
        for aid, pos in homes.items():
            if pos in static_obs:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]:
                    nx, ny = cx + dr, cy + dc
                    if 0 <= nx < self.env.grid_size and 0 <= ny < self.env.grid_size and (nx, ny) not in static_obs:
                        self.home_positions[aid] = (nx, ny)
                        break
            else:
                self.home_positions[aid] = pos

    def update_plan_if_needed(self):
        """環境の変化を検知し、必要なら全員の経路を再計算する"""
        # エピソードがリセットされた場合の検知
        if self.env.num_moves < self.last_update_step:
            self.paths = {'agent_0': [], 'agent_1': []}
            self.targets = {'agent_0': None, 'agent_1': None}
            self.reservations.clear()
            self.last_update_step = -1

        # 同一ステップ内では1回だけ計算
        if self.env.num_moves == self.last_update_step:
            return
        self.last_update_step = self.env.num_moves

        # 現状の環境データから、各自が向かうべきターゲットを決定
        desired_targets = self._determine_all_targets()

        replan_needed = False
        for aid in self.env.possible_agents:
            # ターゲットが変わったか、パスが尽きたら再計算
            if self.targets[aid] != desired_targets[aid]:
                replan_needed = True
            if not self.paths[aid] and desired_targets[aid][0] not in ('home_reached', 'delivery_reached', 'pickup_reached'):
                replan_needed = True
        
        if replan_needed:
            self._replan_all(desired_targets)

    def _determine_all_targets(self):
        """タスク重複を防ぎながら全員のターゲットを決定"""
        targets = {}
        assigned_seats = set()

        # 料理を持っているエージェントを優先して処理
        def get_agent_priority(aid):
            return 0 if len(self.env.agent_inventory[aid]) > 0 else 1

        agents_sorted = sorted(self.env.possible_agents, key=lambda a: (get_agent_priority(a), a))

        customer_waits = {c.seat_position: c.wait_time 
                          for c in self.env.customer_manager.customers if c.state == 'ordered'}
        
        for aid in agents_sorted:
            # 1. 料理を持っている -> 配膳へ
            inventory = self.env.agent_inventory[aid]
            if len(inventory) > 0:
                dish = inventory[0]
                targets[aid] = ('delivery', dish['target_seat'])
                continue
            
            # 2. 料理を持っていない -> 誰にも割り当てられていない一番待っている客の料理を探す
            best_dish = None
            max_wait = -1
            for dish in self.env.ready_dishes:
                seat = dish['target_seat']
                if seat in customer_waits and seat not in assigned_seats:
                    wait_time = customer_waits[seat]
                    if wait_time > max_wait:
                        max_wait = wait_time
                        best_dish = dish
            
            if best_dish:
                targets[aid] = ('pickup', self.env.counter_pos)
                assigned_seats.add(best_dish['target_seat'])
            else:
                # 3. やることがなければホーム（待機場所）へ
                home_pos = self.home_positions.get(aid, (0, 0))
                agent_pos = self.env.agent_positions[aid]
                if agent_pos == home_pos:
                    targets[aid] = ('home_reached', home_pos)
                else:
                    targets[aid] = ('home', home_pos)
                    
        return targets

    def _replan_all(self, desired_targets):
        """優先順位に基づいて時空間カレンダーを引き直す"""
        self.reservations.clear()
        self.paths = {'agent_0': [], 'agent_1': []}
        self.targets = desired_targets

        def get_priority(aid):
            t_type = desired_targets[aid][0]
            if t_type == 'delivery': return 0
            if t_type == 'pickup': return 1
            return 2

        agents_sorted = sorted(self.env.possible_agents, key=lambda a: (get_priority(a), a))

        for aid in agents_sorted:
            path = self._space_time_astar(aid, desired_targets[aid])
            if path:
                self.paths[aid] = path
                self._reserve_path(aid, path)
            else:
                self.paths[aid] = []
                self._reserve_path(aid, []) # 動けない場合はその場に居座る予約をする

    def _space_time_astar(self, agent_id, target):
        """時間を考慮した4次元(x, y, dir, t)のA*探索"""
        t_type, t_pos = target
        start_x, start_y = self.env.agent_positions[agent_id]
        start_dir = self.env.agent_directions[agent_id]

        if t_type == 'home_reached':
            return []

        static_obstacles = set(self.env.obstacles) | set(self.env.seats)
        if self.env.counter_pos:
            static_obstacles.add(self.env.counter_pos)

        def get_adj(pos):
            r, c = pos
            return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

        # ゴール設定 (配達・ピックアップは隣接マスがゴール)
        if t_type == 'home':
            valid_goals = [t_pos] if t_pos not in static_obstacles else [p for p in get_adj(t_pos) if p not in static_obstacles]
        else:
            valid_goals = [p for p in get_adj(t_pos) if p not in static_obstacles]

        if not valid_goals:
            return []
        if (start_x, start_y) in valid_goals:
            return []

        counter = itertools.count()

        def heuristic(x, y):
            return min(abs(x - gx) + abs(y - gy) for gx, gy in valid_goals)

        open_set = []
        # (f_score, g_score, id, x, y, dir, time, action_taken, parent_node)
        heapq.heappush(open_set, (heuristic(start_x, start_y), 0, next(counter), start_x, start_y, start_dir, 0, None, None))
        
        visited = set()
        max_time = 60 # 探索の最大時間（深さ）

        while open_set:
            f, g, _, x, y, curr_dir, t, action_taken, parent = heapq.heappop(open_set)

            if t > max_time:
                continue

            state = (x, y, curr_dir, t)
            if state in visited:
                continue
            visited.add(state)

            if (x, y) in valid_goals:
                # ゴールから逆順に経路を復元
                path = []
                curr = (x, y, curr_dir, t, action_taken, parent)
                while curr[5] is not None:
                    path.append((curr[4], curr[0], curr[1], curr[2], curr[3]))
                    curr = curr[5]
                return path[::-1]

            # --- アクション0: Forward ---
            dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][curr_dir]
            nx, ny = x + dx, y + dy
            nt = t + 1

            can_forward = True
            if not (0 <= nx < self.env.grid_size and 0 <= ny < self.env.grid_size):
                can_forward = False
            elif (nx, ny) in static_obstacles:
                can_forward = False
            elif (nx, ny, nt) in self.reservations: # 頂点衝突
                can_forward = False
            else:
                # すれ違い衝突（交差衝突）の判定
                occ_next = self.reservations.get((x, y, nt))
                occ_now = self.reservations.get((nx, ny, t))
                if occ_now is not None and occ_now == occ_next:
                    can_forward = False
            
            if can_forward and (nx, ny, curr_dir, nt) not in visited:
                h = heuristic(nx, ny)
                heapq.heappush(open_set, (g + 1 + h, g + 1, next(counter), nx, ny, curr_dir, nt, 0, (x, y, curr_dir, t, action_taken, parent)))

            # --- アクション1: Turn Right ---
            ndir_r = (curr_dir + 1) % 4
            if (x, y, ndir_r, nt) not in visited and (x, y, nt) not in self.reservations:
                h = heuristic(x, y)
                heapq.heappush(open_set, (g + 1 + h, g + 1, next(counter), x, y, ndir_r, nt, 1, (x, y, curr_dir, t, action_taken, parent)))

            # --- アクション2: Turn Left ---
            ndir_l = (curr_dir - 1) % 4
            if (x, y, ndir_l, nt) not in visited and (x, y, nt) not in self.reservations:
                h = heuristic(x, y)
                heapq.heappush(open_set, (g + 1 + h, g + 1, next(counter), x, y, ndir_l, nt, 2, (x, y, curr_dir, t, action_taken, parent)))

            # --- アクション4: Wait ---
            if (x, y, curr_dir, nt) not in visited and (x, y, nt) not in self.reservations:
                h = heuristic(x, y)
                heapq.heappush(open_set, (g + 1 + h, g + 1, next(counter), x, y, curr_dir, nt, 4, (x, y, curr_dir, t, action_taken, parent)))

        return []

    def _reserve_path(self, agent_id, path):
        """カレンダーに自分の予定を書き込む"""
        for step in path:
            action, x, y, curr_dir, t = step
            self.reservations[(x, y, t)] = agent_id
        
        if path:
            final_x, final_y = path[-1][1], path[-1][2]
            final_t = path[-1][4]
            # ゴール到着後も、他のエージェントに轢かれないように占有し続ける
            for t in range(final_t + 1, final_t + 60):
                self.reservations[(final_x, final_y, t)] = agent_id
        else:
            # 動けない場合は、現在地に居座ることを宣言する
            x, y = self.env.agent_positions[agent_id]
            for t in range(60):
                self.reservations[(x, y, t)] = agent_id

    def get_next_action(self, agent_id):
        if not self.paths[agent_id]:
            return 4 # 予定がない、または目的地到着済みの場合は待機(インタラクト)
        
        step = self.paths[agent_id].pop(0)
        return step[0] # 計算されたアクション(0, 1, 2, 4)を返す


class RuleBasedAgent:
    """
    時空間A* (Space-Time A*) を用いたルールベースエージェント
    """
    shared_planner = None

    def __init__(self, agent_id, env):
        self.id = agent_id
        self.env = env
        
        # クラス変数として共通のプランナーを初期化
        if RuleBasedAgent.shared_planner is None or RuleBasedAgent.shared_planner.env != env:
            RuleBasedAgent.shared_planner = SpaceTimePlanner(env)

    def get_action(self, obs, other_agent=None):
        planner = RuleBasedAgent.shared_planner
        
        # 状況が変わっていれば、全エージェントの経路を一括で引き直す
        planner.update_plan_if_needed()
        
        # 自分のカレンダーから次の1手を取り出して実行
        return planner.get_next_action(self.id)