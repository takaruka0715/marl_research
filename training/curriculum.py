class Curriculum:
    """
    タイムアウト機能付き適応的カリキュラム（配膳数ベース）
    修正版: 距離ベースの難易度調整を導入
    """
    def __init__(self):
        self.current_stage_idx = 0
        self.stages = [
            # Stage 1: 遠距離限定 (Long Distance Only)
            # [cite_start]カウンター(7,1)からマンハッタン距離10以上離れた席のみ出現 [cite: 244]
            # これにより、必然的にマップ奥地への移動を学習する
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 40,
                'min_customer_dist': 10,  # ★新規パラメータ: 近場の客を禁止
                'description': 'Stage 1: Long Distance Delivery',
                'threshold': 2.0,        # 平均2皿以上でクリア
                'timeout_episodes': 3000 
            },
            # Stage 2: 中距離以上 (Medium + Long)
            # 距離制限を緩和し、中距離の客も出現させる
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 5,   # ★緩和
                'description': 'Stage 2: Medium & Long Distance',
                'threshold': 4.0,        # 平均4皿以上でクリア [cite: 246]
                'timeout_episodes': 3000 
            },
            # Stage 3: 全範囲 (Full Random)
            # 距離制限なし（近距離も解禁）
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 0,   # ★制限なし [cite: 247]
                'description': 'Stage 3: Full Random Distribution',
                'threshold': 6.0,        # 平均6皿以上でクリア
                'timeout_episodes': 4000
            },
            # Stage 4: High Load (Final)
            # 高負荷設定（スパム間隔短縮）
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 20,
                'min_customer_dist': 0,
                'description': 'Stage 4: High Load Efficiency',
                'threshold': float('inf'),
                'timeout_episodes': float('inf')
            },
        ]
    
    def get_current_stage(self):
        return self.stages[self.current_stage_idx] # [cite: 249]

    def check_progression(self, recent_served_avg, episodes_spent_in_stage):
        """
        進捗判定（配膳数ベース）
        Args:
            [cite_start]recent_served_avg (float): 直近エピソードの平均配膳数 [cite: 250]
            episodes_spent_in_stage (int): 現在のステージでの滞在エピソード数
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False, None
            
        current_config = self.stages[self.current_stage_idx]
        target = current_config['threshold'] # 配膳数の目標値
        limit = current_config['timeout_episodes']
        
        # [cite_start]1. 目標配膳数達成パターン [cite: 251]
        if recent_served_avg >= target:
            self.current_stage_idx += 1
            return True, "SUCCESS (Served Target Reached)"
            
        # 2. タイムアウトパターン
        if episodes_spent_in_stage >= limit:
            self.current_stage_idx += 1
            return True, "TIMEOUT (Forced Progression)"
            
        return False, None