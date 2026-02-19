class Curriculum:
    """
    タイムアウト機能付き適応的カリキュラム（配膳数ベース）
    修正版v2: 距離制限をより細かく刻み、難易度上昇を滑らかに設定
    """
    def __init__(self):
        self.current_stage_idx = 0
        self.stages = [
            # Stage 1: 超遠距離 (Very Long Distance)
            # カウンター(7,1)からマンハッタン距離12以上
            # 四隅などの本当に遠い席のみ出現させ、確実に遠出を学習させる
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 40,
                'min_customer_dist': 12,  # ★かなり厳しく制限
                'description': 'Stage 1: Very Long Distance',
                'threshold': 2.0,        # 2皿でクリア
                'timeout_episodes': 4000
            },
            # Stage 2: 遠距離 (Long Distance)
            # 距離9以上。外周エリア全体を含める
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 40,
                'min_customer_dist': 8,   # ★少し緩和
                'description': 'Stage 2: Long Distance',
                'threshold': 3.0,        # 3皿でクリア
                'timeout_episodes': 3000
            },
            # Stage 3: 中距離 (Medium Distance)
            # 距離6以上。カウンター近辺以外はほぼ解禁
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 6,   # ★さらに緩和
                'description': 'Stage 3: Medium Distance',
                'threshold': 4.0,        # 4皿でクリア
                'timeout_episodes': 3000
            },
            # Stage 4: 全範囲 (Full Random)
            # 距離制限なし（近距離も解禁）
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 0,   # ★全解禁
                'description': 'Stage 4: Full Random Distribution',
                'threshold': 6.0,        # 6皿でクリア
                'timeout_episodes': 4000
            },
            # Stage 5: High Load (Final)
            # 高負荷設定
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 20,
                'min_customer_dist': 0,
                'description': 'Stage 5: High Load Efficiency',
                'threshold': float('inf'),
                'timeout_episodes': float('inf')
            },
        ]
    
    def get_current_stage(self):
        return self.stages[self.current_stage_idx]

    def check_progression(self, recent_served_avg, episodes_spent_in_stage):
        """
        進捗判定（配膳数ベース）
        Args:
            recent_served_avg (float): 直近エピソードの平均配膳数
            episodes_spent_in_stage (int): 現在のステージでの滞在エピソード数
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False, None
            
        current_config = self.stages[self.current_stage_idx]
        target = current_config['threshold']
        limit = current_config['timeout_episodes']
        
        # 1. 目標配膳数達成
        if recent_served_avg >= target:
            self.current_stage_idx += 1
            return True, "SUCCESS (Served Target Reached)"
            
        # 2. タイムアウト
        if episodes_spent_in_stage >= limit:
            self.current_stage_idx += 1
            return True, "TIMEOUT (Forced Progression)"
            
        return False, None