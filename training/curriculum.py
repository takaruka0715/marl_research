class Curriculum:
    """
    タイムアウト機能付き適応的カリキュラム（配膳数ベース）
    修正版: 全範囲解禁スタートの簡略化カリキュラム
    """
    def __init__(self):
        self.current_stage_idx = 0
        self.stages = [
            # Stage 1: 全範囲 (Full Random)
            # 最初から距離制限なし（近距離も遠距離も解禁）
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,     # 適度な注文ペース
                'min_customer_dist': 0,   # 全解禁
                'description': 'Stage 1: Full Random Distribution',
                'threshold': 10.0,        # 10皿でクリア
                'timeout_episodes': 5000
            },
            # Stage 2: High Load (Final)
            # 高負荷設定（注文ペースが速い）
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 15,
                'min_customer_dist': 0,
                'description': 'Stage 2: High Load Efficiency',
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