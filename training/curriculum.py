class Curriculum:
    """
    タイムアウト機能付き適応的カリキュラム（配膳数ベース）
    修正版: ドーナツ型（スライディング・ウィンドウ）マッピング強制カリキュラム
    """
    def __init__(self):
        self.current_stage_idx = 0
        self.stages = [
            # Stage 1: 近距離での基礎マッピング (Basic Mapping)
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 0,
                'max_customer_dist': 5,   # 近い客のみ出現
                'description': 'Stage 1: Close-range Basic Mapping',
                'threshold': 5.0,         # 目標5.0皿に引き下げ
                'timeout_episodes': 10000 # 学習期間を延長
            },
            # Stage 2: 遠距離への強制マッピング (Donut / Forced Far Delivery)
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 8,   # 手前の客を消去
                'max_customer_dist': 15,  # 遠くの客のみ出現
                'description': 'Stage 2: Forced Far-range Mapping (Donut)',
                'threshold': 5.0,
                'timeout_episodes': 10000
            },
            # Stage 3: 全範囲解禁・高負荷 (Full Range / High Load)
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 15,     # 注文ペースを上げる
                'min_customer_dist': 0,
                'max_customer_dist': float('inf'), # 距離制限を完全解除
                'description': 'Stage 3: Full Range Integration (High Load)',
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