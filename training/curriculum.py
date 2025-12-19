class Curriculum:
    """
    案A：配達数（Served Count）ベースの適応的カリキュラム
    """
    def __init__(self):
        self.current_stage_idx = 0
        self.stages = [
            # Stage 1: 家具はあるが壁はない広場
            {
                'layout': 'empty', 'customers': True, 'spawn_interval': 60,
                'description': 'Stage 1: Basic Delivery (No Walls)',
                'threshold': 3.0,        # 平均3件配膳できたら次へ
                'timeout_episodes': 2000 # セーフティネット
            },
            # Stage 2: 障害物回避
            {
                'layout': 'basic', 'customers': True, 'spawn_interval': 40,
                'description': 'Stage 2: Obstacle Navigation',
                'threshold': 5.0,        # 平均5件配膳
                'timeout_episodes': 3000
            },
            # Stage 3: 複雑な環境
            {
                'layout': 'complex', 'customers': True, 'spawn_interval': 40,
                'description': 'Stage 3: Complex Environment',
                'threshold': 7.0,        # 平均7件配膳
                'timeout_episodes': 4000
            },
            # Stage 4: 高負荷（最終段階）
            {
                'layout': 'complex', 'customers': True, 'spawn_interval': 20,
                'description': 'Stage 4: High Load Efficiency',
                'threshold': float('inf'), 
                'timeout_episodes': float('inf')
            },
        ]
    
    def get_current_stage(self):
        """現在のステージ設定を取得"""
        return self.stages[self.current_stage_idx]

    def check_progression(self, avg_served_count, episodes_spent_in_stage):
        """
        配達数に基づいた進捗判定
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False, None
        
        current_config = self.stages[self.current_stage_idx]
        target = current_config['threshold']
        limit = current_config['timeout_episodes']
        
        # 目標配達数達成による進行
        if avg_served_count >= target:
            self.current_stage_idx += 1
            return True, "SUCCESS (Delivery Target Reached)"
            
        # タイムアウトによる強制進行
        if episodes_spent_in_stage >= limit:
            self.current_stage_idx += 1
            return True, "TIMEOUT (Forced Progression)"
            
        return False, None