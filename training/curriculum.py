# training/curriculum.py [cite: 125, 126, 127, 128, 129, 130, 131, 132, 133]

class Curriculum:
    """
    タイムアウト機能付き適応的カリキュラム（配膳数ベース）
    """
    def __init__(self):
        self.current_stage_idx = 0
        self.stages = [
            # Stage 1: Basic
            {
                'layout': 'empty', 'customers': True, 'spawn_interval': 60,
                'description': 'Stage 1: Basic Mechanics',
                'threshold': 1.0,        # ★平均1皿以上の配膳でクリア [cite: 126]
                'timeout_episodes': 2000 
            },
            # Stage 2: Navigation
            {
                'layout': 'basic', 'customers': True, 'spawn_interval': 40,
                'description': 'Stage 2: Obstacle Navigation',
                'threshold': 3.0,        # ★平均3皿以上の配膳でクリア [cite: 127]
                'timeout_episodes': 3000 
            },
            # Stage 3: Complex
            {
                'layout': 'complex', 'customers': True, 'spawn_interval': 40,
                'description': 'Stage 3: Complex Environment',
                'threshold': 5.0,        # ★平均5皿以上の配膳でクリア [cite: 128]
                'timeout_episodes': 4000
            },
            # Stage 4: High Load (Final)
            {
                'layout': 'complex', 'customers': True, 'spawn_interval': 20,
                'description': 'Stage 4: High Load Efficiency',
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
            recent_served_avg (float): 直近エピソードの平均配膳数 [cite: 130]
            episodes_spent_in_stage (int): 現在のステージでの滞在エピソード数
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False, None
            
        current_config = self.stages[self.current_stage_idx]
        target = current_config['threshold'] # 配膳数の目標値
        limit = current_config['timeout_episodes']
        
        # 1. 目標配膳数達成パターン [cite: 132]
        if recent_served_avg >= target:
            self.current_stage_idx += 1
            return True, "SUCCESS (Served Target Reached)"
            
        # 2. タイムアウトパターン [cite: 133]
        if episodes_spent_in_stage >= limit:
            self.current_stage_idx += 1
            return True, "TIMEOUT (Forced Progression)"
            
        return False, None