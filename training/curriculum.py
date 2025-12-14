class Curriculum:
    """
    タイムアウト機能付き適応的カリキュラム
    """
    def __init__(self):
        self.current_stage_idx = 0
        self.stages = [
            # Stage 1: Basic
            {
                'layout': 'empty', 'customers': True, 'spawn_interval': 60,
                'description': 'Stage 1: Basic Mechanics',
                'threshold': 10.0,       # クリア目標
                'timeout_episodes': 2000 # ★制限時間（これを超えたら強制進行）
            },
            # Stage 2: Navigation
            {
                'layout': 'basic', 'customers': True, 'spawn_interval': 40,
                'description': 'Stage 2: Obstacle Navigation',
                'threshold': 20.0,
                'timeout_episodes': 3000 # 3000エピソード猶予を与える
            },
            # Stage 3: Complex
            {
                'layout': 'complex', 'customers': True, 'spawn_interval': 40,
                'description': 'Stage 3: Complex Environment',
                'threshold': 30.0,
                'timeout_episodes': 4000
            },
            # Stage 4: High Load (Final)
            {
                'layout': 'complex', 'customers': True, 'spawn_interval': 20,
                'description': 'Stage 4: High Load Efficiency',
                'threshold': float('inf'), # 最終ステージなのでクリアなし
                'timeout_episodes': float('inf')
            },
        ]
    
    def get_current_stage(self):
        return self.stages[self.current_stage_idx]

    def check_progression(self, recent_performance, episodes_spent_in_stage):
        """
        進捗判定
        Args:
            recent_performance (float): 最近の成績
            episodes_spent_in_stage (int): このステージに滞在してからの経過エピソード数
        Returns:
            (bool, str): (進むかどうか, 理由)
        """
        # 最終ステージなら何もしない
        if self.current_stage_idx >= len(self.stages) - 1:
            return False, None
            
        current_config = self.stages[self.current_stage_idx]
        target = current_config['threshold']
        limit = current_config['timeout_episodes']
        
        # 1. 目標達成パターン (Ideal)
        if recent_performance >= target:
            self.current_stage_idx += 1
            return True, "SUCCESS (Score Reached)"
            
        # 2. タイムアウトパターン (Safety Net)
        # 局所解にハマった場合、ここで強制的に環境を変えることで脱出を図る
        if episodes_spent_in_stage >= limit:
            self.current_stage_idx += 1
            return True, "TIMEOUT (Forced Progression)"
            
        return False, None