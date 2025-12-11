class Curriculum:
    """カリキュラム学習段階管理"""
    def __init__(self):
        self.stages = [
            {'episodes': (0, 1500), 'layout': 'empty', 'customers': False, 
             'spawn_interval': 999, 'description': 'Basic Movement'},
            {'episodes': (1500, 3000), 'layout': 'empty', 'customers': False, 
             'spawn_interval': 10, 'description': 'Random Delivery'},
            {'episodes': (3000, 5500), 'layout': 'basic', 'customers': False, 
             'spawn_interval': 15, 'description': 'Navigation Obstacles'},
            {'episodes': (5500, 10000), 'layout': 'basic', 'customers': True, 
             'spawn_interval': 30, 'description': 'Simple Service'},
            {'episodes': (10000, 15000), 'layout': 'complex', 'customers': False, 
             'spawn_interval': 15, 'description': 'Complex Navigation'},
            {'episodes': (15000, 30000), 'layout': 'complex', 'customers': True, 
             'spawn_interval': 20, 'description': 'Full Service'},
        ]
    
    def get_stage(self, episode):
        """エピソード番号から現在のステージを取得"""
        for stage in self.stages:
            if stage['episodes'][0] <= episode < stage['episodes'][1]:
                return stage
        return None
