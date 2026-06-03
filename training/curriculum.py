# ================================================
# FILE: training/curriculum.py
# ================================================
class Curriculum:
    """
    タイムアウト機能付き適応的カリキュラム（配膳数ベース）
    料理の種類数（num_food_types）を段階的に増やす機能を追加
    """
    def __init__(self):
        self.current_stage_idx = 0
        self.stages = [
            # Stage 0: 料理1種類 ＆ カウンターの目の前限定 (超簡単)
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 0,
                'max_customer_dist': 3,   # ★カウンターのすぐ隣にしか客が出ない
                'num_food_types': 1,      # ★料理は1種類のみ
                'description': 'Stage 0: Super Easy - Near Counter & 1 Food',
                'threshold': 6.0,         # 目標3.0皿
                'timeout_episodes': 3000
            },
            # Stage 1: 料理1種類 ＆ 近距離の基礎マッピング (Basic Mapping)
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 0,
                'max_customer_dist': 6,   # ★少し広げる
                'num_food_types': 1,      # ★料理はまだ1種類
                'description': 'Stage 1: Close-range Basic Mapping & 1 Food',
                'threshold': 10.0,
                'timeout_episodes': 5000
            },
            # Stage 2: 料理3種類 ＆ 遠距離への強制マッピング (Donut / Forced Far Delivery)
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 30,
                'min_customer_dist': 8,   # 手前の客を消去
                'max_customer_dist': 15,  # 遠くの客のみ出現
                'num_food_types': 3,      # ★料理を3種類に増やす
                'description': 'Stage 2: Forced Far-range Mapping & 3 Foods',
                'threshold': 10.0,
                'timeout_episodes': 10000 
            },
            # ★追加 Stage 2.5: 遠距離特化の猛特訓（バッファを遠距離経験で満たす）
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 20,     # 注文ペースを少し上げる
                'min_customer_dist': 8,   # 引き続き遠くの客のみ
                'max_customer_dist': 15,
                'num_food_types': 3,
                'description': 'Stage 2.5: Intensive Far-range Training (Buffer Filling)',
                'threshold': 16.0,         # 非常に高いハードル（実質的にタイムアウトまで遠距離を強制）
                'timeout_episodes': 3000  # 3000エピソードでバッファを遠距離の経験で完全に塗り替える
            },
            # Stage 3: 全範囲解禁・高負荷 (Full Range / High Load)
            {
                'layout': 'complex', 
                'customers': True, 
                'spawn_interval': 15,     # 注文ペースを上げる
                'min_customer_dist': 0,
                'max_customer_dist': float('inf'), # 距離制限を完全解除
                'num_food_types': 3,
                # ★追加: 最終ステージでは距離ではなく到達難易度カテゴリごとに均等に顧客を出す
                'spawn_mode': 'difficulty_balanced',
                'difficulty_categories': ['easy', 'medium', 'hard'],
                'description': 'Stage 4: Full Range Integration (Difficulty Balanced)',
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
