from collections import deque
import random

class SharedReplayBuffer:
    """共有経験リプレイバッファ"""
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        """経験を追加"""
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """ランダムにサンプル"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
