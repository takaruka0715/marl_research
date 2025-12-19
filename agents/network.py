import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """従来のDueling DQN (変更なし)"""
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x, task_features=None):
        # 互換性のため引数 task_features を受け取るが無視する
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class UrgencyAwareDQN(nn.Module):
    """
    4. 緊急度アテンション付き Qネットワーク
    Query: エージェント状態
    Key/Value: タスク（注文）情報 + 緊急度
    """
    def __init__(self, input_dim, output_dim, attn_config):
        super(UrgencyAwareDQN, self).__init__()
        self.attn_config = attn_config
        hidden_dim = 256
        embed_dim = attn_config.task_embed_dim
        
        # --- 4.1 ネットワーク入力 ---
        
        # Agent State Encoder (Queryの元)
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim) # h_i
        )
        
        # Task Feature Encoder (Key/Valueの元)
        # task_dim = 3 (pos_x, pos_y, urgency)
        self.task_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU()
        ) # x_j -> embedding
        
        # --- 4.2 アテンション計算 ---
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))
        
        # --- 4.3 Q値の算出 ---
        # Concat[h_i, c_i] -> Q
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state, task_features=None):
        """
        Args:
            state: (Batch, StateDim) -> o_i
            task_features: (Batch, MaxTasks, 3) -> [pos_x, pos_y, urgency]
                           ※ タスクがない場合は0パディングされている想定
        """
        self.scale = self.scale.to(state.device)
        
        # 1. Encode Agent State (Query)
        h_i = self.state_encoder(state) # (Batch, Embed)
        query = self.w_q(h_i).unsqueeze(1) # (Batch, 1, Embed)
        
        # 2. Encode Tasks (Key/Value)
        if task_features is None or task_features.sum() == 0:
            # タスク情報がない場合、コンテキストはゼロベクトル
            context = torch.zeros_like(h_i)
        else:
            # task_features: (Batch, M, 3)
            task_emb = self.task_encoder(task_features) # (Batch, M, Embed)
            
            key = self.w_k(task_emb)   # (Batch, M, Embed)
            value = self.w_v(task_emb) # (Batch, M, Embed)
            
            # 3. Attention Calculation
            # score = Q * K^T / sqrt(d)
            # (Batch, 1, Embed) @ (Batch, Embed, M) -> (Batch, 1, M)
            scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
            
            # Masking (オプション): パディング部分(全て0の特徴量)のスコアを下げる処理が必要ならここで行う
            # 簡易実装として、0パディングされたタスクはKeyも0に近いのでAttentionが低くなることを期待する
            
            attn_weights = F.softmax(scores, dim=-1) # (Batch, 1, M)
            
            # Context = Weights * V
            # (Batch, 1, M) @ (Batch, M, Embed) -> (Batch, 1, Embed)
            context = torch.matmul(attn_weights, value).squeeze(1) # (Batch, Embed)

        # 4. Final Q Calculation
        combined = torch.cat([h_i, context], dim=1) # (Batch, Embed * 2)
        q_values = self.head(combined)
        
        return q_values