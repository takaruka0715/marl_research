import torch
import torch.nn as nn

class VDNNetwork(nn.Module):
    """
    Value Decomposition Networks (VDN)
    各エージェントのローカルQ値を加算して全体のQ値を構成する
    """
    def __init__(self, input_dim, output_dim, num_agents=2):
        super(VDNNetwork, self).__init__()
        self.num_agents = num_agents
        self.output_dim = output_dim
        
        # 各エージェント用のローカルQ値ネットワーク
        self.local_q_networks = nn.ModuleList([
            self._build_q_network(input_dim, output_dim) 
            for _ in range(num_agents)
        ])
    
    def _build_q_network(self, input_dim, output_dim):
        """ローカルQ値ネットワークの構築"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, states):
        """
        Args:
            states: List[torch.Tensor] (各エージェントの状態)
        Returns:
            q_tot: torch.Tensor (全体のQ値)
            q_locals: List[torch.Tensor] (各エージェントのローカルQ値)
        """
        q_locals = []
        for i, state in enumerate(states):
            q_local = self.local_q_networks[i](state)
            q_locals.append(q_local)
        
        # VDN: 全体のQ値 = 各エージェントのローカルQ値の合計
        q_tot = torch.zeros_like(q_locals[0])
        for q_local in q_locals:
            q_tot = q_tot + q_local
        
        return q_tot, q_locals
    
    def get_local_q(self, agent_idx, state):
        """特定エージェントのローカルQ値を取得"""
        return self.local_q_networks[agent_idx](state)

class VDNTargetNetwork(nn.Module):
    """VDN用ターゲットネットワーク"""
    def __init__(self, input_dim, output_dim, num_agents=2):
        super(VDNTargetNetwork, self).__init__()
        self.num_agents = num_agents
        self.output_dim = output_dim
        
        self.local_q_networks = nn.ModuleList([
            self._build_q_network(input_dim, output_dim) 
            for _ in range(num_agents)
        ])
    
    def _build_q_network(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, states):
        """ターゲットネットワークのフォワードパス"""
        q_locals = []
        for i, state in enumerate(states):
            q_local = self.local_q_networks[i](state)
            q_locals.append(q_local)
        
        q_tot = torch.zeros_like(q_locals[0])
        for q_local in q_locals:
            q_tot = q_tot + q_local
        
        return q_tot, q_locals
