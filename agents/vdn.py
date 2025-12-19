import torch
import torch.nn as nn
from .network import DuelingDQN, UrgencyAwareDQN

class VDNNetwork(nn.Module):
    """
    Value Decomposition Networks (VDN)
    各エージェントのローカルQ値を計算する。
    """
    def __init__(self, input_dim, output_dim, num_agents=2, agent_config=None):
        super(VDNNetwork, self).__init__()
        self.num_agents = num_agents
        self.agent_config = agent_config
        
        # 各エージェント用のローカルQ値ネットワーク
        self.local_q_networks = nn.ModuleList([
            self._build_q_network(input_dim, output_dim) 
            for _ in range(num_agents)
        ])
    
    def _build_q_network(self, input_dim, output_dim):
        # 提案手法のフラグが立っている場合、Attention付きネットワークを使用
        if self.agent_config and hasattr(self.agent_config, 'attn_config') and self.agent_config.attn_config.use_attention:
            return UrgencyAwareDQN(input_dim, output_dim, self.agent_config.attn_config)
        else:
            return DuelingDQN(input_dim, output_dim)
    
    def forward(self, states, task_features=None):
        """
        Args:
            states: List[torch.Tensor] (各エージェントの状態)
            task_features: torch.Tensor (Batch, MaxTasks, 3) 共通のタスク情報
        Returns:
            q_locals: List[torch.Tensor] (各エージェントのローカルQ値 [Batch, Action])
        """
        q_locals = []
        for i, state in enumerate(states):
            # Attentionネットワークは task_features も受け取る
            if isinstance(self.local_q_networks[i], UrgencyAwareDQN):
                q_local = self.local_q_networks[i](state, task_features)
            else:
                q_local = self.local_q_networks[i](state) # 従来のDQNはタスク特徴量を受け取らない
            q_locals.append(q_local)
        
        return q_locals
    
    def get_local_q(self, agent_idx, state, task_features=None):
        """特定エージェントのローカルQ値を取得"""
        net = self.local_q_networks[agent_idx]
        if isinstance(net, UrgencyAwareDQN):
            return net(state, task_features)
        else:
            return net(state)

# エラーの原因となっていたクラスを追加
class VDNTargetNetwork(VDNNetwork):
    pass