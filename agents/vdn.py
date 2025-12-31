import torch
import torch.nn as nn

class VDNNetwork(nn.Module):
    """
    Value Decomposition Networks (VDN)
    各エージェントのローカルQ値を計算する。
    Q_totの計算はAgentクラス側で行うため、ここではリストを返す。
    """
    def __init__(self, input_dim, output_dim, num_agents=2):
        super(VDNNetwork, self).__init__()
        self.num_agents = num_agents
        # 共有ネットワークを1つだけ定義
        self.shared_q_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
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
    
    def forward(self, states_list):
        # 入力された全エージェントの状態に対して、同じネットワークを適用
        q_locals = []
        for state in states_list:
            q_locals.append(self.shared_q_network(state))
        return q_locals
    
    def get_local_q(self, agent_idx, state):
        # どのエージェントに対しても同じネットワークで計算
        return self.shared_q_network(state)

# TargetNetworkも構造は同じなので継承または同様に修正
class VDNTargetNetwork(VDNNetwork):
    pass