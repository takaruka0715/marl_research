import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

class TAR2Network(nn.Module):
    """
    Temporal-Agent Reward Redistribution (TAR²) Model
    論文 に基づく実装
    """
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=64, n_layers=2):
        super(TAR2Network, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Embedding(action_dim, hidden_dim)
        
        # Final Outcome Embedding (最終状態Zの条件付け)
        self.outcome_embed = nn.Linear(state_dim * num_agents, hidden_dim) 

        # 2. Transformer Blocks (Temporal & Agent Attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 3. Score Network (Unnormalized Scores)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # concat(features, outcome)
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 出力 c_{i,t}
        )

        # 4. Inverse Dynamics Head (Regularizer)
        self.inv_dynamics = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.to(self.device)

    def forward(self, states, actions, final_states):
        """
        Args:
            states: (Batch, Time, Agents, StateDim)
            actions: (Batch, Time, Agents)
            final_states: (Batch, Agents, StateDim)
        """
        B, T, N, S = states.shape
        
        # --- Embedding ---
        s_emb = self.state_embed(states) # (B, T, N, H)
        a_emb = self.action_embed(actions) # (B, T, N, H)
        
        # 結合して (B, T*N, H) のシーケンスにする
        x = (s_emb + a_emb).view(B, T * N, self.hidden_dim)
        
        # Final Outcome Conditioning
        flat_final = final_states.view(B, -1)
        z = self.outcome_embed(flat_final).unsqueeze(1) # (B, 1, H)
        z_expanded = z.expand(-1, T * N, -1)

        # --- Transformer Processing ---
        features = self.transformer(x) # (B, T*N, H)
        
        # --- Score Calculation ---
        combined = torch.cat([features, z_expanded], dim=-1)
        raw_scores = self.score_head(combined) # (B, T*N, 1)
        scores = raw_scores.view(B, T, N) # (B, T, N)に整形

        # --- Inverse Dynamics Prediction ---
        # t と t+1 の特徴量から行動を予測
        curr_feat = features.view(B, T, N, -1)[:, :-1, :, :] # t=0 to T-1
        next_feat = features.view(B, T, N, -1)[:, 1:, :, :]  # t=1 to T
        inv_input = torch.cat([curr_feat, next_feat], dim=-1)
        pred_actions = self.inv_dynamics(inv_input) # (B, T-1, N, ActionDim)

        return scores, pred_actions

    def get_redistributed_rewards(self, scores, total_episode_reward):
        """
        決定論的正規化による報酬再分配
        Returns: shaped_rewards (B, T, N)
        """
        epsilon = 1e-8
        B, T, N = scores.shape
        
        # 1. Temporal Weights
        c_agg = scores.sum(dim=2) # (B, T)
        min_c_agg = c_agg.min(dim=1, keepdim=True)[0]
        numerator_temp = c_agg - min_c_agg
        denominator_temp = numerator_temp.sum(dim=1, keepdim=True) + epsilon
        w_temp = numerator_temp / denominator_temp # (B, T)
        
        # 2. Agent Weights
        min_c_agent = scores.min(dim=2, keepdim=True)[0]
        numerator_agent = scores - min_c_agent
        denominator_agent = (numerator_agent).sum(dim=2, keepdim=True) + epsilon
        w_agent = numerator_agent / denominator_agent # (B, T, N)

        # 3. Final Redistributed Reward
        R_total = total_episode_reward.view(B, 1, 1)
        shaped_rewards = w_temp.unsqueeze(2) * w_agent * R_total
        
        return shaped_rewards

    def update(self, batch_states, batch_actions, batch_rewards, lambda_id=0.1):
        """TAR2モデルの学習 (Algorithm 2)"""
        final_states = batch_states[:, -1, :, :]
        
        scores, pred_actions_logits = self.forward(batch_states, batch_actions, final_states)
        
        # Reward Regression Loss: (R_episode - sum(scores))^2
        total_predicted_score = scores.sum(dim=(1, 2))
        regression_loss = ((batch_rewards - total_predicted_score) ** 2).mean()
        
        # Inverse Dynamics Loss
        if batch_actions.shape[1] > 1: # 少なくとも2ステップ以上ないとIDロスは計算不可
            target_actions = batch_actions[:, :-1, :].reshape(-1)
            pred_logits_flat = pred_actions_logits.reshape(-1, self.action_dim)
            id_loss = nn.CrossEntropyLoss()(pred_logits_flat, target_actions)
        else:
            id_loss = torch.tensor(0.0).to(self.device)
        
        total_loss = regression_loss + lambda_id * id_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

def collate_trajectories(trajectories, device):
    """エピソードデータのパディングとバッチ化"""
    batch_states = []
    batch_actions = []
    batch_rewards = []
    
    for traj in trajectories:
        batch_states.append(torch.FloatTensor(traj['states']).to(device))
        batch_actions.append(torch.LongTensor(traj['actions']).to(device))
        batch_rewards.append(torch.FloatTensor([traj['total_reward']]).to(device))

    # パディング (Batch, MaxTime, N, ...)
    padded_states = pad_sequence(batch_states, batch_first=True)
    padded_actions = pad_sequence(batch_actions, batch_first=True)
    stacked_rewards = torch.stack(batch_rewards).squeeze(1)

    return padded_states, padded_actions, stacked_rewards, None # final_statesは内部生成