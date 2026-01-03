# main.py の全書き換え案

import torch
import argparse
import os
from confs import Config, AgentConfig, TrainingConfig
from training import Trainer
from envs import RestaurantEnv
from agents import DQNAgent, VDNAgent, QMIXAgent
from visualization import plot_learning_curves, create_restaurant_gif

def get_args():
    parser = argparse.ArgumentParser(description="MARL Restaurant Service")
    parser.add_argument('--train', action='store_true', help='Run training mode')
    parser.add_argument('--eval', action='store_true', help='Run evaluation/visualization mode only')
    parser.add_argument('--use_vdn', action='store_true', help='Use VDN architecture')
    parser.add_argument('--use_qmix', action='store_true', help='Use QMIX architecture')
    parser.add_argument('--use_tar2', action='store_true', help='Use TAR2 reward shaping')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save/load models')
    return parser.parse_args()

def main():
    args = get_args()
    
    # アルゴリズム名の決定（ファイル名サフィックス用）
    if args.use_qmix:
        mode_suffix = "qmix" + ("_tar2" if args.use_tar2 else "")
    elif args.use_vdn:
        mode_suffix = "vdn" + ("_tar2" if args.use_tar2 else "")
    else:
        mode_suffix = "dqn"

    # --- Config設定 ---
    config = Config()
    if args.use_tar2:
        print(">> TAR2 mode: Overwriting rewards for sparse setting")
        config.pickup_reward = 10.0
        config.delivery_reward = 400.0
        config.collision_penalty = -10.0
        config.step_cost = -0.01
        config.wait_penalty = 0.0

    # ==========================================
    # 学習モード (Training Mode)
    # ==========================================
    if args.train:
        print(f"=== Starting Training [{mode_suffix}] ===")
        agent_config = AgentConfig(use_vdn=args.use_vdn, use_tar2=args.use_tar2)
        training_config = TrainingConfig(agent_config=agent_config)

        trainer = Trainer(
            num_episodes=training_config.num_episodes,
            use_shared_replay=training_config.use_shared_replay,
            use_vdn=args.use_vdn,
            use_qmix=args.use_qmix,
            use_tar2=args.use_tar2,
            config=config
        )

        # 学習実行
        agents, ep_rewards, avg_rewards, served_stats, col_rates, wait_times, final_env = trainer.train()

        # モデル保存
        print("\nSaving models...")
        trainer.save_agents(directory=args.model_dir, suffix=f"_{mode_suffix}")

        # グラフ保存
        plot_filename = f"learning_curves_{mode_suffix}.png"
        plot_learning_curves(ep_rewards, avg_rewards, served_stats, col_rates, wait_times, filename=plot_filename)
        
        # GIF生成 (学習直後の状態)
        gif_filename = f"restaurant_{mode_suffix}.gif"
        print(f"Generating animation to {gif_filename}...")
        create_restaurant_gif(final_env, agents, filename=gif_filename)

    # ==========================================
    # 評価・可視化モード (Evaluation Mode)
    # ==========================================
    elif args.eval:
        print(f"=== Starting Evaluation (GIF Generation) [{mode_suffix}] ===")
        
        # 1. 環境の準備 (可視化用にComplexなどの難しいレイアウトを指定も可能)
        # 評価時は customers=True, layout='complex' などを想定
        env = RestaurantEnv(
            layout_type='complex', 
            enable_customers=True, 
            customer_spawn_interval=20,
            local_obs_size=5,
            config=config
        )
        
        state_dim = env.observation_space('agent_0').shape[0]
        action_dim = 5 # envs/restaurant_env.py で spaces.Discrete(5) になっているため
        
        # 2. エージェントの初期化とロード
        agents = {}
        
        if args.use_qmix:
            global_state_dim = state_dim * 2
            # バッファは推論時には不要なのでNone
            agent = QMIXAgent(state_dim, action_dim, global_state_dim, num_agents=2, shared_buffer=None)
            model_path = f"{args.model_dir}/qmix_agent_{mode_suffix}.pth"
            if os.path.exists(model_path):
                agent.load_model(model_path)
                agents['qmix'] = agent
                print(f"Loaded QMIX model from {model_path}")
            else:
                print(f"Error: Model file not found {model_path}")
                return

        elif args.use_vdn:
            agent = VDNAgent(state_dim, action_dim, num_agents=2, shared_buffer=None)
            model_path = f"{args.model_dir}/vdn_agent_{mode_suffix}.pth"
            if os.path.exists(model_path):
                agent.load_model(model_path)
                agents['vdn'] = agent
                print(f"Loaded VDN model from {model_path}")
            else:
                print(f"Error: Model file not found {model_path}")
                return
        else:
            # Independent DQN
            for agent_name in env.possible_agents:
                agent = DQNAgent(state_dim, action_dim, shared_buffer=None)
                model_path = f"{args.model_dir}/{agent_name}_{mode_suffix}.pth"
                if os.path.exists(model_path):
                    agent.load_model(model_path)
                    agents[agent_name] = agent
                    print(f"Loaded {agent_name} from {model_path}")
                else:
                    print(f"Error: Model file not found {model_path}")
                    return

        # 3. GIF生成
        gif_filename = f"restaurant_eval_{mode_suffix}.gif"
        print(f"\nGenerating animation using loaded models to {gif_filename}...")
        
        # Epsilonを0にして純粋な活用(Exploitation)を行う場合
        # 各エージェントクラスの属性を直接書き換える
        for k, ag in agents.items():
            ag.epsilon = 0.0
            
        create_restaurant_gif(env, agents, filename=gif_filename)
        print("Done.")

    else:
        print("Please specify --train or --eval argument.")

if __name__ == "__main__":
    main()