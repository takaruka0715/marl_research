import torch
from confs import Config, AgentConfig, TrainingConfig
from training import Trainer
from visualization import plot_learning_curves, create_restaurant_gif

def main(use_vdn=False, use_qmix=False, use_tar2=False):
    """
    メイン実行関数 (ParallelEnv対応)
    Args:
        use_vdn (bool): VDNを使用するか
        use_qmix (bool): QMIXを使用するか
        use_tar2 (bool): TAR2を使用するか
    """
    algo_name = "Independent DQN"
    if use_qmix: algo_name = "QMIX"
    elif use_vdn: algo_name = "VDN"

    print(f"Algorithm: {algo_name}")
    print(f"Reward Shaping (TAR2): {'ON' if use_tar2 else 'OFF'}")
    
    # 1. 基本設定の読み込み
    config = Config()
    
    # 2. TAR2を使用する場合の設定上書き
    # 途中報酬をカットして、スパース報酬（結果重視）の設定にする
    if use_tar2:
        print(">> TAR2 mode detected: Overwriting intermediate rewards to 0.0 (Sparse Reward Setting)")
        config.pickup_reward = 10.0      # 最小限のピックアップ報酬
        config.delivery_reward = 400.0   # 大きな達成報酬
        config.collision_penalty = -10.0 
        config.step_cost = -0.01         # 移動コストは小さく
        config.wait_penalty = 0.0
    else:
        print(">> Standard mode: Using dense rewards defined in config.py")

    # 3. エージェントと学習設定の初期化
    agent_config = AgentConfig(use_vdn=use_vdn, use_tar2=use_tar2)
    training_config = TrainingConfig(agent_config=agent_config)
    
    # 4. トレーナーの初期化
    # ParallelEnv対応版のTrainerを呼び出します
    trainer = Trainer(
        num_episodes=training_config.num_episodes,
        use_shared_replay=training_config.use_shared_replay,
        use_vdn=use_vdn,
        use_qmix=use_qmix,
        use_tar2=use_tar2,
        config=config
    )
    
    # 5. 学習実行
    # Trainer内部で ParallelEnv がインスタンス化され、学習が進みます
    agents, episode_rewards, avg_rewards, served_stats, final_env = trainer.train()
    
    # 6. 出力ファイル名の決定ロジック
    if use_qmix:
        mode_suffix = "qmix" + ("_tar2" if use_tar2 else "")
    elif use_vdn:
        if use_tar2:
            mode_suffix = "vdn_tar2"
        else:
            mode_suffix = "vdn"
    else:
        # VDN/QMIXを使っていない (Independent DQN)
        mode_suffix = "dqn"

    plot_filename = f"learning_curves_{mode_suffix}.png"
    gif_filename = f"restaurant_{mode_suffix}.gif"

    # 7. 結果の可視化と保存
    print(f"\nPlotting learning curves to {plot_filename}...")
    plot_learning_curves(episode_rewards, avg_rewards, served_stats, filename=plot_filename)
    
    print(f"\nGenerating animation to {gif_filename}...")
    # 新しい create_restaurant_gif (ParallelEnv対応) を呼び出します
    create_restaurant_gif(final_env, agents, filename=gif_filename)
    
    print("\nAll Done.")

if __name__ == "__main__":
    main(use_vdn=False, use_qmix=True, use_tar2=False)