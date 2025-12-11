了解しました。
先ほど提示した **完全モジュール分割リファクタリング版（A構成）** を対象とした、
**プロ仕様の「ソフトウェア設計書（設計ドキュメント）」** を作成します。

以下は、実際の研究プロジェクトや共同開発で使用できるレベルの
**包括的設計書（要件・構成・仕様・シーケンス・データ構造・責務分離）** です。

---

# 📘 **Multi-Agent Restaurant RL System — 設計書（v1.0）**

---

# 1. **概要（Overview）**

本システムは、マルチエージェント強化学習 (MARL) を用いてレストランにおける配膳タスクを学習する環境および学習フレームワークである。
エージェントは料理のピックアップ・配膳・顧客対応を協調して行い、報酬最大化を目指す。

本設計書は、
**リファクタ後のモジュール構造（A案）に基づいた完全設計仕様** を示す。

---

# 2. **目的（Purpose）**

* 大規模・複雑化した環境コードを、拡張性・再利用性の高い構造へ分割する。
* MARL 研究用途として、実験パラメータ、ネットワーク構造、環境構造を独立させ、管理しやすくする。
* 保守性の高い構造を提供し、複数開発者でも扱いやすい形にする。

---

# 3. **全体構成（Whole Architecture）**

```
marl_research_refactored/
├── env/
│   ├── restaurant_env.py   ← 環境本体
│   ├── customers.py         ← 顧客生成・注文管理
│   ├── layout.py            ← ステージの静的レイアウト
│   └── utils_env.py         ← 座標操作・衝突判定
│
├── agents/
│   ├── network.py           ← Dueling DQN（神経ネットワーク）
│   ├── dqn_agent.py         ← エージェント本体
│   └── replay_buffer.py     ← 経験リプレイバッファ
│
├── training/
│   ├── trainer.py           ← 学習ループ（探索・更新・可視化フック）
│   └── curriculum.py        ← カリキュラム学習管理
│
├── visualization/
│   ├── plot.py              ← 学習曲線可視化
│   └── gif_maker.py         ← 環境の GIF 生成
│
├── config.py                ← ハイパーパラメータ・環境設定
└── main.py                  ← エントリーポイント
```

---

# 4. **機能要件（Functional Requirements）**

## **環境機能**

* 2エージェントが 12×12 グリッド内を移動できる
* エージェントは以下の行動を取る：

  * 上下左右への移動（0–3）
  * インタラクト（4）

    * 料理のピックアップ
    * 配膳
    * 顧客注文の完了
* 顧客のランダム生成と注文管理
* ステージ（empty/basic/complex）のレイアウト切替
* 報酬設計：

  * 配膳成功：+1.0
  * 顧客注文完了：+1.0
  * ピックアップ：+0.1
  * 壁衝突：0
* 観測：

  * エージェント座標（正規化）
  * 所持フラグ

## **学習機能**

* Dueling DQN を用いた Q 学習
* ε-greedy による探索
* ターゲットネットワークの更新
* リプレイバッファによる経験再利用
* カリキュラム学習による段階的難易度上昇

## **可視化機能**

* 学習曲線（報酬）プロット
* シミュレーション GIF 生成

---

# 5. **非機能要件（Non-Functional Requirements）**

* **保守性**：機能ごとに明確に分離されたモジュール構成
* **再利用性**：環境・エージェントコードを他研究に転用可能
* **性能**：1エピソードが 0.1 秒以内で処理可能（Python基準）
* **拡張性**：行動空間、状態空間の拡張が容易

---

# 6. **モジュール設計（Module Design）**

---

# 6.1 env モジュール

## 6.1.1 restaurant_env.py

### **責務**

* 環境のメインロジック
* エージェント移動・報酬・状態更新
* 顧客とのインタラクション

### **クラス図（簡略）**

```
RestaurantEnv
  - positions: Dict[str, Tuple[int,int]]
  - inventory: Dict[str, bool]
  - walls, tables, serving_area
  - customer_manager: CustomerManager
```

### **主なメソッド**

| メソッド          | 説明             |
| ------------- | -------------- |
| reset()       | 環境初期化          |
| step(actions) | 行動処理・顧客生成・報酬計算 |
| _move_agent() | 壁/境界判定つき移動     |
| _interact()   | ピックアップ/配膳/注文完了 |
| _get_obs()    | 観測生成           |

---

## 6.1.2 customers.py

### **責務**

* 顧客生成（spawn）
* 注文生成
* 注文完了判定

---

## 6.1.3 layout.py

### **責務**

* ステージごとのレイアウト（壁・テーブル・提供エリア）定義

---

## 6.1.4 utils_env.py

### **責務**

* 移動処理
* 衝突/範囲チェック

---

# 6.2 agents モジュール

## 6.2.1 network.py

### **責務**

* Dueling DQN のモデル定義

---

## 6.2.2 replay_buffer.py

### **責務**

* 経験メモリリングバッファ

---

## 6.2.3 dqn_agent.py

### **責務**

* 行動選択（ε-greedy）
* Q値更新
* ターゲットネット更新

---

# 6.3 training モジュール

## 6.3.1 trainer.py

### **責務**

* 1エピソードの実行
* 報酬記録
* epsilon 更新
* モデルの学習ステップ管理

---

## 6.3.2 curriculum.py

### **責務**

* 学習ステージ管理
* 難易度上昇ロジック

---

# 6.4 visualization モジュール

* plot.py：Matplotlib で学習曲線
* gif_maker.py：環境遷移を画像化して GIF を生成

---

# 7. **データ構造設計（Data Design）**

---

## 7.1 状態（Observation）

```
obs[a] = [
    agent_x / 12,
    agent_y / 12,
    inventory_flag
]
```

---

## 7.2 行動（Action）

```
0: up
1: right
2: down
3: left
4: interact
```

---

## 7.3 経験（Replay Buffer）

```
(state, action, reward, next_state, done)
```

---

# 8. **シーケンス図（主要な流れ）**

## 8.1 1 ステップの流れ

```
Trainer → Env: step(actions)
Env → CustomerManager: generate_orders()
Env → CustomerManager: spawn()
Env → Agents: reward
Trainer → Agents: learn()
```

---

## 8.2 学習ループ

```
for episode in episodes:
    state = env.reset()
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        buffer.add()
        agent.learn()
```

---

# 9. **拡張計画（Extendability）**

* Grid の大規模化（12→20）
* 部分観測 → CNN への拡張
* マルチエージェント強化学習手法の追加（MAPPO, QMIX）
* 報酬の細分化（衝突ペナルティ、歩行距離ペナルティ）
* レストラン作業工程の追加（調理ステップ）

---

# 10. **まとめ（Conclusion）**

この設計書は、元の巨大コードを：

* 明確な責務分割
* 読みやすいモジュール構成
* 拡張可能なアーキテクチャ

へと再構成したものであり、
今後の MARL 研究における高い拡張性と保守性を実現する。