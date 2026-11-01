\# Autonomous Robot Navigation with Gemini API



ROS2 (TurtleBot3) と Google Gemini API を用いた自律探索ロボットシステムの実行手順書です。



\---



\## File Structure（ファイル構成）


- `README.md`: システム概要および実行手順書
- `llm_hybrid_controller.py`: ハイブリッド制御用Pythonスクリプト
- `run_robot_advanced.py`: メインスクリプト（Gemini API連携、自律探索・移動制御）
- `docker-compose.yml`: Docker起動設定
- `Dockerfile`: コンテナ定義



\---



\## Setup \& Execution（実行方法）



\### 1. APIキーの設定



環境変数にGemini APIキーを設定します。



```bash

export GEMINI\_API\_KEY="あなたのGemini\_APIキー"

```



\### 2. 実行方法



\#### Dockerを用いる場合（推奨）



```bash

docker compose up --build

```



\#### ローカル環境で直接実行する場合



```bash

python3 llm\_mission\_controller.py

```



\---



\## Interactive CLI Guide（端末操作マニュアル）



起動後のターミナルから、日本語の自然言語指示を直接入力できます。



\### 指示の入力例



\- `探索して`

\- `3分間探索して、その後スタートに戻って`

\- `前方に2m進んで`

\- `待って` / `止まって`



\### 専用コマンド



\- `help`: 使い方を表示

\- `status`: 現在の状態（走行距離、探索面積など）を確認

\- `quit`: 安全に終了

