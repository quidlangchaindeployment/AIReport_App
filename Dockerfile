# ベースイメージとして公式のPython 3.10スリム版を使用
FROM python:3.10-slim

# (★) --- 日本語フォントとグラフ用ライブラリのインストール ---
# gcc (コンパイラ) と 日本語フォント (fonts-ipafont-gothic) を追加
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    fonts-ipafont-gothic \
    && rm -rf /var/lib/apt/lists/*
# (★) --- ここまでが変更点 ---

# 作業ディレクトリを設定
WORKDIR /app

# 依存ライブラリリストをコンテナにコピー
COPY requirements.txt .

# pipをアップグレードし、依存ライブラリをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# spaCyの日本語モデルをダウンロード
RUN python -m spacy download ja_core_news_sm

# アプリケーションコードをコンテナにコピー
COPY app.py .
COPY geography_db.py .

# Streamlitが使用するデフォルトポートを公開
EXPOSE 8501

# コンテナ起動時にStreamlitアプリケーションを実行するコマンド
# (★) 警告が出ていた --server.enableCORS を削除
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless", "true"]