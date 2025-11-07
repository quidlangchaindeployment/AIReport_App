import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
import logging
import time
import spacy
import altair as alt  # L11: Altair (L630から移動)
import networkx as nx
from networkx.algorithms import community 
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import StringIO, BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# L17-L22: 外部ライブラリ ( requirements.txt に必要 )
# 必要なライブラリ (Excel)
try:
    import openpyxl
except ImportError:
    st.error("Excel (openpyxl) がインストールされていません。`pip install openpyxl` してください。")
# 必要なライブラリ (spaCy)
try:
    import ja_core_news_sm
except ImportError:
    st.error("spaCy日本語モデル (ja_core_news_sm) が見つかりません。`python -m spacy download ja_core_news_sm` してください。")

# L27: 定数 (KISS)
# AIモデルを定数化 (KISS)
# ( gemini-1.5-flash-latest や gemini-2.5-flash-lite など)
AI_MODEL_NAME = "gemini-2.5-flash-lite"
# L31: バッチサイズと待機時間も定数化 (KISS)
FILTER_BATCH_SIZE = 50
FILTER_SLEEP_TIME = 6.1 
TAGGING_BATCH_SIZE = 10
TAGGING_SLEEP_TIME = 6.1

# L37: 地名辞書
# geography_db.py が見つからない場合のエラーハンドリング (KISS)
try:
    from geography_db import JAPAN_GEOGRAPHY_DB
except ImportError:
    st.error("地名辞書ファイル (geography_db.py) が見つかりません。")
    JAPAN_GEOGRAPHY_DB = {}  # 実行時エラーを避けるため、空の辞書を定義

# --- L42-L59: ロガー設定 ---
class StreamlitLogHandler(logging.Handler):
    """Streamlitのセッションステートにログメッセージを追加するハンドラ"""
    def __init__(self):
        super().__init__()
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []

    def emit(self, record):
        """ログメッセージをセッションステートに追加"""
        log_entry = self.format(record)
        st.session_state.log_messages.append(log_entry)
        # ログが溜まりすぎないように制御 (例: 最新500件)
        st.session_state.log_messages = st.session_state.log_messages[-500:]

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# --- L63: キャッシュ (KISS / SRP) ---
# LLMとspaCyモデルのロードを @st.cache_resource でキャッシュする
# これにより、手動での session_state 管理 (L1385など) が不要になる

@st.cache_resource  # キャッシュ
def get_llm():
    """LLM (Google Gemini) モデルをロード・キャッシュする"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY がありません。")
            # st.error("APIキーがありません") # 関数内でのUI表示は避ける (SRP)
            return None
            
        llm = ChatGoogleGenerativeAI(
            model=AI_MODEL_NAME, 
            temperature=0.0,
            convert_system_message_to_human=True,
            api_key=api_key
        )
        logger.info(f"LLM Model ({AI_MODEL_NAME}) loaded successfully.")
        return llm
    except Exception as e:
        logger.error(f"LLMの初期化に失敗: {e}", exc_info=True)
        return None

@st.cache_resource  # キャッシュ
def load_spacy_model():
    """spaCyの日本語モデル(ja_core_news_sm)をロード・キャッシュする"""
    try:
        logger.info("Loading spaCy model (ja_core_news_sm)...")
        nlp = spacy.load("ja_core_news_sm")
        logger.info("spaCy model loaded successfully.")
        return nlp
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}", exc_info=True)
        # st.error は main / render 関数で行う (SRP)
        return None

# --- L106-L138: ファイル読み込みヘルパー (read_file) ---
# (既存の L106-L138 をそのままここに貼り付け)
def read_file(file):
    """アップロードされたファイル(Excel/CSV)をPandas DataFrameとして読み込む"""
    file_name = file.name
    logger.info(f"ファイル読み込み開始: {file_name}")
    try:
        if file_name.endswith('.csv'):
            # 文字コードを自動判別 (KISS)
            try:
                # 最初にUTF-8-SIG (BOM付き) を試す
                content = file.getvalue().decode('utf-8-sig')
                df = pd.read_csv(StringIO(content))
            except UnicodeDecodeError:
                # Shift_JIS (CP932) で再試行
                logger.warning(f"UTF-8-SIGデコード失敗。CP932で再試行: {file_name}")
                content = file.getvalue().decode('cp932')
                df = pd.read_csv(StringIO(content))
        elif file_name.endswith(('.xlsx', '.xls')):
            # BytesIO を使用 (KISS)
            df = pd.read_excel(BytesIO(file.getvalue()), engine='openpyxl')
        else:
            logger.warning(f"サポート外のファイル形式: {file_name}")
            return None, f"サポート外のファイル形式: {file_name}"
        logger.info(f"ファイル読み込み成功: {file_name}")
        return df, None
    except Exception as e:
        logger.error(f"ファイル読み込みエラー ({file_name}): {e}", exc_info=True)
        st.error(f"ファイル「{file_name}」の読み込み中にエラー: {e}")
        return None, f"読み込みエラー: {e}"

# --- L140: AI関数 (キャッシュ利用版) ---

def get_dynamic_categories(analysis_prompt):  # llm 引数を削除 (SRP)
    """
    ユーザーの分析指針に基づき、AIが動的なカテゴリをJSON形式で生成する。
    """
    llm = get_llm()  # キャッシュされたLLMを直接呼び出し
    if llm is None:
        logger.error("get_dynamic_categories: LLM is not available.")
        st.error("AIモデルが利用できません。サイドバーでAPIキーを設定してください。")
        return None  #
        
    logger.info("動的カテゴリ生成AIを呼び出し...")
    prompt = PromptTemplate.from_template(
        """
        あなたはデータ分析のスキーマ設計者です。「分析指針」を読み、テキストから抽出するべき「トピックのカテゴリ」を考案してください。「市区町村」は必須カテゴリとして自動で追加されるため、それ以外のカテゴリを定義してください。
        # 指示: 1.「分析指針」のトピックをカテゴリ化 2.各カテゴリの説明記述 3.厳格なJSON辞書出力 4.地名カテゴリ禁止 5.該当なければ空JSON
        # 分析指針:{user_prompt}
        # 回答 (JSON辞書形式):
        """
    )
    chain = prompt | llm | StrOutputParser()
    try:
        response_str = chain.invoke({"user_prompt": analysis_prompt})
        logger.debug(f"AIカテゴリ定義(生): {response_str}")
        # ( ... 既存の L161-L176 のパースロジック ... )
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not match:
            logger.warning("AIがJSON形式で応答しませんでした。")
            return None
        json_str = match.group(0).replace("'", '"')
        try:
            categories = json.loads(json_str)
            return categories
        except json.JSONDecodeError as json_e:
            logger.error(f"AI応答のJSONパース失敗: {json_e} - Raw: {json_str}")
            return None
    except Exception as e:
        logger.error(f"AIカテゴリ生成中にエラー: {e}", exc_info=True)
        st.error(f"AIカテゴリ生成中にエラーが発生しました: {e}")
        return None

def filter_relevant_data_by_ai(df_batch, analysis_prompt):  # llm 引数を削除 (SRP)
    """
    AIを使い、分析指針と無関係な行をフィルタリングする (relevant: true/false)。
    """
    llm = get_llm()  # キャッシュされたLLMを直接呼び出し
    if llm is None:
        logger.error("filter_relevant_data_by_ai: LLM is not available.")
        st.error("AIモデルが利用できません。APIキーを確認してください。")
        return pd.DataFrame()  # 空のDF (フィルタリング失敗)

    logger.debug(f"{len(df_batch)}件 AI関連性フィルタリング開始...")
    
    # ( ... 既存の L209-L248 のロジック (input_texts_jsonl, prompt, chain.invoke, パース処理) ... )
    input_texts_jsonl = df_batch.apply(lambda row: json.dumps({"id": row['id'], "text": str(row['ANALYSIS_TEXT_COLUMN'])[:500]}, ensure_ascii=False), axis=1).tolist()
    prompt = PromptTemplate.from_template(
        """
        あなたはデータ分析のキュレーターです。「分析指針」に基づき、「テキストデータ(JSONL)」の各行が分析対象として【関連しているか (relevant: true)】、【無関係か (relevant: false)】を判定してください。
        # 分析指針 (Analysis Scope):
        {analysis_prompt}
        # 指示:
        1. 「分析指針」と【強く関連】する投稿のみを `true` とする。
        2. 単なる宣伝（例: "セール開催中！"）、挨拶のみ（例: "あけましておめでとう"）、指針と無関係な地域の言及（例: 指針が「広島」なのに「北海道」の話のみ）は `false` とする。
        3. 出力は【JSONL形式のみ】（id と relevant (boolean) を含む辞書）。
        # テキストデータ (JSONL):
        {text_data_jsonl}
        # 回答 (JSONL形式のみ):
        """
    )
    chain = prompt | llm | StrOutputParser()
    try:
        invoke_params = {
            "analysis_prompt": analysis_prompt,
            "text_data_jsonl": "\n".join(input_texts_jsonl)
        }
        logger.debug(f"AI Filtering - Invoking LLM...")
        response_str = chain.invoke(invoke_params)
        logger.debug(f"AI Filtering - Raw response received.")
        results = []
        match = re.search(r'```(?:jsonl|json)?\s*([\s\S]*?)\s*```', response_str, re.DOTALL)
        jsonl_content = match.group(1).strip() if match else response_str.strip()
        for line in jsonl_content.strip().split('\n'):
            cleaned_line = line.strip()
            if not cleaned_line: continue
            try:
                data = json.loads(cleaned_line)
                if isinstance(data.get("relevant"), bool):
                    results.append({"id": data.get("id"), "relevant": data.get("relevant")})
                else:
                    results.append({"id": data.get("id"), "relevant": str(data.get("relevant")).lower() == 'true'})
            except json.JSONDecodeError as json_e:
                logger.warning(f"AIフィルタリング回答パース失敗: {cleaned_line} - Error: {json_e}")
                id_match = re.search(r'"id":\s*(\d+)', cleaned_line)
                if id_match:
                    results.append({"id": int(id_match.group(1)), "relevant": True})
        return pd.DataFrame(results) if results else pd.DataFrame(columns=['id', 'relevant'])
    except Exception as e:
        logger.error(f"AIフィルタリングバッチ処理中エラー: {e}", exc_info=True)
        st.error(f"AIフィルタリング処理エラー: {e}")
        return df_batch[['id']].copy().assign(relevant=True)

def perform_ai_tagging(df_batch, categories_to_tag, analysis_prompt=""):  # llm 引数を削除 (SRP)
    """テキストのバッチを受け取り、AIが【指定されたカテゴリ定義】に基づいて直接タグ付けを行う"""
    llm = get_llm()  # キャッシュされたLLMを直接呼び出し
    if llm is None:
        logger.error("perform_ai_tagging: LLM is not available.")
        st.error("AIモデルが利用できません。APIキーを確認してください。")
        return pd.DataFrame()  # 空のDF (タグ付け失敗)

    logger.debug(f"AI Tagging - Received categories: {json.dumps(categories_to_tag, ensure_ascii=False)}")
    logger.info(f"{len(df_batch)}件 AIタグ付け開始 (カテゴリ: {list(categories_to_tag.keys())})")
    
    # ( ... 既存の L258-L321 のロジック (geography_context, input_texts_jsonl, prompt, chain.invoke, パース処理) ... )
    relevant_geo_db = {}
    if JAPAN_GEOGRAPHY_DB:
        prompt_lower = analysis_prompt.lower()
        keys_found = [
            key for key in JAPAN_GEOGRAPHY_DB.keys() 
            if any(hint in key for hint in [
                "広島", "福岡", "大阪", "東京", "北海道", "愛知", "宮城", "札幌", "横浜", "名古屋", "京都", "神戸", "仙台"
            ]) and any(hint in prompt_lower for hint in [
                "広島", "福岡", "大阪", "東京", "北海道", "愛知", "宮城", "札幌", "横浜", "名古屋", "京都", "神戸", "仙台"
            ])
        ]
        if "広島" in prompt_lower: keys_found.extend(["広島県", "広島市"])
        if "東京" in prompt_lower: keys_found.extend(["東京都", "東京23区"])
        if "大阪" in prompt_lower: keys_found.extend(["大阪府", "大阪市"])
        for key in set(keys_found):
            if key in JAPAN_GEOGRAPHY_DB:
                relevant_geo_db[key] = JAPAN_GEOGRAPHY_DB[key]
        if not relevant_geo_db:
            logger.warning("地名辞書の絞り込みヒントなし。主要都市のみ渡します。")
            default_keys = ["東京都", "東京23区", "大阪府", "大阪市", "広島県", "広島市"]
            for key in default_keys:
                 if key in JAPAN_GEOGRAPHY_DB:
                     relevant_geo_db[key] = JAPAN_GEOGRAPHY_DB[key]
        geo_context_str = json.dumps(relevant_geo_db, ensure_ascii=False, indent=2)
        if len(geo_context_str) > 5000:
            logger.warning(f"地名辞書が大きすぎ ({len(geo_context_str)}B)。キーのみに縮小。")
            geo_context_str = json.dumps(list(relevant_geo_db.keys()), ensure_ascii=False)
    else:
        geo_context_str = "{}"
    logger.info(f"AIに渡す地名辞書(絞込済): {list(relevant_geo_db.keys())}")
    
    input_texts_jsonl = df_batch.apply(lambda row: json.dumps({"id": row['id'], "text": str(row['ANALYSIS_TEXT_COLUMN'])[:500]}, ensure_ascii=False), axis=1).tolist()
    logger.debug(f"AI Tagging - Input sample: {input_texts_jsonl[0] if input_texts_jsonl else 'None'}")
    
    prompt = PromptTemplate.from_template(
        """
        あなたは高精度データ分析アシスタントです。「カテゴリ定義」「地名辞書」「分析指針」に基づき、キーワードを抽出します。
        # 分析指針 (Analysis Scope): {analysis_prompt}
        # 地名辞書 (JAPAN_GEOGRAPHY_DB): {geo_context}
        # カテゴリ定義 (categories): {categories}
        # テキストデータ (JSONL): {text_data_jsonl}
        # 指示:
        1. 「テキストデータ(JSONL)」の各行を処理する。
        2. 「カテゴリ定義」のキー名を【厳格に】使用し、全カテゴリを抽出する。
        3. 【"市区町村キーワード" 以外のカテゴリ】:
           - 値は必ず【リスト形式】で出力（該当なければ空リスト []）。
        4. 【"市区町村キーワード" (最重要・単一回答)】:
           - 値は【単一の文字列】で出力する (該当なければ空文字列 "")。リスト形式は【厳禁】。
           - 抽出ルール:
             a. 「地名辞書」の【値】(例: "呉市", "廿日市市", "中区") または【キー】(例: "広島市") に一致する、最も文脈に関連性の高いものを【1つだけ】選ぶ。
             b. (例: "広島市" と "中区" が両方言及されていれば、より詳細な "中区" を優先する)
             c. "宮島" のようなランドマーク名は、それが属する「地名辞書」の市区町村名 (例: "廿日市市") に【必ず変換】して回答する。
             d. "広島" のような曖昧な表現は、文脈から (a) のいずれかに特定できる場合のみ (例: "広島市") 抽出し、特定できなければ【空文字列 ""】とする。
             e. 都道府県名 (例: "広島県")、および「観光地」のような地名以外の単語は【絶対に抽出しない】。
             f. 「分析指針」と無関係な地域の地名（例: 指針が「広島」なのにテキストが「滋賀県」）は【抽出しない】。
        5. ハルシネーション（情報の捏造）禁止。
        6. 出力は【JSONL形式のみ】（id と categories を含む辞書）。
        # 回答 (JSONL形式のみ):
        """
    )
    chain = prompt | llm | StrOutputParser()
    try:
        invoke_params = {
            "categories": json.dumps(categories_to_tag, ensure_ascii=False), 
            "geo_context": geo_context_str,
            "text_data_jsonl": "\n".join(input_texts_jsonl),
            "analysis_prompt": analysis_prompt
        }
        logger.debug(f"AI Tagging - Invoking LLM...")
        logger.info(f"Attempting AI call for ID: {df_batch.iloc[0]['id']}...")
        response_str = chain.invoke(invoke_params)
        logger.debug(f"AI Tagging - Raw response received.")
        
        results = []
        expected_keys = list(categories_to_tag.keys())
        match = re.search(r'```(?:jsonl|json)?\s*([\s\S]*?)\s*```', response_str, re.DOTALL)
        jsonl_content = match.group(1).strip() if match else response_str.strip()

        for line in jsonl_content.strip().split('\n'):
            cleaned_line = line.strip()
            if not cleaned_line: continue
            try:
                data = json.loads(cleaned_line)
                row_result = {"id": data.get("id")}
                tag_source = data.get('categories', data)
                
                for key in expected_keys:
                    found_key = None
                    for resp_key in tag_source.keys():
                        if resp_key.strip() == key:
                            found_key = resp_key
                            break
                    raw_value = tag_source.get(found_key) if found_key else None
                    
                    if key == "市区町村キーワード":
                        processed_value = ""
                        if isinstance(raw_value, list) and raw_value:
                            processed_value = str(raw_value[0]).strip()
                        elif raw_value is not None and str(raw_value).strip():
                            processed_value = str(raw_value).strip()
                        if processed_value.lower() in ["該当なし", "none", "null", ""]:
                            row_result[key] = "" 
                        else:
                            row_result[key] = processed_value
                    else:
                        processed_values = [] 
                        if isinstance(raw_value, list):
                            processed_values = sorted(list(set(str(val).strip() for val in raw_value if str(val).strip())))
                        elif raw_value is not None and str(raw_value).strip():
                            processed_values = [str(raw_value).strip()]
                        
                        # (★) --- 括弧除去修正: リストをカンマ区切りの文字列に変換 ---
                        row_result[key] = ", ".join(processed_values)
                results.append(row_result)
            except json.JSONDecodeError as json_e:
                logger.warning(f"AIタグ付け回答パース失敗: {cleaned_line} - Error: {json_e}")
                id_match = re.search(r'"id":\s*(\d+)', cleaned_line)
                if id_match:
                    results.append({"id": int(id_match.group(1))})
        return pd.DataFrame(results) if results else pd.DataFrame(columns=['id'] + list(expected_keys))
    except Exception as e:
        logger.error(f"AIタグ付けバッチ処理中エラー: {e}", exc_info=True)
        st.error(f"AIタグ付け処理エラー: {e}")
        return pd.DataFrame() # 失敗時は空のDFを返す

# --- L322-L438: Step B (分析手法提案) ---
# (既存の L322-L438 (suggest_analysis_techniques 関数) をそのままここに貼り付け)
def suggest_analysis_techniques(df):
    """
    フラグ付きデータフレームを分析し、適切な分析手法を優先度順に提案する。
    """
    suggestions = []
    if df is None or df.empty: # 空のDFもチェック
        logger.error("suggest_analysis_techniques に None または空のDataFrame"); return suggestions
    try:
        # データ型の再確認と列の特定 (より確実に)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include='object').columns.tolist() # object型をまず取得
        datetime_cols = []
        possible_dt_cols = [col for col in object_cols] # object列から候補を探す
        # 日付型への変換を試みる (欠損が多い列は除外)
        for col in possible_dt_cols:
             if df[col].isnull().sum() / len(df) > 0.5: continue # 欠損が5割超ならスキップ
             sample = df[col].dropna().head(50)
             if sample.empty: continue
             try:
                 pd.to_datetime(sample, errors='raise')
                 # 変換成功 → 全体を変換して確認
                 temp_dt = pd.to_datetime(df[col], errors='coerce').dropna()
                 # 年月日のいずれかが複数存在するか、特定のキーワードを含むかなどで判断
                 if not temp_dt.empty and (temp_dt.dt.year.nunique() > 1 or temp_dt.dt.month.nunique() > 1 or temp_dt.dt.day.nunique() > 1 or col.lower() in ['date', 'time', 'timestamp', '日付', '日時']):
                     datetime_cols.append(col)
                     logger.info(f"列 '{col}' を日時列として認識しました。")
             except (ValueError, TypeError, OverflowError, pd.errors.ParserError): pass # エラーが出ても無視

        numeric_cols = [col for col in numeric_cols if col != 'id'] # id列除外
        # ANALYSIS_TEXT_COLUMN と日時列を除いたものがカテゴリ列候補
        categorical_cols = [col for col in object_cols if col != 'ANALYSIS_TEXT_COLUMN' and col not in datetime_cols]
        # キーワード列（フラグ列）を特定
        flag_cols = [col for col in categorical_cols if col.endswith('キーワード')]
        other_categorical = [col for col in categorical_cols if not col.endswith('キーワード')]
        logger.info(f"提案分析 - 数値:{numeric_cols}, カテゴリ(フラグ):{flag_cols}, カテゴリ(他):{other_categorical}, 日時:{datetime_cols}")

        # --- 提案リスト (優先度順) ---
        potential_suggestions = []

        # 優先度1: 基本集計 (ほぼ必須)
        if flag_cols:
            potential_suggestions.append({
                "priority": 1, "name": "単純集計（頻度分析）",
                "description": "各キーワード（カテゴリ）がどのくらいの頻度で出現したかトップNを表示し、全体像を把握します。",
                "reason": f"キーワード列({len(flag_cols)}個)あり。まず見るべき基本指標です。",
                "suitable_cols": flag_cols
            })
        if numeric_cols:
             potential_suggestions.append({
                 "priority": 1, "name": "基本統計量",
                 "description": f"数値データ({', '.join(numeric_cols)})の平均、中央値、最大/最小値などを算出し、データの分布を確認します。",
                 "reason": f"数値列({len(numeric_cols)}個)あり。データの基本特性把握に。",
                 "suitable_cols": numeric_cols
             })

        # 優先度2: 関係性の分析 (クロス集計)
        if len(flag_cols) >= 2:
            potential_suggestions.append({
                "priority": 2, "name": "クロス集計（キーワード間）",
                "description": "キーワード間の組み合わせで多く出現するパターンを探ります（例: 特定の市区町村と観光地の組み合わせ）。",
                "reason": f"複数キーワード列({len(flag_cols)}個)あり、関連性の発見に。",
                "suitable_cols": flag_cols
            })
        if flag_cols and other_categorical:
             potential_suggestions.append({
                "priority": 2, "name": "クロス集計（キーワード×属性）",
                "description": f"キーワード({flag_cols[0]}など)と他の属性({', '.join(other_categorical)})の関係性を分析します（例: 年代別によく出る観光地）。",
                "reason": f"キーワード列と他カテゴリ列({len(other_categorical)}個)あり、属性ごとの傾向把握に。",
                "suitable_cols": flag_cols + other_categorical
            })

        # 優先度3: 共起ネットワーク分析 (L438の指示)
        all_categorical_cols = [col for col in df.select_dtypes(include='object').columns if col != 'ANALYSIS_TEXT_COLUMN']
        
        if len(all_categorical_cols) >= 2:
            potential_suggestions.append({
                "priority": 3, "name": "共起ネットワーク分析",
                "description": "CSV内のカテゴリ列（例: 「市区町村」と「年代」）を選び、それらの共起関係をネットワークとして可視化します。",
                "reason": f"分析可能なカテゴリ列が2つ以上({len(all_categorical_cols)}個)見つかりました。属性間の関連性発見に。",
                "suitable_cols": all_categorical_cols # ★ flag_cols から all_categorical_cols に変更
            })

        # 優先度4: グループ比較 (L438の指示)
        if numeric_cols and flag_cols:
            potential_suggestions.append({
                "priority": 4, "name": "カテゴリ別集計（グループ比較）",
                "description": f"キーワードカテゴリ（{flag_cols[0]}など）ごとに数値データ({numeric_cols[0]}など)の平均値や合計値に差があるか比較します。",
                "reason": f"キーワード列と数値列({len(numeric_cols)}個)あり、グループ間の特徴比較に。",
                "suitable_cols": {"numeric": numeric_cols, "grouping": flag_cols}
            })

        # 優先度5: 時系列分析 (L438の指示)
        if datetime_cols and flag_cols:
            potential_suggestions.append({
                "priority": 5, "name": "時系列キーワード分析",
                "description": f"特定のキーワードの出現数が時間（{datetime_cols[0]}など）とともにどう変化したかトレンドを可視化します。",
                "reason": f"キーワード列と日時列({len(datetime_cols)}個)あり、時間変化の把握に。",
                "suitable_cols": {"datetime": datetime_cols, "keywords": flag_cols}
            })
            
        # (★) --- (L405) 'if' ブロックの外側、'try' ブロックの内側 (インデント修正) ---
        if datetime_cols: # (★) 日時列がある場合のみ提案
            potential_suggestions.append({
               "priority": 5, "name": "投稿量分析",
               "description": f"全体の投稿数が時間（{datetime_cols[0]}など）とともにどう変化したかトレンドを可視化します。",
               "reason": f"日時列({len(datetime_cols)}個)あり、時間変化の把握に。",
               "suitable_cols": {"datetime": datetime_cols, "keywords": flag_cols} # (★) 同じ suitable_cols を渡す
            })
        # 優先度6: テキストマイニング (L438の指示)
        potential_suggestions.append({
            "priority": 6, "name": "テキストマイニング（頻出単語など）",
            "description": "原文テキストから頻出する単語を抽出し、どのような言葉が多く使われているか全体像を把握します。",
            "reason": "原文テキストがあり、タグ付け以外の観点からのインサイト発見に。",
            "suitable_cols": ['ANALYSIS_TEXT_COLUMN']
        })

        # 優先度7: 多変量解析 (L438の指示)
        if len(numeric_cols) >= 3:
             potential_suggestions.append({
                 "priority": 7, "name": "主成分分析 (PCA) / 因子分析",
                 "description": f"複数の数値データ({', '.join(numeric_cols)})間の相関関係から、背後にある共通の要因（主成分/因子）を探ります。",
                 "reason": f"複数数値列({len(numeric_cols)}個)があり、変数間の複雑な関係性の縮約や解釈に。",
                 "suitable_cols": numeric_cols
             })

        # 優先度でソートし、上位8件程度を返す (L438の指示)
        suggestions = sorted(potential_suggestions, key=lambda x: x['priority'])
        logger.info(f"提案手法(ソート後): {[s['name'] for s in suggestions]}")
        return suggestions[:8] # 上限を 8 に変更

    except Exception as e:
        logger.error(f"分析手法提案中にエラー: {e}", exc_info=True); st.warning(f"分析手法提案中にエラー: {e}")
    return suggestions

def get_suggestions_from_prompt(user_prompt, df, existing_suggestions):  # llm 引数を削除 (SRP)
    """
    ユーザーの自由記述プロンプトとデータ構造に基づき、AIが追加の分析手法を提案する。
    """
    logger.info("AIプロンプトベースの分析提案を開始...")
    llm = get_llm()  # キャッシュされたLLMを直接呼び出し
    if llm is None:
        logger.error("get_suggestions_from_prompt: LLM is not available.")
        return []
    
    try:
        # ( ... 既存の L439-L498 のロジック (column_info_str, prompt, chain.invoke) ... )
        col_info = []
        for col in df.columns:
            col_info.append(f"- {col} (型: {df[col].dtype})")
        column_info_str = "\n".join(col_info)
        existing_names = [s['name'] for s in existing_suggestions]
        prompt = PromptTemplate.from_template(
            """
            あなたはデータ分析のスキーマ設計者です。ユーザーの「分析指示」を解釈し、それをJSONリスト形式の「分析手法」に変換してください。
            # データ構造 (利用可能な列名):
            {column_info}
            # ユーザーの分析指示 (このテキストを解釈対象とします):
            {user_prompt}
            # 指示:
            1. 「ユーザーの分析指示」に含まれる分析項目を【1つずつ】解釈し、それぞれを「分析手法」として定義する。 (例: 「投稿数分析」は「投稿数分析」という名前の手法にする)
            2. 各提案に `priority` (優先度: 6固定), `name` (手法名), `description` (手法の簡潔な説明), `reason` (提案理由: 「ユーザー指示に基づく」と記述) を含むJSONリスト形式で回答する。(★)
            3. 指示が空、または解釈不能な場合は、空リスト [] を返してください。
            """
        )
        chain = prompt | llm | StrOutputParser()
        response_str = chain.invoke({
            "column_info": column_info_str,
            "user_prompt": user_prompt
        })
        
        # ( ... 既存の L502-L534 のロジック (パース処理) ... )
        logger.info(f"AI追加提案(生): {response_str}")
        match = re.search(r'\[.*\]', response_str, re.DOTALL)
        if not match:
            logger.warning("AIがJSONリスト形式で応答しませんでした。")
            return []
        json_str = match.group(0)
        ai_suggestions = json.loads(json_str)
        for suggestion in ai_suggestions:
            suggestion['priority'] = 6 # ユーザー指示は優先度を低く設定
        logger.info(f"AI追加提案(パース済): {len(ai_suggestions)}件")
        return ai_suggestions
        
    except Exception as e:
        logger.error(f"AI追加提案の生成中にエラー: {e}", exc_info=True)
        st.warning(f"AI追加提案の生成中にエラーが発生しました: {e}")
        return []

# --- L468: Step B (提案表示UI) ---
# --- L468: Step B (提案表示UI) ---
def display_suggestions(suggestions, df):
    """
    提案された分析手法を表示し、ユーザーが選択できるようにする (★ チェックボックス版)
    """
    if not suggestions:
        st.info("提案可能な分析手法がありません。")
        return

    st.subheader("提案された分析手法:")
    st.markdown("---")
    
    # L497 のロジック (デフォルト5件選択)
    default_selection_names = [s['name'] for s in suggestions[:min(len(suggestions), 5)]] 
    
    st.markdown("実行したい分析手法を選択（複数可）:")
    selected_technique_names = []
    
    for suggestion in suggestions:
        name = suggestion['name']
        is_default_checked = name in default_selection_names
        is_checked = st.checkbox(
            name, 
            value=is_default_checked, 
            key=f"cb_{name}"
        )
        if is_checked:
            selected_technique_names.append(name)
    
    # L515-L519: 不要なコメントアウトを削除 (KISS)
    
    if selected_technique_names:
        st.markdown("---")
        st.subheader("選択された手法の詳細:")
        selected_suggestions = [s for s in suggestions if s['name'] in selected_technique_names]
        
        # (★) --- UI BUG FIX (L513) ---
        # session_state.suggestions_B に重複がある場合に備え、描画時に重複を除外する
        seen_names = set()
        unique_selected_suggestions = []
        for s in selected_suggestions:
            if s['name'] not in seen_names:
                unique_selected_suggestions.append(s)
                seen_names.add(s['name'])
        # --- FIX END ---
        
        for suggestion in unique_selected_suggestions: # (★) リストを変数に変更
            with st.expander(f"{suggestion['name']} (優先度: {suggestion['priority']})"):
                st.markdown(f"**<説明>**\n{suggestion['description']}")
                st.markdown(f"**<提案理由>**\n{suggestion['reason']}")
    
    st.markdown("---")

    # (★) --- (Problem 2) ボタンのインデント修正 (L525) ---
    # 以下のボタンは for ループの外に、1回だけ定義する
    if st.button("選択した手法で分析を実行 (Step Cへ)", key="execute_button_C_v2", disabled=not selected_technique_names, type="primary"):
         if selected_technique_names:
             st.session_state.chosen_analysis_list = selected_technique_names
             st.session_state.current_step = 'C'
             st.rerun()
         else:
             st.error("分析を実行するには、少なくとも1つの手法を選択してください。")

# --- L537: Step C (AIサマリープロンプト) ---
# (既存の L537-L578 (generate_ai_summary_prompt 関数) をそのままここに貼り付け)
def generate_ai_summary_prompt(results_dict, df):
    """
    Step C-1 で得られた分析結果(DataFrame)をAI用のプロンプトに変換する。
    """
    logger.info("AIサマリー用プロンプトの生成開始...")
    if not results_dict:
        logger.warning("AIサマリーの元になる分析結果がありません。")
        return "エラー: AIサマリーの元になる分析結果がありません。Step C-1を先に実行してください。"
    
    context_str = f"## 分析対象データの概要\n"
    context_str += f"- 総行数: {len(df)}\n"
    context_str += f"- 列リスト: {', '.join(df.columns.tolist())}\n\n"
    context_str += "## 個別分析の結果サマリー\n"
    context_str += "（注：トークン数節約のため、各分析結果は最大5件のみ抜粋しています）\n\n"
    
    for name, data in results_dict.items():
        context_str += f"### {name}\n"
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if data.empty:
                context_str += "(データなし)\n\n"
            else:
                if len(data) > 5:
                    context_str += f"上位5件:\n{data.head(5).to_string()}\n\n"
                else:
                    context_str += f"全件:\n{data.to_string()}\n\n"
        else:
            context_str += f"{str(data)}\n\n"
    
    final_prompt = f"""
あなたは、データ分析結果をクライアント（日本のビジネスマン）向けのパワーポイント資料にまとめる、優秀なコンサルタントです。
以下の「分析コンテキスト」には、各分析手法（例：単純集計、共起ネットワーク）から得られた生データ（上位5件の抜粋）が含まれています。

# 指示 (最重要):
1.  **生データをコピーしない:** 「分析コンテキスト」内の表形式のテキスト（.to_string() の結果）を、あなたの回答に【絶対に】含めないでください。
2.  **結果の「構造化」:** 以下の「# 回答フォーマット」に【厳格に】従ってください。
3.  **結論ファースト:** まず「総括（Key Takeaways）」として、ビジネス上の最も重要な発見や提案を2～3点の箇条書きで記述してください。
4.  **個別分析の要約:** 次に、「個別分析の要点」として、`results_dict` 内の各分析手法から得られた【最も重要なインサイト1点】だけを抜き出し、簡潔な1文の箇条書きにしてください。

---
[分析コンテキスト]
{context_str}
---
[あなたの回答]

# 回答フォーマット (この構造を厳守)

## 分析サマリーレポート

### 総括 (Key Takeaways)
* [ここには、全分析結果から導かれる最も重要な「ビジネス上の結論」または「次のアクション提案」を2～3点で記述]
* [例：〇〇（地名）では「食」への関心が最も高く、特に「〇〇（単語）」との組み合わせが鍵となる。]

---

### 個別分析の要点
* **[単純集計]**: [単純集計の結果から読み取れる最も重要な事実を1行で記述。 例：「投稿数では『広島市』が突出して1位であった。」]
* **[共起ネットワーク]**: [共起ネットワーク分析から読み取れる最も重要な「単語の組み合わせ」や「クラスタの特徴」を1行で記述。 例：「『原爆ドーム』クラスタは『平和』や『歴史』と強く結びついていた。」]
* **[時系列分析]**: [時系列分析の結果から読み取れる最も重要な「時期的なトレンド」を1行で記述。]
* **[ハッシュタグ分析]**: [ハッシュタグ分析の結果から読み取れる最も重要な「トレンド」を1行で記述。]
* **[その他、実行された分析]**: [同様に、各分析の最重要ポイントを1行で記述]
"""
    logger.info("AIサマリー用プロンプト生成完了。")
    return final_prompt

# --- L580: Step C (可視化ヘルパー) ---
def run_simple_count(df, flag_cols):
    """単純集計（頻度分析）を実行し、Streamlitで可視化する"""
    if not flag_cols:
        st.warning("集計対象のキーワード列（suitable_cols）が見つかりません。")
        return None #
    
    col_to_analyze = st.selectbox(
        "集計するキーワード列を選択:", 
        flag_cols, 
        key=f"sc_select_{flag_cols[0]}"
    )
    
    if not col_to_analyze or col_to_analyze not in df.columns:
        st.error(f"列 '{col_to_analyze}' がデータに存在しません。")
        return None #
    try:
        s = df[col_to_analyze].astype(str).str.split(', ').explode()
        s = s[s.str.strip() != ''] # 空白を除去
        s = s.str.strip() # 前後の空白を除去
        
        if s.empty:
            st.info("集計対象のキーワードがありませんでした。")
            return None #
            
        counts = s.value_counts().head(20) # 上位20件
        st.bar_chart(counts)
        with st.expander("詳細データ（上位20件）"):
            st.dataframe(counts)
        return counts # 
            
    except Exception as e:
        st.error(f"単純集計の処理中にエラー: {e}")
        logger.error(f"run_simple_count error: {e}", exc_info=True)
    return None #

def run_basic_stats(df, numeric_cols):
    """基本統計量を実行し、Streamlitで表示する"""
    if not numeric_cols:
        st.warning("集計対象の数値列（suitable_cols）が見つかりません。")
        return None #
    
    existing_cols = [col for col in numeric_cols if col in df.columns]
    if not existing_cols:
        st.error("指定された数値列がデータに存在しません。")
        return None #
        
    stats_df = df[existing_cols].describe()
    st.dataframe(stats_df)
    
    with st.expander("各項目の説明"):
        st.markdown("""
        - **count**: 件数（データの個数）
        - **mean**: 平均値
        - **std**: 標準偏差（データのばらつき度合い）
        - **min**: 最小値
        - **25% (Q1)**: 第1四分位数（データを小さい順に並べたとき、下から25%地点の値）
        - **50% (Q2)**: 中央値（median）（50%地点の値）
        - **75% (Q3)**: 第3四分位数（75%地点の値）
        - **max**: 最大値
        """)
    
    return stats_df #

def run_crosstab(df, suitable_cols):
    """クロス集計を実行し、Streamlitで表示する"""
    if not suitable_cols or len(suitable_cols) < 2:
        st.warning("クロス集計には2つ以上の列が必要です。")
        return None #

    existing_cols = [col for col in suitable_cols if col in df.columns]
    if len(existing_cols) < 2:
        st.error(f"データ内に存在する分析対象列が2つ未満です: {existing_cols}")
        return None #

    st.info(f"分析可能な列: {', '.join(existing_cols)}")
    
    key_base = suitable_cols[0]
    col1 = st.selectbox("行 (Index) に設定する列:", existing_cols, key=f"ct_idx_{key_base}")
    
    options_col2 = [c for c in existing_cols if c != col1]
    if not options_col2:
        st.error("2つ目の列を選択できません。")
        return None #
        
    col2 = st.selectbox("列 (Column) に設定する列:", options_col2, key=f"ct_col_{key_base}")

    if not col1 or not col2:
        return None #

    try:
        crosstab_df = pd.crosstab(df[col1].astype(str), df[col2].astype(str))
        
        if crosstab_df.empty:
            st.info("クロス集計の結果、データがありませんでした。")
            return None
        
        st.dataframe(crosstab_df)
        
        if st.checkbox("ヒートマップで表示", key=f"ct_heatmap_{key_base}"):    
            return crosstab_df # 
    except Exception as e:
        st.error(f"クロス集計の処理中にエラー: {e}")
        logger.error(f"run_crosstab error: {e}", exc_info=True)
    return None #

def run_timeseries(df, suitable_cols_dict, name=""): # (★) name 引数を追加
    """時系列分析を実行し、Streamlitで可視化する"""
    if not isinstance(suitable_cols_dict, dict) or 'datetime' not in suitable_cols_dict or 'keywords' not in suitable_cols_dict:
        st.warning("時系列分析のための列情報（datetime, keywords）が不十分です。")
        return None #
        
    dt_cols = [col for col in suitable_cols_dict['datetime'] if col in df.columns]
    kw_cols = [col for col in suitable_cols_dict['keywords'] if col in df.columns]

    if not dt_cols: st.error("日時列が見つかりません。"); return None #

    # (★) --- (Problem 2) 重複キーエラー修正 ---
    key_base = dt_cols[0] + "_" + name # (★) name を使って key_base を一意にする
    
    dt_col = st.selectbox("使用する日時列:", dt_cols, key=f"ts_dt_{key_base}")
    
    # (★) --- キーワード選択を任意（オプショナル）に変更 ---
    kw_options = ["(全体の投稿量)"] + kw_cols
    kw_col = st.selectbox("集計対象:", kw_options, key=f"ts_kw_{key_base}", help="「(全体の投稿量)」を選ぶと、キーワードに関わらず全ての投稿数を集計します。")

    if not dt_col:
        return None #

    try:
        # (★) --- (Problem 1) KeyError 修正: インデントを修正 ---
        if kw_col == "(全体の投稿量)":
            df_copy = df[[dt_col]].copy()
        else:
            df_copy = df[[dt_col, kw_col]].copy()
            # (★) 以下のクリーニング処理は 'else' ブロックの内側
            df_copy[kw_col] = df_copy[kw_col].astype(str).str.replace(r"[\[\]'\"]", "", regex=True)
            df_copy = df_copy[df_copy[kw_col].str.strip() != ''] 
            if df_copy.empty: st.info(f"「{kw_col}」に有効なキーワードがありませんでした。"); return None #
        
        df_copy[dt_col] = pd.to_datetime(df_copy[dt_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[dt_col])
        if df_copy.empty: st.info("有効な日時データがありません。"); return None #

        time_df = df_copy.set_index(dt_col).resample('D').size().rename("投稿数")
        
        if time_df.empty: st.info("時系列集計の結果、データがありませんでした。"); return None #
        
        time_df.index.name = "日時"
        
        st.line_chart(time_df)
        with st.expander("詳細データ"):
            st.dataframe(time_df)
        
        return time_df
            
    except Exception as e:
        st.error(f"時系列分析の処理中にエラー: {e}")
        logger.error(f"run_timeseries error: {e}", exc_info=True)
    return None #

def run_text_mining(df, text_col='ANALYSIS_TEXT_COLUMN'):
    """
    spaCyを使用してテキストマイニング（頻出単語分析）を実行し、可視化する。
    APIは使用しない。
    """
    if text_col not in df.columns or df[text_col].empty:
        st.warning(f"分析対象のテキスト列 '{text_col}' がないか、空です。")
        return None #

    nlp = load_spacy_model() #キャッシュされたモデルを直接呼び出し
    if nlp is None:
        st.error("spaCy日本語モデルのロードに失敗しました。")
        return None
            
    st.info("テキストマイニング処理中（データ量によって時間がかかる場合があります）...")

    try:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            st.warning("分析対象のテキストがありません。")
            return None #
            
        words = []
        target_pos = {'NOUN', 'PROPN', 'ADJ'}
        stop_words = {
        'の', 'に', 'は', 'を', 'が', 'で', 'て', 'です', 'ます', 'こと', 'もの', 'それ', 'あれ',
        'これ', 'ため', 'いる', 'する', 'ある', 'ない', 'いう', 'よう', 'そう', 'など', 'さん',
        '的', '人', '自分', '私', '僕', '何', 'その', 'この', 'あの',
        '思う', '行く', '見る', '来る', '感じ', '良い', '良い', 'なる', 'てる', 'られる', 'れる',
        '場所', '感じ', '時間', '今回', '色々', '中', 'ところ', 'たち', '人達', '多い', 'スポット'
    }
        for doc in nlp.pipe(texts, disable=["parser", "ner"]):
            for token in doc:
                if (token.pos_ in target_pos) and (not token.is_stop) and (token.lemma_ not in stop_words) and (len(token.lemma_) > 1):
                    words.append(token.lemma_)

        if not words:
            st.warning("抽出可能な有効な単語が見つかりませんでした。")
            return None #

        word_counts = pd.Series(words).value_counts().head(30) # 上位30件

        st.subheader("頻出単語 Top 30")
        st.bar_chart(word_counts)
        with st.expander("詳細データ（Top 30）"):
            st.dataframe(word_counts.reset_index(name="出現回数").rename(columns={"index": "単語"}))

        # L727: 重複した dataframe 呼び出しを削除 (KISS)
        
        return word_counts # 
    except Exception as e:
        st.error(f"テキストマイニング処理中にエラー: {e}")
        logger.error(f"run_text_mining error: {e}", exc_info=True)
    return None #

def run_cooccurrence_network(df, suitable_cols):
    """(★変更) フィルタリングされた自由記述列内の「単語同士」の共起ネットワークを可視化する"""
    
    all_cols = df.columns.tolist()
    
    if 'ANALYSIS_TEXT_COLUMN' not in all_cols:
        st.error("分析対象の自由記述列（ANALYSIS_TEXT_COLUMN）が見つかりません。")
        return None

    # (★) --- (Problem 2) カラーパレットの定義 ---
    # 凡例と色をマッピングするための固定カラーリスト
    COLOR_PALETTE = [
        "#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#33FFF6",
        "#F3FF33", "#FF8C33", "#8C33FF", "#33FF8C", "#FF338C"
    ]
    
    # (★) UIを「フィルタ列」「フィルタキーワード」の選択に変更
    st.info("分析したい「キーワード」で投稿を絞り込み、その投稿内容の共起ネットワークを作成します。")
    
    flag_col_options = [col for col in suitable_cols if col in df.columns] 
    if not flag_col_options:
         st.warning("分析可能なフィルタ列（カテゴリ列）が見つかりません。")
         return None

    # 1. フィルタ列の選択 (デフォルト: 市区町村キーワード)
    default_flag_col_index = 0
    if "市区町村キーワード" in flag_col_options:
        default_flag_col_index = flag_col_options.index("市区町村キーワード")
    flag_col = st.selectbox(
        "1. 絞り込みに使用する列:",
        flag_col_options,
        index=default_flag_col_index,
        key="cn_filter_col",
        help="ここで選んだ列のキーワード（例：市区町村キーワード）で、分析対象の投稿を絞り込みます。"
    )
    
    # 2. フィルタキーワードの入力 (例: 広島市)
    try:
        # (★) --- (Problem 1) nan 除外ロジック (BUG FIX) ---
        s = df[flag_col].dropna().astype(str).str.split(',').explode().str.strip()
        s = s[~s.isin(['', 'nan', 'Nan', 'NaN'])] # (★) nan を除外
        
        keyword_counts = s.value_counts()
        
        # 選択肢 (多すぎると重いためTop50に制限)
        options = keyword_counts.index.tolist()[:50] 
        # デフォルト (Top10)
        default_options = keyword_counts.index.tolist()[:10] 
    except Exception as e:
        st.error(f"キーワードの頻度計算中にエラー: {e}")
        options = []
        default_options = []

    # セッションステートキー (列名ごとに選択状態を保持)
    session_key = f'cn_selected_keywords_{flag_col}'
    
    # セッションステートが未初期化の場合、デフォルト値(Top10)を設定
    if session_key not in st.session_state:
        st.session_state[session_key] = default_options

    st.markdown(f"**2. 絞り込むキーワード（「{flag_col}」列）:**")
    
    # --- 全選択 / 全解除ボタン ---
    def select_all_keywords():
        st.session_state[session_key] = options
    def deselect_all_keywords():
        st.session_state[session_key] = []

    btn_cols = st.columns([1, 1, 3])
    with btn_cols[0]:
        st.button("全選択", on_click=select_all_keywords, key=f"btn_all_{flag_col}", use_container_width=True)
    with btn_cols[1]:
        st.button("全解除", on_click=deselect_all_keywords, key=f"btn_none_{flag_col}", use_container_width=True)

    # --- キーワード複数選択チェックボックス ---
    st.multiselect(
        f"（頻度順 Top 50）:",
        options,
        key=session_key, # (★) セッションステートに直接バインド
        label_visibility="collapsed"
    )

    # 3. テキスト列の選択
    text_col_options = [col for col in all_cols if df[col].dtype == 'object']
    default_text_col_index = text_col_options.index('ANALYSIS_TEXT_COLUMN') if 'ANALYSIS_TEXT_COLUMN' in text_col_options else 0
    text_col = st.selectbox(
        "3. 分析対象の自由記述列（サウンドバイト）:",
        text_col_options,
        index=default_text_col_index,
        key="cn_text_col_v2"
    )

    # (★) --- (Problem 2/3) カラム比率を 25:75 に変更 ---
    st.markdown("---")
    ui_cols = st.columns([0.25, 0.75]) # (★) 15:85 -> 25:75

    with ui_cols[0]:
        st.subheader("グラフ詳細設定")
        
        # 1. レイアウトの選択
        solver = st.selectbox(
            "レイアウト (layout)", 
            ['barnesHut', 'fruchterman_reingold', 'repulsion'], 
            index=0,
            key="cn_solver",
            help="グラフの配置アルゴリズム。'barnesHut' は高速で安定性が高い（推奨）"
        )

        # 2. (★) --- (Problem 2) UIレイアウト崩れ修正: 横並びをやめて縦積みに ---
        st.markdown("---")
        st.markdown("**物理演算パラメータ**")
        
        gravity = st.slider(
            "重力 (Gravity)", 
            min_value=-50000, max_value=-1000, value=-2000, step=1000, 
            key="cn_gravity",
            help="グラフの中心にノードを引き寄せる力。負の値を大きくすると、グラフが中央にまとまります。"
        )
        node_distance = st.slider(
            "ノード間の反発力", 100, 500, 200, key="cn_distance", 
            help="ノード同士が反発する力（距離）。値を大きくすると、各ノードが離れます。"
        )
        spring_length = st.slider(
            "エッジの長さ", 50, 500, 250, key="cn_spring", 
            help="ノード間を繋ぐ線の基本の長さ。"
        )
        
        # 3. (★) フィルタリングパラメータ
        st.markdown("---")
        st.markdown("**フィルタ設定**")
        
        top_n_words_limit = st.slider(
            "分析対象の単語数 (Top N)", 
            min_value=50, max_value=300, value=100, 
            key="cn_top_n",
            help="分析対象とする単語の最大数。値を小さくすると、出現頻度が最も高い単語群に絞り込まれ、グラフのノイズが減ります。"
        )
        max_degree_cutoff = st.slider(
            "最大接続数 (Exclude Hubs)", 10, 100, 50, key="cn_max_degree",
            help="接続数がこれより多いノード（スーパーハブ）をグラフから除外します。放射状グラフを解消するのに役立ちます。"
        )
        min_occurrence = st.slider(
            "最小共起回数 (Min Freq)", 1, 30, 10, key="cn_slider_v3", 
            help="ノード間を接続する最小の共起回数。値を大きくすると、関連性の強い線だけが残り、グラフがシンプルになります。"
        ) 
        
        # 4. (★) デザインパラメータ
        st.markdown("---")
        st.markdown("**デザイン設定**")
        default_node_size = st.slider(
            "基準ノードサイズ", 5, 50, 15, key="cn_node_size_v2", 
            help="ノード（円）の基本サイズ。"
        )
        default_text_size = st.slider(
            "テキストサイズ", 
            min_value=10, max_value=100, value=50, # (★) デフォルト 50, レンジ 10-100
            key="cn_text_size_v2", 
            help="ノードに表示されるテキストのサイズ。"
        )


    with ui_cols[1]:
        # (★) L994 から L1102 までのグラフ生成ロジックを「ui_cols[1]」内に移動
        
        nlp = load_spacy_model()
        if nlp is None:
            st.error("spaCy日本語モデルのロードに失敗しました。")
            return None
                
        # (★) target_pos に 'VERB' (動詞) を追加し、「行動」を抽出
        target_pos = {'NOUN', 'PROPN', 'ADJ', 'VERB'} 
        
        # (★) stop_words を更新 (分析のノイズとなる汎用語のみに限定)
        stop_words = {
            'の', 'に', 'は', 'を', 'が', 'で', 'て', 'です', 'ます', 'こと', 'もの', 'それ', 'あれ',
            'これ', 'ため', 'いる', 'する', 'ある', 'ない', 'いう', 'よう', 'そう', 'など', 'さん',
            '的', '人', '自分', '私', '僕', '何', 'その', 'この', 'あの', 'れる', 'られる',
            'てる', 'なる', '中', 'ところ', 'たち', '人達', '今回', '本当', 'とても', '色々'
            # (★) '食べる', '美味しい', '楽しい', '好き', '行く', '見る', '思う' などを削除
        }
        
        selected_keywords = st.session_state.get(session_key, [])
        
        if not selected_keywords:
            st.warning(f"「2. 絞り込むキーワード」を1つ以上選択してください。")
            return None
        
        # (★) --- タイトル重複修正 (L1023) ---
        st.subheader(f"共起ネットワーク (トピック: {len(selected_keywords)}個のキーワード)")
        
        # (★) --- スコープバグ修正: 変数を try の前に初期化 ---
        communities_with_words = {}
        degrees = {}
            
        try:
            G = nx.Graph()
            
            # (★) 1. キーワードでDataFrameをフィルタリング
            escaped_keywords = [re.escape(k) for k in selected_keywords]
            # | (OR) でパターンを作成
            pattern = '|'.join(escaped_keywords)
            
            df_filtered = df[df[flag_col].astype(str).str.contains(pattern, na=False)]
            
            if df_filtered.empty:
                st.warning(f"選択したキーワードを含む投稿が見つかりませんでした。")
                return None

            # 2. フィルタされた投稿のテキスト列を処理
            texts_to_analyze = df_filtered[text_col].dropna().astype(str)
            
            # (★) --- (Problem 1a) Top N 単語リストの作成 ---
            all_words = []
            for text in texts_to_analyze: # (★) 1回目のループ (TopN計算用)
                doc = nlp(text)
                for token in doc:
                    if (token.pos_ in target_pos) and (not token.is_stop) and (token.lemma_ not in stop_words) and (len(token.lemma_) > 1):
                        if token.lemma_ not in selected_keywords:
                            all_words.append(token.lemma_)
            
            if not all_words:
                st.warning("フィルタ結果から分析対象の単語が見つかりませんでした。")
                return None
            
            # (★) 上位N件の単語セットを作成
            top_n_words_set = set(pd.Series(all_words).value_counts().head(top_n_words_limit).index)
            logger.info(f"Top {top_n_words_limit} words calculated. Set size: {len(top_n_words_set)}")
            
            # (★) itertools をインポート
            from itertools import combinations
            
            for text in texts_to_analyze: # (★) 2回目のループ (ペア作成用)
                doc = nlp(text)
                words_in_text = set()
                for token in doc:
                    # (★) --- (Problem 1b.b) Top N 単語リストの適用 ---
                    # TopNセットに含まれる単語のみを抽出
                    if (token.pos_ in target_pos) and (token.lemma_ in top_n_words_set):
                        words_in_text.add(token.lemma_)
                
                # (★) 1つの投稿内の単語同士でペアを作成
                for word1, word2 in combinations(sorted(list(words_in_text)), 2):
                    if G.has_edge(word1, word2):
                        G[word1][word2]['weight'] += 1
                    else:
                        G.add_edge(word1, word2, weight=1) 

            if G.number_of_nodes() == 0:
                st.info("共起ネットワークを構築できませんでした（有効なキーワードペアなし）。")
                return None

            # (★) 可視化ロジック (コミュニティ検出)
            edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data['weight'] < min_occurrence]
            G.remove_edges_from(edges_to_remove)
            G.remove_nodes_from(list(nx.isolates(G))) 

            # (★) --- スーパーハブの除外 (NEW) ---
            degrees = dict(G.degree()) # (★) degrees をここで計算
            nodes_to_remove = [node for node, degree in degrees.items() if degree > max_degree_cutoff]
            G.remove_nodes_from(nodes_to_remove)
            G.remove_nodes_from(list(nx.isolates(G))) # 再度孤立ノードを削除
            logger.info(f"スーパーハブ除外: {len(nodes_to_remove)} 個のノードを削除しました (しきい値: {max_degree_cutoff})")
            # --- 除外ここまで ---

            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                st.info(f"最小共起回数 ({min_occurrence}) / 最大接続数 ({max_degree_cutoff}) の条件でペアが見つかりませんでした。")
                return None
            
            # (★) --- (Problem 3) UX/Height 修正 (L1062) ---
            net = Network(height="700px", width="100%", cdn_resources='in_line') # (★) height="700px" に変更
            
            degrees = dict(G.degree()) # (★) グラフ修正後に degrees を再計算
            min_degree, max_degree = (min(degrees.values()) or 1), (max(degrees.values()) or 1)
            
            # (★) --- (Problem 2) コミュニティ検出と色分け ---
            community_map = {}
            try:
                communities = community.greedy_modularity_communities(G)
                # (★) 凡例表示のため、コミュニティを単語リストとして保存
                communities_with_words = {i: list(comm) for i, comm in enumerate(communities)}
                community_map = {node: i for i, comm in communities_with_words.items() for node in comm}
                community_count = len(communities)
                logger.info(f"コミュニティ検出成功。{community_count}個のクラスタを発見。")
            except Exception as e:
                logger.warning(f"コミュニティ検出に失敗: {e}。色分けなしで続行します。")

            
            for node in G.nodes():
                if node not in degrees: continue
                size_factor = degrees.get(node, 0)
                size = default_node_size + 30 * (size_factor - min_degree) / (max_degree - min_degree + 1e-6)
                group_id = community_map.get(node, 0) # 属するクラスタ番号
                color = COLOR_PALETTE[group_id % len(COLOR_PALETTE)] # (★) 固定パレットから色を決定
                
                net.add_node(
                    node, label=node, size=size, title=f"{node} (クラスタ: {group_id}, 結合数: {size_factor})", 
                    color=color, # (★) group= ではなく color= を使用
                    font={"size": default_text_size}
                )
            # --- (Problem 2) END ---

            for u, v, data in G.edges(data=True):
                weight = data['weight']
                net.add_edge(u, v, title=f"共起回数: {weight}", value=weight)

            # (★) --- L1139 (エラー修正) ---
            # (★) solver の値に応じて、呼び出す関数を変更
            if solver == 'barnesHut':
                net.barnes_hut(
                    gravity=gravity,
                    overlap=0.1
                )
            else: # fruchterman_reingold or repulsion
                net.repulsion(
                    node_distance=node_distance, 
                    spring_length=spring_length
                )
            
            net.solver = solver 
            net.show_buttons(filter_=['physics', 'nodes', 'layout'])
            
            # (★) --- インデント修正 (L1145) ---
            html_file = "cooccurrence_network.html"
            net.save_graph(html_file)
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            # (★) --- (Problem 3) UX/Height 修正 (L1149) ---
            components.html(html_content, height=710) # (★) height=710 に変更
            
            edge_list = pd.DataFrame(G.edges(data=True), columns=["source", "target", "data"])
            edge_list['weight'] = edge_list['data'].apply(lambda x: x['weight'])
            
            # (★) --- (Problem 2) 凡例生成ボタンとロジック (tryブロックの内側に移動) ---
            st.markdown("---")
            st.subheader("凡例 (AIによるトピック推定)")
            
            legend_session_key = f"cn_legend_{flag_col}_{len(selected_keywords)}_{top_n_words_limit}"
            
            if legend_session_key not in st.session_state:
                st.session_state[legend_session_key] = {}
            
            if st.button("🤖 AIで凡例を生成 (β)", key="gen_legend_btn"):
                llm = get_llm()
                if llm and communities_with_words:
                    with st.spinner(f"{len(communities_with_words)}個のクラスタのトピックをAIが分析中..."):
                        legend_map = {}
                        
                        prompt = PromptTemplate.from_template(
                            """
                            以下の「単語リスト」は、あるトピックに関するコミュニティです。
                            このコミュニティの共通テーマを最もよく表す「凡例ラベル」（例：食事・グルメ、平和・歴史、移動手段）を【3語以内】で考案してください。
                            # 単語リスト (上位10件): {word_list_str}
                            # 回答 (3語以内):
                            """
                        )
                        chain = prompt | llm | StrOutputParser()
                        
                        for group_id, words in communities_with_words.items():
                            if not words: continue
                            # AIへの入力は代表的な10単語に絞る
                            words_top10 = sorted(words, key=lambda w: degrees.get(w, 0), reverse=True)[:10]
                            words_str = ", ".join(words_top10)
                            
                            try:
                                raw_label = chain.invoke({"word_list_str": words_str})
                                # (★) --- (Problem 1) AI G3 Clean-up ---
                                cleaned_label = re.sub(r'^(#|回答)\s*\(.*?\)\s*:\s*', '', raw_label.strip())
                                legend_map[group_id] = cleaned_label # (★) 修正
                            except Exception as e:
                                logger.error(f"AI凡例生成エラー (Group {group_id}): {e}")
                                legend_map[group_id] = "(AIエラー)"
                            
                            time.sleep(1) # API Rate Limit 対策
                            
                        st.session_state[legend_session_key] = legend_map
                else:
                    st.error("AIモデルが利用できないか、コミュニティが検出されませんでした。")
            
            # 凡例の表示
            if st.session_state[legend_session_key]:
                st.markdown("##### AIによる推定トピック:")
                
                legend_items = []
                for group_id, topic in st.session_state[legend_session_key].items():
                    color = COLOR_PALETTE[group_id % len(COLOR_PALETTE)]
                    # (★) CSSで「タグ」のようなインラインブロック要素として定義
                    legend_html = f"""
                    <span style="
                        display: inline-block;
                        margin: 4px;
                        padding: 8px 12px;
                        border-radius: 8px;
                        background-color: #f0f2f6;
                        border: 1px solid #e0e0e0;
                    ">
                        <span style='color:{color}; font-size: 20px; font-weight: bold; vertical-align: middle;'>■</span>
                        <span style="vertical-align: middle; margin-left: 8px; font-size: 14px;">{topic} (G{group_id})</span>
                    </span>
                    """
                    legend_items.append(legend_html.replace("\n", "")) # (★) 念のため改行を削除
                
                # (★) --- (Problem 1) L1188 修正: unsafe_allow_html=True を追加 ---
                st.markdown("<div style='line-height: 1.8;'>" + " ".join(legend_items) + "</div>", unsafe_allow_html=True)

            else:
                st.info("「AIで凡例を生成」ボタンを押すと、各色のトピックがAIによって推定されます。")
            # --- (Problem 2) END ---
            
            return edge_list[['source', 'target', 'weight']].sort_values(by="weight", ascending=False)

        # (★) --- インデント修正 (L1155) ---
        except Exception as e:
            st.error(f"共起ネットワーク分析中にエラー: {e}")
            logger.error(f"run_cooccurrence_network error: {e}", exc_info=True)
    
    # (★) --- インデント修正 (L1158) ---
    return None

def run_hashtag_analysis(df, text_col='ANALYSIS_TEXT_COLUMN'):
    """ハッシュタグを抽出し、頻度分析を実行する"""
    if text_col not in df.columns or df[text_col].empty:
        st.warning(f"分析対象のテキスト列 '{text_col}' がないか、空です。")
        return None #

    st.info("テキスト列からハッシュタグを抽出・集計中...")

    try:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            st.warning("分析対象のテキストがありません。")
            return None #
            
        # 正規表現でハッシュタグを抽出
        hashtag_pattern = r'#(\S+)'
        hashtags = texts.str.findall(hashtag_pattern).explode()
        
        # 抽出できなかった場合
        if hashtags.empty or hashtags.isnull().all():
            st.warning("テキスト内に有効なハッシュタグ（#...）が見つかりませんでした。")
            return None #

        # 小文字に統一して集計
        hashtags = hashtags.str.lower()
        hashtag_counts = hashtags.value_counts().head(30) # 上位30件

        st.subheader("頻出ハッシュタグ Top 30")
        
        # ハッシュタグの前に # を付け直す
        hashtag_counts.index = "#" + hashtag_counts.index.astype(str)
        
        st.bar_chart(hashtag_counts)
        with st.expander("詳細データ（Top 30）"):
            st.dataframe(hashtag_counts.reset_index(name="出現回数").rename(columns={"index": "ハッシュタグ"}))
        
        return hashtag_counts # 

    except Exception as e:
        st.error(f"ハッシュタグ分析処理中にエラー: {e}")
        logger.error(f"run_hashtag_analysis error: {e}", exc_info=True)
    return None

# --- (★) L843付近に追加: 人気投稿分析 ---
def run_engagement_analysis(df, text_col='ANALYSIS_TEXT_COLUMN'):
    """エンゲージメント（いいね数など）に基づき人気投稿を分析する"""
    if text_col not in df.columns:
        st.warning(f"テキスト列 '{text_col}' が見つかりません。")
        return None

    # エンゲージメントに関連する可能性のある列名を推測
    possible_cols = [
        'likes', 'like', 'いいね', 'いいね数', 
        'retweets', 'retweet', 'リツイート', 'リツイート数',
        'comments', 'comment', 'コメント', 'コメント数',
        'engagement', 'エンゲージメント'
    ]
    
    # 存在する数値列をフィルタリング
    numeric_cols = df.select_dtypes(include=np.number).columns
    engagement_cols = [col for col in numeric_cols if any(c_name in col.lower() for c_name in possible_cols)]

    if not engagement_cols:
        st.warning("データに「いいね数」「リツイート数」「コメント数」などのエンゲージメントを示す数値列が見つかりません。")
        st.info(f"（分析可能な数値列: {', '.join(numeric_cols.tolist())}）")
        return None

    # ユーザーにソート基準の列を選んでもらう
    sort_col = st.selectbox(
        "人気投稿の基準（ソートキー）にする列を選択してください:",
        engagement_cols,
        key="eng_select_col"
    )
    
    top_n = st.slider("表示件数", 5, 50, 10, key="eng_slider")

    if not sort_col:
        return None

    try:
        # ソート対象の列とテキスト列を抽出
        cols_to_show = [sort_col, text_col]
        # 他のエンゲージメント列もあれば表示に追加
        cols_to_show.extend([col for col in engagement_cols if col != sort_col and col in df.columns])
        
        popular_posts_df = df[cols_to_show].copy()
        popular_posts_df = popular_posts_df.sort_values(by=sort_col, ascending=False).head(top_n)
        
        st.subheader(f"「{sort_col}」に基づく人気投稿 Top {top_n}")
        
        st.dataframe(popular_posts_df)
        
        with st.expander("投稿詳細（テキスト全体）"):
            for idx, row in popular_posts_df.iterrows():
                st.markdown(f"**{idx+1}. {sort_col}: {row[sort_col]}**")
                st.text_area(f"Text (Row {idx})", row[text_col], height=100, disabled=True, key=f"eng_text_{idx}")
                st.markdown("---")

        return popular_posts_df

    except Exception as e:
        st.error(f"人気投稿分析処理中にエラー: {e}")
        logger.error(f"run_engagement_analysis error: {e}", exc_info=True)
    return None

# --- (★) L843付近に追加: 市区町村別 概要・投稿数・センチメント分析 (LLM使用) ---
def run_geo_summary_llm(df, geo_col="市区町村キーワード", text_col="ANALYSIS_TEXT_COLUMN"):
    """
    市区町村キーワードごとにグループ化し、投稿数集計、センチメント分析、投稿概要の要約をLLMで行う。
    """
    if geo_col not in df.columns:
        st.warning(f"分析の軸となる「{geo_col}」列が見つかりません。Step AまたはBでタグ付けされたデータを使用してください。")
        return None
    if text_col not in df.columns:
        st.warning(f"分析対象のテキスト列「{text_col}」が見つかりません。")
        return None

    # 1. 投稿数の集計 (run_simple_count と同じロジック)
    st.subheader(f"1. 「{geo_col}」別 投稿数")
    try:
        s = df[geo_col].astype(str).str.split(', ').explode()
        s = s[s.str.strip() != ''] # 空白を除去
        s = s.str.strip() # 前後の空白を除去
        
        if s.empty:
            st.info("集計対象の市区町村キーワードがありませんでした。")
            return None #
            
        geo_counts = s.value_counts().head(20) # 上位20件
        st.bar_chart(geo_counts)
        with st.expander("投稿数データ（上位20件）"):
            st.dataframe(geo_counts)
    except Exception as e:
        st.error(f"市区町村別の投稿数集計中にエラー: {e}")
        logger.error(f"run_geo_summary_llm (Count) error: {e}", exc_info=True)
        return None # 投稿数がなければ続行不可

    # 2. センチメント分析 と 3. 投稿概要の要約 (LLM使用)
    st.markdown("---")
    st.subheader(f"2. & 3. 「{geo_col}」別 センチメントと投稿概要（AI分析）")

    # 分析対象の市区町村を投稿数TopNから選択
    target_geos = geo_counts.index.tolist()
    selected_geos = st.multiselect(
        "AI分析（センチメント・概要）を実行する市区町村を選択（APIコールが発生します）:",
        target_geos,
        default=target_geos[:min(len(target_geos), 3)], # デフォルトTop3
        key="geo_llm_select"
    )

    if not selected_geos:
        st.info("AI分析を実行する市区町村を選択してください。")
        return geo_counts # 投稿数データのみ返す

    llm = get_llm()
    if llm is None:
        st.error("AIモデルが利用できません。サイドバーでAPIキーを設定してください。")
        return geo_counts

    # プロンプトテンプレートの準備
    prompt = PromptTemplate.from_template(
        """
        あなたはデータアナリストです。以下の「{geo_col}」に関する「サンプル投稿テキスト群」を読み、その地域の「センチメント」と「投稿概要」を分析してください。
        
        # 分析対象の市区町村:
        {target_geo_name}
        
        # サンプル投稿テキスト群 (最大10件):
        {text_samples}
        
        # 指示:
        1. 「センチメント」: テキスト群全体の雰囲気を「ポジティブ」「ネガティブ」「ニュートラル」の3択で判定し、その理由も簡潔に記述してください。
        2. 「投稿概要」: これらの投稿で主に何が話題にされているか、重要なトピックを300文字程度で要約してください。
        
        # 回答フォーマット (厳格なJSON辞書形式のみ):
        {{
          "geo_name": "{target_geo_name}",
          "sentiment": "（ポジティブ/ネガティブ/ニュートラル）",
          "sentiment_reason": "（判定理由）",
          "summary": "（300文字程度の要約）"
        }}
        """
    )
    chain = prompt | llm | StrOutputParser()

    results = []
    progress_bar = st.progress(0, text="AI分析待機中...")
    
    if 'geo_summary_results' not in st.session_state:
        st.session_state.geo_summary_results = {}

    run_button = st.button(f"選択した {len(selected_geos)} 件のAI分析を実行", key="run_geo_llm", type="primary")

    if run_button:
        # 実行時にキャッシュをクリア
        st.session_state.geo_summary_results = {}
        
        for i, geo_name in enumerate(selected_geos):
            progress_bar.progress((i) / len(selected_geos), text=f"AI分析中: {geo_name} ({i+1}/{len(selected_geos)})")
            
            # geo_name が含まれる行を抽出
            try:
                mask = df.apply(lambda row: isinstance(row[geo_col], str) and geo_name in row[geo_col], axis=1)
                geo_texts = df.loc[mask, text_col].dropna().sample(n=min(10, mask.sum()), random_state=1).tolist()
            except Exception:
                geo_texts = df.loc[mask, text_col].dropna().tolist()[:10]

            if not geo_texts:
                logger.warning(f"「{geo_name}」のテキストが見つかりません。スキップします。")
                results.append({"geo_name": geo_name, "sentiment": "データなし", "sentiment_reason": "-", "summary": "投稿テキストが見つかりませんでした。"})
                continue
            
            # テキストサンプルを文字列に
            text_samples_str = "\n".join([f"- {text[:200]}..." for text in geo_texts]) 

            try:
                response_str = chain.invoke({
                    "geo_col": geo_col,
                    "target_geo_name": geo_name,
                    "text_samples": text_samples_str
                })
                
                logger.debug(f"AI Geo Summary (Raw) for {geo_name}: {response_str}")
                match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    data = json.loads(json_str)
                    results.append(data)
                else:
                    raise Exception("JSON形式の応答がありません。")
                
            except Exception as e:
                logger.error(f"AI Geo Summary Error (LLM) for {geo_name}: {e}", exc_info=True)
                st.error(f"「{geo_name}」のAI分析中にエラー: {e}")
                results.append({"geo_name": geo_name, "sentiment": "分析エラー", "sentiment_reason": str(e), "summary": "AIの応答取得に失敗しました。"})
            
            # APIレート制限のための待機 (app.py L34 の値)
            time.sleep(TAGGING_SLEEP_TIME) 
        
        progress_bar.progress(1.0, text="AI分析完了！")
        st.session_state.geo_summary_results = results
        
    # 結果の表示
    if st.session_state.geo_summary_results:
        results_df = pd.DataFrame(st.session_state.geo_summary_results)
        
        st.subheader("センチメント分析結果")
        sentiment_counts = results_df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        
        st.subheader("市区町村別 概要")
        for res in st.session_state.geo_summary_results:
            with st.expander(f"**{res.get('geo_name')}** (センチメント: {res.get('sentiment')})"):
                st.markdown(f"**<AIによる投稿概要>**\n{res.get('summary')}")
                st.caption(f"センチメント判定理由: {res.get('sentiment_reason')}")
        
        return results_df 
    
    return geo_counts

# --- L752: UI更新ヘルパー (DRY原則) ---
def update_progress_ui(progress_placeholder, log_placeholder, processed_rows, total_rows, message_prefix):
    """
    Step A の進捗バーとログエリアを更新する (DRY)
    """
    try:
        progress_percent = min(processed_rows / total_rows, 1.0)
        progress_text = f"[{message_prefix}] 処理中: {processed_rows}/{total_rows} 件 ({progress_percent:.0%})"
        progress_placeholder.progress(progress_percent, text=progress_text)
        
        # ログ表示 (最新50件)
        log_text_for_ui = "\n".join(st.session_state.log_messages[-50:])
        log_placeholder.text_area("実行ログ (最新50件):", log_text_for_ui, height=200, key=f"log_update_{message_prefix}_{processed_rows}", disabled=True)
    except Exception as e:
        logger.warning(f"UI update failed: {e}") # UIエラーは処理を止めない

# --- L752: Step A (タグ付けUI) ---
def render_step_a():
    """Step A: タグ付け処理のUIを描画する"""
    st.title("🏷️ テキストデータのAIタグ付け (Step A)")

    # Step A 固有のセッションステートをここで初期化 (SRP)
    if 'cancel_analysis' not in st.session_state: st.session_state.cancel_analysis = False
    if 'generated_categories' not in st.session_state: st.session_state.generated_categories = {}
    if 'selected_categories' not in st.session_state: st.session_state.selected_categories = set()
    if 'api_key_A' not in st.session_state: st.session_state.api_key_A = "" # L1096 (旧 L1383) から移動
    if 'analysis_prompt_A' not in st.session_state: st.session_state.analysis_prompt_A = "" # L1092 (旧 L1379) から移動
    if 'selected_text_col' not in st.session_state: st.session_state.selected_text_col = {} # L1094 (旧 L1381) から移動
    if 'tagged_df_A' not in st.session_state: st.session_state.tagged_df_A = pd.DataFrame() # L1090 (旧 L1377) から移動

    # L754-L757: 不要なコメントアウトを削除 (KISS)
    
    st.header("Step 1: 分析対象ファイルのアップロード")
    uploaded_files = st.file_uploader("分析したい Excel / CSV ファイル（複数可）", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True, key="uploader_A")
    
    if not uploaded_files:
        st.info("分析を開始するには、ExcelまたはCSVファイルをアップロードしてください。")
        return # ファイルがなければここで終了 (KISS)
    
    valid_files_data = {}
    error_messages = []
    for f in uploaded_files:
        df, err = read_file(f)
        if err: error_messages.append(f"**{f.name}**: {err}")
        else: valid_files_data[f.name] = df
    if error_messages: st.error("以下のファイルは読み込めませんでした:\n" + "\n".join(error_messages))
    if not valid_files_data: st.warning("読み込み可能なファイルがありません。"); return

    st.header("Step 2: 分析指針の入力")
    analysis_prompt = st.text_area(
        "AIがタグ付けとクレンジングを行う際の指針を入力してください（必須）:",
        value=st.session_state.analysis_prompt_A,
        height=100,
        placeholder="例: 広島県の観光に関するInstagramの投稿。無関係な地域の投稿や、単なる挨拶・宣伝は除外したい。",
        key="analysis_prompt_input_A"
    )
    st.session_state.analysis_prompt_A = analysis_prompt # L781: セッションに保存

    if not analysis_prompt.strip():
        st.warning("分析指針は必須です。AIがデータを理解するために目的を入力してください。")
        return # 指針がなければここで終了 (KISS)

    st.header("Step 3: AIによるカテゴリ候補の生成")
    if st.button("AIにカテゴリ候補を生成させる", key="gen_cat_button", type="primary"):
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Google APIキーが設定されていません。（サイドバーで設定してください）")
        else:
            with st.spinner("AIが分析指針を読み解き、カテゴリを考案中..."):
                logger.info("AIカテゴリ生成ボタンクリック")
                st.session_state.generated_categories = {"市区町村キーワード": "地名辞書(JAPAN_GEOGRAPHY_DB)から抽出された市区町村名"}
                # L796: キャッシュ利用版 (llm引数削除)
                ai_categories = get_dynamic_categories(analysis_prompt) 
                if ai_categories:
                    st.session_state.generated_categories.update(ai_categories)
                    logger.info(f"AIカテゴリ生成成功: {list(ai_categories.keys())}")
                    st.success("AIによるカテゴリ候補の生成が完了しました。")
                else:
                    st.error("AIによるカテゴリ生成に失敗しました。AIの応答を確認してください。")

    st.header("Step 4: 分析カテゴリの選択")
    if not st.session_state.generated_categories:
        st.info("Step 3 でカテゴリを生成してください。")
        return
    st.markdown("タグ付けしたいカテゴリを以下から選択してください（「市区町村キーワード」は必須です）")
    selected_cats = []
    cols = st.columns(3)
    categories_to_show = st.session_state.generated_categories.items()
    for i, (cat, desc) in enumerate(categories_to_show):
        with cols[i % 3]:
            is_checked = st.checkbox(
                cat, 
                value=(cat == "市区町村キーワード" or cat in st.session_state.selected_categories), 
                help=desc, 
                key=f"cat_cb_{cat}",
                disabled=(cat == "市区町村キーワード") # 必須項目は無効化
            )
            if is_checked:
                selected_cats.append(cat)
    st.session_state.selected_categories = set(selected_cats)

    st.header("Step 5: 分析対象テキスト列の指定")
    selected_text_col_map = {}
    st.markdown("ファイルごとに、タグ付け対象のテキストが含まれる列を指定してください。")
    for f_name, df in valid_files_data.items():
        cols_list = list(df.columns)
        default_index = 0
        if st.session_state.selected_text_col.get(f_name) in cols_list:
            default_index = cols_list.index(st.session_state.selected_text_col.get(f_name))
        elif 'ANALYSIS_TEXT_COLUMN' in cols_list:
             default_index = cols_list.index('ANALYSIS_TEXT_COLUMN')
        selected_col = st.selectbox(f"**{f_name}** のテキスト列:", cols_list, index=default_index, key=f"col_select_{f_name}")
        selected_text_col_map[f_name] = selected_col
    st.session_state.selected_text_col = selected_text_col_map

    st.header("Step 6: 分析実行")
    if st.button("キャンセル", key="cancel_button_A"):
        st.session_state.cancel_analysis = True
        logger.warning("分析キャンセルボタンが押されました。")
        st.warning("次のバッチ処理後に分析をキャンセルします...")
        
    if st.button("分析実行", type="primary", key="run_analysis_A"):
        st.session_state.cancel_analysis = False
        st.session_state.log_messages = [] # ログリセット
        st.session_state.tagged_df_A = pd.DataFrame() # 結果リセット
        
        try:
            with st.spinner("Step A: AI分析処理中..."):
                logger.info("Step A 分析実行ボタンクリック")
                progress_placeholder = st.progress(0.0, text="処理待機中...")
                log_placeholder = st.empty()
                
                temp_dfs = []
                for f_name, df in valid_files_data.items():
                    col_name = selected_text_col_map[f_name]
                    temp_df = df.rename(columns={col_name: 'ANALYSIS_TEXT_COLUMN'})
                    temp_dfs.append(temp_df)
                
                logger.info(f"{len(temp_dfs)} 個ファイルを結合..."); 
                master_df = pd.concat(temp_dfs, ignore_index=True, sort=False); 
                master_df['id'] = master_df.index; 
                total_rows = len(master_df); 
                logger.info(f"結合完了。総行数: {total_rows}")
                if master_df.empty: logger.error("結合後DF空"); raise Exception("分析対象データ空")

                logger.info("Step A-2: 重複削除 開始...")
                initial_row_count = len(master_df)
                master_df.drop_duplicates(subset=['ANALYSIS_TEXT_COLUMN'], keep='first', inplace=True)
                deduped_row_count = len(master_df)
                logger.info(f"重複削除 完了。 {initial_row_count}行 -> {deduped_row_count}行 ({initial_row_count - deduped_row_count}行削除)")
                
                logger.info("Step A-3: AI関連性フィルタリング 開始...")
                total_filter_rows = len(master_df)
                total_filter_batches = (total_filter_rows + FILTER_BATCH_SIZE - 1) // FILTER_BATCH_SIZE
                all_filtered_results = []
                
                for i in range(0, total_filter_rows, FILTER_BATCH_SIZE): # L1033: 定数
                    if st.session_state.cancel_analysis: logger.warning(f"フィルタリングキャンセル (バッチ {i//FILTER_BATCH_SIZE + 1})"); st.warning("分析キャンセル"); break
                    
                    batch_df = master_df.iloc[i:i+FILTER_BATCH_SIZE] # L1036: 定数
                    current_batch_num = i // FILTER_BATCH_SIZE + 1 # L1037: 定数
                    logger.info(f"AIフィルタリング バッチ {current_batch_num}/{total_filter_batches} 処理中...")
                    
                    # L1048: UI更新をヘルパー関数で呼び出し (DRY)
                    update_progress_ui(
                        progress_placeholder, log_placeholder, 
                        min(i + FILTER_BATCH_SIZE, total_filter_rows), total_filter_rows, 
                        "AIフィルタリング"
                    )
                    
                    # L1053: キャッシュ利用版 (llm引数削除)
                    filtered_df = filter_relevant_data_by_ai(batch_df, analysis_prompt)
                    if filtered_df is not None and not filtered_df.empty:
                        all_filtered_results.append(filtered_df)
                    else:
                        logger.warning(f"AIフィルタリング バッチ {current_batch_num} 結果空")
                        
                    time.sleep(FILTER_SLEEP_TIME) # L1060: 定数
                
                if st.session_state.cancel_analysis:
                    logger.warning("AIフィルタリング処理がキャンセルされました。")
                    raise Exception("分析がキャンセルされました") 
                if not all_filtered_results:
                    logger.error("全バッチAIフィルタリング失敗"); raise Exception("AIフィルタリング処理失敗")
                logger.info("全AIフィルタリング結果結合...");
                filter_results_df = pd.concat(all_filtered_results, ignore_index=True)
                relevant_ids = filter_results_df[filter_results_df['relevant'] == True]['id']
                filtered_master_df = master_df[master_df['id'].isin(relevant_ids)].copy()
                filtered_row_count = len(filtered_master_df)
                logger.info(f"AIフィルタリング 完了。 {deduped_row_count}行 -> {filtered_row_count}行 ({deduped_row_count - filtered_row_count}行削除)")
                if filtered_master_df.empty:
                    logger.error("AIフィルタリング後、データが0件になりました。"); raise Exception("分析対象データ空")
                
                logger.info("Step A-4: AIタグ付け処理開始..."); 
                selected_category_definitions = { cat: desc for cat, desc in st.session_state.generated_categories.items() if cat in st.session_state.selected_categories }; 
                logger.info(f"選択カテゴリ: {list(selected_category_definitions.keys())}")
                
                master_df_for_tagging = filtered_master_df
                total_rows = len(master_df_for_tagging) # L1082: 総行数を更新
                
                all_tagged_results = []; 
                total_batches = (total_rows + TAGGING_BATCH_SIZE - 1) // TAGGING_BATCH_SIZE; 
                logger.info(f"バッチサイズ {TAGGING_BATCH_SIZE}, 総バッチ数: {total_batches}")
                
                for i in range(0, total_rows, TAGGING_BATCH_SIZE): # L1085: 定数
                    if st.session_state.cancel_analysis: logger.warning(f"ループキャンセル (バッチ {i//TAGGING_BATCH_SIZE + 1})"); st.warning("分析キャンセル"); break
                    
                    batch_df = master_df_for_tagging.iloc[i:i+TAGGING_BATCH_SIZE]; # L1088: 定数
                    current_batch_num = i // TAGGING_BATCH_SIZE + 1; 
                    logger.info(f"バッチ {current_batch_num}/{total_batches} 処理中...")
                    
                    # L1089: UI更新をヘルパー関数で呼び出し (DRY)
                    update_progress_ui(
                        progress_placeholder, log_placeholder, 
                        min(i + TAGGING_BATCH_SIZE, total_rows), total_rows, 
                        "AIタグ付け"
                    )
                    
                    logger.info(f"Calling perform_ai_tagging batch {current_batch_num}...")
                    # L1094: キャッシュ利用版 (llm引数削除)
                    tagged_df = perform_ai_tagging(batch_df, selected_category_definitions, analysis_prompt)
                    logger.info(f"Finished perform_ai_tagging batch {current_batch_num}.")
                    if tagged_df is not None and not tagged_df.empty: all_tagged_results.append(tagged_df)
                    
                    time.sleep(TAGGING_SLEEP_TIME) # L1098: 定数
                
                if st.session_state.cancel_analysis:
                    logger.warning("AIタグ付け処理がキャンセルされました。")
                    raise Exception("分析がキャンセルされました")
                if not all_tagged_results: logger.error("全バッチAIタグ付け失敗"); raise Exception("AIタグ付け処理失敗")
                
                logger.info("全AIタグ付け結果結合..."); 
                tagged_results_df = pd.concat(all_tagged_results, ignore_index=True)
                
                logger.info("最終マージ処理開始..."); 
                cols_to_drop_from_master = [col for col in tagged_results_df.columns if col in master_df_for_tagging.columns and col != 'id']
                if cols_to_drop_from_master: 
                    logger.warning(f"重複列削除: {cols_to_drop_from_master}"); 
                    master_df_for_merge = master_df_for_tagging.drop(columns=cols_to_drop_from_master)
                else: 
                    master_df_for_merge = master_df_for_tagging
                
                final_df = pd.merge(master_df_for_merge, tagged_results_df, on='id', how='right')
                st.session_state.tagged_df_A = final_df
                logger.info("分析処理 正常終了"); 
                st.success("AIによる分析処理が完了しました。"); 
                progress_placeholder.progress(1.0, text="処理完了")
                log_text_for_ui = "\n".join(st.session_state.log_messages)
                log_placeholder.text_area("実行ログ:", log_text_for_ui, height=200, key=f"log_update_A_final", disabled=True)
                
        except Exception as e:
            logger.error(f"Step A 分析実行中にエラー: {e}", exc_info=True)
            st.error(f"分析実行中にエラーが発生しました: {e}")
            if 'progress_placeholder' in locals():
                progress_placeholder.progress(1.0, text="エラーにより処理中断")
    
    if st.session_state.cancel_analysis:
        st.session_state.cancel_analysis = False # L1126: 状態をリセット
    
    if not st.session_state.tagged_df_A.empty:
        st.header("Step 7: 分析結果の確認とエクスポート")
        st.dataframe(st.session_state.tagged_df_A.head(50))
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(encoding="utf-8-sig", index=False).encode("utf-8-sig")
        csv_data = convert_df_to_csv(st.session_state.tagged_df_A)
        st.download_button(
            label="分析結果CSVをダウンロード",
            data=csv_data,
            file_name="keyword_extraction_result.csv",
            mime="text/csv",
        )

# --- L833: Step C (可視化UI) ---
def render_step_c():
    """Step C: 分析結果の可視化を描画する"""
    st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
    st.title("🔬 分析結果の可視化 (Step C)")
    
    # Step C 固有のセッションステートをここで初期化 (SRP)
    if 'step_c_results' not in st.session_state: st.session_state.step_c_results = {}
    if 'ai_summary_prompt' not in st.session_state: st.session_state.ai_summary_prompt = None
    if 'ai_summary_result' not in st.session_state: st.session_state.ai_summary_result = None

    if 'chosen_analysis_list' not in st.session_state or not st.session_state.chosen_analysis_list:
        st.warning("実行する分析が選択されていません。Step Bに戻ってください。")
        if st.button("Step B に戻る"):
            st.session_state.current_step = 'B'; st.rerun()
        return

    if 'df_flagged_B' not in st.session_state or st.session_state.df_flagged_B.empty:
        st.warning("分析対象のデータが見つかりません。Step BでCSVをアップロードしてください。")
        if st.button("Step B に戻る"):
            st.session_state.current_step = 'B'; st.rerun()
        return
        
    if 'suggestions_B' not in st.session_state or not st.session_state.suggestions_B:
        st.warning("分析手法の提案リストが見つかりません。Step Bで再提案してください。")
        if st.button("Step B に戻る"):
            st.session_state.current_step = 'B'; st.rerun()
        return

    df = st.session_state.df_flagged_B
    selected_names = st.session_state.chosen_analysis_list
    all_suggestions = st.session_state.suggestions_B
    
    analyses_to_run = [s for s in all_suggestions if s['name'] in selected_names]
    
    st.info(f"**実行する分析:** {', '.join(selected_names)}")
    st.markdown("---")

    st.session_state.step_c_results = {}
    
    for suggestion in analyses_to_run:
        name = suggestion['name']
        cols = suggestion.get('suitable_cols', []) 
        
        with st.container(border=True):
            if name != "共起ネットワーク分析": # (★) この if 文を追加
                st.subheader(f"📈 分析結果: {name}")
            
            try:
                result_data = None # 結果格納用
                if name == "単純集計（頻度分析）":
                    result_data = run_simple_count(df, cols) 
                elif name == "基本統計量":
                    result_data = run_basic_stats(df, cols) 
                elif name == "クロス集計（キーワード間）":
                    result_data = run_crosstab(df, cols) 
                elif name == "クロス集計（キーワード×属性）":
                    result_data = run_crosstab(df, cols)
                elif name == "共起ネットワーク分析":
                    result_data = run_cooccurrence_network(df, cols) 
                
                elif name == "市区町村地域ごとの投稿概要と投稿数":
                    st.info("AIによる分析（センチメント・概要）が含まれます。")
                    result_data = run_geo_summary_llm(df, geo_col="市区町村キーワード", text_col="ANALYSIS_TEXT_COLUMN")

                elif name == "投稿量分析":
                    st.info("「時系列キーワード分析」を実行します。")
                    if isinstance(cols, dict) and 'datetime' in cols and 'keywords' in cols:
                        result_data = run_timeseries(df, cols, name) # (★) name を追加
                    else:
                        st.warning("この分析には「日時列」と「キーワード列」が必要です。Step Bの提案定義を確認してください。")

                elif name == "センチメント分析":
                    st.warning("この分析は「市区町村地域ごとの投稿概要と投稿数」分析に含まれています。そちらを実行してください。")

                elif name == "ハッシュタグ分析":
                    result_data = run_hashtag_analysis(df, text_col='ANALYSIS_TEXT_COLUMN')

                elif name == "カテゴリ分類":
                    st.info("「単純集計（頻度分析）」を実行します。")
                    result_data = run_simple_count(df, cols) 

                elif name == "エンゲージメントに基づく人気投稿分析":
                    result_data = run_engagement_analysis(df, text_col='ANALYSIS_TEXT_COLUMN')

                elif name == "二群間の比較分析":
                    st.info("「カテゴリ別集計（グループ比較）」または「クロス集計」を実行します。")
                    if isinstance(cols, dict) and 'numeric' in cols and 'grouping' in cols:
                         st.markdown("（カテゴリ別集計（グループ比較）を実行）")
                         # (L895-L933 のロジックを流用)
                         grouping_cols = cols['grouping']
                         numeric_cols_to_desc = [col for col in cols['numeric'] if col in df.columns]
                         
                         if not numeric_cols_to_desc: st.warning("分析対象の数値列がデータにありません。")
                         elif not grouping_cols: st.warning("分析対象のグループ列がありません。")
                         else:
                             if not isinstance(grouping_cols, list):
                                 grouping_cols = [grouping_cols]
                             existing_grouping_cols = [col for col in grouping_cols if col in df.columns]
                             if not existing_grouping_cols:
                                 st.warning(f"グループ化列 {grouping_cols} がデータに存在しません。")
                             else:
                                 try:
                                     df_copy = df.copy()
                                     for col in existing_grouping_cols:
                                         df_copy[col] = df_copy[col].astype(str)
                                         
                                     result_df = df_copy.groupby(existing_grouping_cols)[numeric_cols_to_desc].describe()
                                     
                                     flat_cols = []
                                     for col in result_df.columns:
                                         flat_cols.append(f"{col[0]}_{col[1]}") 
                                     result_df.columns = flat_cols
                                     
                                     final_result_df = result_df.reset_index()
                                     st.dataframe(final_result_df) 
                                     result_data = final_result_df 
                                 except Exception as group_e:
                                     st.error(f"グループ別集計エラー: {group_e}")
                                     logger.error(f"Groupby describe error: {group_e}", exc_info=True)
                    else:
                         st.markdown("（クロス集計を実行）")
                         result_data = run_crosstab(df, cols) 

                elif name == "エリア別抽出パターン分析":
                    st.info("「クロス集計（キーワード間）」を実行します。")
                    result_data = run_crosstab(df, cols)
                
                elif name == "カテゴリ別集計（グループ比較）":
                    # (L895-L933 のロジック)
                    if isinstance(cols, dict) and 'numeric' in cols and 'grouping' in cols:
                         grouping_cols = cols['grouping']
                         numeric_cols_to_desc = [col for col in cols['numeric'] if col in df.columns]
                         if not numeric_cols_to_desc: st.warning("分析対象の数値列がデータにありません。")
                         elif not grouping_cols: st.warning("分析対象のグループ列がありません。")
                         else:
                             if not isinstance(grouping_cols, list):
                                 grouping_cols = [grouping_cols]
                             existing_grouping_cols = [col for col in grouping_cols if col in df.columns]
                             if not existing_grouping_cols:
                                 st.warning(f"グループ化列 {grouping_cols} がデータに存在しません。")
                             else:
                                 try:
                                     df_copy = df.copy()
                                     for col in existing_grouping_cols:
                                         df_copy[col] = df_copy[col].astype(str)
                                     result_df = df_copy.groupby(existing_grouping_cols)[numeric_cols_to_desc].describe()
                                     flat_cols = []
                                     for col in result_df.columns:
                                         flat_cols.append(f"{col[0]}_{col[1]}") 
                                     result_df.columns = flat_cols
                                     final_result_df = result_df.reset_index()
                                     st.dataframe(final_result_df) 
                                     result_data = final_result_df 
                                 except Exception as group_e:
                                     st.error(f"グループ別集計エラー: {group_e}")
                                     logger.error(f"Groupby describe error: {group_e}", exc_info=True)
                    else:
                         st.warning(f"「{name}」の列定義が不適切です: {cols}")

                elif name == "時系列キーワード分析":
                    if isinstance(cols, dict) and 'datetime' in cols and 'keywords' in cols:
                        result_data = run_timeseries(df, cols, name) # (★) name を追加
                    else:
                         st.warning(f"「{name}」の列定義が不適切です: {cols}")

                elif name == "テキストマイニング（頻出単語など）":
                    if cols and isinstance(cols, list) and cols[0] == 'ANALYSIS_TEXT_COLUMN':
                        result_data = run_text_mining(df, 'ANALYSIS_TEXT_COLUMN')
                    else:
                        st.warning(f"「{name}」の列定義が不適切です: {cols}")

                elif name == "主成分分析 (PCA) / 因子分析":
                    result_data = run_pca(df, cols)
                
                else:
                    st.warning(f"「{name}」の可視化ロジックはまだ実装されていません。")
                
                if result_data is not None and not result_data.empty:
                    st.session_state.step_c_results[name] = result_data
            
            except Exception as e:
                st.error(f"「{name}」の分析中に予期せぬエラーが発生しました: {e}")
                logger.error(f"Step C Analysis Error ({name}): {e}", exc_info=True)

    st.markdown("---")
    st.success("Step C-1 (可視化) が完了しました。")
    
    st.header("Step C-2: AIによる分析サマリー")
    
    if not st.session_state.step_c_results:
        st.warning("AIサマリーの元になる分析結果がありません。Step C-1で有効な分析を実行してください。")
    else:
        st.info("上記で実行された分析結果をAIに入力し、総合的なサマリーレポートを生成します。")

        if st.button("🤖 AIサマリー用のプロンプトを生成", key="gen_prompt_c2"):
            st.session_state.ai_summary_prompt = generate_ai_summary_prompt(st.session_state.step_c_results, df)
            st.session_state.ai_summary_result = None 
            st.rerun()

        if st.session_state.ai_summary_prompt:
            st.subheader("AIへの指示プロンプト（確認・編集可）")
            prompt_input = st.text_area(
                "以下のプロンプトをAIに送信します:",
                value=st.session_state.ai_summary_prompt,
                height=300,
                key="ai_prompt_c2_input"
            )
            
            if st.button("🚀 この内容でAIに指示を送信", key="send_prompt_c2", type="primary"):
                if not os.getenv("GOOGLE_API_KEY"):
                    st.error("AIの実行には Google APIキー が必要です。（サイドバーで設定してください）")
                else:
                    with st.spinner("AIがサマリーを生成中... (Rate Limitに注意)"):
                        llm = get_llm() # キャッシュされたLLMを呼び出し
                        if llm:
                            try:
                                response = llm.invoke(prompt_input) 
                                st.session_state.ai_summary_result = response.content
                            except Exception as e:
                                st.error(f"AIの呼び出しに失敗しました: {e}")
                                logger.error(f"AI summary failed: {e}", exc_info=True)
                        else:
                            st.error("AIモデルの初期化に失敗しました。")
            
        if st.session_state.ai_summary_result:
            st.subheader("AIによる分析サマリーレポート")
            st.markdown(st.session_state.ai_summary_result)
    
    st.markdown("---")
    if st.button("⬅️ Step B に戻る", key="back_to_b_c2"):
        st.session_state.current_step = 'B'; st.rerun()

# --- L1002: Step B (分析提案UI) ---
def render_step_b():
    """Step B: 分析手法の提案UIを描画する"""
    st.title("📊 分析手法の提案 (Step B)")
    
    # Step B 固有のセッションステートをここで初期化 (SRP)
    if 'df_flagged_B' not in st.session_state: st.session_state.df_flagged_B = pd.DataFrame()
    if 'suggestions_B' not in st.session_state: st.session_state.suggestions_B = []
    if 'chosen_analysis_list' not in st.session_state: st.session_state.chosen_analysis_list = []
    
    st.header("Step 1: フラグ付きCSVのアップロード")
    uploaded_flagged_file = st.file_uploader("フラグ付け済みCSVファイルをアップロード", type=['csv'], key="step_b_uploader")
    
    analysis_prompt_B = st.text_area(
        "（任意）追加の分析指示:", 
        placeholder="例: 特定の市区町村（広島市など）と観光施設の相関関係を深掘りしたい。",
        key="step_b_prompt"
    )

    if uploaded_flagged_file:
        try:
            uploaded_flagged_file.seek(0)
            df_flagged = pd.read_csv(uploaded_flagged_file, encoding="utf-8-sig")
            st.session_state.df_flagged_B = df_flagged # L1017: Step C のためにセッションに保存
            st.success(f"ファイル「{uploaded_flagged_file.name}」読込完了")
            st.dataframe(df_flagged.head())

            if st.button("💡 分析手法を提案させる", key="suggest_button_B"):
                with st.spinner("データ構造と指示内容を分析し、手法を提案中..."):
                    base_suggestions = suggest_analysis_techniques(df_flagged)
                    
                    ai_suggestions = []
                    if analysis_prompt_B.strip():
                        # L1028: キャッシュ利用版 (llm引数削除)
                        ai_suggestions = get_suggestions_from_prompt(
                            analysis_prompt_B, df_flagged, base_suggestions
                        )

                    base_suggestion_names = {s['name'] for s in base_suggestions} 
                    filtered_ai_suggestions = [
                        s for s in ai_suggestions if s['name'] not in base_suggestion_names 
                    ]
                    all_suggestions = sorted(base_suggestions + filtered_ai_suggestions, key=lambda x: x['priority']) 
                    st.session_state.suggestions_B = all_suggestions
                    # L1041: 提案時に古いCの結果をクリア (KISS)
                    st.session_state.step_c_results = {}
                    st.session_state.ai_summary_prompt = None
                    st.session_state.ai_summary_result = None

            if 'suggestions_B' in st.session_state and st.session_state.suggestions_B:
                display_suggestions(st.session_state.suggestions_B, df_flagged)
            
            # L1070-L1077: 致命的バグ (NameError) 修正
            # L1070 (旧 L1418) の if st.button(...) ブロック全体を削除

        except Exception as e:
            logger.error(f"ファイル読込/分析提案中にエラー: {e}", exc_info=True)
            st.error(f"ファイル読込/分析提案中にエラー: {e}")

# --- L1078: Main (アプリケーション実行) ---
def main():
    """Streamlitアプリケーションのメイン実行関数"""
    st.set_page_config(page_title="AI Data Analysis App", layout="wide")
    
    # L1082: グローバルなセッションステートのみ初期化 (SRP)
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'A' # 初期ステップ
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []

    # L1090-L1099 (旧 L1377-L1385): ステップ固有の初期化を削除 (SRP)

    with st.sidebar:
        st.title("Navigation")
        st.markdown("---")
        
        st.header("⚙️ AI 設定")
        google_api_key = st.text_input("Google API Key", type="password", key="api_key_global")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # L1109: APIキーがない場合の警告を強化 (KISS)
        if not os.getenv("GOOGLE_API_KEY"):
            st.warning("AI機能を利用するには Google APIキー を設定してください。")
        else:
            # L1113: アプリ起動時にLLMとspaCyのロードを試みる (KISS)
            if get_llm() is None:
                st.error("LLMの初期化に失敗。APIキーが正しいか確認してください。")
            if load_spacy_model() is None:
                st.error("spaCyモデルのロードに失敗。Dockerイメージを再確認してください。")
        
        st.markdown("---")
        
        st.header("🔄 Step 選択")
        current_step = st.session_state.current_step
        
        if st.button("Step A: タグ付け", key="nav_A", use_container_width=True, type=("primary" if current_step == 'A' else "secondary")):
            if st.session_state.current_step != 'A':
                st.session_state.current_step = 'A'; st.rerun()

        if st.button("Step B: 分析手法提案", key="nav_B", use_container_width=True, type=("primary" if current_step == 'B' else "secondary")):
            if st.session_state.current_step != 'B':
                st.session_state.current_step = 'B'; st.rerun()

    # --- ステップに応じて描画関数を呼び出し ---
    if st.session_state.current_step == 'A':
        render_step_a()
    elif st.session_state.current_step == 'B':
        render_step_b()
    elif st.session_state.current_step == 'C': 
        render_step_c() 

if __name__ == "__main__":
    main()