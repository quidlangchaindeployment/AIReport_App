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
FILTER_SLEEP_TIME = 4.1  # 15 RPM (60s / 15)
TAGGING_BATCH_SIZE = 10
TAGGING_SLEEP_TIME = 4.1  # 15 RPM

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
                        row_result[key] = processed_values
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
        if len(flag_cols) >= 2:
            potential_suggestions.append({
                "priority": 3, "name": "共起ネットワーク分析",
                "description": "テキスト内で同時に出現するキーワード（例: 「広島市」と「厳島神社」）の関係性を線で結び、どの単語が中心的な役割を果たしているかを可視化します。",
                "reason": f"複数のキーワード列({len(flag_cols)}個)あり。単語間の隠れたつながりを発見できます。",
                "suitable_cols": flag_cols
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
        
        for suggestion in selected_suggestions:
            with st.expander(f"{suggestion['name']} (優先度: {suggestion['priority']})"):
                st.markdown(f"**<説明>**\n{suggestion['description']}")
                st.markdown(f"**<提案理由>**\n{suggestion['reason']}")
    
    st.markdown("---")

    # L525: キー名変更済みのボタン (execute_button_C_v2)
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
あなたは優秀なデータアナリストです。
以下の「分析対象データの概要」と「個別分析の結果サマリー」を読み解き、プロの視点から総合的な「分析サマリーレポート」を作成してください。
# 指示:
1. 各分析結果を横断的に解釈し、重要なインサイト（洞察）を抽出する。
2. 単なる結果の羅列ではなく、ビジネス上の示唆（例: どのキーワードが重要か、どの属性に注目すべきか）を導き出す。
3. レポートは日本のビジネスマン向けに、見やすいマークダウン形式（見出し、箇条書き）で構成する。
4. 結論から先に述べ、その後に詳細な根拠を説明する。
---
[分析コンテキスト]
{context_str}
---
[あなたの回答]
# 分析サマリーレポート
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

def run_timeseries(df, suitable_cols_dict):
    """時系列分析を実行し、Streamlitで可視化する"""
    if not isinstance(suitable_cols_dict, dict) or 'datetime' not in suitable_cols_dict or 'keywords' not in suitable_cols_dict:
        st.warning("時系列分析のための列情報（datetime, keywords）が不十分です。")
        return None #
        
    dt_cols = [col for col in suitable_cols_dict['datetime'] if col in df.columns]
    kw_cols = [col for col in suitable_cols_dict['keywords'] if col in df.columns]

    if not dt_cols: st.error("日時列が見つかりません。"); return None #
    if not kw_cols: st.error("キーワード列が見つかりません。"); return None #

    key_base = dt_cols[0]
    dt_col = st.selectbox("使用する日時列:", dt_cols, key=f"ts_dt_{key_base}")
    kw_col = st.selectbox("集計するキーワード列:", kw_cols, key=f"ts_kw_{key_base}")

    if not dt_col or not kw_col:
        return None #

    try:
        df_copy = df[[dt_col, kw_col]].copy()
        
        df_copy[dt_col] = pd.to_datetime(df_copy[dt_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[dt_col])
        if df_copy.empty: st.info("有効な日時データがありません。"); return None #

        df_copy[kw_col] = df_copy[kw_col].astype(str)
        df_copy = df_copy[df_copy[kw_col].str.strip() != ''] 
        if df_copy.empty: st.info(f"「{kw_col}」に有効なキーワードがありませんでした。"); return None #

        time_df = df_copy.set_index(dt_col).resample('D').size().rename("投稿数")
        
        if time_df.empty: st.info("時系列集計の結果、データがありませんでした。"); return None #
        
        time_df.index.name = "日時"
        
        st.line_chart(time_df)
        with st.expander("詳細データ"):
            st.dataframe(time_df)
        
        return time_df # 
            
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

    nlp = load_spacy_model() # キャッシュされたモデルを直接呼び出し
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
            '的', '的', '的', '的', '人', '自分', '私', '僕', '何', 'その', 'この', 'あの'
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

# --- L752: Part 2 (render関数, main) は次のチャットで提案します ---
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
                    st.warning("「共起ネットワーク分析」は現在実装中です。") #
                elif name == "カテゴリ別集計（グループ比較）":
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
                                         
                                     # L874: 致命的バグ (NameError) 修正
                                     # L874 (旧) を L871 の前に移動
                                     result_df = df_copy.groupby(existing_grouping_cols)[numeric_cols_to_desc].describe()
                                     
                                     flat_cols = []
                                     for col in result_df.columns:
                                         flat_cols.append(f"{col[0]}_{col[1]}") 
                                     result_df.columns = flat_cols
                                     
                                     final_result_df = result_df.reset_index()
                                     st.dataframe(final_result_df) 
                                     result_data = final_result_df # 
                                 except Exception as group_e:
                                     st.error(f"グループ別集計エラー: {group_e}")
                                     logger.error(f"Groupby describe error: {group_e}", exc_info=True)
                    else:
                         st.warning(f"「{name}」の列定義が不適切です: {cols}")
                elif name == "時系列キーワード分析":
                    if isinstance(cols, dict) and 'datetime' in cols and 'keywords' in cols:
                        result_data = run_timeseries(df, cols)
                    else:
                         st.warning(f"「{name}」の列定義が不適切です: {cols}")
                elif name == "テキストマイニング（頻出単語など）":
                    if cols and isinstance(cols, list) and cols[0] == 'ANALYSIS_TEXT_COLUMN':
                        result_data = run_text_mining(df, 'ANALYSIS_TEXT_COLUMN')
                    else:
                        st.warning(f"「{name}」の列定義が不適切です: {cols}")
                elif name == "主成分分析 (PCA) / 因子分析":
                    st.warning("「主成分分析」は現在実装中です。")
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