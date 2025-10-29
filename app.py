import streamlit as st
import pandas as pd
import numpy as np
import spacy
from logging import getLogger, StreamHandler, DEBUG, Formatter, Handler, LogRecord
import re
from io import BytesIO
import os
import json
import sys
import time
from collections import deque

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

try:
    from geography_db import JAPAN_GEOGRAPHY_DB
except ImportError:
    JAPAN_GEOGRAPHY_DB = None

# --- ロガー設定 ---
logger = getLogger(__name__)
logger.setLevel(DEBUG)

# --- UIログ表示用カスタムハンドラ ---
class StreamlitLogHandler(Handler):
    def __init__(self, max_lines=15):
        super().__init__()
        # セッションステート 'log_messages' がなければ初期化
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = deque(maxlen=max_lines)

    def emit(self, record: LogRecord):
        log_entry = self.format(record)
        # セッションステート 'log_messages' があれば追記
        if 'log_messages' in st.session_state:
            st.session_state.log_messages.append(log_entry)
# ------------------------------------

# --- LLM初期化関数 ---
def get_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Google APIキーが設定されていません。", file=sys.stderr)
        return None
    try:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
    except Exception as e:
        print(f"LLM (gemini-2.5-flash-lite) の初期化に失敗: {e}", file=sys.stderr)
        # エラー発生時はログに詳細を記録 (UIには表示しない)
        logger.error(f"LLM (gemini-2.5-flash-lite) initialization failed: {e}", exc_info=True)
        return None

# --- spaCyモデルの読み込み ---
@st.cache_resource # Streamlitのキャッシュ機能を利用
def load_spacy_model():
    """spaCyの日本語モデルを読み込む"""
    try:
        nlp = spacy.load("ja_core_news_sm") # モデル名を確認
    except OSError:
        st.info("初回起動: 日本語NLPモデルをダウンロードします...")
        try:
            from spacy.cli import download
            download("ja_core_news_sm")
            nlp = spacy.load("ja_core_news_sm")
        except Exception as e:
            st.error(f"spaCyモデルのダウンロード/ロードに失敗: {e}")
            logger.error(f"spaCy model download/load failed: {e}", exc_info=True)
            nlp = None # 失敗した場合はNoneを返す
    return nlp


# --- データ読み込み & 前処理ツール ---
def load_and_preprocess_data(uploaded_file, text_column):
    """アップロードされた「1つの」ファイルを読み込み、前処理を行う"""
    logger.info(f"ファイル {uploaded_file.name} 読込・前処理開始...")
    df = None
    try:
        uploaded_file.seek(0) # ファイルポインタを先頭に
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.warning(f"サポート外形式: {uploaded_file.name}"); return None, None
    except Exception as e:
        st.error(f"ファイル {uploaded_file.name} 読込失敗: {e}"); logger.error(f"File reading failed: {e}", exc_info=True); return None, None # トレースバック追加

    if df is None: st.error(f"ファイル {uploaded_file.name} 読込失敗"); return None, None
    if text_column not in df.columns: st.error(f"ファイル {uploaded_file.name} に列 '{text_column}' なし"); return None, None

    # テキスト列の欠損値処理と空行チェック
    df.dropna(subset=[text_column], inplace=True)
    if df.empty: logger.warning(f"{uploaded_file.name} はテキスト列 '{text_column}' の欠損値除去後、0行"); return None, []

    df = df.reset_index(drop=True)
    metadata_columns = [col for col in df.columns if col != text_column]

    # 型変換 (エラーを無視し、日付変換は試行)
    for col in metadata_columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
        if df[col].dtype == 'object':
            try: df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError, OverflowError): pass # エラーが出ても処理を続ける

    logger.info(f"ファイル {uploaded_file.name} 処理完了 ({len(df)} 行)")
    return df, metadata_columns


# --- データ構造分析ツール ---
def analyze_data_structure(df):
    """DataFrameの構造を分析し、サマリーを生成する"""
    logger.info("データ構造分析開始...")
    if df is None or df.empty:
        logger.warning("分析対象のDataFrameが空です。")
        return {"total_rows": 0, "total_columns": 0, "column_details": {}}

    structure_info = {"total_rows": len(df), "total_columns": len(df.columns)}
    column_details = {}
    for col in df.columns:
        try: # 列ごとの処理でエラーが発生しても継続
             column_details[col] = {"type": str(df[col].dtype), "unique_values": df[col].nunique(), "missing_values": int(df[col].isnull().sum())} # int()追加
        except Exception as e:
             logger.error(f"列 '{col}' の分析中にエラー: {e}")
             column_details[col] = {"type": "Error", "unique_values": -1, "missing_values": -1}

    structure_info["column_details"] = column_details
    logger.info("データ構造分析完了")
    return structure_info

# --- AIによる動的カテゴリ定義 ---
def get_dynamic_categories(analysis_prompt, llm):
    """LLMを使い、指針から「抽出すべきカテゴリ」のスキーマを動的に生成する"""
    logger.info("AI動的カテゴリ定義生成開始...")
    if llm is None: return {"市区町村キーワード": "テキストが言及している日本の市区町村名。"} # LLMがない場合はデフォルトのみ
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
        logger.info(f"AIカテゴリ定義(生): {response_str}")
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not match: raise ValueError("Invalid JSON format")
        json_str = match.group(0); category_map = json.loads(json_str)
        if not isinstance(category_map, dict): raise ValueError("Response is not a dict")
        final_categories = {"市区町村キーワード": "テキストが言及している日本の市区町村名。"}
        # (★) AIが生成したキーの空白を除去
        cleaned_category_map = {k.strip(): v for k, v in category_map.items()}
        final_categories.update(cleaned_category_map)
        logger.info(f"最終カテゴリ定義: {final_categories}")
        return final_categories
    except Exception as e:
        logger.error(f"AI動的カテゴリ定義失敗: {e}", exc_info=True)
        st.warning(f"AI動的カテゴリ定義失敗: {e}")
        return {"市区町村キーワード": "テキストが言及している日本の市区町村名。"}
# AIフィルタリングロジック
def filter_relevant_data_by_ai(df_batch, analysis_prompt, llm):
    """
    AIを使い、分析指針と無関係な行をフィルタリングする (relevant: true/false)。
    """
    if llm is None:
        logger.error("filter_relevant_data_by_ai: LLMが初期化されていません。")
        st.error("AIモデルが利用できません。APIキーを確認してください。")
        return pd.DataFrame() # 空のDF (フィルタリング失敗)

    logger.debug(f"{len(df_batch)}件 AI関連性フィルタリング開始...")
    
    # テキストデータをJSONL形式に変換 (IDとテキストのみ)
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
        # レスポンス (JSONL) のパース
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
                    # AIが true/false 以外 (例: "true") で返した場合のフォールバック
                    results.append({"id": data.get("id"), "relevant": str(data.get("relevant")).lower() == 'true'})
            except json.JSONDecodeError as json_e:
                logger.warning(f"AIフィルタリング回答パース失敗: {cleaned_line} - Error: {json_e}")
                # パース失敗時は安全のため relevant=True (残す) として扱う
                id_match = re.search(r'"id":\s*(\d+)', cleaned_line)
                if id_match:
                    results.append({"id": int(id_match.group(1)), "relevant": True})
        
        return pd.DataFrame(results) if results else pd.DataFrame(columns=['id', 'relevant'])
        
    except Exception as e:
        logger.error(f"AIフィルタリングバッチ処理中エラー: {e}", exc_info=True)
        st.error(f"AIフィルタリング処理エラー: {e}")
        # エラー時は全件 True (残す) として返す
        return df_batch[['id']].copy().assign(relevant=True)

# --- AIによる直接タグ付け ---
def perform_ai_tagging(df_batch, categories_to_tag, llm, analysis_prompt=""): # (★) analysis_prompt を引数に追加
    """テキストのバッチを受け取り、AIが【指定されたカテゴリ定義】に基づいて直接タグ付けを行う"""
    if llm is None: # LLMがない場合は処理不可
        logger.error("perform_ai_tagging: LLMが初期化されていません。")
        st.error("AIモデルが利用できません。APIキーを確認してください。")
        return pd.DataFrame() # 空のDFを返す

    logger.debug(f"AI Tagging - Received categories: {json.dumps(categories_to_tag, ensure_ascii=False)}")
    logger.info(f"{len(df_batch)}件 AIタグ付け開始 (カテゴリ: {list(categories_to_tag.keys())})")
    
    # (★) 分析指針 (analysis_prompt) に基づいて、AIに渡す地名辞書を絞り込む
    relevant_geo_db = {}
    if JAPAN_GEOGRAPHY_DB:
        prompt_lower = analysis_prompt.lower()
        # (例: "広島" が指針にあれば、"広島県" と "広島市" のデータのみ渡す)
        keys_found = [
            key for key in JAPAN_GEOGRAPHY_DB.keys() 
            if any(hint in key for hint in [
                "広島", "福岡", "大阪", "東京", "北海道", "愛知", "宮城", "札幌", "横浜", "名古屋", "京都", "神戸", "仙台"
            ]) and any(hint in prompt_lower for hint in [
                "広島", "福岡", "大阪", "東京", "北海道", "愛知", "宮城", "札幌", "横浜", "名古屋", "京都", "神戸", "仙台"
            ])
        ]
        # (★) 指針から関連キーを推測 (簡易版)
        if "広島" in prompt_lower: keys_found.extend(["広島県", "広島市"])
        if "東京" in prompt_lower: keys_found.extend(["東京都", "東京23区"])
        if "大阪" in prompt_lower: keys_found.extend(["大阪府", "大阪市"])
        
        for key in set(keys_found):
            if key in JAPAN_GEOGRAPHY_DB:
                relevant_geo_db[key] = JAPAN_GEOGRAPHY_DB[key]
        
        # (★) もし何も見つからなければ (または指針が空なら)、主要都市のみ
        if not relevant_geo_db:
            logger.warning("地名辞書の絞り込みヒントなし。主要都市のみ渡します。")
            default_keys = ["東京都", "東京23区", "大阪府", "大阪市", "広島県", "広島市"]
            for key in default_keys:
                 if key in JAPAN_GEOGRAPHY_DB:
                     relevant_geo_db[key] = JAPAN_GEOGRAPHY_DB[key]
        
        geo_context_str = json.dumps(relevant_geo_db, ensure_ascii=False, indent=2)
        # (★) トークン数削減のため、5000文字を超える場合は縮小
        if len(geo_context_str) > 5000:
            logger.warning(f"地名辞書が大きすぎ ({len(geo_context_str)}B)。キーのみに縮小。")
            geo_context_str = json.dumps(list(relevant_geo_db.keys()), ensure_ascii=False)
    else:
        geo_context_str = "{}"
        
    logger.info(f"AIに渡す地名辞書(絞込済): {list(relevant_geo_db.keys())}")
    
    input_texts_jsonl = df_batch.apply(lambda row: json.dumps({"id": row['id'], "text": str(row['ANALYSIS_TEXT_COLUMN'])[:500]}, ensure_ascii=False), axis=1).tolist() # (★) str() で囲む
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
             a. 「地名辞書」の【値】(例: "呉市", "廿日市市", "広島市中区") または【キー】(例: "広島市") に一致する、最も文脈に関連性の高いものを【1つだけ】選ぶ。
             b. (例: "広島市" と "広島市中区" が両方言及されていれば、より詳細な "広島市中区" を優先する)
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
        # (★) 正しい invoke_params を定義
        invoke_params = {
            "categories": json.dumps(categories_to_tag, ensure_ascii=False), 
            "geo_context": geo_context_str, # (★) 変更
            "text_data_jsonl": "\n".join(input_texts_jsonl),
            "analysis_prompt": analysis_prompt # (★) 追加
        }
        
        # (★) invoke_params の定義直後から、そのまま AI 呼び出し処理を続ける
        logger.debug(f"AI Tagging - Invoking LLM...")
        logger.info(f"Attempting AI call for ID: {df_batch.iloc[0]['id']}...")
        response_str = chain.invoke(invoke_params)
        logger.debug(f"AI Tagging - Raw response received.")
        logger.info(f"AI応答受信完了。")
        results = []
        match = re.search(r'```(?:jsonl|json)?\s*([\s\S]*?)\s*```', response_str, re.DOTALL)
        jsonl_content = match.group(1).strip() if match else response_str.strip()
        logger.debug(f"AI Tagging - Cleaned JSONL ready.")
        expected_keys = list(categories_to_tag.keys())
        for line in jsonl_content.strip().split('\n'):
            cleaned_line = line.strip();
            if not cleaned_line: continue
            try:
                data = json.loads(cleaned_line); row_result = {"id": data.get("id")}
                tag_source = data.get("categories") if isinstance(data.get("categories"), dict) else data
                for key in expected_keys:
                    # (★) AIが返すキーの揺らぎに対応するため、空白除去して比較
                    found_key = None
                    for resp_key in tag_source.keys():
                        if resp_key.strip() == key:
                            found_key = resp_key
                            break
                    raw_value = tag_source.get(found_key) if found_key else None # 見つかったキーで値を取得

                    # (★) --- 新しいロジック (ここから) ---
                    if key == "市区町村キーワード":
                        # (★) "市区町村キーワード" は【単一文字列】として処理
                        processed_value = "" # デフォルトは空文字列
                        if isinstance(raw_value, list) and raw_value:
                            # AIが指示に反してリストで返した場合、最初の有効な値
                            processed_value = str(raw_value[0]).strip()
                        elif raw_value is not None and str(raw_value).strip():
                            # AIが指示通り単一文字列で返した場合
                            processed_value = str(raw_value).strip()
                        
                        # "該当なし" 等のAIの返答を空文字列に正規化
                        if processed_value.lower() in ["該当なし", "none", "null", ""]:
                            row_result[key] = "" 
                        else:
                            row_result[key] = processed_value
                    
                    else:
                        # (★) "市区町村キーワード" 以外のキーは【リスト】として処理 (従来のロジック)
                        processed_values = [] 
                        if isinstance(raw_value, list):
                            processed_values = sorted(list(set(str(val).strip() for val in raw_value if str(val).strip())))
                        elif raw_value is not None and str(raw_value).strip():
                            processed_values = [str(raw_value).strip()]
                        
                        row_result[key] = processed_values
                    # (★) --- 新しいロジック (ここまで) ---
                    
                results.append(row_result)
            except json.JSONDecodeError as json_e:
                logger.warning(f"AI回答JSONLパース失敗: {cleaned_line} - Error: {json_e}")
                try: # IDだけでも取得試行
                    id_match = re.search(r'"id":\s*(\d+)', cleaned_line)
                    if id_match:
                         failed_id = int(id_match.group(1)); empty_result = {"id": failed_id}
                         for key in expected_keys: empty_result[key] = []
                         results.append(empty_result); logger.warning(f"パース失敗行 ID:{failed_id} 補完")
                    else: logger.error(f"パース失敗行 ID抽出不可: {cleaned_line}")
                except Exception as id_extract_e: logger.error(f"ID抽出中エラー: {id_extract_e}")
                continue
        logger.info(f"{len(results)}件タグ付け結果パース完了。")
        if results: logger.debug(f"AI Tagging - Parsed sample: {results[0]}")
        return pd.DataFrame(results) if results else pd.DataFrame(columns=['id'] + expected_keys)
    except Exception as e:
        logger.error(f"AIタグ付けバッチ処理中エラー: {e}", exc_info=True)
        st.error(f"AIバッチ処理エラー: {e}")
        return pd.DataFrame()

# --- 分析手法提案ロジック ---
# --- (★) 分析手法提案ロジック (提案拡充版) ---
# --- (★) 分析手法提案ロジック (提案拡充版) ---
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

        # (★) 優先度3: 共起ネットワーク分析 (新規追加)
        if len(flag_cols) >= 2:
            potential_suggestions.append({
                "priority": 3, "name": "共起ネットワーク分析",
                "description": "テキスト内で同時に出現するキーワード（例: 「広島市」と「厳島神社」）の関係性を線で結び、どの単語が中心的な役割を果たしているかを可視化します。",
                "reason": f"複数のキーワード列({len(flag_cols)}個)あり。単語間の隠れたつながりを発見できます。",
                "suitable_cols": flag_cols
            })
        

        # (★) 優先度4: グループ比較 (優先度変更 3 -> 4)
        if numeric_cols and flag_cols:
            potential_suggestions.append({
                "priority": 4, "name": "カテゴリ別集計（グループ比較）",
                "description": f"キーワードカテゴリ（{flag_cols[0]}など）ごとに数値データ({numeric_cols[0]}など)の平均値や合計値に差があるか比較します。",
                "reason": f"キーワード列と数値列({len(numeric_cols)}個)あり、グループ間の特徴比較に。",
                "suitable_cols": {"numeric": numeric_cols, "grouping": flag_cols}
            })

        # (★) 優先度5: 時系列分析 (優先度変更 4 -> 5)
        if datetime_cols and flag_cols:
             potential_suggestions.append({
                "priority": 5, "name": "時系列キーワード分析",
                "description": f"特定のキーワードの出現数が時間（{datetime_cols[0]}など）とともにどう変化したかトレンドを可視化します。",
                "reason": f"キーワード列と日時列({len(datetime_cols)}個)あり、時間変化の把握に。",
                "suitable_cols": {"datetime": datetime_cols, "keywords": flag_cols}
            })

        # (★) 優先度6: テキストマイニング (優先度変更 5 -> 6)
        potential_suggestions.append({
            "priority": 6, "name": "テキストマイニング（頻出単語など）",
            "description": "原文テキストから頻出する単語を抽出し、どのような言葉が多く使われているか全体像を把握します。",
            "reason": "原文テキストがあり、タグ付け以外の観点からのインサイト発見に。",
            "suitable_cols": ['ANALYSIS_TEXT_COLUMN']
        })

        # (★) 優先度7: 多変量解析 (優先度変更 6 -> 7)
        if len(numeric_cols) >= 3:
             potential_suggestions.append({
                 "priority": 7, "name": "主成分分析 (PCA) / 因子分析",
                 "description": f"複数の数値データ({', '.join(numeric_cols)})間の相関関係から、背後にある共通の要因（主成分/因子）を探ります。",
                 "reason": f"複数数値列({len(numeric_cols)}個)があり、変数間の複雑な関係性の縮約や解釈に。",
                 "suitable_cols": numeric_cols
             })

        # 優先度でソートし、上位8件程度を返す
        suggestions = sorted(potential_suggestions, key=lambda x: x['priority'])
        logger.info(f"提案手法(ソート後): {[s['name'] for s in suggestions]}")
        return suggestions[:8] # (★) 上限を 8 に変更

    except Exception as e:
        logger.error(f"分析手法提案中にエラー: {e}", exc_info=True); st.warning(f"分析手法提案中にエラー: {e}")
    return suggestions

# (★ ここから新規追加)
def get_suggestions_from_prompt(user_prompt, llm, df, existing_suggestions):
    """
    ユーザーの自由記述プロンプトとデータ構造に基づき、AIが追加の分析手法を提案する。
    """
    logger.info("AIプロンプトベースの分析提案を開始...")
    if llm is None:
        logger.error("get_suggestions_from_prompt: LLMが初期化されていません。")
        return []
    
    try:
        # AIに渡すためのデータ構造サマリーを作成
        col_info = []
        for col in df.columns:
            col_info.append(f"- {col} (型: {df[col].dtype})")
        column_info_str = "\n".join(col_info)
        
        # 既存の提案名リスト
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
        
        # 回答 (JSONリスト形式のみ):
        """
    )
        
        chain = prompt | llm | StrOutputParser()
        response_str = chain.invoke({
            "column_info": column_info_str,
            "user_prompt": user_prompt,
        })
        
        logger.info(f"AI追加提案(生): {response_str}")
        
        # AIの応答 (JSONリスト) をパース
        match = re.search(r'\[.*\]', response_str, re.DOTALL)
        if not match:
            logger.warning("AIがJSONリスト形式で応答しませんでした。")
            return []
            
        json_str = match.group(0)
        ai_suggestions = json.loads(json_str)
        
        # priorityが6未満のものは、既存の提案と競合しないよう調整
        # (★) AIが生成した提案の優先度を 1 (最高) に設定
        for suggestion in ai_suggestions:
            suggestion['priority'] = 6 # ユーザー指示を最優先
                
        logger.info(f"AI追加提案(パース済): {len(ai_suggestions)}件")
        return ai_suggestions
        
    except Exception as e:
        logger.error(f"AI追加提案の生成中にエラー: {e}", exc_info=True)
        st.warning(f"AI追加提案の生成中にエラーが発生しました: {e}")
        return []

# (★ ここから Step C 用の新規関数群)

def run_simple_count(df, flag_cols):
    """単純集計（頻度分析）を実行し、Streamlitで可視化する"""
    if not flag_cols:
        st.warning("集計対象のキーワード列（suitable_cols）が見つかりません。")
        return
    
    # ユーザーに集計対象の列を1つ選んでもらう
    col_to_analyze = st.selectbox(
        "集計するキーワード列を選択:", 
        flag_cols, 
        key=f"sc_select_{flag_cols[0]}" # (★) キーをユニークにする
    )
    
    if not col_to_analyze or col_to_analyze not in df.columns:
        st.error(f"列 '{col_to_analyze}' がデータに存在しません。")
        return

    # カンマ区切りの文字列を個別の行に分解
    try:
        # (★) .strアクセス前に .astype(str) を挟み、数値型などでのエラーを回避
        s = df[col_to_analyze].astype(str).str.split(', ').explode()
        s = s[s.str.strip() != ''] # 空白を除去
        s = s.str.strip() # 前後の空白を除去
        
        if s.empty:
            st.info("集計対象のキーワードがありませんでした。")
            return
            
        counts = s.value_counts().head(20) # 上位20件
        st.bar_chart(counts)
        with st.expander("詳細データ（上位20件）"):
            st.dataframe(counts)
            
        return counts # (★) 正常終了時に集計結果を返す
            
    except Exception as e:
        st.error(f"単純集計の処理中にエラー: {e}")
        logger.error(f"run_simple_count error: {e}", exc_info=True)
    return None # (★) エラー時は None を返す


def run_basic_stats(df, numeric_cols):
    """基本統計量を実行し、Streamlitで表示する"""
    if not numeric_cols:
        st.warning("集計対象の数値列（suitable_cols）が見つかりません。")
        return
    
    # 存在する列のみを対象
    existing_cols = [col for col in numeric_cols if col in df.columns]
    if not existing_cols:
        st.error("指定された数値列がデータに存在しません。")
        return
        
    stats_df = df[existing_cols].describe() # (★) 結果を一旦変数に
    st.dataframe(stats_df)
    return stats_df # (★) 結果を返す


def run_crosstab(df, suitable_cols):
    """クロス集計を実行し、Streamlitで表示する"""
    if not suitable_cols or len(suitable_cols) < 2:
        st.warning("クロス集計には2つ以上の列が必要です。")
        return

    # 存在する列のみを対象
    existing_cols = [col for col in suitable_cols if col in df.columns]
    if len(existing_cols) < 2:
        st.error(f"データ内に存在する分析対象列が2つ未満です: {existing_cols}")
        return

    st.info(f"分析可能な列: {', '.join(existing_cols)}")
    
    # (★) キーが重複しないよう、suitable_cols[0] をキーに含める
    key_base = suitable_cols[0]
    col1 = st.selectbox("行 (Index) に設定する列:", existing_cols, key=f"ct_idx_{key_base}")
    
    # col1 以外の列を col2 の候補とする
    options_col2 = [c for c in existing_cols if c != col1]
    if not options_col2:
        st.error("2つ目の列を選択できません。")
        return
        
    col2 = st.selectbox("列 (Column) に設定する列:", options_col2, key=f"ct_col_{key_base}")

    if not col1 or not col2:
        return

    try:
        # (★) 注意: この実装はカンマ区切りの文字列を「そのまま」集計します (例: "広島市, 呉市")
        # これを分解（explode）すると組み合わせ爆発が起きるため、まず簡易版として実装
        
        # (★) .astype(str) を挟んで安全に処理
        crosstab_df = pd.crosstab(df[col1].astype(str), df[col2].astype(str))
        
        if crosstab_df.empty:
            st.info("クロス集計の結果、データがありませんでした。")
            return
        
        st.dataframe(crosstab_df)
        
        # (★) ヒートマップ表示 (オプション)
        if st.checkbox("ヒートマップで表示", key=f"ct_heatmap_{key_base}"):
             try:
                 import altair as alt
                 # データフレームをAltairが扱える形式（long format）に変換
                 ct_long = crosstab_df.stack().reset_index().rename(columns={0: 'count'})
                 heatmap = alt.Chart(ct_long).mark_rect().encode(
                     x=alt.X(col2, type='ordinal', title=col2), # (★) f-string形式をやめ、type='ordinal' で型を指定
                     y=alt.Y(col1, type='ordinal', title=col1), # (★) f-string形式をやめ、type='ordinal' で型を指定
                     color=alt.Color('count', type='quantitative', title='Count', scale=alt.Scale(range='heatmap')), # (★) 同様に変更
                     tooltip=[col1, col2, 'count']
                 ).properties(
                     title=f"クロス集計: {col1} vs {col2}"
                 ).interactive() # ズームやパンを可能にする
                 st.altair_chart(heatmap, use_container_width=True)
             except ImportError:
                 st.warning("ヒートマップ表示には `altair` が必要です。`st.dataframe` で表示します。")
             except Exception as he:
                 st.warning(f"ヒートマップ描画エラー: {he}。`st.dataframe` で表示します。")
        return crosstab_df # (★) 正常終了時に集計結果を返す
    except Exception as e:
        st.error(f"クロス集計の処理中にエラー: {e}")
        logger.error(f"run_crosstab error: {e}", exc_info=True)
    return None # (★) エラー時は None を返す


def run_timeseries(df, suitable_cols_dict):
    """時系列分析を実行し、Streamlitで可視化する"""
    if not isinstance(suitable_cols_dict, dict) or 'datetime' not in suitable_cols_dict or 'keywords' not in suitable_cols_dict:
        st.warning("時系列分析のための列情報（datetime, keywords）が不十分です。")
        return
        
    dt_cols = [col for col in suitable_cols_dict['datetime'] if col in df.columns]
    kw_cols = [col for col in suitable_cols_dict['keywords'] if col in df.columns]

    if not dt_cols: st.error("日時列が見つかりません。"); return
    if not kw_cols: st.error("キーワード列が見つかりません。"); return

    # (★) キーが重複しないよう、dt_cols[0] をキーに含める
    key_base = dt_cols[0]
    dt_col = st.selectbox("使用する日時列:", dt_cols, key=f"ts_dt_{key_base}")
    kw_col = st.selectbox("集計するキーワード列:", kw_cols, key=f"ts_kw_{key_base}")

    if not dt_col or not kw_col:
        return

    try:
        df_copy = df[[dt_col, kw_col]].copy()
        
        # 日時列を変換
        df_copy[dt_col] = pd.to_datetime(df_copy[dt_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[dt_col])
        if df_copy.empty: st.info("有効な日時データがありません。"); return

        # (★) キーワード列を .astype(str) に
        df_copy[kw_col] = df_copy[kw_col].astype(str)
        
        # キーワード列が空でない行のみをカウント
        df_copy = df_copy[df_copy[kw_col].str.strip() != ''] 
        if df_copy.empty: st.info(f"「{kw_col}」に有効なキーワードがありませんでした。"); return

        # 日(Day)単位でリサンプリングしてカウント
        # (★) .resample('D') の前に .set_index() が必要
        time_df = df_copy.set_index(dt_col).resample('D').size().rename("投稿数")
        
        if time_df.empty: st.info("時系列集計の結果、データがありませんでした。"); return
            
        st.line_chart(time_df)
        with st.expander("詳細データ"):
            st.dataframe(time_df)
            
        return time_df # (★) 正常終了時に集計結果を返す
            
    except Exception as e:
        st.error(f"時系列分析の処理中にエラー: {e}")
        logger.error(f"run_timeseries error: {e}", exc_info=True)
    return None # (★) エラー時は None を返す

def run_text_mining(df, text_col='ANALYSIS_TEXT_COLUMN'):
    """
    spaCyを使用してテキストマイニング（頻出単語分析）を実行し、可視化する。
    APIは使用しない。
    """
    if text_col not in df.columns or df[text_col].empty:
        st.warning(f"分析対象のテキスト列 '{text_col}' がないか、空です。")
        return

    # Step A でロード済みのspaCyモデルをセッションから取得
    if 'nlp' not in st.session_state or st.session_state.nlp is None:
        # もしStep AをスキップしてStep B/Cに来た場合、モデルをロード
        st.session_state.nlp = load_spacy_model()
        if st.session_state.nlp is None:
            st.error("spaCy日本語モデルのロードに失敗しました。Step Aからやり直してください。")
            return
            
    nlp = st.session_state.nlp
    
    st.info("テキストマイニング処理中（データ量によって時間がかかる場合があります）...")

    try:
        # (★) 欠損値を除外し、文字列に変換
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            st.warning("分析対象のテキストがありません。")
            return
            
        words = []
        # (★) nlp.pipe で高速処理
        # 品詞 (pos_) が 名詞(NOUN), 固有名詞(PROPN), 形容詞(ADJ) のみ抽出
        target_pos = {'NOUN', 'PROPN', 'ADJ'}
        
        # ストップワード（一般的すぎる単語）のリスト (簡易版)
        stop_words = {
            'の', 'に', 'は', 'を', 'が', 'で', 'て', 'です', 'ます', 'こと', 'もの', 'それ', 'あれ',
            'これ', 'ため', 'いる', 'する', 'ある', 'ない', 'いう', 'よう', 'そう', 'など', 'さん',
            '的', '的', '的', '的', '人', '自分', '私', '僕', '何', 'その', 'この', 'あの'
        }

        # nlp.pipe でバッチ処理
        for doc in nlp.pipe(texts, disable=["parser", "ner"]): # 構文解析と固有表現抽出は不要
            for token in doc:
                # (★) 見出し語(lemma_)を使い、品詞チェック、ストップワード除外、1文字除外
                if (token.pos_ in target_pos) and (not token.is_stop) and (token.lemma_ not in stop_words) and (len(token.lemma_) > 1):
                    words.append(token.lemma_)

        if not words:
            st.warning("抽出可能な有効な単語が見つかりませんでした。")
            return

        # Pandasで頻度集計
        word_counts = pd.Series(words).value_counts().head(30) # 上位30件

        st.subheader("頻出単語 Top 30")
        st.bar_chart(word_counts)
        with st.expander("詳細データ（Top 30）"):
            st.dataframe(word_counts.reset_index(name="出現回数").rename(columns={"index": "単語"}))        
        return word_counts # (★) 正常終了時に集計結果を返す

    except Exception as e:
        st.error(f"テキストマイニング処理中にエラー: {e}")
        logger.error(f"run_text_mining error: {e}", exc_info=True)
    return None # (★) エラー時は None を返す

def generate_ai_summary_prompt(results_dict, df):
    """
    Step C-1 で得られた分析結果(DataFrame)をAI用のプロンプトに変換する。
    """
    logger.info("AIサマリー用プロンプトの生成開始...")
    if not results_dict:
        logger.warning("AIサマリーの元になる分析結果がありません。")
        return "エラー: AIサマリーの元になる分析結果がありません。Step C-1を先に実行してください。"
    
    # データ全体の概要
    context_str = f"## 分析対象データの概要\n"
    context_str += f"- 総行数: {len(df)}\n"
    context_str += f"- 列リスト: {', '.join(df.columns.tolist())}\n\n"
    context_str += "## 個別分析の結果サマリー\n"
    context_str += "（注：トークン数節約のため、各分析結果は最大5件のみ抜粋しています）\n\n"
    
    # 各分析結果を文字列に変換
    for name, data in results_dict.items():
        context_str += f"### {name}\n"
        if isinstance(data, (pd.DataFrame, pd.Series)):
            if data.empty:
                context_str += "(データなし)\n\n"
            else:
                # (★) レートリミット/トークン数節約のため、.head(5) と .info() だけ渡す
                if len(data) > 5:
                    context_str += f"上位5件:\n{data.head(5).to_string()}\n\n"
                else:
                    context_str += f"全件:\n{data.to_string()}\n\n"
        else:
            context_str += f"{str(data)}\n\n"
    
    # AIへの最終的な指示プロンプト
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

def render_step_c():
    """Step C: 分析結果の可視化を描画する"""
    st.title("🔬 分析結果の可視化 (Step C)")
    
    # (★) Step C-2 (AIサマリー) 用のセッションステートを初期化
    if 'step_c_results' not in st.session_state: st.session_state.step_c_results = {}
    if 'ai_summary_prompt' not in st.session_state: st.session_state.ai_summary_prompt = None
    if 'ai_summary_result' not in st.session_state: st.session_state.ai_summary_result = None
    
    # --- 必要なデータがセッションにあるか確認 ---
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
    
    # 選択された分析手法の「完全な定義（suitable_cols含む）」を取得
    analyses_to_run = [s for s in all_suggestions if s['name'] in selected_names]
    
    st.info(f"**実行する分析:** {', '.join(selected_names)}")
    st.markdown("---")
    # (★) AIサマリー生成に備え、結果をリセット
    st.session_state.step_c_results = {}
    # --- 各分析手法をループして実行・描画 ---
    for suggestion in analyses_to_run:
        name = suggestion['name']
        cols = suggestion.get('suitable_cols', []) # (★) 提案ロジックから渡された列
        
        with st.container(border=True):
            st.subheader(f"📈 分析結果: {name}")
            
            try:
                result_data = None # (★) 結果格納用
                if name == "単純集計（頻度分析）":
                    result_data = run_simple_count(df, cols) # (★)
                elif name == "基本統計量":
                    result_data = run_basic_stats(df, cols) # (★)
                elif name == "クロス集計（キーワード間）":
                    result_data = run_crosstab(df, cols) # (★)
                elif name == "クロス集計（キーワード×属性）":
                    result_data = run_crosstab(df, cols) # (★)
                elif name == "カテゴリ別集計（グループ比較）":
                    # (★) suitable_cols が dict {"numeric": [], "grouping": []} のはず
                    if isinstance(cols, dict) and 'numeric' in cols and 'grouping' in cols:
                         # (★) GroupBy オブジェクトに .describe() を直接呼び出す
                         grouping_cols = cols['grouping']
                         numeric_cols_to_desc = [col for col in cols['numeric'] if col in df.columns]
                         
                         if not numeric_cols_to_desc:
                             st.warning("分析対象の数値列がデータにありません。")
                         elif not grouping_cols:
                             st.warning("分析対象のグループ列がありません。")
                         else:
                             # (★) 複数のグループ列に対応するため、リストであることを確認
                             if not isinstance(grouping_cols, list):
                                 grouping_cols = [grouping_cols]
                                 
                             # グループ列がデータに存在するか確認
                             existing_grouping_cols = [col for col in grouping_cols if col in df.columns]
                             if not existing_grouping_cols:
                                 st.warning(f"グループ化列 {grouping_cols} がデータに存在しません。")
                             else:
                                try:
                                     # (★) グループ列の型を文字列に変換（安全のため） - このブロックを戻す
                                     df_copy = df.copy()
                                     for col in existing_grouping_cols:
                                         df_copy[col] = df_copy[col].astype(str)
                                         
                                     # (★) .describe() の結果を取得
                                     result_df = df_copy.groupby(existing_grouping_cols)[numeric_cols_to_desc].describe()
                                     flat_cols = []
                                     for col in result_df.columns:
                                         # col is a tuple, e.g., ('評価およびスコア', 'mean')
                                         flat_cols.append(f"{col[0]}_{col[1]}") # e.g., "評価およびスコア_mean"
                                     result_df.columns = flat_cols
                                     
                                     # (★) 行のIndexをリセットして列に変換し、表示
                                     final_result_df = result_df.reset_index() # (★)
                                     
                                     # (★) フラット化されたDataFrameを表示
                                     st.dataframe(final_result_df) 
                                     
                                     # (★) AIサマリー用に結果を保存
                                     result_data = final_result_df # (★)
                                except Exception as group_e:
                                     st.error(f"グループ別集計エラー: {group_e}")
                elif name == "時系列キーワード分析":
                    # ... (時系列分析のロジックは L418-L425 までそのまま) ...
                    if isinstance(cols, dict) and 'datetime' in cols and 'keywords' in cols:
                        result_data = run_timeseries(df, cols) # (★)
                    else:
                         st.warning(f"「{name}」の列定義が不適切です: {cols}")
                elif name == "テキストマイニング（頻出単語など）":
                    if cols and isinstance(cols, list) and cols[0] == 'ANALYSIS_TEXT_COLUMN':
                        result_data = run_text_mining(df, 'ANALYSIS_TEXT_COLUMN') # (★)
                    else:
                        st.warning(f"「{name}」の列定義が不適切です: {cols}")
                elif name == "主成分分析 (PCA) / 因子分析":
                    st.warning("「主成分分析」は現在実装中です。")
                else:
                    st.warning(f"「{name}」の可視化ロジックはまだ実装されていません。")
                
                # (★) 正常に実行された結果をセッションに保存
                if result_data is not None and not result_data.empty:
                    st.session_state.step_c_results[name] = result_data
            
            except Exception as e:
                st.error(f"「{name}」の分析中に予期せぬエラーが発生しました: {e}")
                logger.error(f"Step C Analysis Error ({name}): {e}", exc_info=True)

    st.markdown("---")
    st.success("Step C-1 (可視化) が完了しました。")
    
    # (★ ここから Step C-2 (AIサマリー) のUI)
    st.header("Step C-2: AIによる分析サマリー")
    
    if not st.session_state.step_c_results:
        st.warning("AIサマリーの元になる分析結果がありません。Step C-1で有効な分析を実行してください。")
    else:
        st.info("上記で実行された分析結果をAIに入力し、総合的なサマリーレポートを生成します。")

        if st.button("🤖 AIサマリー用のプロンプトを生成", key="gen_prompt_c2"):
            # プロンプト生成関数を呼び出し
            st.session_state.ai_summary_prompt = generate_ai_summary_prompt(st.session_state.step_c_results, df)
            st.session_state.ai_summary_result = None # 既存のAI結果をリセット
            st.rerun() # プロンプトを表示するために再描画

        # (★) プロンプトが生成されたら、確認エリアを表示
        if st.session_state.ai_summary_prompt:
            st.subheader("AIへの指示プロンプト（確認・編集可）")
            prompt_input = st.text_area(
                "以下のプロンプトをAIに送信します:",
                value=st.session_state.ai_summary_prompt,
                height=300,
                key="ai_prompt_c2_input" # ユーザーが編集できるようにキーを設定
            )
            
            # (★) AI実行ボタン
            if st.button("🚀 この内容でAIに指示を送信", key="send_prompt_c2", type="primary"):
                if not os.getenv("GOOGLE_API_KEY"):
                    st.error("AIの実行には Google APIキー が必要です。（サイドバーで設定してください）")
                else:
                    with st.spinner("AIがサマリーを生成中... (Rate Limitに注意)"):
                        llm = get_llm()
                        if llm:
                            try:
                                # (★) ユーザーが編集した可能性のあるテキストエリアの内容を送信
                                response = llm.invoke(prompt_input) 
                                st.session_state.ai_summary_result = response.content
                            except Exception as e:
                                st.error(f"AIの呼び出しに失敗しました: {e}")
                                logger.error(f"AI summary failed: {e}", exc_info=True)
                        else:
                            st.error("AIモデルの初期化に失敗しました。")
            
        # (★) AIの実行結果がセッションにあれば表示
        if st.session_state.ai_summary_result:
            st.subheader("AIによる分析サマリーレポート")
            st.markdown(st.session_state.ai_summary_result)
    
    # (★ ここまで Step C-2 (AIサマリー) のUI)

    st.markdown("---")
    if st.button("⬅️ Step B に戻る", key="back_to_b_c2"):
        st.session_state.current_step = 'B'; st.rerun()

# (★ ここまで Step C 用の新規関数群)

def display_suggestions(suggestions, df):
    """
    提案された分析手法を表示し、ユーザーが選択できるようにする (★ チェックボックス版)
    """
    if not suggestions:
        st.info("提案可能な分析手法がありません。")
        return

    st.subheader("提案された分析手法:")
    st.markdown("---")
    
    # (★) デフォルトで選択される手法名リスト (上位5件)
    # のロジックをここに移動
    default_selection_names = [s['name'] for s in suggestions[:min(len(suggestions), 5)]] 
    
    # (★) --- Checkbox UI (MultiSelectの代わり) ---
    st.markdown("実行したい分析手法を選択（複数可）:")
    
    selected_technique_names = []
    
    # (★) 提案された手法をループしてチェックボックスを生成
    for suggestion in suggestions:
        name = suggestion['name']
        
        # デフォルトでチェックを入れるか
        is_default_checked = name in default_selection_names
        
        # チェックボックスを作成
        is_checked = st.checkbox(
            name, 
            value=is_default_checked, 
            key=f"cb_{name}" # キーをユニークに
        )
        
        # チェックされたらリストに追加
        if is_checked:
            selected_technique_names.append(name)
    
    # (★) --- MultiSelectのあった場所 (L499-L505) は削除 ---

    # (★) --- 選択された手法の詳細を表示 ---
    if selected_technique_names:
        st.markdown("---")
        st.subheader("選択された手法の詳細:")
        
        # 選択された手法の定義を取得
        selected_suggestions = [s for s in suggestions if s['name'] in selected_technique_names]
        
        for suggestion in selected_suggestions:
            with st.expander(f"{suggestion['name']} (優先度: {suggestion['priority']})"):
                st.markdown(f"**<説明>**\n{suggestion['description']}")
                st.markdown(f"**<提案理由>**\n{suggestion['reason']}")

    st.markdown("---")
    # (★) ボタンの有効/無効を設定 (手法が1つ以上選択されている場合のみ有効)
    # (★) --- 実行ボタン (このブロックが1つだけ存在することを確認) ---
    if st.button("選択した手法で分析を実行 (Step Cへ)", key="execute_button_C_v2", disabled=not selected_technique_names, type="primary"):
         if selected_technique_names: # 念のため再チェック
             # st.info(f"ステップCは現在実装中です。選択された手法: {', '.join(selected_technique_names)}")
             st.session_state.current_step = 'C' # (★) 変更
             st.rerun() # (★) 変更
         else:
             st.error("分析を実行するには、少なくとも1つの手法を選択してください。")

# --- Step A の UI とロジック ---
def render_step_a():
    # ... (Step A のコード全体 - 変更なし) ...
    st.title("🤖 [AI対応] データ分析・フラグ付けツール (Step A: タグ付け)")
    st.markdown("ファイルをアップロードし、分析指針に基づいてキーワードでタグ付けを行います。")
    logger = getLogger(__name__)
    if 'generated_categories' not in st.session_state: st.session_state.generated_categories = None
    if 'selected_categories' not in st.session_state: st.session_state.selected_categories = []
    if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
    if 'cancel_analysis' not in st.session_state: st.session_state.cancel_analysis = False
    st.header("Step 1: 分析対象ファイルのアップロード"); uploaded_files = st.file_uploader("分析したい Excel / CSV ファイル（複数可）", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True, key="uploader_A")
    if uploaded_files:
        st.header("Step 2: 分析指針の入力"); analysis_prompt = st.text_area("データの補足やフラグ付けの指針を入力:", placeholder="例：広島の観光データ...", key="analysis_prompt_A")
        st.header("Step 3: 抽出カテゴリ候補の生成")
        if st.button("📝 カテゴリ候補を生成", key="generate_cat_button_A"):
            logger.info("カテゴリ候補生成ボタンクリック");
            if not os.getenv("GOOGLE_API_KEY"): st.error("APIキー未設定")
            elif not analysis_prompt: st.error("分析指針未入力")
            else:
                with st.spinner("AIカテゴリ候補生成中..."):
                    llm = get_llm()
                    if llm: logger.info("get_dynamic_categories 呼び出し..."); st.session_state.generated_categories = get_dynamic_categories(analysis_prompt, llm); logger.info(f"生成候補: {st.session_state.generated_categories}"); st.session_state.selected_categories = ["市区町村キーワード"]; st.session_state.analysis_done = False
                    else: st.error("LLM初期化失敗")
        if st.session_state.generated_categories:
            st.header("Step 4: タグ付けするカテゴリを選択"); all_possible_categories = list(st.session_state.generated_categories.keys()); mandatory_category = "市区町村キーワード"; selectable_categories = [cat for cat in all_possible_categories if cat != mandatory_category]; default_selection = [cat for cat in selectable_categories if cat in st.session_state.generated_categories]
            selected_dynamic_categories = st.multiselect("タグ付けしたいカテゴリを選択:", options=selectable_categories, default=default_selection, key="category_multiselect_A"); st.session_state.selected_categories = [mandatory_category] + selected_dynamic_categories; st.write("---")
        st.header("Step 5: 分析設定 (ファイルごと)"); st.info("ファイルごとに、分析対象のテキスト列を指定してください。")
        text_column_map, valid_files_present = {}, False
        for i, uploaded_file in enumerate(uploaded_files):
             preview_df = None
             try:
                uploaded_file.seek(0); preview_df = pd.read_csv(uploaded_file, encoding="utf-8-sig", nrows=5) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, nrows=5); uploaded_file.seek(0)
                if preview_df is not None and not preview_df.empty:
                    valid_files_present = True
                    with st.container(border=True):
                        st.markdown(f"**ファイル名: `{uploaded_file.name}`**"); columns = preview_df.columns.tolist(); session_key = f"text_col_{i}_{uploaded_file.name}"
                        default_col_index = 0; lower_columns = [c.lower() for c in columns]
                        if 'text' in lower_columns: default_col_index = lower_columns.index('text')
                        elif '本文' in columns: default_col_index = columns.index('本文')
                        elif 'content' in lower_columns: default_col_index = lower_columns.index('content')
                        selected_col = st.selectbox(f"分析テキスト列 ({uploaded_file.name}):", columns, index=default_col_index, key=session_key); text_column_map[uploaded_file.name] = selected_col
                    st.dataframe(preview_df)
             except Exception as e: st.error(f"ファイル {uploaded_file.name} プレビュー失敗: {e}")
        if valid_files_present:
            st.header("Step 6: 分析実行"); progress_placeholder = st.empty(); log_placeholder = st.empty(); cancel_button_placeholder = st.empty(); col1, col2 = st.columns([1, 5]); start_analysis = col1.button("📈 分析実行", type="primary", key="analyze_button_A")
            if start_analysis:
                logger.critical("--- ★★★ 分析実行ボタン CLICKED (Step A) ★★★ ---"); st.session_state.analysis_done = False; st.session_state.cancel_analysis = False; st_handler = None
                try: # UIログハンドラ設定
                    logger = getLogger(__name__);
                    for h in logger.handlers[:]:
                        if isinstance(h, StreamlitLogHandler): logger.removeHandler(h)
                    st.session_state.log_messages.clear(); st_handler = StreamlitLogHandler(); formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s'); st_handler.setFormatter(formatter); logger.addHandler(st_handler)
                    log_placeholder.text_area("実行ログ:", "分析を開始します...", height=200, key="log_display_A_init", disabled=True); print("--- UIログハンドラ設定完了 ---", file=sys.stderr)
                except Exception as handler_e: print(f"--- FATAL ERROR logger setup --- {handler_e}", file=sys.stderr); import traceback; traceback.print_exc(file=sys.stderr); st.error(f"ログ設定エラー: {handler_e}"); st.stop()
                with col2: # キャンセルボタン表示
                    if cancel_button_placeholder.button("⏹️ 分析をキャンセル", key="cancel_button_A"): st.session_state.cancel_analysis = True; logger.warning("キャンセルボタン押下"); st.warning("分析キャンセル中...")
                logger.info("分析処理開始...");
                if not st.session_state.generated_categories or not st.session_state.selected_categories: logger.error("カテゴリ未選択"); st.error("カテゴリ生成・選択要"); st.stop()
                if not os.getenv("GOOGLE_API_KEY"): logger.error("APIキー未設定"); st.error("APIキー設定要"); st.stop()
                if not analysis_prompt: logger.error("分析指針未入力"); st.error("分析指針入力要"); st.stop()
                logger.info("初期チェック完了。"); all_dfs = []; total_rows = 0; processed_rows = 0; st_handler_in_use = st_handler
                try: # メインの分析処理
                    logger.info("AIモデル初期化開始..."); llm = get_llm();
                    if llm is None: raise Exception("AIモデル初期化失敗")
                    logger.info("AIモデル初期化完了。"); st.session_state.nlp = load_spacy_model(); logger.info("spaCyモデル読込完了。"); logger.info("データ読込・結合開始...")
                    temp_dfs = []; valid_file_count = 0
                    for up_file in uploaded_files:
                        if up_file.name not in text_column_map: logger.warning(f"{up_file.name} スキップ"); continue
                        text_col_name = text_column_map[up_file.name]; logger.info(f"{up_file.name} ({text_col_name}) 処理中...")
                        df_single, _ = load_and_preprocess_data(up_file, text_col_name)
                        if df_single is None or df_single.empty: logger.warning(f"{up_file.name} スキップ"); continue
                        df_single.rename(columns={text_col_name: 'ANALYSIS_TEXT_COLUMN'}, inplace=True); temp_dfs.append(df_single); valid_file_count += 1; logger.info(f"{up_file.name} 処理完了 ({len(df_single)} 行)")
                    if not temp_dfs: logger.error("有効DFなし"); raise Exception("有効データなし...")
                    logger.info(f"{valid_file_count} 個ファイルを結合..."); master_df = pd.concat(temp_dfs, ignore_index=True, sort=False); master_df['id'] = master_df.index; total_rows = len(master_df); logger.info(f"結合完了。総行数: {total_rows}")
                    if master_df.empty: logger.error("結合後DF空"); raise Exception("分析対象データ空")

                    # (★ ここから新規追加: 重複削除とAIフィルタリング)
                    logger.info("Step A-2: 重複削除 開始...")
                    initial_row_count = len(master_df)
                    master_df.drop_duplicates(subset=['ANALYSIS_TEXT_COLUMN'], keep='first', inplace=True)
                    deduped_row_count = len(master_df)
                    logger.info(f"重複削除 完了。 {initial_row_count}行 -> {deduped_row_count}行 ({initial_row_count - deduped_row_count}行削除)")
                    
                    logger.info("Step A-3: AI関連性フィルタリング 開始...")
                    # (★) フィルタリング専用のバッチサイズと待機時間
                    filter_batch_size = 50 # (★) 50件ずつフィルタリング
                    filter_sleep_time = 4.1 # (★) 60s / 15 RPM = 4s。マージン込みで 4.1s
                    
                    total_filter_rows = len(master_df)
                    total_filter_batches = (total_filter_rows + filter_batch_size - 1) // filter_batch_size
                    all_filtered_results = []
                    
                    for i in range(0, total_filter_rows, filter_batch_size):
                        if st.session_state.cancel_analysis: logger.warning(f"フィルタリングキャンセル (バッチ {i//filter_batch_size + 1})"); st.warning("分析キャンセル"); break
                        
                        batch_df = master_df.iloc[i:i+filter_batch_size]
                        current_batch_num = i // filter_batch_size + 1
                        logger.info(f"AIフィルタリング バッチ {current_batch_num}/{total_filter_batches} 処理中...")
                        
                        # (★) UI更新
                        progress_percent = min((i + filter_batch_size) / total_filter_rows, 1.0)
                        progress_text = f"[AIフィルタリング] 処理中: {min(i + filter_batch_size, total_filter_rows)}/{total_filter_rows} 件 ({progress_percent:.0%})"
                        progress_placeholder.progress(progress_percent, text=progress_text)
                        log_text_for_ui = "\n".join(st.session_state.log_messages)
                        log_placeholder.text_area("実行ログ:", log_text_for_ui, height=200, key=f"log_update_A_filter_{i}", disabled=True)
                        
                        # (★) AIフィルタリング実行
                        filtered_df = filter_relevant_data_by_ai(batch_df, analysis_prompt, llm)
                        if filtered_df is not None and not filtered_df.empty:
                            all_filtered_results.append(filtered_df)
                        else:
                            logger.warning(f"AIフィルタリング バッチ {current_batch_num} 結果空")
                            
                        time.sleep(filter_sleep_time) # (★) フィルタリング待機
                    
                    if st.session_state.cancel_analysis:
                        logger.warning("AIフィルタリング処理がキャンセルされました。")
                        raise Exception("分析がキャンセルされました") # (★) キャンセル時は処理中断

                    if not all_filtered_results:
                        logger.error("全バッチAIフィルタリング失敗"); raise Exception("AIフィルタリング処理失敗")

                    logger.info("全AIフィルタリング結果結合...");
                    filter_results_df = pd.concat(all_filtered_results, ignore_index=True)
                    
                    # (★) relevant=True の ID リストを取得
                    relevant_ids = filter_results_df[filter_results_df['relevant'] == True]['id']
                    
                    # (★) master_df をフィルタリング
                    filtered_master_df = master_df[master_df['id'].isin(relevant_ids)].copy()
                    filtered_row_count = len(filtered_master_df)
                    logger.info(f"AIフィルタリング 完了。 {deduped_row_count}行 -> {filtered_row_count}行 ({deduped_row_count - filtered_row_count}行削除)")
                    
                    if filtered_master_df.empty:
                        logger.error("AIフィルタリング後、データが0件になりました。"); raise Exception("分析対象データ空")
                    
                    # (★ ここまで新規追加)

                    logger.info("Step A-4: AIタグ付け処理開始..."); selected_category_definitions = { cat: desc for cat, desc in st.session_state.generated_categories.items() if cat in st.session_state.selected_categories }; logger.info(f"選択カテゴリ: {list(selected_category_definitions.keys())}")
                    
                    # (★) AIタグ付けの対象を `filtered_master_df` に変更
                    master_df_for_tagging = filtered_master_df
                    total_rows = len(master_df_for_tagging) # (★) 総行数を更新
                    # (★) 効率化: バッチサイズ10、待機時間4.1秒 (約15RPM) に変更
                    batch_size = 10; all_tagged_results = []; total_batches = (total_rows + batch_size - 1) // batch_size; logger.info(f"バッチサイズ {batch_size}, 総バッチ数: {total_batches}")
                    for i in range(0, total_rows, batch_size):
                        if st.session_state.cancel_analysis: logger.warning(f"ループキャンセル (バッチ {i//batch_size + 1})"); st.warning("分析キャンセル"); break
                        
                        # (★) バッチ取得元を変更
                        batch_df = master_df_for_tagging.iloc[i:i+batch_size]; current_batch_num = i // batch_size + 1; logger.info(f"バッチ {current_batch_num}/{total_batches} 処理中...")
                        
                        # (★) UI更新 (プログレスバーのテキスト変更)
                        progress_percent = min((i + batch_size)/total_rows, 1.0); progress_text = f"[AIタグ付け] 処理中: {min(i + batch_size, total_rows)}/{total_rows} 件 ({progress_percent:.0%})"
                        progress_placeholder.progress(progress_percent, text=progress_text); log_text_for_ui = "\n".join(st.session_state.log_messages); log_placeholder.text_area("実行ログ:", log_text_for_ui, height=200, key=f"log_update_A_{i}", disabled=True)
                        logger.info(f"Calling perform_ai_tagging batch {current_batch_num}...")
                        # (★) analysis_prompt を渡すように変更
                        tagged_df = perform_ai_tagging(batch_df, selected_category_definitions, llm, analysis_prompt)
                        logger.info(f"Finished perform_ai_tagging batch {current_batch_num}.")
                        if tagged_df is not None and not tagged_df.empty: all_tagged_results.append(tagged_df)
                        else: logger.warning(f"バッチ {current_batch_num} AI処理結果空")
                        processed_rows += len(batch_df); time.sleep(4.1) # 待機時間
                    if not st.session_state.cancel_analysis:
                        if not all_tagged_results: logger.error("全バッチAI処理失敗"); raise Exception("AIタグ付け処理失敗")
                        logger.info("全バッチ結果結合..."); results_df = pd.concat(all_tagged_results, ignore_index=True); logger.info("結合完了")
                        logger.info("AI応答文字列変換..."); cols_to_convert = [col for col in results_df.columns if col != 'id']
                        for col in cols_to_convert:
                             if col in results_df: results_df[col] = results_df[col].apply( lambda x: ', '.join(x) if isinstance(x, list) else (str(x) if pd.notna(x) else '') )
                        logger.info("変換完了")
                        logger.info("元データと結果マージ...")
                        cols_in_results_df_except_id = [col for col in results_df.columns if col != 'id']; cols_to_drop_from_master = [col for col in cols_in_results_df_except_id if col in master_df.columns]
                        if cols_to_drop_from_master: logger.warning(f"重複列削除: {cols_to_drop_from_master}"); master_df_for_merge = master_df_for_tagging.drop(columns=cols_to_drop_from_master) # (★) 変更
                        else: master_df_for_merge = master_df_for_tagging
                        logger.info(f"Merging master ({master_df_for_merge.shape}) with results ({results_df.shape})"); final_df = master_df_for_merge.merge(results_df, on='id', how='left'); logger.info(f"マージ後形状: {final_df.shape}")
                        logger.info(f"選択カテゴリ列確認・整形: {st.session_state.selected_categories}")
                        for cat in st.session_state.selected_categories:
                            if cat not in final_df.columns: logger.warning(f"列 '{cat}' なし。空列追加"); final_df[cat] = ''
                            else: final_df[cat] = final_df[cat].fillna('').astype(str); logger.info(f"列 '{cat}' 確認/作成完了")
                        logger.info("マージ完了")
                        logger.info("構造分析開始..."); st.session_state.structure_info = analyze_data_structure(final_df); st.session_state.df_flagged = final_df
                        st.session_state.analysis_done = True # 正常完了
                        logger.info("分析完了！"); progress_placeholder.progress(1.0, text="完了！")
                        log_text_final = "\n".join(st.session_state.log_messages); log_placeholder.text_area("実行ログ:", log_text_final, height=200, key="log_final_A", disabled=True)
                    else: # キャンセルされた場合
                        progress_placeholder.empty(); log_text_cancel = "\n".join(st.session_state.log_messages); log_placeholder.text_area("実行ログ:", log_text_cancel, height=200, key="log_cancel_A", disabled=True)
                except Exception as e:
                    error_message = f"分析処理中エラー: {e}"; st.error(error_message)
                    print(f"--- FATAL ERROR --- {error_message}", file=sys.stderr); import traceback; traceback.print_exc(file=sys.stderr)
                    logger.error(error_message, exc_info=True); st.session_state.analysis_done = False # 異常終了
                    progress_placeholder.empty(); log_text_error = "\n".join(st.session_state.log_messages); log_placeholder.text_area("実行ログ:", log_text_error, height=200, key="log_error_A", disabled=True)
                finally:
                    cancel_button_placeholder.empty()
                    if 'st_handler_in_use' in locals() and st_handler_in_use in logger.handlers:
                         logger.removeHandler(st_handler_in_use); print("--- UIログハンドラ削除 ---", file=sys.stderr)
                    st.session_state.cancel_analysis = False # リセット
        else:
             if uploaded_files: st.warning("アップロードされたファイルを正しく読み込めませんでした。")

    # --- Step 7: 結果表示 (Step A) ---
    if st.session_state.get('analysis_done') and st.session_state.current_step == 'A':
        st.header("🎉 分析が完了しました (Step A)")
        st.subheader("A-1: データ構造の分析結果"); info = st.session_state.structure_info; st.metric(label="総行数", value=info['total_rows']); st.metric(label="総列数", value=info['total_columns'])
        st.markdown("#### 列ごとの詳細:"); st.dataframe(pd.DataFrame(info['column_details']).T)
        st.subheader("A-2: キーワード抽出の結果 (先頭100件)"); df_result = st.session_state.df_flagged; dynamic_columns = st.session_state.get('selected_categories', ['市区町村キーワード'])
        def highlight_flags(row):
            color = 'background-color: #FFFFE0'
            for col in dynamic_columns:
                if row.get(col, ''): return [color] * len(row)
            return [''] * len(row)
        text_col = ['ANALYSIS_TEXT_COLUMN'] if 'ANALYSIS_TEXT_COLUMN' in df_result.columns else []
        other_cols = [col for col in df_result.columns if col not in dynamic_columns and col not in text_col and col != 'id']
        display_cols = other_cols + text_col + dynamic_columns; existing_display_cols = [col for col in display_cols if col in df_result.columns]
        st.dataframe(df_result[existing_display_cols].head(100).style.apply(highlight_flags, axis=1))
        @st.cache_data
        def convert_df_to_csv(df):
            output = BytesIO(); cols_to_download = [col for col in df.columns if col != 'id']
            df.to_csv(output, columns=cols_to_download, encoding="utf-8-sig", index=False)
            return output.getvalue()
        csv_data = convert_df_to_csv(df_result); st.download_button(label="📥 抽出結果をCSVでダウンロード", data=csv_data, file_name="keyword_extraction_result.csv", mime="text/csv", key="download_button_A")
        st.markdown("---"); st.header("Step B へ進む")
        if st.button("📊 分析手法の提案へ進む", key="goto_step_B"): st.session_state.current_step = 'B'; st.rerun()

# --- Step B の UI とロジック ---
def render_step_b():
    st.title("📊 分析手法の提案 (Step B)")
    st.info("Step A でフラグ付けし、エクスポートしたCSVファイルをアップロードしてください。データに適した分析手法を提案します。")
    logger = getLogger(__name__)

    uploaded_flagged_file = st.file_uploader("フラグ付け済みCSVファイルをアップロード", type=['csv'], key="step_b_uploader")
    
    # (★) 追加の分析指示用テキストエリアを追加
    analysis_prompt_B = st.text_area(
        "（任意）追加の分析指示:", 
        placeholder="例: 特定の市区町村（広島市など）と観光施設の相関関係を深掘りしたい。",
        key="step_b_prompt"
    )

    if uploaded_flagged_file:
        try:
            uploaded_flagged_file.seek(0)
            df_flagged = pd.read_csv(uploaded_flagged_file, encoding="utf-8-sig")
            st.session_state.df_flagged_B = df_flagged # (★) Step C のためにセッションに保存
            st.success(f"ファイル「{uploaded_flagged_file.name}」読込完了")
            st.dataframe(df_flagged.head())

            if st.button("💡 分析手法を提案させる", key="suggest_button_B"):
                with st.spinner("データ構造と指示内容を分析し、手法を提案中..."):
                    # 1. 構造ベースの提案
                    base_suggestions = suggest_analysis_techniques(df_flagged)
                    
                    ai_suggestions = []
                    # 2. プロンプトベースの提案 (指示がある場合のみ)
                    if analysis_prompt_B.strip():
                        llm = get_llm()
                        if llm:
                            # (★) existing_suggestions を渡さないように変更
                            ai_suggestions = get_suggestions_from_prompt(
                                analysis_prompt_B, llm, df_flagged, [] 
                            )
                        else:
                            st.error("AIモデルの初期化に失敗しました。APIキーを確認してください。")

                    # 3. (★) 提案のマージと重複排除ロジック
                    base_suggestion_names = {s['name'] for s in base_suggestions} # (★) 構造ベースの名前リスト
                    
                    # ユーザー指示(AI提案)から、構造ベースと名前が重複するものを除外
                    filtered_ai_suggestions = [
                        s for s in ai_suggestions if s['name'] not in base_suggestion_names # (★) ロジック逆転
                    ]
                    
                    # 構造ベース(優先度1-5) + ユーザー指示(重複除外済, 優先度6) を結合し、優先度順にソート
                    all_suggestions = sorted(base_suggestions + filtered_ai_suggestions, key=lambda x: x['priority']) # (★) 順序変更
                    st.session_state.suggestions_B = all_suggestions

            if 'suggestions_B' in st.session_state and st.session_state.suggestions_B:
                 display_suggestions(st.session_state.suggestions_B, df_flagged)

        except Exception as e:
            st.error(f"ファイル読込/分析提案中にエラー: {e}")
            logger.error("Step B error", exc_info=True)


# --- ▼▼▼ ここが重要 ▼▼▼ ---
# Streamlit アプリケーション実行
# main 関数を定義し、その中でステップ分岐を行う
def main():
    st.set_page_config(page_title="データ分析・フラグ付けツール", layout="wide")
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = deque(maxlen=15)
    # --- 基本ロガー設定 ---
    logger = getLogger(__name__)
    if not any(isinstance(h, StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        if logger.hasHandlers(): logger.handlers.clear()
        handler_stdout = StreamHandler(sys.stdout)
        formatter_stdout = Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler_stdout.setFormatter(formatter_stdout)
        logger.addHandler(handler_stdout)
        logger.setLevel(DEBUG)
        # 起動ログは __main__ ブロックで出す

    if JAPAN_GEOGRAPHY_DB is None:
        st.error("重大なエラー: 地名辞書ファイル (geography_db.py) が見つかりません。")
        logger.critical("geography_db.py が見つからないため停止します。")
        st.stop()

    # --- ステップ管理用のセッションステート初期化 ---
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'A' # 初期ステップ

    # (★) --- サイドバー (共通ナビゲーション) ---
    with st.sidebar:
        st.title("Navigation")
        st.markdown("---")
        
        # APIキー設定 (全ステップで共通化)
        st.header("⚙️ AI 設定")
        # (★) キーを "api_key_A" から "api_key_global" に変更
        google_api_key = st.text_input("Google API Key", type="password", key="api_key_global")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        else:
            st.warning("AI機能を利用するには Google APIキー を設定してください。")
        
        st.markdown("---")
        
        # (★) ステップ選択
        st.header("🔄 Step 選択")
        current_step = st.session_state.current_step
        
        if st.button("Step A: タグ付け", key="nav_A", use_container_width=True, type=("primary" if current_step == 'A' else "secondary")):
            if st.session_state.current_step != 'A':
                st.session_state.current_step = 'A'
                st.rerun() # ページを即時再描画

        if st.button("Step B: 分析手法提案", key="nav_B", use_container_width=True, type=("primary" if current_step == 'B' else "secondary")):
            if st.session_state.current_step != 'B':
                st.session_state.current_step = 'B'
                st.rerun() # ページを即時再描画

    # --- ステップに応じて描画関数を呼び出し ---
    if st.session_state.current_step == 'A':
        render_step_a()
    elif st.session_state.current_step == 'B':
        render_step_b()
    elif st.session_state.current_step == 'C':
        render_step_c()


if __name__ == '__main__':
    # --- 基本ロガー設定 (重複するが、直接実行時にも必要) ---
    logger = getLogger(__name__)
    if not any(isinstance(h, StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        if logger.hasHandlers(): logger.handlers.clear()
        handler_stdout = StreamHandler(sys.stdout)
        formatter_stdout = Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler_stdout.setFormatter(formatter_stdout)
        logger.addHandler(handler_stdout)
        logger.setLevel(DEBUG)
        logger.info("--- Application Start ---") # 起動ログ

    # --- メイン関数呼び出し ---
    main()
