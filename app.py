# ---
# app.py (AI Data Analysis App - Refactored Version)
#
# このコードは、商用利用が容易な寛容な（permissive）ライセンス
# (例: MIT, Apache License 2.0, BSD) の下で利用可能な
# ライブラリ、またはライセンスに依存しないコードのみを使用して実装されています。
# GPL, AGPL, SSPLなどのコピーレフト効果を持つライブラリは使用していません。
# ---

# --- 1. ライブラリのインポート ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
import logging
import time
import spacy
import altair as alt
import networkx as nx
from networkx.algorithms import community
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import StringIO, BytesIO
from typing import Optional, Dict, List, Any, Union

# (★) LangChain / Google Generative AI のインポート
# ライセンス: Apache License 2.0
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# (★) Step D (PowerPoint生成) で必要となるライブラリ
# ライセンス: MIT License
try:
    import pptx
    from pptx import Presentation
    from pptx.util import Inches, Pt
except ImportError:
    st.error(
        "PowerPoint生成ライブラリ(python-pptx)が見つかりません。"
        "pip install python-pptx を実行してください。"
    )

# (★) Step D (ドラッグ＆ドロップUI) で必要となるライブラリ
# ライセンス: MIT License
try:
    from streamlit_sortables import sort_items
except ImportError:
    st.error(
        "UIライブラリ(streamlit-sortables)が見つかりません。"
        "pip install streamlit-sortables を実行してください。"
    )

# 既存のライブラリ (openpyxl, ja_core_news_sm) のインポート
try:
    import openpyxl
except ImportError:
    st.error("Excel (openpyxl) がインストールされていません。`pip install openpyxl` してください。")
try:
    import ja_core_news_sm
except ImportError:
    st.error("spaCy日本語モデル (ja_core_news_sm) が見つかりません。`python -m spacy download ja_core_news_sm` してください。")

# --- 2. (★) 定数定義 ---

# (★) 要件に基づき、使用するAIモデルを定数として定義
MODEL_FLASH_LITE = "gemini-2.5-flash-lite" # Step A, B (高速・効率的)
MODEL_FLASH = "gemini-2.5-flash"         # Step D (代替)
MODEL_PRO = "gemini-2.5-pro"             # Step C, D (高品質)

# バッチサイズと待機時間 (KISS)
FILTER_BATCH_SIZE = 50
FILTER_SLEEP_TIME = 6.1  # Rate Limit 対策 (10 requests per 60 seconds)
TAGGING_BATCH_SIZE = 10
TAGGING_SLEEP_TIME = 6.1  # Rate Limit 対策

# 地名辞書
try:
    from geography_db import JAPAN_GEOGRAPHY_DB
except ImportError:
    st.error("地名辞書ファイル (geography_db.py) が見つかりません。")
    JAPAN_GEOGRAPHY_DB = {}


# --- 3. ロガー設定 ---
class StreamlitLogHandler(logging.Handler):
    """Streamlitのセッションステートにログメッセージを追加するハンドラ"""
    def __init__(self):
        super().__init__()
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []

    def emit(self, record):
        log_entry = self.format(record)
        st.session_state.log_messages.append(log_entry)
        st.session_state.log_messages = st.session_state.log_messages[-500:]

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


# --- 4. (★) AIモデル・NLPモデルのキャッシュ管理 ---

# (★) 要件に基づき、異なるモデル名を指定してLLMをロードする関数に刷新
@st.cache_resource(ttl=3600)  # 1時間キャッシュ
def get_llm(model_name: str, temperature: float = 0.0) -> Optional[ChatGoogleGenerativeAI]:
    """
    指定されたモデル名と温度でLLM (Google Gemini) モデルをロード・キャッシュする。
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error(f"get_llm: GOOGLE_API_KEY がありません (Model: {model_name})")
            return None

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True,
            api_key=api_key
        )
        logger.info(f"LLM Model ({model_name}) loaded successfully.")
        return llm
    except Exception as e:
        logger.error(f"LLM ({model_name}) の初期化に失敗: {e}", exc_info=True)
        st.error(f"AIモデル ({model_name}) のロードに失敗しました: {e}")
        return None

@st.cache_resource
def load_spacy_model() -> Optional[spacy.language.Language]:
    """spaCyの日本語モデル(ja_core_news_sm)をロード・キャッシュする"""
    try:
        logger.info("Loading spaCy model (ja_core_news_sm)...")
        nlp = spacy.load("ja_core_news_sm")
        logger.info("spaCy model loaded successfully.")
        return nlp
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}", exc_info=True)
        return None


# --- 5. ファイル読み込みヘルパー ---
def read_file(file: st.runtime.uploaded_file_manager.UploadedFile) -> (Optional[pd.DataFrame], Optional[str]):
    """アップロードされたファイル(Excel/CSV)をPandas DataFrameとして読み込む"""
    file_name = file.name
    logger.info(f"ファイル読み込み開始: {file_name}")
    try:
        if file_name.endswith('.csv'):
            # 文字コードを自動判別
            try:
                content = file.getvalue().decode('utf-8-sig')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8-SIGデコード失敗。CP932で再試行: {file_name}")
                content = file.getvalue().decode('cp932')
            df = pd.read_csv(StringIO(content))

        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(file.getvalue()), engine='openpyxl')
        else:
            msg = f"サポート外のファイル形式: {file_name}"
            logger.warning(msg)
            return None, msg

        logger.info(f"ファイル読み込み成功: {file_name}")
        return df, None
    except Exception as e:
        logger.error(f"ファイル読み込みエラー ({file_name}): {e}", exc_info=True)
        st.error(f"ファイル「{file_name}」の読み込み中にエラー: {e}")
        return None, f"読み込みエラー: {e}"


# --- 6. (★) Step A: AIタグ付け関連関数 ---
# (要件: Step Aは gemini-2.5-flash-lite を使用)

def get_dynamic_categories(analysis_prompt: str) -> Optional[Dict[str, str]]:
    """
    (Step A) ユーザーの分析指針に基づき、AIが動的なカテゴリをJSON形式で生成する。
    (★) モデル: MODEL_FLASH_LITE (gemini-2.5-flash-lite)
    """
    # (★) Step A の要件に基づき、FLASH_LITE モデルを明示的に指定
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None:
        logger.error("get_dynamic_categories: LLM (Flash Lite) が利用できません。")
        st.error("AIモデル(Flash Lite)が利用できません。サイドバーでAPIキーを設定してください。")
        return None

    logger.info("動的カテゴリ生成AI (Flash Lite) を呼び出し...")
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

        # マークダウンや不要なテキストを除去し、JSONのみを抽出
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

def filter_relevant_data_by_ai(df_batch: pd.DataFrame, analysis_prompt: str) -> pd.DataFrame:
    """
    (Step A) AIを使い、分析指針と無関係な行をフィルタリングする (relevant: true/false)。
    (★) モデル: MODEL_FLASH_LITE (gemini-2.5-flash-lite)
    (★) 要件: 進捗表示 (この関数はバッチ処理の一部として呼ばれ、呼び出し元の
          `render_step_a` 内の `update_progress_ui` で進捗が表示される)
    """
    # (★) Step A の要件に基づき、FLASH_LITE モデルを明示的に指定
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None:
        logger.error("filter_relevant_data_by_ai: LLM (Flash Lite) が利用できません。")
        st.error("AIモデル(Flash Lite)が利用できません。APIキーを確認してください。")
        return pd.DataFrame()  # 空のDF

    logger.debug(f"{len(df_batch)}件 AI関連性フィルタリング (Flash Lite) 開始...")

    # テキストが長すぎる場合、先頭500文字に切り詰める
    input_texts_jsonl = df_batch.apply(
        lambda row: json.dumps(
            {"id": row['id'], "text": str(row['ANALYSIS_TEXT_COLUMN'])[:500]},
            ensure_ascii=False
        ),
        axis=1
    ).tolist()

    prompt = PromptTemplate.from_template(
        """
        あなたはデータ分析のキュレーターです。「分析指針」に基づき、「テキストデータ(JSONL)」の各行が分析対象として【関連しているか (relevant: true)】、【無関係か (relevant: false)】を判定してください。
        # 分析指針 (Analysis Scope):
        {analysis_prompt}
        # 指示:
        1. 「分析指針」と【強く関連】する投稿のみを `true` とする。
        2. 単なる宣伝、挨拶のみ、指針と無関係な地域の言及は `false` とする。
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
        response_str = chain.invoke(invoke_params)
        
        results = []
        # マークダウン ````jsonl ... ``` を除去
        match = re.search(r'```(?:jsonl|json)?\s*([\s\S]*?)\s*```', response_str, re.DOTALL)
        jsonl_content = match.group(1).strip() if match else response_str.strip()

        for line in jsonl_content.strip().split('\n'):
            cleaned_line = line.strip()
            if not cleaned_line: continue
            try:
                data = json.loads(cleaned_line)
                # relevant が "true" (str) や true (bool) など揺らぎがあるため堅牢に処理
                is_relevant = False
                if isinstance(data.get("relevant"), bool):
                    is_relevant = data.get("relevant")
                elif isinstance(data.get("relevant"), str):
                    is_relevant = data.get("relevant").lower() == 'true'
                
                results.append({"id": data.get("id"), "relevant": is_relevant})
            except (json.JSONDecodeError, AttributeError) as json_e:
                logger.warning(f"AIフィルタリング回答パース失敗: {cleaned_line} - Error: {json_e}")
                # パース失敗時は、IDが特定できれば関連あり(True)としてフォールバック
                id_match = re.search(r'"id":\s*(\d+)', cleaned_line)
                if id_match:
                    results.append({"id": int(id_match.group(1)), "relevant": True})

        return pd.DataFrame(results) if results else pd.DataFrame(columns=['id', 'relevant'])
        
    except Exception as e:
        logger.error(f"AIフィルタリングバッチ処理中エラー: {e}", exc_info=True)
        st.error(f"AIフィルタリング処理エラー: {e}")
        # エラー時は安全側に倒し、すべて関連あり(True)として返す
        return df_batch[['id']].copy().assign(relevant=True)

def perform_ai_tagging(
    df_batch: pd.DataFrame,
    categories_to_tag: Dict[str, str],
    analysis_prompt: str = ""
) -> pd.DataFrame:
    """
    (Step A) テキストのバッチを受け取り、AIが【指定されたカテゴリ定義】に基づいて直接タグ付けを行う
    (★) モデル: MODEL_FLASH_LITE (gemini-2.5-flash-lite)
    (★) 要件: 進捗表示 (この関数はバッチ処理の一部として呼ばれ、呼び出し元の
          `render_step_a` 内の `update_progress_ui` で進捗が表示される)
    """
    # (★) Step A の要件に基づき、FLASH_LITE モデルを明示的に指定
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.0)
    if llm is None:
        logger.error("perform_ai_tagging: LLM (Flash Lite) が利用できません。")
        st.error("AIモデル(Flash Lite)が利用できません。APIキーを確認してください。")
        return pd.DataFrame()

    logger.info(f"{len(df_batch)}件 AIタグ付け (Flash Lite) 開始 (カテゴリ: {list(categories_to_tag.keys())})")

    # 地名辞書のコンテキストを準備 (分析指針に関連する地名のみをAIに渡す)
    geo_context_str = "{}"
    if JAPAN_GEOGRAPHY_DB and "市区町村キーワード" in categories_to_tag:
        try:
            relevant_geo_db = {}
            prompt_lower = analysis_prompt.lower()
            
            # (地名辞書のキーとプロンプトの両方に含まれる主要なヒント)
            hints = ["広島", "福岡", "大阪", "東京", "北海道", "愛知", "宮城", "札幌", "横浜", "名古屋", "京都", "神戸", "仙台"]
            keys_found = [
                key for key in JAPAN_GEOGRAPHY_DB.keys()
                if any(h in key.lower() for h in hints) and any(h in prompt_lower for h in hints)
            ]
            # 特定のキーワードが含まれている場合は、関連するキーを強制的に追加
            if "広島" in prompt_lower: keys_found.extend(["広島県", "広島市"])
            if "東京" in prompt_lower: keys_found.extend(["東京都", "東京23区"])
            if "大阪" in prompt_lower: keys_found.extend(["大阪府", "大阪市"])

            for key in set(keys_found): # 重複削除
                if key in JAPAN_GEOGRAPHY_DB:
                    relevant_geo_db[key] = JAPAN_GEOGRAPHY_DB[key]
            
            #  relevant_geo_db が空の場合、フォールバック (主要都市)
            if not relevant_geo_db:
                logger.warning("地名辞書の絞り込みヒントなし。主要都市のみ渡します。")
                default_keys = ["東京都", "東京23区", "大阪府", "大阪市", "広島県", "広島市", "福岡県", "福岡市"]
                for key in default_keys:
                    if key in JAPAN_GEOGRAPHY_DB:
                        relevant_geo_db[key] = JAPAN_GEOGRAPHY_DB[key]

            geo_context_str = json.dumps(relevant_geo_db, ensure_ascii=False, indent=2)
            
            # コンテキストが大きすぎる場合、キーのみに縮小
            if len(geo_context_str) > 5000:
                logger.warning(f"地名辞書が大きすぎ ({len(geo_context_str)}B)。キーのみに縮小。")
                geo_context_str = json.dumps(list(relevant_geo_db.keys()), ensure_ascii=False)
                
            logger.info(f"AIに渡す地名辞書(絞込済): {list(relevant_geo_db.keys())}")
        except Exception as e:
            logger.error(f"地名辞書の準備中にエラー: {e}", exc_info=True)
            geo_context_str = "{}" # エラー時は空の辞書

    # テキストが長すぎる場合、先頭500文字に切り詰める
    input_texts_jsonl = df_batch.apply(
        lambda row: json.dumps(
            {"id": row['id'], "text": str(row['ANALYSIS_TEXT_COLUMN'])[:500]},
            ensure_ascii=False
        ),
        axis=1
    ).tolist()

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
             a. 「地名辞書」の【値】(例: "呉市", "中区") または【キー】(例: "広島市") に一致する、最も文脈に関連性の高いものを【1つだけ】選ぶ。
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
        response_str = chain.invoke(invoke_params)
        logger.debug(f"AI Tagging - Raw response received.")

        results = []
        expected_keys = list(categories_to_tag.keys())
        
        # マークダウン ````jsonl ... ``` を除去
        match = re.search(r'```(?:jsonl|json)?\s*([\s\S]*?)\s*```', response_str, re.DOTALL)
        jsonl_content = match.group(1).strip() if match else response_str.strip()

        for line in jsonl_content.strip().split('\n'):
            cleaned_line = line.strip()
            if not cleaned_line: continue
            try:
                data = json.loads(cleaned_line)
                row_result = {"id": data.get("id")}
                # AIが 'categories' でラップする場合と、しない場合の両方に対応
                tag_source = data.get('categories', data)
                
                if not isinstance(tag_source, dict):
                    raise json.JSONDecodeError(f"tag_source is not a dict: {tag_source}", "", 0)

                for key in expected_keys:
                    # AIの回答キーが ' カテゴリ ' のように空白を含む場合に対応
                    found_key = None
                    for resp_key in tag_source.keys():
                        if str(resp_key).strip() == key:
                            found_key = resp_key
                            break
                    
                    raw_value = tag_source.get(found_key) if found_key else None

                    # --- "市区町村キーワード" の処理 (単一文字列) ---
                    if key == "市区町村キーワード":
                        processed_value = ""
                        if isinstance(raw_value, list) and raw_value:
                            # リストで返ってきた場合、最初の要素を採用
                            processed_value = str(raw_value[0]).strip()
                        elif raw_value is not None and str(raw_value).strip():
                            # 文字列で返ってきた場合
                            processed_value = str(raw_value).strip()
                        
                        # 該当なし等の表現を空文字に統一
                        if processed_value.lower() in ["該当なし", "none", "null", "", "n/a"]:
                            row_result[key] = ""
                        else:
                            row_result[key] = processed_value
                    
                    # --- その他のカテゴリの処理 (カンマ区切り文字列) ---
                    else:
                        processed_values = []
                        if isinstance(raw_value, list):
                            processed_values = sorted(list(set(
                                str(val).strip() for val in raw_value if str(val).strip()
                            )))
                        elif raw_value is not None and str(raw_value).strip():
                            # 単一文字列で返ってきた場合もリストに格納
                            processed_values = [str(raw_value).strip()]
                        
                        # 既存コード (L304) に合わせ、カンマ区切りの文字列として格納
                        row_result[key] = ", ".join(processed_values)
                
                results.append(row_result)
                
            except (json.JSONDecodeError, AttributeError) as json_e:
                logger.warning(f"AIタグ付け回答パース失敗: {cleaned_line} - Error: {json_e}")
                id_match = re.search(r'"id":\s*(\d+)', cleaned_line)
                if id_match:
                    # パース失敗時は、IDのみの空の行を追加 (マージが失敗しないように)
                    results.append({"id": int(id_match.group(1))})
                    
        return pd.DataFrame(results) if results else pd.DataFrame(columns=['id'] + list(expected_keys))

    except Exception as e:
        logger.error(f"AIタグ付けバッチ処理中エラー: {e}", exc_info=True)
        st.error(f"AIタグ付け処理エラー: {e}")
        return pd.DataFrame()  # 失敗時は空のDF


# --- 7. (★) Step A: UI描画関数 ---

def update_progress_ui(
    progress_placeholder: st.delta_generator.DeltaGenerator,
    log_placeholder: st.delta_generator.DeltaGenerator,
    processed_rows: int,
    total_rows: int,
    message_prefix: str
):
    """
    (Step A) の進捗バーとログエリアを更新する (DRY)
    (★) 要件: AI読み込み時間の進捗を0～100％で表示
    """
    try:
        # total_rows が 0 の場合 DivisionByZero を防ぐ
        if total_rows == 0:
            progress_percent = 1.0
        else:
            progress_percent = min(processed_rows / total_rows, 1.0)
            
        progress_text = f"[{message_prefix}] 処理中: {processed_rows}/{total_rows} 件 ({progress_percent:.0%})"
        progress_placeholder.progress(progress_percent, text=progress_text)

        # ログ表示 (最新50件)
        log_text_for_ui = "\n".join(st.session_state.log_messages[-50:])
        log_placeholder.text_area(
            "実行ログ (最新50件):",
            log_text_for_ui,
            height=200,
            key=f"log_update_{message_prefix}_{processed_rows}", # 重複キーを避ける
            disabled=True
        )
    except Exception as e:
        # UIの更新エラーはログに警告のみ残し、処理は続行
        logger.warning(f"UI update failed: {e}")

def render_step_a():
    """(Step A) タグ付け処理のUIを描画する"""
    st.title("🏷️ Step A: AIタグ付け & キュレーション")

    # Step A 固有のセッションステートを初期化
    if 'cancel_analysis' not in st.session_state:
        st.session_state.cancel_analysis = False
    if 'generated_categories' not in st.session_state:
        st.session_state.generated_categories = {}
    if 'selected_categories' not in st.session_state:
        st.session_state.selected_categories = set()
    if 'analysis_prompt_A' not in st.session_state:
        st.session_state.analysis_prompt_A = ""
    if 'selected_text_col' not in st.session_state:
        st.session_state.selected_text_col = {}
    if 'tagged_df_A' not in st.session_state:
        st.session_state.tagged_df_A = pd.DataFrame()

    st.header("Step 1: 分析対象ファイルのアップロード")
    uploaded_files = st.file_uploader(
        "分析したい Excel / CSV ファイル（複数可）",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key="uploader_A"
    )

    if not uploaded_files:
        st.info("分析を開始するには、ExcelまたはCSVファイルをアップロードしてください。")
        return

    # ファイル読み込み処理
    valid_files_data = {}
    error_messages = []
    for f in uploaded_files:
        df, err = read_file(f)
        if err:
            error_messages.append(f"**{f.name}**: {err}")
        else:
            valid_files_data[f.name] = df
            
    if error_messages:
        st.error("以下のファイルは読み込めませんでした:\n" + "\n".join(error_messages))
    if not valid_files_data:
        st.warning("読み込み可能なファイルがありません。")
        return

    st.header("Step 2: 分析指針の入力")
    analysis_prompt = st.text_area(
        "AIがタグ付けとキュレーションを行う際の指針を入力してください（必須）:",
        value=st.session_state.analysis_prompt_A,
        height=100,
        placeholder="例: 広島県の観光に関するInstagramの投稿。無関係な地域の投稿や、単なる挨拶・宣伝は除外したい。",
        key="analysis_prompt_input_A"
    )
    st.session_state.analysis_prompt_A = analysis_prompt

    if not analysis_prompt.strip():
        st.warning("分析指針は必須です。AIがデータを理解するために目的を入力してください。")
        return

    st.header("Step 3: AIによるカテゴリ候補の生成")
    st.markdown(f"（(★) 使用モデル: `{MODEL_FLASH_LITE}`）")
    if st.button("AIにカテゴリ候補を生成させる (Step 3)", key="gen_cat_button", type="primary"):
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Google APIキーが設定されていません。（サイドバーで設定してください）")
        else:
            with st.spinner(f"AI ({MODEL_FLASH_LITE}) が分析指針を読み解き、カテゴリを考案中..."):
                logger.info("AIカテゴリ生成ボタンクリック")
                # 「市区町村キーワード」は必須カテゴリとして固定
                st.session_state.generated_categories = {"市区町村キーワード": "地名辞書(JAPAN_GEOGRAPHY_DB)から抽出された市区町村名"}
                
                # (★) Step A の AI カテゴリ生成を呼び出し
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
                disabled=(cat == "市区町村キーワード")  # 必須項目は無効化
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
        
        # 以前の選択を記憶
        if st.session_state.selected_text_col.get(f_name) in cols_list:
            default_index = cols_list.index(st.session_state.selected_text_col.get(f_name))
        # 'text' や 'body' といった一般的な列名を推測
        elif any(c in cols_list for c in ['text', 'body', 'content', '投稿', '本文']):
            try:
                default_index = next(i for i, c in enumerate(cols_list) if c in ['text', 'body', 'content', '投稿', '本文'])
            except StopIteration:
                default_index = 0 # フォールバック
                
        selected_col = st.selectbox(f"**{f_name}** のテキスト列:", cols_list, index=default_index, key=f"col_select_{f_name}")
        selected_text_col_map[f_name] = selected_col
    st.session_state.selected_text_col = selected_text_col_map

    st.header("Step 6: 分析実行")
    st.markdown(f"（(★) 使用モデル: `{MODEL_FLASH_LITE}`）")
    
    col_run, col_cancel = st.columns([1, 1])
    with col_cancel:
        if st.button("キャンセル", key="cancel_button_A", use_container_width=True):
            st.session_state.cancel_analysis = True
            logger.warning("分析キャンセルボタンが押されました。")
            st.warning("次のバッチ処理後に分析をキャンセルします...")
    
    with col_run:
        if st.button("分析実行 (Step 6)", type="primary", key="run_analysis_A", use_container_width=True):
            st.session_state.cancel_analysis = False
            st.session_state.log_messages = []  # ログリセット
            st.session_state.tagged_df_A = pd.DataFrame()  # 結果リセット
            
            try:
                with st.spinner(f"Step A: AI分析処理中 ({MODEL_FLASH_LITE})..."):
                    logger.info("Step A 分析実行ボタンクリック")
                    # (★) 要件: 進捗を0～100％で表示
                    progress_placeholder = st.progress(0.0, text="処理待機中...")
                    log_placeholder = st.empty()

                    # --- 1. ファイル結合 ---
                    update_progress_ui(progress_placeholder, log_placeholder, 0, 100, "ファイル結合")
                    temp_dfs = []
                    for f_name, df in valid_files_data.items():
                        col_name = selected_text_col_map[f_name]
                        temp_df = df.rename(columns={col_name: 'ANALYSIS_TEXT_COLUMN'})
                        temp_dfs.append(temp_df)
                    
                    master_df = pd.concat(temp_dfs, ignore_index=True, sort=False)
                    master_df['id'] = master_df.index
                    if master_df.empty:
                        raise Exception("分析対象のデータがありません。")

                    # --- 2. 重複削除 ---
                    initial_row_count = len(master_df)
                    master_df.drop_duplicates(subset=['ANALYSIS_TEXT_COLUMN'], keep='first', inplace=True)
                    deduped_row_count = len(master_df)
                    logger.info(f"重複削除 完了。 {initial_row_count}行 -> {deduped_row_count}行")

                    # --- 3. (★) AI関連性フィルタリング (キュレーション) ---
                    total_filter_rows = len(master_df)
                    total_filter_batches = (total_filter_rows + FILTER_BATCH_SIZE - 1) // FILTER_BATCH_SIZE
                    all_filtered_results = []
                    
                    for i in range(0, total_filter_rows, FILTER_BATCH_SIZE):
                        if st.session_state.cancel_analysis:
                            raise Exception("分析がキャンセルされました")
                        
                        batch_df = master_df.iloc[i:i + FILTER_BATCH_SIZE]
                        current_batch_num = (i // FILTER_BATCH_SIZE) + 1
                        
                        # (★) 進捗表示 (0-100%)
                        update_progress_ui(
                            progress_placeholder, log_placeholder,
                            min(i + FILTER_BATCH_SIZE, total_filter_rows), total_filter_rows,
                            f"AIキュレーション (バッチ {current_batch_num}/{total_filter_batches})"
                        )
                        
                        filtered_df = filter_relevant_data_by_ai(batch_df, analysis_prompt)
                        if filtered_df is not None and not filtered_df.empty:
                            all_filtered_results.append(filtered_df)
                        
                        time.sleep(FILTER_SLEEP_TIME) # (★) Rate Limit 対策
                    
                    if not all_filtered_results:
                        raise Exception("AIフィルタリング処理に失敗しました。")

                    filter_results_df = pd.concat(all_filtered_results, ignore_index=True)
                    relevant_ids = filter_results_df[filter_results_df['relevant'] == True]['id']
                    filtered_master_df = master_df[master_df['id'].isin(relevant_ids)].copy()
                    filtered_row_count = len(filtered_master_df)
                    logger.info(f"AIフィルタリング 完了。 {deduped_row_count}行 -> {filtered_row_count}行")

                    if filtered_master_df.empty:
                        st.warning("AIキュレーションの結果、分析対象のデータが0件になりました。")
                        st.session_state.tagged_df_A = pd.DataFrame() # 空のDFをセット
                        progress_placeholder.progress(1.0, text="処理完了 (対象データ0件)")
                        return # 処理中断

                    # --- 4. (★) AIタグ付け ---
                    selected_category_definitions = {
                        cat: desc for cat, desc in st.session_state.generated_categories.items()
                        if cat in st.session_state.selected_categories
                    }
                    
                    master_df_for_tagging = filtered_master_df
                    total_rows = len(master_df_for_tagging)
                    all_tagged_results = []
                    total_batches = (total_rows + TAGGING_BATCH_SIZE - 1) // TAGGING_BATCH_SIZE
                    
                    for i in range(0, total_rows, TAGGING_BATCH_SIZE):
                        if st.session_state.cancel_analysis:
                            raise Exception("分析がキャンセルされました")
                        
                        batch_df = master_df_for_tagging.iloc[i:i + TAGGING_BATCH_SIZE]
                        current_batch_num = (i // TAGGING_BATCH_SIZE) + 1
                        
                        # (★) 進捗表示 (0-100%)
                        update_progress_ui(
                            progress_placeholder, log_placeholder,
                            min(i + TAGGING_BATCH_SIZE, total_rows), total_rows,
                            f"AIタグ付け (バッチ {current_batch_num}/{total_batches})"
                        )

                        tagged_df = perform_ai_tagging(batch_df, selected_category_definitions, analysis_prompt)
                        if tagged_df is not None and not tagged_df.empty:
                            all_tagged_results.append(tagged_df)
                        
                        time.sleep(TAGGING_SLEEP_TIME) # (★) Rate Limit 対策

                    if not all_tagged_results:
                        raise Exception("AIタグ付け処理に失敗しました。")

                    # --- 5. 最終マージ ---
                    logger.info("全AIタグ付け結果結合...");
                    tagged_results_df = pd.concat(all_tagged_results, ignore_index=True)

                    logger.info("最終マージ処理開始...");
                    # 元データとタグ付け結果を 'id' でマージ
                    final_df = pd.merge(master_df_for_tagging, tagged_results_df, on='id', how='right')
                    
                    # 'id' が重複してマージされた場合 (e.g., tagging_df に id 以外の列が重複)
                    # 最終的な列セットを定義
                    final_cols = list(master_df_for_tagging.columns) + [col for col in tagged_results_df.columns if col not in master_df_for_tagging.columns]
                    final_df = final_df[final_cols]

                    st.session_state.tagged_df_A = final_df
                    logger.info("Step A 分析処理 正常終了");
                    st.success("AIによる分析処理が完了しました。");
                    progress_placeholder.progress(1.0, text="処理完了")
                    update_progress_ui(progress_placeholder, log_placeholder, total_rows, total_rows, "処理完了")

            except Exception as e:
                logger.error(f"Step A 分析実行中にエラー: {e}", exc_info=True)
                st.error(f"分析実行中にエラーが発生しました: {e}")
                if 'progress_placeholder' in locals():
                    progress_placeholder.progress(1.0, text="エラーにより処理中断")

    # (★) 要件④: エクスポートリンクを表示
    if not st.session_state.tagged_df_A.empty:
        st.header("Step 7: 分析結果の確認とエクスポート")
        st.dataframe(st.session_state.tagged_df_A.head(50))

        @st.cache_data
        def convert_df_to_csv(df: pd.DataFrame) -> bytes:
            """DataFrameをUTF-8-SIGエンコードのCSV (bytes) に変換する"""
            return df.to_csv(encoding="utf-8-sig", index=False).encode("utf-8-sig")

        csv_data = convert_df_to_csv(st.session_state.tagged_df_A)
        st.download_button(
            label="分析結果CSV (Curated_Data.csv) をダウンロード",
            data=csv_data,
            file_name="Curated_Data.csv",
            mime="text/csv",
        )
        st.info("このCSVファイルを、Step B でアップロードして分析を続けてください。")

import networkx as nx # (★) Step B (共起ネットワーク) で必要
from itertools import combinations # (★) Step B (共起ネットワーク) で必要

def suggest_analysis_techniques_py(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    (Step B) データフレームを分析し、Pythonで実行可能な基本的な分析手法を提案する。
    (旧 `suggest_analysis_techniques` をリファクタリング)
    """
    suggestions = []
    if df is None or df.empty:
        logger.error("suggest_analysis_techniques_py: DFが空です。")
        return suggestions
        
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include='object').columns.tolist()
        datetime_cols = []

        # 日付列の候補を object 型から探す
        for col in object_cols:
             if df[col].isnull().sum() / len(df) > 0.5: continue # 欠損が5割超ならスキップ
             sample = df[col].dropna().head(50)
             if sample.empty: continue
             try:
                 # サンプルで日付変換を試みる
                 pd.to_datetime(sample, errors='raise')
                 # 成功したら全体をチェック
                 temp_dt = pd.to_datetime(df[col], errors='coerce').dropna()
                 if not temp_dt.empty and (temp_dt.dt.year.nunique() > 1 or temp_dt.dt.month.nunique() > 1 or temp_dt.dt.day.nunique() > 1 or col.lower() in ['date', 'time', 'timestamp', '日付', '日時']):
                     datetime_cols.append(col)
                     logger.info(f"列 '{col}' を日時列として認識しました。")
             except (ValueError, TypeError, OverflowError, pd.errors.ParserError):
                 pass # 日付でなければ無視

        numeric_cols = [col for col in numeric_cols if col != 'id'] # id列除外
        categorical_cols = [col for col in object_cols if col != 'ANALYSIS_TEXT_COLUMN' and col not in datetime_cols]
        flag_cols = [col for col in categorical_cols if col.endswith('キーワード')]
        other_categorical = [col for col in categorical_cols if not col.endswith('キーワード')]
        
        logger.info(f"提案分析(PY) - 数値:{numeric_cols}, カテゴリ(フラグ):{flag_cols}, カテゴリ(他):{other_categorical}, 日時:{datetime_cols}")

        potential_suggestions = []

        # (★) --- 1. 全体メトリクス ---
        potential_suggestions.append({
            "priority": 1, "name": "全体のメトリクス",
            "description": "投稿数、エンゲージメント、センチメント傾向など、データセット全体の概要を計算します。",
            "reason": "データ全体の状況把握に必須です。",
            "suitable_cols": [col for col in df.columns if 'センチメント' in col or 'いいね' in col or 'エンゲージメント' in col],
            "type": "python"
        })

        # 優先度1: 基本集計
        if flag_cols:
            potential_suggestions.append({
                "priority": 1, "name": "単純集計（頻度分析）",
                "description": "各キーワード（カテゴリ）がどのくらいの頻度で出現したかトップNを分析します。",
                "reason": f"キーワード列({len(flag_cols)}個)あり。基本指標です。",
                "suitable_cols": flag_cols,
                "type": "python" # (★) Python実行フラグ
            })

        # (★) --- 1. 市区町村別投稿数 (単純集計の具体化) ---
        if '市区町村キーワード' in flag_cols:
            potential_suggestions.append({
                "priority": 1, "name": "市区町村別投稿数",
                "description": "「市区町村キーワード」列の出現頻度を分析します。",
                "reason": "地域別の投稿ボリュームを把握します。",
                "suitable_cols": ['市区町村キーワード'],
                "type": "python"
            })

        # 優先度2: クロス集計
        if len(flag_cols) >= 2:
            potential_suggestions.append({
                "priority": 2, "name": "クロス集計（キーワード間）",
                "description": "キーワード間の組み合わせで多く出現するパターンを探ります。",
                "reason": f"複数キーワード列({len(flag_cols)}個)あり、関連性の発見に。",
                "suitable_cols": flag_cols,
                "type": "python"
            })
        if flag_cols and other_categorical:
             potential_suggestions.append({
                "priority": 2, "name": "クロス集計（キーワード×属性）",
                "description": f"キーワード({flag_cols[0]}など)と他の属性({', '.join(other_categorical)})の関係性を分析します。",
                "reason": f"キーワード列と他カテゴリ列({len(other_categorical)}個)あり。",
                "suitable_cols": flag_cols + other_categorical,
                "type": "python"
            })
            
        # (★) --- 2. 話題カテゴリ別 観光地TOP10 (クロス集計の具体化) ---
        # (★) Step Aで '話題カテゴリ' と '観光地キーワード' が生成されている前提
        if '話題カテゴリ' in df.columns and '観光地キーワード' in df.columns:
            potential_suggestions.append({
                "priority": 2, "name": "話題カテゴリ別 観光地TOP10",
                "description": "「話題カテゴリ」と「観光地キーワード」をクロス集計し、カテゴリ別の人気観光地を分析します。",
                "reason": "カテゴリと観光地の関連性を分析します。",
                "suitable_cols": ['話題カテゴリ', '観光地キーワード'],
                "type": "python"
            })

        # 優先度3: 時系列分析
        if datetime_cols and flag_cols:
            potential_suggestions.append({
                "priority": 3, "name": "時系列キーワード分析",
                "description": f"特定のキーワードの出現数が時間（{datetime_cols[0]}など）とともにどう変化したかトレンドを分析します。",
                "reason": f"キーワード列と日時列({len(datetime_cols)}個)あり。",
                "suitable_cols": {"datetime": datetime_cols, "keywords": flag_cols},
                "type": "python"
            })
            
        # (★) --- 3. 共起ネットワーク ---
        if 'ANALYSIS_TEXT_COLUMN' in df.columns:
            potential_suggestions.append({
                "priority": 3, "name": "共起ネットワーク",
                "description": "投稿テキスト内の単語の出現パターンを分析し、関連性の高い単語のネットワークを構築します。",
                "reason": "テキストデータから隠れたトピックや関連性を発見します。",
                "suitable_cols": ['ANALYSIS_TEXT_COLUMN'],
                "type": "python"
            })
            
        # 優先度4: テキストマイニング
        if 'ANALYSIS_TEXT_COLUMN' in df.columns:
            potential_suggestions.append({
                "priority": 4, "name": "テキストマイニング（頻出単語）",
                "description": "原文テキストから頻出する単語を抽出し、どのような言葉が多く使われているか全体像を把握します。",
                "reason": "原文テキストがあり、タグ付け以外のインサイト発見に。",
                "suitable_cols": ['ANALYSIS_TEXT_COLUMN'],
                "type": "python"
            })

        # (★) --- 4. 話題カテゴリ別 サマリ (Python + AI) ---
        if '話題カテゴリ' in df.columns and 'ANALYSIS_TEXT_COLUMN' in df.columns:
            potential_suggestions.append({
                "priority": 4, "name": "話題カテゴリ別 投稿数とサマリ",
                "description": "指定された話題カテゴリ（グルメ、自然など）ごとに投稿数を集計し、AIが投稿内容のサマリを生成します。",
                "reason": "カテゴリごとの主要な話題を把握します。",
                "suitable_cols": ['話題カテゴリ', 'ANALYSIS_TEXT_COLUMN'],
                "type": "python" # (★) AIを呼び出すが、メインロジックはPython
            })

        # (★) --- 4. 話題カテゴリ別 エンゲージメントTOP5 (Python + AI) ---
        engagement_cols = [col for col in numeric_cols if any(c in col.lower() for c in ['いいね', 'like', 'エンゲージメント', 'engagement'])]
        if '話題カテゴリ' in df.columns and 'ANALYSIS_TEXT_COLUMN' in df.columns and engagement_cols:
            potential_suggestions.append({
                "priority": 4, "name": "話題カテゴリ別 エンゲージメントTOP5と概要",
                "description": f"指定された話題カテゴリごとに、エンゲージメント（{engagement_cols[0]}）が高いTOP5投稿を抽出し、AIがその概要を生成します。",
                "reason": "カテゴリごとに「バズった」投稿の内容を把握します。",
                "suitable_cols": ['話題カテゴリ', 'ANALYSIS_TEXT_COLUMN'] + engagement_cols,
                "type": "python" # (★) AIを呼び出すが、メインロジックはPython
            })

        suggestions = sorted(potential_suggestions, key=lambda x: x['priority'])
        logger.info(f"Pythonベース提案(ソート後): {[s['name'] for s in suggestions]}")
        return suggestions

    except Exception as e:
        logger.error(f"Python分析手法提案中にエラー: {e}", exc_info=True)
        st.warning(f"分析手法提案中にエラー: {e}")
    return suggestions

def suggest_analysis_techniques_ai(
    user_prompt: str,
    df: pd.DataFrame,
    existing_suggestions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    (Step B) ユーザーの自由記述プロンプトに基づき、AIが追加の分析手法を提案する。
    (★) モデル: MODEL_FLASH_LITE (gemini-2.5-flash-lite)
    (旧 `get_suggestions_from_prompt` をリファクタリング)
    """
    logger.info("AIプロンプトベースの分析提案 (Flash Lite) を開始...")
    
    # (★) Step B の要件に基づき、FLASH_LITE モデルを明示的に指定
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.1) # 少しだけ創造性を持たせる
    if llm is None:
        logger.error("suggest_analysis_techniques_ai: LLM (Flash Lite) が利用できません。")
        return []

    try:
        col_info = []
        for col in df.columns:
            col_info.append(f"- {col} (型: {df[col].dtype}, 例: {df[col].dropna().iloc[0] if not df[col].dropna().empty else 'N/A'})")
        column_info_str = "\n".join(col_info[:15]) # 列が多すぎても困るので最大15
        
        existing_names = [s['name'] for s in existing_suggestions]
        
        prompt = PromptTemplate.from_template(
            """
            あなたはデータ分析の専門家です。ユーザーの「分析指示」と「データ構造」を読み、実行可能な「分析タスク」をJSONリスト形式で提案してください。
            
            # データ構造 (利用可能な列名):
            {column_info}
            
            # 既に提案済みのタスク (これらは提案しないでください):
            {existing_tasks}
            
            # ユーザーの分析指示:
            {user_prompt}
            
            # 指示:
            1. 「ユーザーの分析指示」を解釈し、具体的な分析タスク（例：「広島市と観光地の相関分析」）に分解する。
            2. 各タスクを以下のJSON形式で定義する。
            3. `name`はタスク名、`description`はAI（あなた自身）がこの後実行するタスクの具体的な指示（プロンプト）とする。
            4. `priority`は 5 固定、`type`は "ai" 固定とする。
            5. 指示が空、または解釈不能な場合は、空リスト [] を返す。
            
            # 回答 (JSONリスト形式のみ):
            [
              {{
                "priority": 5,
                "name": "（ユーザー指示に基づくタスク名1）",
                "description": "（このタスクを実行するためのAIへの具体的な指示プロンプト1）",
                "reason": "ユーザー指示に基づく",
                "suitable_cols": [],
                "type": "ai"
              }}
            ]
            """
        )
        chain = prompt | llm | StrOutputParser()
        response_str = chain.invoke({
            "column_info": column_info_str,
            "user_prompt": user_prompt,
            "existing_tasks": ", ".join(existing_names)
        })

        logger.info(f"AI追加提案(生): {response_str}")
        match = re.search(r'\[.*\]', response_str, re.DOTALL)
        if not match:
            logger.warning("AIがJSONリスト形式で応答しませんでした。")
            return []
            
        json_str = match.group(0)
        ai_suggestions = json.loads(json_str)
        
        # 'type': 'ai' が付与されているか確認 (AIの揺らぎ吸収)
        for s in ai_suggestions:
            s['type'] = 'ai'
            if 'priority' not in s: s['priority'] = 5
            
        logger.info(f"AI追加提案(パース済): {len(ai_suggestions)}件")
        return ai_suggestions

    except Exception as e:
        logger.error(f"AI追加提案の生成中にエラー: {e}", exc_info=True)
        st.warning(f"AI追加提案の生成中にエラーが発生しました: {e}")
        return []

# --- 8.1. (★) Step B: 分析実行ヘルパー (Python) ---
# (要件⑥: Pythonでできる部分はPythonで)

def run_simple_count(df: pd.DataFrame, suggestion: Dict[str, Any]) -> pd.DataFrame:
    """(Step B) 単純集計（頻度分析）を実行し、DataFrameを返す"""
    flag_cols = suggestion.get('suitable_cols', [])
    if not flag_cols:
        logger.warning("run_simple_count: 集計対象の列が見つかりません。")
        return pd.DataFrame()
    
    # 複数の列が対象の場合、最初の列を使用
    col_to_analyze = flag_cols[0]
    
    if col_to_analyze not in df.columns:
        logger.warning(f"run_simple_count: 列 '{col_to_analyze}' がDFに存在しません。")
        return pd.DataFrame()
        
    try:
        # Step A の出力 (", "区切り) を前提に、explode (分解) する
        s = df[col_to_analyze].astype(str).str.split(', ').explode()
        s = s[s.str.strip().isin(['', 'nan', 'None', 'N/A']) == False] # 空白やNaNを除去
        s = s.str.strip()
        
        if s.empty:
            logger.info("run_simple_count: 集計対象のキーワードがありませんでした。")
            return pd.DataFrame()
            
        counts = s.value_counts().head(50) # (★) Step C のために上位50件を返す
        counts_df = counts.reset_index()
        counts_df.columns = [col_to_analyze, 'count']
        return counts_df
            
    except Exception as e:
        logger.error(f"run_simple_count error: {e}", exc_info=True)
    return pd.DataFrame()

def run_crosstab(df: pd.DataFrame, suggestion: Dict[str, Any]) -> pd.DataFrame:
    """(Step B) クロス集計を実行し、DataFrameを返す"""
    cols = suggestion.get('suitable_cols', [])
    if len(cols) < 2:
        logger.warning("run_crosstab: クロス集計には2列以上必要です。")
        return pd.DataFrame()

    # 存在する列から2列選択
    existing_cols = [col for col in cols if col in df.columns]
    if len(existing_cols) < 2:
        logger.warning(f"run_crosstab: DF内に存在する列が2未満: {existing_cols}")
        return pd.DataFrame()

    col1, col2 = existing_cols[0], existing_cols[1]
    
    try:
        # (★) "市区町村キーワード" は explode して集計
        if col1 == "市区町村キーワード":
            df_exploded_1 = df.assign(**{col1: df[col1].str.split(', ')}).explode(col1)
        else:
            df_exploded_1 = df
            
        if col2 == "市区町村キーワード":
            df_exploded_2 = df_exploded_1.assign(**{col2: df_exploded_1[col2].str.split(', ')}).explode(col2)
        else:
            df_exploded_2 = df_exploded_1

        crosstab_df = pd.crosstab(df_exploded_2[col1].astype(str), df_exploded_2[col2].astype(str))
        
        # データを (col1, col2, count) のロングフォーマットに変換 (JSONにしやすいため)
        crosstab_long = crosstab_df.stack().reset_index()
        crosstab_long.columns = [col1, col2, 'count']
        crosstab_long = crosstab_long[crosstab_long['count'] > 0].sort_values(by='count', ascending=False)
        
        return crosstab_long.head(100) # (★) Step C のために上位100件を返す
        
    except Exception as e:
        logger.error(f"run_crosstab error: {e}", exc_info=True)
    return pd.DataFrame()

def run_timeseries(df: pd.DataFrame, suggestion: Dict[str, Any]) -> pd.DataFrame:
    """(Step B) 時系列分析を実行し、DataFrameを返す"""
    cols_dict = suggestion.get('suitable_cols', {})
    if not isinstance(cols_dict, dict) or 'datetime' not in cols_dict or 'keywords' not in cols_dict:
        logger.warning(f"run_timeseries: 列情報（datetime, keywords）が不十分です。")
        return pd.DataFrame()
        
    dt_cols = [col for col in cols_dict['datetime'] if col in df.columns]
    kw_cols = [col for col in cols_dict['keywords'] if col in df.columns]

    if not dt_cols:
        logger.warning("run_timeseries: 日時列が見つかりません。"); return pd.DataFrame()
    if not kw_cols:
        logger.warning("run_timeseries: キーワード列が見つかりません。"); return pd.DataFrame()

    dt_col = dt_cols[0]
    kw_col = kw_cols[0] # (★) 代表として最初のキーワード列を使用

    try:
        df_copy = df[[dt_col, kw_col]].copy()
        
        # 日付列をパース
        df_copy[dt_col] = pd.to_datetime(df_copy[dt_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[dt_col])
        
        # キーワード列を explode
        df_exploded = df_copy.assign(**{kw_col: df_copy[kw_col].str.split(', ')}).explode(kw_col)
        df_exploded[kw_col] = df_exploded[kw_col].str.strip()
        df_exploded = df_exploded[df_exploded[kw_col].isin(['', 'nan', 'None', 'N/A']) == False]
        
        if df_exploded.empty:
            logger.info("run_timeseries: 有効な日時/キーワードデータがありません。"); return pd.DataFrame()

        # (★) 日付 x キーワード でグループ化し、日毎の投稿数を集計
        time_df = df_exploded.groupby([pd.Grouper(key=dt_col, freq='D'), kw_col]).size().rename("count").reset_index()
        
        time_df.columns = ['date', 'keyword', 'count']
        time_df['date'] = time_df['date'].dt.strftime('%Y-%m-%d') # (★) JSON用に日付を文字列化
        
        # (★) Step C のため、キーワード別Top50件 + 日付順にソート
        top_keywords = df_exploded[kw_col].value_counts().head(50).index
        time_df_filtered = time_df[time_df['keyword'].isin(top_keywords)]
        
        return time_df_filtered.sort_values(by=['keyword', 'date'])
            
    except Exception as e:
        logger.error(f"run_timeseries error: {e}", exc_info=True)
    return pd.DataFrame()

def run_text_mining(df: pd.DataFrame, suggestion: Dict[str, Any]) -> pd.DataFrame:
    """(Step B) テキストマイニング（頻出単語）を実行し、DataFrameを返す"""
    text_col = suggestion.get('suitable_cols', ['ANALYSIS_TEXT_COLUMN'])[0]
    if text_col not in df.columns or df[text_col].empty:
        logger.warning(f"run_text_mining: テキスト列 '{text_col}' がないか、空です。")
        return pd.DataFrame()

    nlp = load_spacy_model()
    if nlp is None:
        st.error("spaCy日本語モデルのロードに失敗しました。")
        return pd.DataFrame()
            
    try:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            return pd.DataFrame()
            
        words = []
        target_pos = {'NOUN', 'PROPN', 'ADJ'} # (名詞, 固有名詞, 形容詞)
        stop_words = {
            'の', 'に', 'は', 'を', 'が', 'で', 'て', 'です', 'ます', 'こと', 'もの', 'それ', 'あれ',
            'これ', 'ため', 'いる', 'する', 'ある', 'ない', 'いう', 'よう', 'そう', 'など', 'さん'
        }
        
        # (★) 進捗表示のため、st.session_state に進捗を書き込む
        total_texts = len(texts)
        if 'progress_text' not in st.session_state:
             st.session_state.progress_text = ""
             
        st.session_state.progress_text = "テキストマイニング (spaCy) 処理中... 0%"

        for i, doc in enumerate(nlp.pipe(texts, disable=["parser", "ner"], batch_size=50)):
            for token in doc:
                if (token.pos_ in target_pos) and (not token.is_stop) and (token.lemma_ not in stop_words) and (len(token.lemma_) > 1):
                    words.append(token.lemma_)
            
            if (i + 1) % 100 == 0:
                percent = (i + 1) / total_texts
                st.session_state.progress_text = f"テキストマイニング (spaCy) 処理中... {percent:.0%}"

        if not words:
            logger.warning("run_text_mining: 抽出可能な有効な単語が見つかりませんでした。")
            return pd.DataFrame()

        word_counts = pd.Series(words).value_counts().head(100) # (★) Step C のために上位100件
        word_counts_df = word_counts.reset_index()
        word_counts_df.columns = ['word', 'count']
        
        st.session_state.progress_text = "テキストマイニング (spaCy) 完了。"
        return word_counts_df
        
    except Exception as e:
        logger.error(f"run_text_mining error: {e}", exc_info=True)
    return pd.DataFrame()

# (★) --- 新規追加: run_overall_metrics ---
def run_overall_metrics(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) データセット全体のメトリクスを計算する"""
    logger.info("run_overall_metrics 実行...")
    metrics = {}
    try:
        # 1. 投稿数
        metrics["total_posts"] = len(df)

        # 2. エンゲージメント (存在すれば)
        engagement_cols = [col for col in df.columns if any(c in col.lower() for c in ['いいね', 'like', 'エンゲージメント', 'engagement', 'retweet', 'リツイート'])]
        total_engagement = 0
        if engagement_cols:
            for col in engagement_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    total_engagement += df[col].sum()
            metrics["total_engagement"] = int(total_engagement) # JSON用にint化
        else:
            metrics["total_engagement"] = "N/A"

        # 3. センチメント (J列 = センチメント と仮定)
        # (★) J列は10番目の列 (0-indexed)。より堅牢なのは列名 'センチメント' を探すこと。
        sentiment_col = None
        if 'センチメント' in df.columns:
            sentiment_col = 'センチメント'
        elif len(df.columns) > 9 and 'センチメント' in str(df.columns[9]): # J列 (インデックス9)
            sentiment_col = df.columns[9]
            
        if sentiment_col:
            logger.info(f"センチメント列: {sentiment_col} を使用します。")
            pos_count = int(df[df[sentiment_col] == 'ポジティブ'].shape[0])
            neg_count = int(df[df[sentiment_col] == 'ネガティブ'].shape[0])
            
            metrics["positive_posts"] = pos_count
            metrics["negative_posts"] = neg_count
            
            # 4. センチメント傾向
            if (pos_count + neg_count) > 0:
                tendency = ((pos_count - neg_count) / (pos_count + neg_count)) * 100
                metrics["sentiment_tendency_percent"] = int(np.floor(tendency)) # 小数点以下切り捨て
            else:
                metrics["sentiment_tendency_percent"] = 0
        else:
            logger.warning("列 'センチメント' が見つかりませんでした。")
            metrics["positive_posts"] = "N/A (列 'センチメント' が見つかりません)"
            metrics["negative_posts"] = "N/A"
            metrics["sentiment_tendency_percent"] = "N/A"

        return metrics

    except Exception as e:
        logger.error(f"run_overall_metrics error: {e}", exc_info=True)
        return {"error": str(e)}

# (★) --- 新規追加: run_cooccurrence_network ---
def run_cooccurrence_network(df: pd.DataFrame, suggestion: Dict[str, Any]) -> pd.DataFrame:
    """(Step B) 共起ネットワークを構築し、エッジリストのDataFrameを返す"""
    logger.info("run_cooccurrence_network 実行...")
    text_col = suggestion.get('suitable_cols', ['ANALYSIS_TEXT_COLUMN'])[0]
    if text_col not in df.columns or df[text_col].empty:
        logger.warning(f"run_cooccurrence_network: テキスト列 '{text_col}' がないか、空です。")
        return pd.DataFrame()

    nlp = load_spacy_model()
    if nlp is None:
        st.error("spaCy日本語モデルのロードに失敗しました。")
        return pd.DataFrame()

    try:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            return pd.DataFrame()

        target_pos = {'NOUN', 'PROPN', 'ADJ'} # (名詞, 固有名詞, 形容詞)
        stop_words = {
            'の', 'に', 'は', 'を', 'が', 'で', 'て', 'です', 'ます', 'こと', 'もの', 'それ', 'あれ',
            'これ', 'ため', 'いる', 'する', 'ある', 'ない', 'いう', 'よう', 'そう', 'など', 'さん',
            '的', '人', '自分', '私', '僕', '何', 'その', 'この', 'あの'
        }
        
        # 1. 全単語を抽出
        all_words = []
        for doc in nlp.pipe(texts, disable=["parser", "ner"]):
            for token in doc:
                if (token.pos_ in target_pos) and (not token.is_stop) and (token.lemma_ not in stop_words) and (len(token.lemma_) > 1):
                    all_words.append(token.lemma_)
        
        if not all_words:
            logger.warning("run_cooccurrence_network: 抽出可能な単語がありません。")
            return pd.DataFrame()

        # 2. Top 100 単語セットを作成 (Step Bでは固定)
        top_n_words_limit = 100
        top_n_words_set = set(pd.Series(all_words).value_counts().head(top_n_words_limit).index)

        G = nx.Graph()
        
        # 3. ペアを作成
        for text in texts:
            doc = nlp(text)
            words_in_text = set()
            for token in doc:
                if (token.pos_ in target_pos) and (token.lemma_ in top_n_words_set):
                    words_in_text.add(token.lemma_)
            
            for word1, word2 in combinations(sorted(list(words_in_text)), 2):
                if G.has_edge(word1, word2):
                    G[word1][word2]['weight'] += 1
                else:
                    G.add_edge(word1, word2, weight=1)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            logger.info("run_cooccurrence_network: 有効なエッジが構築されませんでした。")
            return pd.DataFrame()

        # 4. エッジリストをDataFrameに変換
        edge_list = []
        for u, v, data in G.edges(data=True):
            edge_list.append({"source": u, "target": v, "weight": data['weight']})
        
        edges_df = pd.DataFrame(edge_list)
        
        # (★) Step C のため、重みでソートし上位500件に制限
        return edges_df.sort_values(by="weight", ascending=False).head(500)

    except Exception as e:
        logger.error(f"run_cooccurrence_network error: {e}", exc_info=True)
        return pd.DataFrame()

# (★) --- 新規追加: run_topic_category_summary ---
def run_topic_category_summary(df: pd.DataFrame, suggestion: Dict[str, Any]) -> pd.DataFrame:
    """(Step B) 話題カテゴリ別に投稿数、サマリ(AI)、上位キーワードを分析する"""
    logger.info("run_topic_category_summary 実行...")
    
    # (★) ユーザー定義のカテゴリリスト
    target_categories = ['グルメ', '自然', '歴史・文化', 'アート', 'イベント', '宿泊・温泉']
    
    # (★) Step Aで生成された列名を推測
    topic_col = '話題カテゴリ' # 仮
    keyword_col = '関連キーワード' # 仮 (存在すれば)
    
    if topic_col not in df.columns:
        logger.warning(f"run_topic_category_summary: 列 '{topic_col}' が見つかりません。")
        return pd.DataFrame([{"error": f"列 '{topic_col}' が見つかりません。"}])
    
    # '関連キーワード' がなければ、 '市区町村キーワード' で代用
    if keyword_col not in df.columns:
        if '市区町村キーワード' in df.columns:
            keyword_col = '市区町村キーワード'
        else:
            keyword_col = None # キーワード分析はスキップ
    
    results = []

    # (★) 進捗表示
    total_cats = len(target_categories)
    if 'progress_text' not in st.session_state:
            st.session_state.progress_text = ""
            
    for i, category in enumerate(target_categories):
        st.session_state.progress_text = f"話題カテゴリ分析中 ({i+1}/{total_cats}): {category}"
        
        # (★) カテゴリでDFをフィルタリング (Step A がカンマ区切り前提)
        df_filtered = df[df[topic_col].astype(str).str.contains(category, na=False)]
        post_count = len(df_filtered)
        
        if post_count == 0:
            results.append({
                "category": category,
                "post_count": 0,
                "summary_ai": "N/A (投稿なし)",
                "top_keywords": []
            })
            continue
        
        # (★) サマリ (AI Flash Lite)
        # サンプルテキストを作成 (最大10件)
        text_samples = df_filtered['ANALYSIS_TEXT_COLUMN'].dropna().sample(n=min(10, post_count), random_state=1).tolist()
        text_samples_str = "\n".join([f"- {text[:200]}..." for text in text_samples])
        
        # AI提案の 'description' を利用して run_ai_summary_batch を呼び出す
        ai_suggestion = {
            "description": f"「{category}」カテゴリに関する以下の投稿サンプルを読み、主要な話題を1～2文で要約してください。\nサンプル:\n{text_samples_str}"
        }
        summary_ai = run_ai_summary_batch(df_filtered, ai_suggestion) # (★) AI呼び出し
        
        # (★) 上位キーワード (Python)
        top_keywords = []
        if keyword_col and keyword_col in df_filtered.columns:
            s = df_filtered[keyword_col].astype(str).str.split(', ').explode()
            s = s[s.str.strip().isin(['', 'nan', 'None', 'N/A']) == False]
            s = s.str.strip()
            if not s.empty:
                top_keywords = s.value_counts().head(5).index.tolist()
        
        results.append({
            "category": category,
            "post_count": post_count,
            "summary_ai": summary_ai,
            "top_keywords": top_keywords
        })
        
        time.sleep(TAGGING_SLEEP_TIME) # (★) AI Rate Limit 対策

    st.session_state.progress_text = "話題カテゴリ分析 完了。"
    return pd.DataFrame(results)

# (★) --- 新規追加: run_topic_engagement_top5 ---
def run_topic_engagement_top5(df: pd.DataFrame, suggestion: Dict[str, Any]) -> pd.DataFrame:
    """(Step B) 話題カテゴリ別にエンゲージメントTOP5投稿と概要(AI)を分析する"""
    logger.info("run_topic_engagement_top5 実行...")

    target_categories = ['グルメ', '自然', '歴史・文化', 'アート', 'イベント', '宿泊・温泉']
    topic_col = '話題カテゴリ'
    text_col = 'ANALYSIS_TEXT_COLUMN'

    if topic_col not in df.columns:
        return pd.DataFrame([{"error": f"列 '{topic_col}' が見つかりません。"}])
    
    # (★) エンゲージメント列を特定
    suitable_cols = suggestion.get('suitable_cols', [])
    engagement_cols = [col for col in suitable_cols if any(c in col.lower() for c in ['いいね', 'like', 'エンゲージメント', 'engagement'])]
    
    if not engagement_cols:
        return pd.DataFrame([{"error": "エンゲージメント列 ('いいね' 等) が見つかりません。"}])
    
    engagement_col = engagement_cols[0] # 最初の列を代表として使用
    if engagement_col not in df.columns or not pd.api.types.is_numeric_dtype(df[engagement_col]):
        return pd.DataFrame([{"error": f"エンゲージメント列 '{engagement_col}' が数値列として存在しません。"}])

    results = []
    
    # (★) 進捗表示
    total_cats = len(target_categories)
    if 'progress_text' not in st.session_state:
            st.session_state.progress_text = ""

    for i, category in enumerate(target_categories):
        st.session_state.progress_text = f"エンゲージメントTOP5分析中 ({i+1}/{total_cats}): {category}"
        
        df_filtered = df[df[topic_col].astype(str).str.contains(category, na=False)]
        post_count = len(df_filtered)
        
        if post_count == 0:
            continue
            
        # (★) エンゲージメントでソートし、TOP5を抽出
        df_top5 = df_filtered.nlargest(5, engagement_col, keep='first')
        
        top5_posts_data = []
        
        if df_top5.empty:
                results.append({
                "category": category,
                "post_count": post_count,
                "top_posts": []
            })
                continue

        # (★) TOP5の投稿ごとに概要をAIで生成
        for _, row in df_top5.iterrows():
            post_text = str(row[text_col])
            engagement_value = row[engagement_col]
            
            # AI提案の 'description' を利用
            ai_suggestion = {
                "description": f"以下の投稿テキストを読み、内容を1文で要約してください。\nテキスト: {post_text[:500]}..."
            }
            summary_ai = run_ai_summary_batch(df_filtered, ai_suggestion) # (★) AI呼び出し
            
            top5_posts_data.append({
                "engagement": int(engagement_value), # JSON用にint化
                "summary_ai": summary_ai,
                "original_text_snippet": post_text[:100] # 確認用
            })
            
            time.sleep(TAGGING_SLEEP_TIME) # (★) AI Rate Limit 対策

        results.append({
            "category": category,
            "post_count": post_count,
            "top_posts": top5_posts_data
        })

    st.session_state.progress_text = "エンゲージメントTOP5分析 完了。"
    return pd.DataFrame(results)


# --- 8.2. (★) Step B: 分析実行ヘルパー (AI) ---
# (要件⑥: AIが必要な場合は gemini-2.5-flash-lite で実行)

def run_ai_summary_batch(df: pd.DataFrame, suggestion: Dict[str, Any]) -> str:
    """
    (Step B) AI提案に基づき、サンプルデータと指示をAIに渡し、分析結果（テキスト）を返す
    (★) モデル: MODEL_FLASH_LITE (gemini-2.5-flash-lite)
    """
    analysis_task_prompt = suggestion.get('description', suggestion.get('name', ''))
    if not analysis_task_prompt:
        logger.warning("run_ai_summary_batch: AIへの指示が空です。")
        return "AIへの指示が空のため、分析を実行できませんでした。"

    # (★) Step B の要件に基づき、FLASH_LITE モデルを明示的に指定
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.1)
    if llm is None:
        logger.error("run_ai_summary_batch: LLM (Flash Lite) が利用できません。")
        return "AIモデル(Flash Lite)が利用できず、分析を実行できませんでした。"

    logger.info(f"AIバッチ分析 (Flash Lite) 実行: {analysis_task_prompt}")

    try:
        # (★) データが多すぎないように、最大1000行をサンプリング
        if len(df) > 1000:
            df_sample = df.sample(1000, random_state=1)
        else:
            df_sample = df.copy()
            
        # (★) AIに渡すコンテキストをJSONL(文字列)として作成
        # テキスト列と、主要なカテゴリ列（キーワード列）を渡す
        text_col = 'ANALYSIS_TEXT_COLUMN'
        flag_cols = [col for col in df_sample.columns if col.endswith('キーワード')]
        cols_to_use = [text_col] + flag_cols
        
        # 存在しない列を除外
        cols_to_use = [col for col in cols_to_use if col in df_sample.columns]
        
        if not cols_to_use:
            return "AI分析に必要なテキスト列またはキーワード列が見つかりません。"
            
        df_sample_subset = df_sample[cols_to_use]
        
        # (★) 1行ずつJSON文字列に変換 (長すぎるテキストは切り詰め)
        sample_data_jsonl_list = []
        for _, row in df_sample_subset.iterrows():
            row_dict = row.to_dict()
            if text_col in row_dict:
                row_dict[text_col] = str(row_dict[text_col])[:300] # 300文字に制限
            sample_data_jsonl_list.append(json.dumps(row_dict, ensure_ascii=False))
        
        # (★) AIに渡すのは最大50件 (トークン制限対策)
        sample_data_context = "\n".join(sample_data_jsonl_list[:50])

        prompt = PromptTemplate.from_template(
            """
            あなたはデータアナリストです。以下の「分析タスク」を実行してください。
            
            # 分析タスク:
            {analysis_task}
            
            # 分析対象データ (JSONL形式のサンプル、最大50件):
            {sample_data}
            
            # 指示:
            1. 「分析対象データ」のサンプルを読み、「分析タスク」を実行する。
            2. 結果は、ビジネス上のインサイト（発見）がわかるように、簡潔なテキスト（箇条書き推奨）で要約して回答する。
            
            # 回答 (テキスト形式):
            """
        )
        chain = prompt | llm | StrOutputParser()
        
        response_str = chain.invoke({
            "analysis_task": analysis_task_prompt,
            "sample_data": sample_data_context
        })
        
        return response_str

    except Exception as e:
        logger.error(f"run_ai_summary_batch error: {e}", exc_info=True)
        return f"AI分析の実行中にエラーが発生しました: {e}"


# --- 8.3. (★) Step B: 分析実行ルーター ---

def execute_analysis(
    analysis_name: str,
    df: pd.DataFrame,
    suggestion: Dict[str, Any]
) -> Union[pd.DataFrame, str, Dict[str, Any]]: # (★) 辞書型も返す
    """
    (Step B) 分析名に基づき、適切なPythonまたはAIの実行関数を呼び出すルーター
    (★) 要件⑥: 「未実装」を防ぐためのキーとなる関数
    """
    try:
        analysis_type = suggestion.get('type', 'python') # デフォルトは 'python'
        
        if analysis_type == 'python':
            # (★) ユーザー指定の分析項目に対応
            if analysis_name == "全体のメトリクス":
                return run_overall_metrics(df, suggestion)
            
            elif analysis_name == "市区町村別投稿数":
                return run_simple_count(df, suggestion) # (★) run_simple_count を再利用
            
            elif analysis_name == "単純集計（頻度分析）":
                return run_simple_count(df, suggestion)
            
            elif analysis_name in ["クロス集計（キーワード間）", "クロス集計（キーワード×属性）", "話題カテゴリ別 観光地TOP10"]:
                return run_crosstab(df, suggestion) # (★) run_crosstab を再利用
            
            elif analysis_name == "時系列キーワード分析":
                return run_timeseries(df, suggestion)
            
            elif analysis_name == "テキストマイニング（頻出単語）":
                return run_text_mining(df, suggestion)
                
            elif analysis_name == "共起ネットワーク":
                return run_cooccurrence_network(df, suggestion)
            
            elif analysis_name == "話題カテゴリ別 投稿数とサマリ":
                return run_topic_category_summary(df, suggestion)
            
            elif analysis_name == "話題カテゴリ別 エンゲージメントTOP5と概要":
                return run_topic_engagement_top5(df, suggestion)
            
            else:
                # Pythonタイプだが未実装の場合 (フォールバック)
                logger.warning(f"Python分析 '{analysis_name}' の実行ロジックが定義されていません。AI分析にフォールバックします。")
                suggestion['description'] = f"データサンプルを使い、'{analysis_name}' を実行してください。"
                return run_ai_summary_batch(df, suggestion)
        
        elif analysis_type == 'ai':
            # (★) AIタイプの場合は、AI実行関数を呼び出す
            return run_ai_summary_batch(df, suggestion)
            
        else:
            return f"不明な分析タイプ ('{analysis_type}') です: {analysis_name}"
            
    except Exception as e:
        logger.error(f"execute_analysis ('{analysis_name}') 実行エラー: {e}", exc_info=True)
        return f"分析 '{analysis_name}' の実行中にエラーが発生しました: {e}"

# --- 8.4. (★) Step B: JSON出力ヘルパー ---
# (要件⑦: Step C の Proモデルに渡しやすい構造化データ)

def convert_results_to_json_string(results_dict: Dict[str, Any]) -> str:
    """
    (Step B) 分析結果の辞書 (DataFrame, str等を含む) を
    JSONL風の文字列に変換する。
    (★) 要件⑦: Step C (gemini-2.5-pro) に最適化された形式
    """
    final_output_lines = []
    
    # (★) 1. 全体サマリー (AIが状況を把握するため)
    summary_info = {
        "analysis_task": "OverallSummary",
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_results_count": len(results_dict),
        "analysis_names": list(results_dict.keys())
    }
    final_output_lines.append(json.dumps(summary_info, ensure_ascii=False))

    # (★) 2. 各分析結果をJSONLの1行として追加
    for name, data in results_dict.items():
        try:
            record = {"analysis_task": name}
            
            if isinstance(data, pd.DataFrame):
                # (★) DataFrameは 'records' 形式 (JSON配列) に変換
                # (ただし、データが大きすぎないように最大1000件に制限)
                if len(data) > 1000:
                    record["data"] = data.head(1000).to_dict(orient='records')
                    record["note"] = f"Data truncated. Showing 1000 of {len(data)} records."
                else:
                    record["data"] = data.to_dict(orient='records')
            
            elif isinstance(data, pd.Series):
                # Seriesは辞書に変換
                record["data"] = data.to_dict()
            
            elif isinstance(data, dict): # (★) 辞書型 (全体のメトリクス用)
                record["data"] = data

            elif isinstance(data, str):
                # AIの回答 (テキスト) はそのまま格納
                record["data"] = data
            
            elif data is None or (hasattr(data, 'empty') and data.empty):
                record["data"] = None
                record["note"] = "No data returned from analysis."
            
            else:
                # その他の型 (numpy intなど) は文字列に変換
                record["data"] = str(data)

            final_output_lines.append(json.dumps(record, ensure_ascii=False, default=str)) # default=strでnumpy型等に対応
        
        except Exception as e:
            logger.error(f"JSON変換エラー ({name}): {e}", exc_info=True)
            error_record = {
                "analysis_task": name,
                "data": None,
                "note": f"Error during JSON serialization: {e}"
            }
            final_output_lines.append(json.dumps(error_record, ensure_ascii=False))

    # (★) 各行を改行で結合した、単一の文字列 (JSONL形式) を返す
    return "\n".join(final_output_lines)


# --- 8.5. (★) Step B: UI描画関数 ---

def render_step_b():
    """(Step B) 分析手法の提案・実行・データ出力UIを描画する"""
    st.title("📊 Step B: 分析の実行とデータ出力")

    # Step B 固有のセッションステートを初期化
    if 'df_flagged_B' not in st.session_state:
        st.session_state.df_flagged_B = pd.DataFrame()
    if 'suggestions_B' not in st.session_state:
        st.session_state.suggestions_B = []
    if 'selected_analysis_B' not in st.session_state:
        st.session_state.selected_analysis_B = []
    if 'step_b_results' not in st.session_state:
        st.session_state.step_b_results = {}
    if 'step_b_json_output' not in st.session_state:
        st.session_state.step_b_json_output = None
    if 'progress_text' not in st.session_state:
         st.session_state.progress_text = ""

    # --- 1. ファイルアップロード (要件⑤) ---
    st.header("Step 1: キュレーション済みCSVのアップロード")
    st.info(f"Step A でエクスポートした CSV (Curated_Data.csv) をアップロードしてください。")
    uploaded_flagged_file = st.file_uploader(
        "フラグ付け済みCSVファイル",
        type=['csv'],
        key="step_b_uploader"
    )

    if uploaded_flagged_file:
        try:
            df, err = read_file(uploaded_flagged_file)
            if err:
                st.error(f"ファイル読み込みエラー: {err}")
                return
            st.session_state.df_flagged_B = df
            st.success(f"ファイル「{uploaded_flagged_file.name}」読込完了 ({len(df)}行)")
            with st.expander("データプレビュー (先頭5行)"):
                st.dataframe(df.head())
        except Exception as e:
            logger.error(f"Step B ファイル読込エラー: {e}", exc_info=True)
            st.error(f"ファイル読み込み中にエラー: {e}")
            return
    else:
        st.warning("分析を続けるには、Step A で生成したCSVファイルをアップロードしてください。")
        return # (★) DFがロードされるまで以下は実行しない

    df_B = st.session_state.df_flagged_B

    # --- 2. 分析手法の提案 (要件⑤) ---
    st.header("Step 2: 分析手法の提案")
    st.markdown(f"（(★) AI提案モデル: `{MODEL_FLASH_LITE}`）")
    
    analysis_prompt_B = st.text_area(
        "（任意）AIに追加で指示したい分析タスクを入力:",
        placeholder="例: 広島市と観光地の相関関係を深掘りしたい。\n例: ポジティブな意見とネガティブな意見の具体例を3つずつ抽出して。",
        key="step_b_prompt"
    )

    if st.button("💡 分析手法を提案させる (Step 2)", key="suggest_button_B", type="primary"):
        with st.spinner(f"データ構造と指示内容を分析し、手法を提案中 ({MODEL_FLASH_LITE})..."):
            # (★) 提案が実行されるたびに、古い結果をクリア
            st.session_state.step_b_results = {}
            st.session_state.step_b_json_output = None
            
            # 1. Pythonベースの提案
            base_suggestions = suggest_analysis_techniques_py(df_B)
            
            # 2. AIベースの提案 (Flash Lite)
            ai_suggestions = []
            if analysis_prompt_B.strip():
                ai_suggestions = suggest_analysis_techniques_ai(
                    analysis_prompt_B, df_B, base_suggestions
                )

            # 重複を除外し、優先度でソート
            base_names = {s['name'] for s in base_suggestions}
            filtered_ai_suggestions = [s for s in ai_suggestions if s['name'] not in base_names]
            
            all_suggestions = sorted(base_suggestions + filtered_ai_suggestions, key=lambda x: x['priority'])
            st.session_state.suggestions_B = all_suggestions
            st.success(f"分析手法の提案が完了しました ({len(all_suggestions)}件)。")

    # --- 3. 分析手法の選択 ---
    if st.session_state.suggestions_B:
        st.header("Step 3: 実行する分析の選択")
        
        default_selection = [s['name'] for s in st.session_state.suggestions_B[:min(len(st.session_state.suggestions_B), 5)]]
        
        st.session_state.selected_analysis_B = st.multiselect(
            "実行したい分析手法を選択（複数可）:",
            options=[s['name'] for s in st.session_state.suggestions_B],
            default=default_selection,
            key="multiselect_B"
        )
        
        with st.expander("選択した手法の詳細を表示"):
            for s in st.session_state.suggestions_B:
                if s['name'] in st.session_state.selected_analysis_B:
                    st.markdown(f"**{s['name']}** (`type: {s.get('type', 'N/A')}`)")
                    st.caption(f"説明: {s.get('description', 'N/A')}")
                    st.caption(f"理由: {s.get('reason', 'N/A')}")
                    st.markdown("---")

    # --- 4. 分析の実行 (要件⑥) ---
    if st.session_state.selected_analysis_B:
        st.header("Step 4: 分析の実行とデータのエクスポート")
        
        if st.button("分析を実行 (Step 4)", key="execute_button_B", type="primary", use_container_width=True):
            selected_names = st.session_state.selected_analysis_B
            all_suggestions_map = {s['name']: s for s in st.session_state.suggestions_B}
            
            # (★) 要件: 読み込み時間 (進捗) を 0-100% で表示
            st.info(f"計 {len(selected_names)} 件の分析を実行します...")
            progress_bar = st.progress(0.0, text="分析待機中...")
            st.session_state.progress_text = ""
            
            # (★) 進捗テキストを表示するプレースホルダ
            progress_text_placeholder = st.empty() 
            
            results_dict = {}
            
            for i, name in enumerate(selected_names):
                progress_percent = (i + 1) / len(selected_names)
                progress_bar.progress(progress_percent, text=f"分析中 ({i+1}/{len(selected_names)}): {name}")
                
                # (★) text_mining 等が st.session_state.progress_text を更新する
                st.session_state.progress_text = f"{name} を実行中..."
                
                # (★) 実行関数の進捗を表示
                with progress_text_placeholder.container():
                        st.info(st.session_state.progress_text) # (★) st.info で進捗を表示
                
                result_data = execute_analysis(name, df_B, all_suggestions_map[name])
                
                results_dict[name] = result_data
            
            # (★) 完了後、進捗テキストをクリア
            progress_text_placeholder.empty()
            
            progress_bar.progress(1.0, text="分析完了！ 構造化データ (JSON) を生成中...")
            
            # (★) 要件⑦: 構造化データ (JSON) の生成
            try:
                json_output_string = convert_results_to_json_string(results_dict)
                st.session_state.step_b_results = results_dict # (★) 生の結果も保存
                st.session_state.step_b_json_output = json_output_string # (★) JSON文字列を保存
                st.success("全ての分析が完了し、Step C用のJSONデータが生成されました。")
            except Exception as e:
                logger.error(f"Step B JSON出力変換エラー: {e}", exc_info=True)
                st.error(f"分析結果のJSON変換中にエラー: {e}")

    # --- 5. エクスポート (要件⑦) ---
    if st.session_state.step_b_json_output:
        st.header("Step 5: 分析データのエクスポート")
        st.info(f"以下のJSONファイルには、Step 4 で実行された {len(st.session_state.step_b_results)} 件の分析結果がすべて含まれています。")
        
        st.download_button(
            label="分析データ (analysis_data.json) をダウンロード",
            data=st.session_state.step_b_json_output,
            file_name="analysis_data.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("出力データ (JSONL) プレビュー")
        st.text_area(
            "JSONL (1行=1分析タスク)",
            value=st.session_state.step_b_json_output,
            height=300,
            key="json_preview_B",
            disabled=True
        )
        st.success("データをダウンロードし、Step C (AIレポート生成) に進んでください。")

# --- 9. (★) Step C: AIレポート生成 (Proモデル) ---
# (要件: Step Cは gemini-2.5-pro を使用)

def generate_step_c_prompt(jsonl_data_string: str) -> str:
    """
    (Step C) アップロードされた Step B の JSONL データを「下読み」し、
    gemini-2.5-pro への高品質な指示プロンプトを自動生成する。
    (★) この「下読み」自体は高速な flash-lite モデルを使用する
    """
    logger.info("Step C プロンプト自動生成 (Flash Lite) 実行...")
    
    # (★) Proモデルへの指示を生成するために、Flash-Liteモデルを使用
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.1)
    if llm is None:
        logger.error("generate_step_c_prompt: LLM (Flash Lite) が利用できません。")
        return "AIモデル(Flash Lite)が利用できませんでした。"

    # (★) トークン数を節約するため、JSONLの先頭100行（または5000文字）のみをコンテキストとする
    context_snippet = "\n".join(jsonl_data_string.splitlines()[:100])
    if len(context_snippet) > 5000:
        context_snippet = context_snippet[:5000] + "\n... (データ省略)"
        
    # (★) Proモデル (高知能) への指示プロンプトを、Flashモデル (高速) に生成させる
    prompt = PromptTemplate.from_template(
        """
        あなたは、シニアデータアナリストのチーフとして、部下（高知能AI）に分析レポートの作成を指示する立場です。
        以下の「分析データ（JSONL形式の抜粋）」を読み、最高のPowerPointレポートを作成させるための「指示プロンプト」を作成してください。

        # 指示プロンプトに含めるべき要素:
        1.  **役割定義**: 高知能AI（gemini-2.5-pro）に、優秀な経営コンサルタントとしての役割を与える。
        2.  **目的**: データからインサイトを抽出し、クライアント（例：観光協会、自治体）への提案を含むPowerPoint資料を作成することが目的であると伝える。
        3.  **データコンテキストの指定**: この後、完全なデータ（JSONL）が提供されることを示唆する。
        4.  **アウトプット形式（最重要）**: 以下の厳格なJSON形式（スライドの配列）で出力するよう【絶対に】指示する。
            `[ { "slide_title": "（スライドのタイトル）", "slide_content": ["（箇条書きの本文1）", "（箇条書きの本文2）"] } ]`
        5.  **必須スライド**: 「エグゼクティブ・サマリー」「分析の概要」「各分析タスク（{analysis_tasks}）の結果と考察」「結論とネクストステップ」を必ず含めるよう指示する。
        6.  **品質**: 「詳細なレポートを生成」「漏れがないように」といった品質要件を強調する。

        # 分析データ（JSONL形式の抜粋）:
        {jsonl_snippet}

        # 作成する「指示プロンプト」 (このプロンプト自体を回答してください):
        """
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        # JSONLから分析タスク名（"analysis_task"）を抽出
        task_names = []
        for line in jsonl_data_string.splitlines():
            try:
                task_name = json.loads(line).get("analysis_task")
                if task_name and task_name != "OverallSummary":
                    task_names.append(task_name)
            except json.JSONDecodeError:
                continue
        
        task_names_str = ", ".join(list(set(task_names))) # 重複削除
        if not task_names_str:
            task_names_str = "（JSONL内の各分析タスク）"

        generated_prompt = chain.invoke({
            "jsonl_snippet": context_snippet,
            "analysis_tasks": task_names_str
        })
        
        # (★) 万が一、AIが余計な前置きを生成した場合に備え、中核部分を抽出
        if "指示プロンプト" in generated_prompt:
             # 「指示プロンプト」以降のテキストを抽出
            generated_prompt = generated_prompt.split("指示プロンプト", 1)[-1]
            generated_prompt = re.sub(r'^.*?#', '#', generated_prompt, flags=re.DOTALL).strip() # 冒頭の不要なテキストを削除

        logger.info("Step C プロンプト自動生成 完了。")
        return generated_prompt

    except Exception as e:
        logger.error(f"generate_step_c_prompt error: {e}", exc_info=True)
        # (★) --- 修正: f-string内の波括弧をエスケープ ---
        # f-string内で { や } を文字列として表示するには、{{ および }} と2重にします。
        return f"# (★) プロンプト自動生成失敗: {e}\n\n# 指示:\nあなたは優秀な経営コンサルタントです。提供される「分析データ（JSONL）」を読み、以下のJSON形式でPowerPoint用の分析レポートを作成してください。\n\n[ {{ \"slide_title\": \"...\", \"slide_content\": [\"...\", \"...\"] }} ]"

def run_step_c_analysis(
    analysis_prompt: str,
    jsonl_data_string: str
) -> str:
    """
    (Step C) gemini-2.5-pro を使用して、分析データから構造化レポート (JSON) を生成する。
    (★) モデル: MODEL_PRO (gemini-2.5-pro)
    """
    logger.info("Step C AIレポート生成 (Pro) 実行...")
    
    # (★) 要件: Step C では Pro モデルを明示的に使用
    llm = get_llm(model_name=MODEL_PRO, temperature=0.2) # 高品質なレポートのため、少し創造性(0.2)を持たせる
    if llm is None:
        logger.error("run_step_c_analysis: LLM (Pro) が利用できません。")
        st.error(f"AIモデル({MODEL_PRO})が利用できません。APIキーを確認してください。")
        return '{"error": "AIモデル(Pro)が利用できませんでした。"}'

    # (★) 最終的に Pro モデルに渡すプロンプト
    # 指示プロンプト + 完全なJSONLデータ
    final_prompt_to_pro = f"""
    {analysis_prompt}
    
    # 分析データ (JSONL):
    {jsonl_data_string}
    
    # 回答 (指示されたJSON形式のみ):
    """
    
    try:
        # (★) 要件: 読み込み時間 (進捗) を表示
        # (Proモデルは時間がかかるため、Streamlitのスピナーで対応)
        response_str = llm.invoke(final_prompt_to_pro)
        
        logger.info("Step C AIレポート生成 (Pro) 完了。")
        
        # (★) 回答から JSON リスト ( [...] ) を抽出
        match = re.search(r'\[.*\]', response_str.content, re.DOTALL)
        if match:
            json_report_str = match.group(0)
            
            # (★) JSONとして有効か検証
            try:
                json.loads(json_report_str)
                return json_report_str # (★) 成功: JSON文字列を返す
            except json.JSONDecodeError as json_e:
                logger.error(f"AI (Pro) の回答がJSONパースに失敗: {json_e}")
                return f'[{{"slide_title": "AI回答パースエラー", "slide_content": ["AIの回答がJSON形式ではありませんでした。", "{str(json_e)}", "Raw: {response_str.content[:500]}..."]}}]'
        
        else:
            logger.error("AI (Pro) の回答にJSONリスト [...] が見つかりません。")
            return f'[{{"slide_title": "AI回答形式エラー", "slide_content": ["AIがJSONリスト形式 [...] で回答しませんでした。", "Raw: {response_str.content[:500]}..."]}}]'

    except Exception as e:
        logger.error(f"run_step_c_analysis error: {e}", exc_info=True)
        st.error(f"AIレポート生成 (Pro) 実行中にエラー: {e}")
        return f'[{{"slide_title": "実行時エラー", "slide_content": ["{str(e)}"]}}]'


def render_step_c():
    """(Step C) AIレポート生成UIを描画する"""
    st.title(f"🖋️ Step C: AI分析レポート生成 (using {MODEL_PRO})")

    # Step C 固有のセッションステート
    if 'step_c_jsonl_data' not in st.session_state:
        st.session_state.step_c_jsonl_data = None
    if 'step_c_prompt' not in st.session_state:
        st.session_state.step_c_prompt = None
    if 'step_c_report_json' not in st.session_state:
        st.session_state.step_c_report_json = None

    # --- 1. ファイルアップロード (要件⑧) ---
    st.header("Step 1: 分析データ (JSON) のアップロード")
    st.info("Step B でエクスポートした `analysis_data.json` をアップロードしてください。")
    uploaded_report_file = st.file_uploader(
        "分析データファイル (analysis_data.json)",
        type=['json', 'jsonl', 'txt'],
        key="step_c_uploader"
    )

    if uploaded_report_file:
        try:
            # (★) ファイル形式の多様性（要件⑧）に対応
            jsonl_data_string = uploaded_report_file.getvalue().decode('utf-8')
            st.session_state.step_c_jsonl_data = jsonl_data_string
            st.success(f"ファイル「{uploaded_report_file.name}」読込完了")
            
            # (★) ファイルがアップロードされたら、プロンプトを自動生成 (要件⑧)
            if st.session_state.step_c_prompt is None: # まだ生成されていない場合のみ
                with st.spinner(f"AI ({MODEL_FLASH_LITE}) が Step B のデータを下読みし、Proモデルへの指示を生成中..."):
                    st.session_state.step_c_prompt = generate_step_c_prompt(jsonl_data_string)

        except Exception as e:
            logger.error(f"Step C ファイル読込エラー: {e}", exc_info=True)
            st.error(f"ファイル読み込み中にエラー: {e}")
            return
    else:
        st.warning("分析を続けるには、Step B で生成した JSON ファイルをアップロードしてください。")
        return

    # --- 2. プロンプトの確認・編集 (要件⑨) ---
    st.header(f"Step 2: AI ({MODEL_PRO}) への指示プロンプト")
    if st.session_state.step_c_prompt:
        st.markdown(f"AI ({MODEL_FLASH_LITE}) が以下の指示プロンプトを自動生成しました。実行前に編集可能です。")
        
        edited_prompt = st.text_area(
            "AIへの指示プロンプト（編集可）:",
            value=st.session_state.step_c_prompt,
            height=300,
            key="step_c_prompt_editor"
        )
        st.session_state.step_c_prompt = edited_prompt # 編集内容を即座に保存
    else:
        st.warning("プロンプトがありません。ファイルを再アップロードしてください。")
        return

    # --- 3. 分析レポートの実行 (要件⑨) ---
    st.header("Step 3: AI分析レポートの実行")
    st.markdown(f"**（(★) 警告: {MODEL_PRO} を使用します。実行には時間がかかります）**")
    
    if st.button(f"分析レポートを生成 (Step 3)", key="execute_button_C", type="primary", use_container_width=True):
        if not st.session_state.step_c_jsonl_data or not st.session_state.step_c_prompt:
            st.error("データまたはプロンプトがありません。")
            return
        
        # (★) 要件: 読み込み時間 (進捗) を表示
        with st.spinner(f"AI ({MODEL_PRO}) が分析レポートを生成中です... (数分かかる場合があります)"):
            st.session_state.step_c_report_json = run_step_c_analysis(
                st.session_state.step_c_prompt,
                st.session_state.step_c_jsonl_data
            )
        st.success("AIによる分析レポートが生成されました！")

    # --- 4. 結果のプレビューとエクスポート (要件⑩) ---
    if st.session_state.step_c_report_json:
        st.header("Step 4: 分析レポート（JSON）の確認とエクスポート")
        st.info("以下の構造化JSONは、Step D (PowerPoint生成) で使用します。")

        # (★) 要件⑩: ダウンロード
        st.download_button(
            label="分析レポート (report_for_powerpoint.json) をダウンロード",
            data=st.session_state.step_c_report_json,
            file_name="report_for_powerpoint.json",
            mime="application/json",
            type="primary",
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("生成されたレポート プレビュー")
        
        try:
            # (★) プレビュー用にJSONをパース
            report_data = json.loads(st.session_state.step_c_report_json)
            if isinstance(report_data, list) and all(isinstance(item, dict) for item in report_data):
                st.text_area(
                    "AIが生成した構造化JSON:",
                    value=st.session_state.step_c_report_json,
                    height=300,
                    key="json_preview_C",
                    disabled=True
                )
                
                st.markdown("---")
                st.subheader("スライド構成 プレビュー")
                for i, slide in enumerate(report_data):
                    with st.expander(f"**スライド {i+1}: {slide.get('slide_title', '（タイトルなし）')}**"):
                        st.markdown("##### コンテンツ:")
                        if isinstance(slide.get('slide_content'), list):
                            for content_item in slide.get('slide_content', []):
                                st.markdown(f"- {content_item}")
                        else:
                            st.write(slide.get('slide_content', 'N/A'))
            else:
                st.error("AIの回答が期待したスライドのリスト形式ではありません。")
                st.text_area("AIの生回答 (JSON):", value=st.session_state.step_c_report_json, height=200, disabled=True)
                
        except Exception as e:
            st.error(f"レポートのプレビュー中にエラー: {e}")
            st.text_area("AIの生回答 (パース失敗):", value=st.session_state.step_c_report_json, height=200, disabled=True)
            
        st.success("データをダウンロードし、Step D (PowerPoint生成) に進んでください。")

# --- 10. (★) Step D: PowerPoint生成 (Pro / Flashモデル) ---
# (要件: Step Dは gemini-2.5-pro または gemini-2.5-flash を使用)

def create_powerpoint_presentation(
    template_file: Optional[BytesIO],
    report_data: List[Dict[str, Any]]
) -> BytesIO:
    """
    (Step D) テンプレート(.pptx)とスライド構成(JSON)に基づき、
    python-pptx を使用して最終的なPowerPointファイルを生成する。
    """
    logger.info("PowerPoint生成処理 開始...")
    
    try:
        # (★) 1. テンプレートの読み込み
        # テンプレートが指定されていればそれを使い、なければデフォルトで起動
        if template_file:
            template_file.seek(0) # BytesIOのポインタをリセット
            prs = Presentation(template_file)
            logger.info("アップロードされたテンプレートを使用してPPTXを生成します。")
        else:
            prs = Presentation() # デフォルトのプレゼンテーション
            logger.info("デフォルトのテンプレートを使用してPPTXを生成します。")

        # (★) 2. スライドレイアウトの選定
        # 最も一般的ない「タイトルとコンテンツ」レイアウトを探す
        # (pptx.slide.SlideLayouts[1] に相当することが多い)
        title_content_layout = None
        for layout in prs.slide_layouts:
            if layout.name.lower() in ["title and content", "タイトルとコンテンツ", "タイトルと内容"]:
                title_content_layout = layout
                break
        
        # 見つからなければ、インデックス[1] をフォールバックとして使用
        if title_content_layout is None:
            try:
                title_content_layout = prs.slide_layouts[1]
                logger.warning(f"「タイトルとコンテンツ」レイアウトが見つかりません。Layout[1] ({title_content_layout.name}) を使用します。")
            except IndexError:
                # それも失敗したら、インデックス[0] (通常はタイトルスライド) を使う
                title_content_layout = prs.slide_layouts[0]
                logger.error("Layout[1] も見つかりません。Layout[0] を使用します。")

        # (★) 3. スライドの生成 (JSONデータをループ)
        for i, slide_data in enumerate(report_data):
            slide_title = slide_data.get("slide_title", f"スライド {i+1}")
            slide_content = slide_data.get("slide_content", ["（コンテンツなし）"])
            
            # スライドをプレゼンテーションに追加
            slide = prs.slides.add_slide(title_content_layout)
            
            # タイトルを設定
            try:
                if slide.shapes.title:
                    slide.shapes.title.text = slide_title
            except Exception as e:
                logger.warning(f"スライド {i+1} のタイトル設定失敗: {e}")

            # コンテンツを設定
            try:
                # プレースホルダを探す (通常、タイトル以外の最初のプレースホルダが本文)
                content_placeholder = None
                for shape in slide.placeholders:
                    if shape.is_placeholder and not shape.has_text_frame:
                        continue
                    if shape.placeholder_format.idx > 0: # 0はタイトルが多い
                        content_placeholder = shape
                        break
                
                if content_placeholder:
                    # コンテンツがリスト形式の場合、箇条書きとして結合
                    if isinstance(slide_content, list):
                        # (★) 箇条書きのレベル設定
                        tf = content_placeholder.text_frame
                        tf.clear() # 既存のテキストをクリア
                        
                        p = tf.paragraphs[0]
                        p.text = str(slide_content[0])
                        p.level = 0
                        
                        for item in slide_content[1:]:
                            p = tf.add_paragraph()
                            p.text = str(item)
                            p.level = 0 # (★) 全てレベル0 (第一階層) に設定
                            
                    else:
                        # リストでない場合 (文字列など)
                        content_placeholder.text = str(slide_content)
                else:
                    logger.warning(f"スライド {i+1} でコンテンツプレースホルダが見つかりません。")
                    
            except Exception as e:
                logger.error(f"スライド {i+1} ('{slide_title}') のコンテンツ設定中にエラー: {e}", exc_info=True)

        logger.info("PowerPoint生成処理 完了。")

        # (★) 4. メモリ（BytesIO）に保存して返す
        file_stream = BytesIO()
        prs.save(file_stream)
        file_stream.seek(0)
        return file_stream

    except Exception as e:
        logger.error(f"create_powerpoint_presentation 全体でエラー: {e}", exc_info=True)
        st.error(f"PowerPointの生成に失敗しました: {e}")
        return None

def run_step_d_ai_correction(
    current_report_json: str,
    correction_prompt: str
) -> str:
    """
    (Step D) ユーザーの修正指示に基づき、AIがスライド構成(JSON)を修正して返す。
    (★) モデル: MODEL_PRO (gemini-2.5-pro) または MODEL_FLASH
    """
    logger.info("Step D AIによるJSON修正 (Pro) 実行...")
    
    # (★) 要件: Step D では Pro (または Flash) を使用
    llm = get_llm(model_name=MODEL_PRO, temperature=0.1)
    if llm is None:
        logger.error("run_step_d_ai_correction: LLM (Pro) が利用できません。")
        st.error(f"AIモデル({MODEL_PRO})が利用できません。APIキーを確認してください。")
        return current_report_json # エラー時は元のJSONを返す

    prompt = PromptTemplate.from_template(
        """
        あなたは、PowerPointレポートのJSON構成データを編集するアシスタントです。
        以下の「現在のレポートJSON」に対し、「修正指示」を適用し、修正後の【JSONのみ】を回答してください。

        # 指示:
        1.  「修正指示」を正確に実行する（例：「スライドを削除」「順番を変更」「文章を要約」）。
        2.  形式は、入力と【全く同じJSONリスト形式】 `[ {{ ... }} ]` を維持する。
        3.  JSON以外の余計なテキスト（「修正しました」など）は【絶対に】含めない。

        # 現在のレポートJSON:
        {current_json}

        # 修正指示:
        {user_prompt}

        # 回答 (修正後のJSONのみ):
        """
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        response_str = chain.invoke({
            "current_json": current_report_json,
            "user_prompt": correction_prompt
        })
        
        # (★) 回答から JSON リスト ( [...] ) を抽出
        match = re.search(r'\[.*\]', response_str, re.DOTALL)
        if match:
            json_report_str = match.group(0)
            
            # (★) JSONとして有効か検証
            try:
                json.loads(json_report_str)
                logger.info("Step D AIによるJSON修正 完了。")
                return json_report_str # (★) 成功: 修正後のJSON文字列を返す
            except json.JSONDecodeError as json_e:
                logger.error(f"AI (Pro) のJSON修正回答がパース失敗: {json_e}")
                st.error("AIがJSON形式で回答しませんでした。修正は適用されていません。")
                return current_report_json # エラー時は元のJSONを返す
        
        else:
            logger.error("AI (Pro) のJSON修正回答に [...] が見つかりません。")
            st.error("AIの回答形式が不正です。修正は適用されていません。")
            return current_report_json # エラー時は元のJSONを返す

    except Exception as e:
        logger.error(f"run_step_d_ai_correction error: {e}", exc_info=True)
        st.error(f"AIによる修正実行中にエラー: {e}")
        return current_report_json # エラー時は元のJSONを返す

def render_step_d():
    """(Step D) PowerPoint生成UIを描画する"""
    st.title(f"プレゼンテーション (PowerPoint) 生成 (Step D)")

    # Step D 固有のセッションステート
    if 'step_d_template_file' not in st.session_state:
        st.session_state.step_d_template_file = None
    if 'step_d_report_data' not in st.session_state:
        st.session_state.step_d_report_data = [] # (★) JSONをパースした「辞書のリスト」
    if 'step_d_generated_pptx' not in st.session_state:
        st.session_state.step_d_generated_pptx = None

    # --- 1. テンプレートのアップロード (要件⑪) ---
    st.header("Step 1: テンプレート PowerPoint のアップロード")
    st.info("（オプション）使用したい .pptx テンプレートがあればアップロードしてください。なければデフォルトデザインで生成されます。")
    template_file = st.file_uploader(
        "PowerPoint テンプレート (.pptx)",
        type=['pptx'],
        key="step_d_template_uploader"
    )
    
    # アップロードされたらセッションに保存
    if template_file:
        st.session_state.step_d_template_file = BytesIO(template_file.getvalue())
        st.success(f"テンプレート「{template_file.name}」を読み込みました。")
    
    # --- 2. Step C 分析結果のアップロード (要件⑫) ---
    st.header("Step 2: Step C 分析レポート (JSON) のアップロード")
    st.info("Step C でエクスポートした `report_for_powerpoint.json` をアップロードしてください。")
    report_file = st.file_uploader(
        "分析レポートファイル (report_for_powerpoint.json)",
        type=['json'],
        key="step_d_report_uploader"
    )

    if report_file:
        try:
            report_json_string = report_file.getvalue().decode('utf-8')
            report_data = json.loads(report_json_string)
            
            # (★) 正常なスライド構成 (辞書のリスト) かをチェック
            if isinstance(report_data, list) and all(isinstance(item, dict) for item in report_data):
                # (★) 読み込んだ内容を「辞書のリスト」としてセッションに保存
                if not st.session_state.step_d_report_data: # まだ読み込んでいない場合のみ
                    st.session_state.step_d_report_data = report_data
                    st.success(f"分析レポート「{report_file.name}」を読み込みました ({len(report_data)}スライド)。")
            else:
                st.error("アップロードされたJSONが期待する形式（スライドのリスト）ではありません。")
                st.session_state.step_d_report_data = []
        except Exception as e:
            logger.error(f"Step D JSONレポート読込エラー: {e}", exc_info=True)
            st.error(f"分析レポートの読み込み中にエラー: {e}")
            st.session_state.step_d_report_data = []
    
    if not st.session_state.step_d_report_data:
        st.warning("PowerPointを生成するには、Step C で生成した JSON レポートをアップロードしてください。")
        return

    # --- 3. 目次（スライド構成）の編集 (要件⑫) ---
    st.header("Step 3: スライド構成の確認・編集")
    st.info("（(★) マウスのドラッグ＆ドロップでスライドの順番を入れ替えることができます）")

    try:
        # (★) --- 修正: sort_items が list[str] を要求するエラーへの対応 ---
        
        # 1. ヘッダー(文字列)のリストと、ヘッダーから元の辞書へのマッピングを作成
        # (★) ヘッダーが一意であることを保証するため、インデックスを付与
        headers_list = []
        header_to_item_map = {}
        
        if not st.session_state.step_d_report_data:
             st.warning("JSONデータが空か、正しく読み込まれていません。")
             return # (★) データがなければここで停止

        for i, item in enumerate(st.session_state.step_d_report_data):
            if not isinstance(item, dict):
                st.error(f"データ形式エラー: {item} は辞書ではありません。Step C のJSON出力を確認してください。")
                continue # (★) 辞書でないデータはスキップ
                
            # (★) 表示するヘッダー (Markdown文字列)。インデックスを付けて一意性を担保
            header_str = f"**{i+1}: {item.get('slide_title', '（タイトルなし）')}**"
            headers_list.append(header_str)
            header_to_item_map[header_str] = item # (★) 元の辞書をマッピング

        # (★) 2. sort_items には「文字列のリスト (list[str])」を渡す
        # (★) このリスト (headers_list) に辞書が含まれていないことを確認
        if not all(isinstance(h, str) for h in headers_list):
            st.error("内部エラー: ヘッダーリストの作成に失敗しました。")
            return

        sorted_headers = sort_items(
            items=headers_list, # (★) list[str] を渡す
            key="sortable_slides_v2" # (★) キーを変更してリフレッシュ
        )
        
        # (★) 3. 並び替えられたヘッダーのリストを使い、元の辞書のリストを再構築
        cleaned_sorted_data = []
        for header in sorted_headers:
            if header in header_to_item_map:
                cleaned_sorted_data.append(header_to_item_map[header]) # (★) マッピングから元の辞書を取得
            else:
                logger.error(f"マッピングエラー: ソート後のヘッダー '{header}' が見つかりません。")
            
        # (★) 4. 並び替えられたクリーンな結果をセッションステートに上書き保存
        st.session_state.step_d_report_data = cleaned_sorted_data
        
    except TypeError as te:
        # (★) TypeError をキャッチ (今回のエラー)
        logger.error(f"streamlit-sortables 引数エラー: {te}", exc_info=True)
        st.error(f"スライド編集UIの描画に失敗: {te}。ライブラリ (streamlit-sortables) が正しくインストールされているか確認してください。")
        # エラーが発生しても、元のデータを表示しようと試みる
        for item in st.session_state.step_d_report_data:
            st.markdown(f"- **{item.get('slide_title', 'N/A')}**")
            
    except Exception as e:
        # その他の予期せぬエラー
        logger.error(f"streamlit-sortables 実行エラー: {e}", exc_info=True)
        st.error(f"スライド編集UIの描画に失敗: {e}。ライブラリ (streamlit-sortables) が正しくインストールされているか確認してください。")
        # エラーが発生しても、元のデータを表示しようと試みる
        if st.session_state.step_d_report_data:
            for item in st.session_state.step_d_report_data:
                st.markdown(f"- **{item.get('slide_title', 'N/A')}**")
            
    except Exception as e:
        # その他の予期せぬエラー
        logger.error(f"streamlit-sortables 実行エラー: {e}", exc_info=True)
        st.error(f"スライド編集UIの描画に失敗: {e}。ライブラリ (streamlit-sortables) が正しくインストールされているか確認してください。")
        # エラーが発生しても、元のデータを表示しようと試みる
        for item in st.session_state.step_d_report_data:
            st.markdown(f"- **{item.get('slide_title', 'N/A')}**")

    # --- 4. 修正指示 (AIによるJSON編集) (要件⑭) ---
    st.header("Step 4: (オプション) AIによる内容の修正指示")
    st.markdown(f"（(★) 使用モデル: `{MODEL_PRO}`）")
    
    with st.expander("AIにスライド内容の修正を指示する"):
        correction_prompt = st.text_area(
            "修正内容を具体的に指示してください:",
            placeholder=(
                "例: 「エグゼクティブ・サマリー」スライドの箇条書きを3点に要約して。\n"
                "例: 「共起ネットワーク」スライドを削除して。\n"
                "例: 全ての「結論」を「提案」という言葉に置き換えて。"
            ),
            key="step_d_correction_prompt"
        )
        
        if st.button("AIでスライド構成を修正", key="run_ai_correction_D", type="secondary"):
            if correction_prompt.strip():
                with st.spinner(f"AI ({MODEL_PRO}) がスライド構成 (JSON) を修正中..."):
                    # 現在のJSON構成（文字列）と指示をAIに渡す
                    current_json_str = json.dumps(st.session_state.step_d_report_data, ensure_ascii=False)
                    
                    # (★) AI (Pro) が JSON を修正
                    corrected_json_str = run_step_d_ai_correction(current_json_str, correction_prompt)
                    
                    # (★) 返ってきたJSON文字列でセッションステートを上書き
                    try:
                        corrected_data = json.loads(corrected_json_str)
                        if isinstance(corrected_data, list):
                            st.session_state.step_d_report_data = corrected_data
                            st.success("AIによるスライド構成の修正が完了しました。Step 3 の構成が更新されています。")
                            st.rerun() # UIを即時更新
                        else:
                            st.error("AIがリスト形式でないデータを返しました。修正はキャンセルされました。")
                    except Exception as e:
                        st.error(f"AIの回答のパースに失敗: {e}。修正はキャンセルされました。")
            else:
                st.warning("修正指示を入力してください。")

    # --- 5. PowerPoint生成・エクスポート (要件⑬, ⑭) ---
    st.header("Step 5: PowerPointの生成とエクスポート")
    
    if st.button("PowerPointを生成 (Step 5)", key="generate_pptx_D", type="primary", use_container_width=True):
        st.session_state.step_d_generated_pptx = None # 古いデータをクリア
        
        with st.spinner("PowerPointファイルを生成中..."):
            generated_file_stream = create_powerpoint_presentation(
                st.session_state.step_d_template_file,
                st.session_state.step_d_report_data
            )
            
            if generated_file_stream:
                st.session_state.step_d_generated_pptx = generated_file_stream.getvalue()
                st.success("PowerPointファイルの生成が完了しました。")
            else:
                st.error("PowerPointファイルの生成に失敗しました。")

    if st.session_state.step_d_generated_pptx:
        st.download_button(
            label="生成された PowerPoint をダウンロード",
            data=st.session_state.step_d_generated_pptx,
            file_name="AI_Analysis_Report.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True
        )
        st.balloons()


# --- 11. (★) Main関数 (アプリケーション実行) ---
def main():
    """Streamlitアプリケーションのメイン実行関数"""
    st.set_page_config(page_title="AI Data Analysis App", layout="wide")
    
    # グローバルなセッションステート
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'A' # 初期ステップ
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []

    # --- サイドバー (ナビゲーション) ---
    with st.sidebar:
        st.title("AI レポーティング App")
        st.markdown("---")
        
        st.header("⚙️ AI 設定 (必須)")
        google_api_key = st.text_input("Google API Key", type="password", key="api_key_global")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        if not os.getenv("GOOGLE_API_KEY"):
            st.warning("AI機能を利用するには Google APIキー を設定してください。")
        else:
            # (★) アプリ起動時にLLMとspaCyのロードを試みる
            if get_llm(MODEL_FLASH_LITE) is None: # (★) 起動確認は最軽量モデル
                st.error("LLMの初期化に失敗。APIキーが正しいか確認してください。")
            if load_spacy_model() is None:
                st.error("spaCyモデルのロードに失敗。")
        
        st.markdown("---")
        
        # (★) --- Step A〜D のナビゲーションボタン ---
        st.header("🔄 ナビゲーション")
        current_step = st.session_state.current_step
        
        # (★) ボタンが押されたら 'current_step' を変更し、st.rerun() で再描画
        if st.button(
            "Step A: AIタグ付け", key="nav_A", use_container_width=True,
            type=("primary" if current_step == 'A' else "secondary")
        ):
            if st.session_state.current_step != 'A':
                st.session_state.current_step = 'A'; st.rerun()

        if st.button(
            "Step B: 分析実行", key="nav_B", use_container_width=True,
            type=("primary" if current_step == 'B' else "secondary")
        ):
            if st.session_state.current_step != 'B':
                st.session_state.current_step = 'B'; st.rerun()

        # (★) Step C (新規追加)
        if st.button(
            "Step C: AIレポート生成", key="nav_C", use_container_width=True,
            type=("primary" if current_step == 'C' else "secondary")
        ):
            if st.session_state.current_step != 'C':
                st.session_state.current_step = 'C'; st.rerun()

        # (★) Step D (新規追加)
        if st.button(
            "Step D: PowerPoint生成", key="nav_D", use_container_width=True,
            type=("primary" if current_step == 'D' else "secondary")
        ):
            if st.session_state.current_step != 'D':
                st.session_state.current_step = 'D'; st.rerun()

    # --- メインコンテンツ (ステップに応じて描画) ---
    if st.session_state.current_step == 'A':
        render_step_a()
    elif st.session_state.current_step == 'B':
        render_step_b()
    elif st.session_state.current_step == 'C':
        render_step_c()
    elif st.session_state.current_step == 'D':
        render_step_d()
    else:
        st.error("不明なステップです。Step Aに戻ります。")
        st.session_state.current_step = 'A'; st.rerun()

if __name__ == "__main__":
    main()