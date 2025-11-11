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
import random  # (★) Tipsローテーションのために追加
from dotenv import load_dotenv  # (★) .env読み込みのために追加
import matplotlib
matplotlib.use('Agg') # (★) Streamlitのバックエンドで動作させるためのおまじない
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import base64
from wordcloud import WordCloud

# (★) --- matplotlib 日本語フォント設定 ---
# DockerfileでインストールしたIPAフォントのパスを指定
# (★) ご注意: ローカル環境で実行する場合、このフォントパスが異なる場合があります。
# (★) Docker環境 (Debianベース) を前提としています。
try:
    # (★) Dockerfile内のパス
    font_path = '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf' 
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'IPAGothic'
        # logger.info(f"日本語フォント '{font_path}' を読み込みました。")
    else:
        # (★) フォールバック (環境によって調整が必要)
        logger.warning(f"日本語フォント '{font_path}' が見つかりません。グラフの日本語が文字化けする可能性があります。")
        # (★) 代替フォントの検索 (やや時間がかかるが堅牢)
        try:
            jp_font = fm.findfont(fm.FontProperties(family='IPAexGothic'))
            plt.rcParams['font.family'] = 'IPAexGothic'
            logger.info(f"代替フォント '{jp_font}' を使用します。")
        except:
             logger.error("代替の日本語フォントも見つかりません。")
             plt.rcParams['font.family'] = 'sans-serif'
except Exception as e:
    logger.error(f"matplotlib日本語フォント設定エラー: {e}")

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
@st.cache_resource(ttl=3600)
def get_llm(
    model_name: str, 
    temperature: float = 0.0,
    timeout_seconds: int = 120  # (★) --- 修正: タイムアウト引数を追加 (デフォルト120秒) ---
) -> Optional[ChatGoogleGenerativeAI]:
    """
    指定されたモデル名、温度、タイムアウトでLLM (Google Gemini) モデルをロード・キャッシュする。
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
            api_key=api_key,
            request_timeout=timeout_seconds # (★) --- 修正: タイムアウトを渡す ---
        )
        logger.info(f"LLM Model ({model_name}) loaded successfully (Timeout: {timeout_seconds}s).")
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

@st.cache_data(ttl=3600) # 1時間キャッシュ
def get_analysis_tips_list_from_ai() -> List[str]:
    """
    (★) 待機時間中に表示する「データ分析に関するTips」をAIで生成する。
    モデル: MODEL_FLASH_LITE (gemini-2.5-flash-lite)
    """
    logger.info("get_analysis_tips_list_from_ai: AI (Flash Lite) でTIPSを生成します。")
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.5)
    if llm is None:
        return ["AIモデルの読み込みに失敗しました。"]

    prompt = PromptTemplate.from_template(
        """
        あなたはデータサイエンティストのメンターです。
        データ分析の初心者から中級者に向けて、役立つ「ヒントやTIPS」を【10個】、JSONのリスト形式で生成してください。
        
        # 指示:
        1. 各TIPSは、1〜2文の簡潔な日本語の文字列にすること。
        2. 例: 「'平均値'だけでなく'中央値'も見ることで、外れ値の影響を把握できます。」
        3. 例: 「データを可視化する前に、まずデータの'欠損値'と'型'を確認しましょう。」
        4. 出力はJSONリスト形式（ ["TIPS1", "TIPS2", ...] ）のみ。
        
        # 回答 (JSONリスト形式のみ):
        """
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        response_str = chain.invoke({})
        logger.debug(f"AI TIPS生成(生): {response_str}")
        
        # マークダウンや不要なテキストを除去し、JSONのみを抽出
        match = re.search(r'\[.*\]', response_str, re.DOTALL)
        if not match:
            logger.warning("AIがTIPSリスト(JSON)の生成に失敗しました。")
            return ["分析TIPSの取得に失敗しました。"]
        
        json_str = match.group(0)
        tips_list = json.loads(json_str)
        
        if isinstance(tips_list, list) and all(isinstance(tip, str) for tip in tips_list):
            logger.info(f"AI TIPS {len(tips_list)}件の生成に成功。")
            return tips_list
        else:
            raise Exception("AIの回答が文字列のリスト形式ではありません。")
            
    except Exception as e:
        logger.error(f"AI TIPS生成中にエラー: {e}", exc_info=True)
        return [
            "TIPSの読み込みに失敗しました。",
            "データ分析は、まず目的（KGI/KPI）を明確にすることから始まります。",
            "「なぜ？」を5回繰り返すことで、分析の真の目的にたどり着くことがあります。",
            "良い分析は、良い「問い」から生まれます。",
            "データは「集める」ことより「どう使うか」が重要です。"
        ]

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
    tip_placeholder: st.delta_generator.DeltaGenerator,  # (★) Tips用プレースホルダ
    processed_rows: int,
    total_rows: int,
    message_prefix: str
):
    """
    (Step A) の進捗バーとログエリアを更新する (DRY)
    (★) 要件: AI読み込み時間の進捗を0～100％で表示
    (★) 要件: AI Tipsをローテーション表示
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
        
        # (★) --- AI Tips のローテーション表示 ---
        if 'tips_list' not in st.session_state or not st.session_state.tips_list:
             # 万が一Tipsが空の場合はAI Tips関数を呼び出す
             st.session_state.tips_list = get_analysis_tips_list_from_ai()
             st.session_state.current_tip_index = 0
             st.session_state.last_tip_time = time.time()

        now = time.time()
        # 60秒ごと（またはTIPSが1件しかない場合）にTIPSを更新
        if (now - st.session_state.last_tip_time > 60) or (len(st.session_state.tips_list) == 1):
            if len(st.session_state.tips_list) > 1:
                st.session_state.current_tip_index = (st.session_state.current_tip_index + 1) % len(st.session_state.tips_list)
            st.session_state.last_tip_time = now
        
        # (★) リストが空でないかチェック
        if st.session_state.tips_list:
            current_tip = st.session_state.tips_list[st.session_state.current_tip_index]
            tip_placeholder.info(f"💡 データ分析TIPS: {current_tip}")
        # (★) --- ここまでが変更点 ---

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

    st.header("Step 2: 分析指針の入力とカテゴリ生成")
    analysis_prompt = st.text_area(
        "AIがタグ付けとキュレーションを行う際の指針を入力してください（必須）:",
        value=st.session_state.analysis_prompt_A,
        height=100,
        placeholder="例: 広島県の観光に関するInstagramの投稿。無関係な地域の投稿や、単なる挨拶・宣伝は除外したい。",
        key="analysis_prompt_input_A"
    )
    st.session_state.analysis_prompt_A = analysis_prompt
    
    # (★) --- 修正: ボタンをStep2に移動 ---
    st.markdown(f"（(★) 使用モデル: `{MODEL_FLASH_LITE}`）")
    if st.button("AIにカテゴリ候補を生成させる (Step 2)", key="gen_cat_button", type="primary"):
        if not analysis_prompt.strip():
            st.warning("分析指針は必須です。AIがデータを理解するために目的を入力してください。")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.error("Google APIキーが設定されていません。（.envファイルを確認してください）")
        else:
            with st.spinner(f"AI ({MODEL_FLASH_LITE}) が分析指針を読み解き、カテゴリを考案中..."):
                logger.info("AIカテゴリ生成ボタンクリック")
                st.session_state.generated_categories = {"市区町村キーワード": "地名辞書(JAPAN_GEOGRAPHY_DB)から抽出された市区町村名"}
                
                ai_categories = get_dynamic_categories(analysis_prompt)
                
                if ai_categories:
                    st.session_state.generated_categories.update(ai_categories)
                    logger.info(f"AIカテゴリ生成成功: {list(ai_categories.keys())}")
                    st.success("AIによるカテゴリ候補の生成が完了しました。Step 3 に進んでください。")
                else:
                    st.error("AIによるカテゴリ生成に失敗しました。AIの応答を確認してください。")
    # (★) --- ここまでが修正点 ---

    if not analysis_prompt.strip():
        st.warning("分析指針は必須です。AIがデータを理解するために目的を入力してください。")
        return

    # (★) --- 修正: Step 3 はカテゴリの選択のみに変更 ---
    st.header("Step 3: 分析カテゴリの選択")
    if not st.session_state.generated_categories:
        st.info("Step 2 で「AIにカテゴリ候補を生成させる」ボタンを押してください。")
        return
        
    st.markdown("タグ付けしたいカテゴリを以下から選択してください（「市区町村キーワード」は必須です）")
    # (★) ... (st.header("Step 4: ...") に名称変更) ...
    # (★) --- 修正: 名称をStep 4, 5, 6, 7 に変更 ---
    
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
                disabled=(cat == "市区町村キーワード")
            )
            if is_checked:
                selected_cats.append(cat)
    st.session_state.selected_categories = set(selected_cats)

    st.header("Step 4: 分析対象テキスト列の指定")
    selected_text_col_map = {}
    st.markdown("ファイルごとに、タグ付け対象のテキストが含まれる列を指定してください。")
    for f_name, df in valid_files_data.items():
        cols_list = list(df.columns)
        default_index = 0
        
        if st.session_state.selected_text_col.get(f_name) in cols_list:
            default_index = cols_list.index(st.session_state.selected_text_col.get(f_name))
        elif any(c in cols_list for c in ['text', 'body', 'content', '投稿', '本文']):
            try:
                default_index = next(i for i, c in enumerate(cols_list) if c in ['text', 'body', 'content', '投稿', '本文'])
            except StopIteration:
                default_index = 0
                
        selected_col = st.selectbox(f"**{f_name}** のテキスト列:", cols_list, index=default_index, key=f"col_select_{f_name}")
        selected_text_col_map[f_name] = selected_col
    st.session_state.selected_text_col = selected_text_col_map

    st.header("Step 5: 分析実行")
    st.markdown(f"（(★) 使用モデル: `{MODEL_FLASH_LITE}`）")
    
    col_run, col_cancel = st.columns([1, 1])
    with col_cancel:
        if st.button("キャンセル", key="cancel_button_A", use_container_width=True):
            st.session_state.cancel_analysis = True
            logger.warning("分析キャンセルボタンが押されました。")
            st.warning("次のバッチ処理後に分析をキャンセルします...")
    
    with col_run:
        if st.button("分析実行 (Step 5)", type="primary", key="run_analysis_A", use_container_width=True):
            st.session_state.cancel_analysis = False
            st.session_state.log_messages = []
            st.session_state.tagged_df_A = pd.DataFrame()
            
            # (★) --- Tips表示の初期化 ---
            tip_placeholder = st.empty()
            try:
                with st.spinner("分析TIPSをAIで生成中..."):
                    if 'tips_list' not in st.session_state or not st.session_state.tips_list:
                        st.session_state.tips_list = get_analysis_tips_list_from_ai()
                
                if not st.session_state.tips_list: # フォールバック
                    st.session_state.tips_list = ["データ分析TIPSの取得に失敗しました。"]

                st.session_state.current_tip_index = random.randint(0, len(st.session_state.tips_list) - 1)
                st.session_state.last_tip_time = time.time()
                tip_placeholder.info(f"💡 データ分析TIPS: {st.session_state.tips_list[st.session_state.current_tip_index]}")
            except Exception as e:
                logger.error(f"Tips初期化エラー: {e}")
            # (★) --- ここまでが変更点 ---

            try:
                with st.spinner(f"Step A: AI分析処理中 ({MODEL_FLASH_LITE})..."):
                    logger.info("Step A 分析実行ボタンクリック")
                    progress_placeholder = st.progress(0.0, text="処理待機中...")
                    log_placeholder = st.empty()

                    # --- 1. ファイル結合 ---
                    update_progress_ui(progress_placeholder, log_placeholder, tip_placeholder, 0, 100, "ファイル結合")
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
                        
                        update_progress_ui(
                            progress_placeholder, log_placeholder, tip_placeholder,
                            min(i + FILTER_BATCH_SIZE, total_filter_rows), total_filter_rows,
                            f"AIキュレーション (バッチ {current_batch_num}/{total_filter_batches})"
                        )
                        
                        filtered_df = filter_relevant_data_by_ai(batch_df, analysis_prompt)
                        if filtered_df is not None and not filtered_df.empty:
                            all_filtered_results.append(filtered_df)
                        
                        time.sleep(FILTER_SLEEP_TIME)
                    
                    if not all_filtered_results:
                        raise Exception("AIフィルタリング処理に失敗しました。")

                    filter_results_df = pd.concat(all_filtered_results, ignore_index=True)
                    relevant_ids = filter_results_df[filter_results_df['relevant'] == True]['id']
                    filtered_master_df = master_df[master_df['id'].isin(relevant_ids)].copy()
                    filtered_row_count = len(filtered_master_df)
                    logger.info(f"AIフィルタリング 完了。 {deduped_row_count}行 -> {filtered_row_count}行")

                    if filtered_master_df.empty:
                        st.warning("AIキュレーションの結果、分析対象のデータが0件になりました。")
                        st.session_state.tagged_df_A = pd.DataFrame()
                        progress_placeholder.progress(1.0, text="処理完了 (対象データ0件)")
                        return

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
                        
                        update_progress_ui(
                            progress_placeholder, log_placeholder, tip_placeholder,
                            min(i + TAGGING_BATCH_SIZE, total_rows), total_rows,
                            f"AIタグ付け (バッチ {current_batch_num}/{total_batches})"
                        )

                        tagged_df = perform_ai_tagging(batch_df, selected_category_definitions, analysis_prompt)
                        if tagged_df is not None and not tagged_df.empty:
                            all_tagged_results.append(tagged_df)
                        
                        time.sleep(TAGGING_SLEEP_TIME)

                    if not all_tagged_results:
                        raise Exception("AIタグ付け処理に失敗しました。")

                    # --- 5. 最終マージ ---
                    logger.info("全AIタグ付け結果結合...");
                    tagged_results_df = pd.concat(all_tagged_results, ignore_index=True)

                    logger.info("最終マージ処理開始...");
                    final_df = pd.merge(master_df_for_tagging, tagged_results_df, on='id', how='right')
                    
                    final_cols = list(master_df_for_tagging.columns) + [col for col in tagged_results_df.columns if col not in master_df_for_tagging.columns]
                    final_df = final_df[final_cols]

                    st.session_state.tagged_df_A = final_df
                    logger.info("Step A 分析処理 正常終了");
                    st.success("AIによる分析処理が完了しました。");
                    progress_placeholder.progress(1.0, text="処理完了")
                    
                    update_progress_ui(
                        progress_placeholder, log_placeholder, tip_placeholder, 
                        total_rows, total_rows, "処理完了"
                    )
                    
                    tip_placeholder.empty() # 処理完了後、Tipsを消す

            except Exception as e:
                logger.error(f"Step A 分析実行中にエラー: {e}", exc_info=True)
                st.error(f"分析実行中にエラーが発生しました: {e}")
                if 'progress_placeholder' in locals():
                    progress_placeholder.progress(1.0, text="エラーにより処理中断")
                if 'tip_placeholder' in locals():
                    tip_placeholder.empty() # エラー時もTipsを消す

    # (★) 要件④: エクスポートリンクを表示
    if not st.session_state.tagged_df_A.empty:
        st.header("Step 6: 分析結果の確認とエクスポート")
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

def find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """DataFrameから、複数のパターンに最初に一致する列名(str)を1つ返す"""
    cols = df.columns
    for pattern in patterns:
        try:
            # 1. 完全一致 (大文字小文字無視)
            for col in cols:
                if col.lower() == pattern.lower():
                    return col
            # 2. 部分一致 (大文字小文字無視)
            for col in cols:
                if re.search(pattern, col, re.IGNORECASE):
                    return col
        except re.error:
            continue # (e.g. invalid regex pattern)
    return None

def find_cols(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """DataFrameから、複数のパターンに一致する列名(list)をすべて返す"""
    cols = df.columns
    found_cols = set()
    for pattern in patterns:
        try:
            # 1. ･･･キーワード (hard-coded rule from old function)
            for col in cols:
                 if col.endswith('キーワード'):
                     found_cols.add(col)
            # 2. 部分一致 (大文字小文字無視)
            for col in cols:
                if re.search(pattern, col, re.IGNORECASE):
                    found_cols.add(col)
        except re.error:
            continue
    return sorted(list(found_cols))

def find_engagement_cols(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """DataFrameから、パターンに一致する「数値」列名(list)をすべて返す"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    found_cols = set()
    for pattern in patterns:
        try:
            for col in numeric_cols: # (★) Only search numeric cols
                if re.search(pattern, col, re.IGNORECASE):
                    found_cols.add(col)
        except re.error:
            continue
    return sorted(list(found_cols))
# (★) --- END NEW HELPER FUNCTIONS ---


def suggest_analysis_techniques_py(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    (Step B) データフレームを分析し、Pythonで実行可能な基本的な分析手法を提案する。
    (★) 修正: 2024/11/10 - re.search を使用し、列名を柔軟に検索するよう堅牢化
    """
    suggestions = []
    if df is None or df.empty:
        logger.error("suggest_analysis_techniques_py: DFが空です。")
        return suggestions
        
    try:
        # (★) --- 1. 柔軟な列名の特定 ---
        all_cols = list(df.columns)
        
        # (★) 主要な列を見つける
        text_col = find_col(df, ['ANALYSIS_TEXT_COLUMN', 'text', 'content', '本文'])
        topic_col = find_col(df, ['話題カテゴリ', 'topic', 'category'])
        location_col = find_col(df, ['市区町村キーワード', 'location', 'city', '地域'])
        tour_spot_col = find_col(df, ['観光地キーワード', 'tourist_spot', 'spot'])
        hashtag_col = find_col(df, ['hash', 'ハッシュタグ'])
        sentiment_col = find_col(df, ['sent', 'センチメント'])
        date_col = find_col(df, ['date', 'time', '日付', '日時'])
        
        # (★) 複数の可能性がある列
        engagement_cols = find_engagement_cols(df, ['eng', 'like', 'いいね', 'エンゲージメント'])
        # (★) `...キーワード` で終わる列 + `topic_col` や `location_col` など
        flag_cols = find_cols(df, ['key', 'keyword', 'キーワード'])
        flag_cols = sorted(list(set(flag_cols + [c for c in [topic_col, location_col, tour_spot_col, hashtag_col] if c])))

        other_categorical = [
            col for col in df.select_dtypes(include='object').columns
            if col not in flag_cols and col != text_col and col != date_col
        ]
        
        logger.info(f"提案分析(PY) - Text:{text_col}, Topic:{topic_col}, Location:{location_col}")
        logger.info(f"提案分析(PY) - FlagCols(All):{flag_cols}")
        logger.info(f"提案分析(PY) - Engagement:{engagement_cols}, Sentiment:{sentiment_col}, Date:{date_col}")

        potential_suggestions = []

        # (★) --- 2. 提案ロジック (堅牢化版) ---

        # --- 1. 全体メトリクス ---
        # (★) センチメント列とエンゲージメント列を `suitable_cols` に渡す
        overall_metric_cols = [c for c in [sentiment_col] + engagement_cols if c]
        potential_suggestions.append({
            "priority": 1, "name": "全体のメトリクス",
            "description": "投稿数、エンゲージメント、センチメント傾向など、データセット全体の概要を計算します。",
            "reason": "データ全体の状況把握に必須です。",
            "suitable_cols": overall_metric_cols, # (★) 
            "type": "python"
        })

        # --- 3. 単純集計（頻度分析）---
        # (★) 見つかったすべての「フラグ列」に対して提案
        if flag_cols:
            for col in flag_cols:
                potential_suggestions.append({
                    "priority": 1, 
                    "name": f"単純集計: {col}", # (★) 例: "単純集計: 市区町村キーワード"
                    "description": f"「{col}」列の出現頻度（TOP50）を分析します。",
                    "reason": f"StepAで生成されたキーワード列({col})の基本指標です。",
                    "suitable_cols": [col], # (★) 1列のみ
                    "type": "python"
                })

        # --- 2. クロス集計 ---
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
            
        # (★) 修正: `topic_col` と `tour_spot_col` が両方見つかった場合
        if topic_col and tour_spot_col:
            potential_suggestions.append({
                "priority": 2, "name": f"{topic_col}別 {tour_spot_col} TOP10",
                "description": f"「{topic_col}」と「{tour_spot_col}」をクロス集計し、カテゴリ別の人気観光地を分析します。",
                "reason": "カテゴリと観光地の関連性を分析します。",
                "suitable_cols": [topic_col, tour_spot_col],
                "type": "python"
            })

        # --- 3. 時系列分析 ---
        if date_col and flag_cols:
            potential_suggestions.append({
                "priority": 3, "name": "時系列キーワード分析",
                "description": f"特定のキーワードの出現数が時間（{date_col}など）とともにどう変化したかトレンドを分析します。",
                "reason": f"キーワード列と日時列({date_col})あり。",
                "suitable_cols": {"datetime": [date_col], "keywords": flag_cols}, # (★) 
                "type": "python"
            })
            
        # --- 3. 共起ネットワーク ---
        if text_col:
            potential_suggestions.append({
                "priority": 3, "name": "共起ネットワーク",
                "description": "投稿テキスト内の単語の出現パターンを分析し、関連性の高い単語のネットワークを構築します。",
                "reason": "テキストデータから隠れたトピックや関連性を発見します。",
                "suitable_cols": [text_col],
                "type": "python"
            })
            
        # --- 4. テキストマイニング ---
        if text_col:
            potential_suggestions.append({
                "priority": 4, "name": "テキストマイニング（頻出単語）",
                "description": "原文テキストから頻出する単語を抽出し、どのような言葉が多く使われているか全体像を把握します。",
                "reason": "原文テキストがあり、タグ付け以外のインサイト発見に。",
                "suitable_cols": [text_col],
                "type": "python"
            })

        # --- 4. 話題カテゴリ別 サマリ (Python + AI) ---
        if topic_col and text_col:
            potential_suggestions.append({
                "priority": 4, "name": "話題カテゴリ別 投稿数とサマリ",
                "description": "指定された話題カテゴリ（グルメ、自然など）ごとに投稿数を集計し、AIが投稿内容のサマリを生成します。",
                "reason": "カテゴリごとの主要な話題を把握します。",
                "suitable_cols": [topic_col, text_col], # (★)
                "type": "python"
            })

        # --- 4. 話題カテゴリ別 エンゲージメントTOP5 (Python + AI) ---
        if topic_col and text_col and engagement_cols:
            potential_suggestions.append({
                "priority": 4, "name": "話題カテゴリ別 エンゲージメントTOP5と概要",
                "description": f"指定された話題カテゴリごとに、エンゲージメント（{engagement_cols[0]}）が高いTOP5投稿を抽出し、AIがその概要を生成します。",
                "reason": "カテゴリごとに「バズった」投稿の内容を把握します。",
                "suitable_cols": {'topic': [topic_col], 'text': [text_col], 'engagement': engagement_cols},
                "type": "python"
            })

        suggestions = sorted(potential_suggestions, key=lambda x: x['priority'])
        
        # (★) 重複する提案を削除 (例: "単純集計: 話題カテゴリ" と "話題カテゴリ別 ...")
        final_suggestions = []
        seen_names = set()
        for s in suggestions:
             if s['name'] not in seen_names:
                 final_suggestions.append(s)
                 seen_names.add(s['name'])
                 
        logger.info(f"Pythonベース提案(ソート後): {[s['name'] for s in final_suggestions]}")
        return final_suggestions

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
    """
    logger.info("AIプロンプトベースの分析提案 (Flash Lite) を開始...")
    
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.1)
    if llm is None:
        logger.error("suggest_analysis_techniques_ai: LLM (Flash Lite) が利用できません。")
        return []

    try:
        col_info = []
        for col in df.columns:
            col_info.append(f"- {col} (型: {df[col].dtype}, 例: {df[col].dropna().iloc[0] if not df[col].dropna().empty else 'N/A'})")
        column_info_str = "\n".join(col_info[:15])
        
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
        
        for s in ai_suggestions:
            s['type'] = 'ai'
            if 'priority' not in s: s['priority'] = 5
            
        logger.info(f"AI追加提案(パース済): {len(ai_suggestions)}件")
        return ai_suggestions

    except Exception as e:
        logger.error(f"AI追加提案の生成中にエラー: {e}", exc_info=True)
        st.warning(f"AI追加提案の生成中にエラーが発生しました: {e}")
        return []

import networkx as nx # (★) Step B (共起ネットワーク) で必要
from itertools import combinations # (★) Step B (共起ネットワーク) で必要
import math # (★) グラフのレイアウト計算用


# --- 8.0. (★) グラフ生成ヘルパー (グラフサイズ修正) ---

def generate_graph_image(
    df: pd.DataFrame,
    plot_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    title: str = "分析グラフ"
) -> Optional[str]:
    """
    (★) DataFrameからmatplotlibグラフを生成し、Base64エンコードされた画像文字列を返す。
    (★) 1. 2. グラフサイズ修正
    """
    logger.info(f"グラフ生成開始: {title} (タイプ: {plot_type})")
    if df is None or df.empty:
        logger.warning("グラフ生成スキップ: DataFrameが空です。")
        return None

    # (★) --- 修正: プロットタイプに応じてFigureサイズを変更 ---
    if plot_type == 'network':
        plt.figure(figsize=(12, 12)) # (★) 2. 共起ネットワーク: 正方形 (12x12)
    elif plot_type == 'timeseries':
        plt.figure(figsize=(15, 7)) # (★) 1. 時系列: 横長 (15x7)
    else:
        plt.figure(figsize=(10, 7)) # (★) デフォルト (棒グラフなど)
    
    plt.rcParams['font.size'] = 12
    # (★) --- ここまでが修正点 ---
    
    try:
        if plot_type == 'bar' and x_col and y_col:
            df_plot = df.nlargest(20, y_col).sort_values(by=y_col, ascending=True)
            if df_plot.empty:
                raise ValueError("グラフ描画対象のデータがありません。")
                
            bars = plt.barh(df_plot[x_col], df_plot[y_col], color='#7280C1')
            plt.xlabel('件数')
            plt.ylabel(x_col)
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            
            for bar in bars:
                plt.text(
                    bar.get_width() + (df_plot[y_col].max() * 0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f' {bar.get_width():.0f}',
                    va='center',
                    ha='left'
                )
        
        elif plot_type == 'timeseries' and x_col and y_col:
            try:
                df[x_col] = pd.to_datetime(df[x_col])
                df_pivot = df.pivot(index=x_col, columns='keyword', values=y_col).fillna(0)
                
                if len(df_pivot.columns) > 6:
                    top_5_keywords = df_pivot.sum().nlargest(5).index
                    df_pivot['その他'] = df_pivot.drop(columns=top_5_keywords).sum(axis=1)
                    df_pivot = df_pivot[list(top_5_keywords) + ['その他']]
                
                df_pivot.plot(kind='line', ax=plt.gca(), linewidth=2.5)
                plt.xlabel('日付')
                plt.ylabel('件数')
                plt.legend(title='キーワード', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(axis='y', linestyle='--', alpha=0.6)
                
            except Exception as e:
                logger.error(f"時系列ピボット/プロットエラー: {e}")
                plt.plot(df[x_col], df[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)

        elif plot_type == 'network':
            df_plot = df.nlargest(100, 'weight')
            if df_plot.empty:
                raise ValueError("ネットワーク描画対象のデータがありません。")

            G = nx.from_pandas_edgelist(df_plot, 'source', 'target', ['weight'])
            
            try:
                partition = community.best_partition(G)
                num_communities = len(set(partition.values()))
                colors = plt.cm.get_cmap('tab20', num_communities)
                node_colors = [colors(partition.get(node)) for node in G.nodes()]
            except Exception:
                node_colors = '#7280C1'
            
            # (★) --- 修正: 2. 共起ネットワークのレイアウト調整 ---
            # (★) k値を調整 (ノードを広げる)
            k_val = 2.5 / math.sqrt(len(G.nodes())) # (★) 1.5 -> 2.5 に変更
            pos = nx.spring_layout(G, k=max(k_val, 0.5), iterations=50, seed=42)
            
            node_sizes = []
            try:
                for node in G.nodes():
                    total_weight = sum(data['weight'] for _, _, data in G.edges(node, data=True))
                    node_sizes.append(total_weight * 20)
            except Exception:
                node_sizes = 500
            
            edge_weights = [d['weight'] / df_plot['weight'].max() * 8 for u, v, d in G.edges(data=True)]

            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
            # (★) 2. エッジのalphaを調整 (細く)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.1, edge_color='grey')
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='IPAGothic')
            # (★) --- ここまでが修正点 ---
            
            plt.axis('off')

        else:
            logger.warning(f"未対応のプロットタイプ: {plot_type}")
            return None

        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=96) # (★) 150 -> 96 (トークン数削減)
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        logger.info(f"グラフ生成成功: {title}")
        return image_base64

    except Exception as e:
        logger.error(f"グラフ生成 ({title}) 失敗: {e}", exc_info=True)
        return None
    finally:
        plt.clf()
        plt.close('all')


# --- 8. (★) Step B: 分析提案関連 ---

def suggest_analysis_techniques_py(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    (Step B) データフレームを分析し、Pythonで実行可能な基本的な分析手法を提案する。
    (★) 3. 単純集計のロジックを修正
    """
    suggestions = []
    if df is None or df.empty:
        logger.error("suggest_analysis_techniques_py: DFが空です。")
        return suggestions
        
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        object_cols = df.select_dtypes(include='object').columns.tolist()
        datetime_cols = []

        for col in object_cols:
             if df[col].isnull().sum() / len(df) > 0.5: continue
             sample = df[col].dropna().head(50)
             if sample.empty: continue
             try:
                 pd.to_datetime(sample, errors='raise')
                 temp_dt = pd.to_datetime(df[col], errors='coerce').dropna()
                 if not temp_dt.empty and (temp_dt.dt.year.nunique() > 1 or temp_dt.dt.month.nunique() > 1 or temp_dt.dt.day.nunique() > 1 or col.lower() in ['date', 'time', 'timestamp', '日付', '日時']):
                     datetime_cols.append(col)
                     logger.info(f"列 '{col}' を日時列として認識しました。")
             except (ValueError, TypeError, OverflowError, pd.errors.ParserError):
                 pass

        numeric_cols = [col for col in numeric_cols if col != 'id']
        categorical_cols = [col for col in object_cols if col != 'ANALYSIS_TEXT_COLUMN' and col not in datetime_cols]
        # (★) 3. _キーワード で終わる列を flag_cols とする
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

        # (★) --- 3. 単純集計（頻度分析）---
        # (★) 修正: StepAで生成した全キーワード列を個別に提案
        if flag_cols:
            for col in flag_cols:
                potential_suggestions.append({
                    "priority": 1, 
                    "name": f"単純集計: {col}", # (★) 例: "単純集計: 市区町村キーワード"
                    "description": f"「{col}」列の出現頻度（TOP50）を分析します。",
                    "reason": f"StepAで生成されたキーワード列({col})の基本指標です。",
                    "suitable_cols": [col], # (★) 1列のみ
                    "type": "python"
                })

        # 優先度2: クロス集計
        if len(flag_cols) >= 2:
            potential_suggestions.append({
                "priority": 2, "name": "クロス集計（キーワード間）",
                "description": "キーワード間の組み合わせで多く出現するパターンを探ります。",
                "reason": f"複数キーワード列({len(flag_cols)}個)あり、関連性の発見に。",
                "suitable_cols": flag_cols, # (★) 編集用に全フラグ列を渡す
                "type": "python"
            })
        if flag_cols and other_categorical:
             potential_suggestions.append({
                "priority": 2, "name": "クロス集計（キーワード×属性）",
                "description": f"キーワード({flag_cols[0]}など)と他の属性({', '.join(other_categorical)})の関係性を分析します。",
                "reason": f"キーワード列と他カテゴリ列({len(other_categorical)}個)あり。",
                "suitable_cols": flag_cols + other_categorical, # (★) 編集用に全列を渡す
                "type": "python"
            })
            
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
                "suitable_cols": {"datetime": datetime_cols, "keywords": flag_cols}, # (★) 編集用に全候補を渡す
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
                "type": "python"
            })

        # (★) --- 4. 話題カテゴリ別 エンゲージメントTOP5 (Python + AI) ---
        engagement_cols = [col for col in numeric_cols if any(c in col.lower() for c in ['いいね', 'like', 'エンゲージメント', 'engagement'])]
        if '話題カテゴリ' in df.columns and 'ANALYSIS_TEXT_COLUMN' in df.columns and engagement_cols:
            potential_suggestions.append({
                "priority": 4, "name": "話題カテゴリ別 エンゲージメントTOP5と概要",
                "description": f"指定された話題カテゴリごとに、エンゲージメント（{engagement_cols[0]}）が高いTOP5投稿を抽出し、AIがその概要を生成します。",
                "reason": "カテゴリごとに「バズった」投稿の内容を把握します。",
                # (★) 編集用に全候補を渡す
                "suitable_cols": {'topic': ['話題カテゴリ'], 'text': ['ANALYSIS_TEXT_COLUMN'], 'engagement': engagement_cols},
                "type": "python"
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
    """
    logger.info("AIプロンプトベースの分析提案 (Flash Lite) を開始...")
    
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.1)
    if llm is None:
        logger.error("suggest_analysis_techniques_ai: LLM (Flash Lite) が利用できません。")
        return []

    try:
        col_info = []
        for col in df.columns:
            col_info.append(f"- {col} (型: {df[col].dtype}, 例: {df[col].dropna().iloc[0] if not df[col].dropna().empty else 'N/A'})")
        column_info_str = "\n".join(col_info[:15])
        
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
        
        for s in ai_suggestions:
            s['type'] = 'ai'
            if 'priority' not in s: s['priority'] = 5
            
        logger.info(f"AI追加提案(パース済): {len(ai_suggestions)}件")
        return ai_suggestions

    except Exception as e:
        logger.error(f"AI追加提案の生成中にエラー: {e}", exc_info=True)
        st.warning(f"AI追加提案の生成中にエラーが発生しました: {e}")
        return []

# --- 8.1. (★) Step B: Python分析ヘルパー (修正反映) ---

def run_simple_count(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) 単純集計（頻度分析）を実行し、DataFrameとグラフ(Base64)を返す"""
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    
    flag_cols = suggestion.get('suitable_cols', [])
    if not flag_cols:
        msg = "集計対象の列が見つかりません。"
        logger.warning(f"run_simple_count: {msg}")
        results["summary"] = msg
        return results
    
    # (★) 3. 提案ロジックの変更により、suitable_cols[0] は分析対象の列名 (e.g., '話題カテゴリ') になっている
    col_to_analyze = flag_cols[0]
    
    if col_to_analyze not in df.columns:
        msg = f"列 '{col_to_analyze}' がDFに存在しません。"
        logger.warning(f"run_simple_count: {msg}")
        results["summary"] = msg
        return results
        
    try:
        s = df[col_to_analyze].astype(str).str.split(', ').explode()
        s = s[s.str.strip().isin(['', 'nan', 'None', 'N/A', '該当なし']) == False]
        s = s.str.strip()
        
        if s.empty:
            msg = "集計対象のキーワードがありませんでした。"
            logger.info(f"run_simple_count: {msg}")
            results["summary"] = msg
            return results
            
        counts = s.value_counts().head(50)
        counts_df = counts.reset_index()
        counts_df.columns = [col_to_analyze, 'count']
        
        results["data"] = counts_df
        results["summary"] = f"'{col_to_analyze}' の単純集計（頻度分析）を実行。上位は {counts_df.iloc[0,0]} ({counts_df.iloc[0,1]}件), {counts_df.iloc[1,0]} ({counts_df.iloc[1,1]}件) でした。"
        
        results["image_base64"] = generate_graph_image(
            df=counts_df,
            plot_type='bar',
            x_col=col_to_analyze,
            y_col='count',
            title=f"「{col_to_analyze}」 頻出TOP20"
        )
        return results
            
    except Exception as e:
        logger.error(f"run_simple_count error: {e}", exc_info=True)
        results["summary"] = f"エラー: {e}"
    return results

def run_crosstab(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) クロス集計を実行し、DataFrameを返す"""
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    
    # (★) UIで編集された列を取得
    cols = suggestion.get('suitable_cols', [])
    if len(cols) < 2:
        msg = "クロス集計には2列以上必要です。"
        logger.warning(f"run_crosstab: {msg}")
        results["summary"] = msg
        return results

    # (★) 編集ロジック (UI側で2列が選択されていることを期待)
    col1, col2 = cols[0], cols[1]

    if col1 not in df.columns or col2 not in df.columns:
        msg = f"選択された列 ({col1}, {col2}) がDFに存在しません。"
        logger.warning(f"run_crosstab: {msg}")
        results["summary"] = msg
        return results
    
    try:
        df_exploded_1 = df.assign(**{col1: df[col1].astype(str).str.split(', ')}).explode(col1)
        df_exploded_2 = df_exploded_1.assign(**{col2: df_exploded_1[col2].astype(str).str.split(', ')}).explode(col2)

        df_exploded_2[col1] = df_exploded_2[col1].str.strip()
        df_exploded_2[col2] = df_exploded_2[col2].str.strip()
        df_exploded_2 = df_exploded_2.replace('', np.nan).replace('nan', np.nan).replace('None', np.nan).dropna(subset=[col1, col2])
        
        crosstab_df = pd.crosstab(df_exploded_2[col1], df_exploded_2[col2])
        
        crosstab_long = crosstab_df.stack().reset_index()
        crosstab_long.columns = [col1, col2, 'count']
        crosstab_long = crosstab_long[crosstab_long['count'] > 0].sort_values(by='count', ascending=False)
        
        results["data"] = crosstab_long.head(100)
        results["summary"] = f"'{col1}' と '{col2}' のクロス集計を実行。最強の組み合わせは {crosstab_long.iloc[0,0]} x {crosstab_long.iloc[0,1]} ({crosstab_long.iloc[0,2]}件) でした。"
        logger.info("run_crosstab: グラフ生成はスキップされました。")
        
        return results
        
    except Exception as e:
        logger.error(f"run_crosstab error: {e}", exc_info=True)
        results["summary"] = f"エラー: {e}"
    return results

def run_timeseries(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) 時系列分析を実行し、DataFrameとグラフ(Base64)を返す"""
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    
    # (★) UIで編集された列を取得
    cols_dict = suggestion.get('suitable_cols', {})
    if not isinstance(cols_dict, dict) or 'datetime' not in cols_dict or 'keywords' not in cols_dict:
        msg = "列情報（datetime, keywords）が不十分です。"
        logger.warning(f"run_timeseries: {msg}")
        results["summary"] = msg
        return results
        
    # (★) 編集ロジック (UI側で1列ずつ選択されていることを期待)
    dt_col = cols_dict['datetime'][0]
    kw_col = cols_dict['keywords'][0]

    if dt_col not in df.columns:
        msg = f"日時列 '{dt_col}' が見つかりません。"
        logger.warning(f"run_timeseries: {msg}"); results["summary"] = msg; return results
    if kw_col not in df.columns:
        msg = f"キーワード列 '{kw_col}' が見つかりません。"
        logger.warning(f"run_timeseries: {msg}"); results["summary"] = msg; return results

    try:
        df_copy = df[[dt_col, kw_col]].copy()
        df_copy[dt_col] = pd.to_datetime(df_copy[dt_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[dt_col])
        
        df_exploded = df_copy.assign(**{kw_col: df_copy[kw_col].astype(str).str.split(', ')}).explode(kw_col)
        df_exploded[kw_col] = df_exploded[kw_col].str.strip()
        df_exploded = df_exploded[df_exploded[kw_col].isin(['', 'nan', 'None', 'N/A', '該当なし']) == False]
        
        if df_exploded.empty:
            msg = "有効な日時/キーワードデータがありません。"
            logger.info(f"run_timeseries: {msg}"); results["summary"] = msg; return results

        time_df = df_exploded.groupby([pd.Grouper(key=dt_col, freq='D'), kw_col]).size().rename("count").reset_index()
        
        time_df.columns = ['date', 'keyword', 'count']
        
        top_keywords = df_exploded[kw_col].value_counts().head(50).index
        time_df_filtered = time_df[time_df['keyword'].isin(top_keywords)]
        
        time_df_for_graph = time_df_filtered.copy()
        
        time_df_for_json = time_df_filtered.sort_values(by=['keyword', 'date'])
        time_df_for_json['date'] = time_df_for_json['date'].dt.strftime('%Y-%m-%d')
        
        results["data"] = time_df_for_json
        
        results["image_base64"] = generate_graph_image(
            df=time_df_for_graph,
            plot_type='timeseries',
            x_col='date',
            y_col='count',
            title=f"「{kw_col}」別 時系列トレンド (TOP5)"
        )
        results["summary"] = f"'{dt_col}' と '{kw_col}' で時系列分析を実行。TOP5キーワードのトレンドグラフを生成しました。"
        return results
            
    except Exception as e:
        logger.error(f"run_timeseries error: {e}", exc_info=True)
        results["summary"] = f"エラー: {e}"
    return results

def run_text_mining(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) テキストマイニング（頻出単語）を実行し、DataFrameとグラフ(Base64)を返す"""
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    
    # (★) UIで編集された列を取得
    text_col = suggestion.get('suitable_cols', ['ANALYSIS_TEXT_COLUMN'])[0]
    
    if text_col not in df.columns or df[text_col].empty:
        msg = f"テキスト列 '{text_col}' がないか、空です。"
        logger.warning(f"run_text_mining: {msg}"); results["summary"] = msg; return results

    nlp = load_spacy_model()
    if nlp is None:
        st.error("spaCy日本語モデルのロードに失敗しました。")
        results["summary"] = "spaCy日本語モデルのロードに失敗しました。"
        return results
            
    try:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            results["summary"] = "テキストデータが空です。"
            return results
            
        words = []
        target_pos = {'NOUN', 'PROPN', 'ADJ'}
        stop_words = {
            'の', 'に', 'は', 'を', 'が', 'で', 'て', 'です', 'ます', 'こと', 'もの', 'それ', 'あれ',
            'これ', 'ため', 'いる', 'する', 'ある', 'ない', 'いう', 'よう', 'そう', 'など', 'さん',
            '的', '人', '自分', '私', '僕', '何', 'その', 'この', 'あの'
        }
        
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
            msg = "抽出可能な有効な単語が見つかりませんでした。"
            logger.warning(f"run_text_mining: {msg}"); results["summary"] = msg; return results

        word_counts = pd.Series(words).value_counts().head(100)
        word_counts_df = word_counts.reset_index()
        word_counts_df.columns = ['word', 'count']
        
        st.session_state.progress_text = "テキストマイニング (spaCy) 完了。"
        
        results["data"] = word_counts_df
        results["summary"] = f"'{text_col}' に対するテキストマイニングを実行。頻出単語は '{word_counts_df.iloc[0,0]}' ({word_counts_df.iloc[0,1]}件) でした。"
        
        results["image_base64"] = generate_graph_image(
            df=word_counts_df,
            plot_type='bar',
            x_col='word',
            y_col='count',
            title=f"「{text_col}」 頻出単語 TOP20"
        )
        return results
        
    except Exception as e:
        logger.error(f"run_text_mining error: {e}", exc_info=True)
        results["summary"] = f"エラー: {e}"
    return results

def run_overall_metrics(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) データセット全体のメトリクスを計算する"""
    # (変更なし)
    logger.info("run_overall_metrics 実行...")
    metrics = {}
    try:
        metrics["total_posts"] = len(df)
        engagement_cols = [col for col in df.columns if any(c in col.lower() for c in ['いいね', 'like', 'エンゲージメント', 'engagement', 'retweet', 'リツイート'])]
        total_engagement = 0
        if engagement_cols:
            for col in engagement_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    total_engagement += df[col].sum()
            metrics["total_engagement"] = int(total_engagement)
        else:
            metrics["total_engagement"] = "N/A"
        sentiment_col = None
        if 'センチメント' in df.columns:
            sentiment_col = 'センチメント'
        elif len(df.columns) > 9 and ('センチメント' in str(df.columns[9]) or 'sentiment' in str(df.columns[9]).lower()):
            sentiment_col = df.columns[9]
        if sentiment_col:
            pos_count = int(df[df[sentiment_col].astype(str).str.contains('ポジティブ|Positive', case=False, na=False)].shape[0])
            neg_count = int(df[df[sentiment_col].astype(str).str.contains('ネガティブ|Negative', case=False, na=False)].shape[0])
            metrics["positive_posts"] = pos_count
            metrics["negative_posts"] = neg_count
            if (pos_count + neg_count) > 0:
                tendency = ((pos_count - neg_count) / (pos_count + neg_count)) * 100
                metrics["sentiment_tendency_percent"] = int(np.floor(tendency))
            else:
                metrics["sentiment_tendency_percent"] = 0
        else:
            logger.warning("列 'センチメント' が見つかりませんでした。")
            metrics["positive_posts"] = "N/A"
            metrics["negative_posts"] = "N/A"
            metrics["sentiment_tendency_percent"] = "N/A"
        summary = f"全体のメトリクスを計算。総投稿数: {metrics['total_posts']}件, 総エンゲージメント: {metrics['total_engagement']}。"
        return {"data": metrics, "image_base64": None, "summary": summary}
    except Exception as e:
        logger.error(f"run_overall_metrics error: {e}", exc_info=True)
        return {"data": {"error": str(e)}, "image_base64": None, "summary": f"エラー: {e}"}

def run_cooccurrence_network(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) 共起ネットワークを構築し、エッジリストのDataFrameとグラフ(Base64)を返す"""
    # (★) UIで編集された列を取得
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    text_col = suggestion.get('suitable_cols', ['ANALYSIS_TEXT_COLUMN'])[0]
    if text_col not in df.columns or df[text_col].empty:
        msg = f"テキスト列 '{text_col}' がないか、空です。"
        logger.warning(f"run_cooccurrence_network: {msg}"); results["summary"] = msg; return results

    nlp = load_spacy_model()
    if nlp is None:
        st.error("spaCy日本語モデルのロードに失敗しました。")
        results["summary"] = "spaCy日本語モデルのロードに失敗しました。"
        return results

    try:
        texts = df[text_col].dropna().astype(str)
        if texts.empty:
            results["summary"] = "テキストデータが空です。"
            return results
        target_pos = {'NOUN', 'PROPN', 'ADJ'}
        stop_words = {
            'の', 'に', 'は', 'を', 'が', 'で', 'て', 'です', 'ます', 'こと', 'もの', 'それ', 'あれ',
            'これ', 'ため', 'いる', 'する', 'ある', 'ない', 'いう', 'よう', 'そう', 'など', 'さん',
            '的', '人', '自分', '私', '僕', '何', 'その', 'この', 'あの'
        }
        st.session_state.progress_text = "共起ネットワーク (spaCy) 処理中... 0%"
        total_texts = len(texts)
        doc_words_list = []
        for i, doc in enumerate(nlp.pipe(texts, disable=["parser", "ner"], batch_size=50)):
            words_in_text = set()
            for token in doc:
                if (token.pos_ in target_pos) and (not token.is_stop) and (token.lemma_ not in stop_words) and (len(token.lemma_) > 1):
                    words_in_text.add(token.lemma_)
            doc_words_list.append(list(words_in_text))
            if (i + 1) % 100 == 0:
                percent = (i + 1) / total_texts
                st.session_state.progress_text = f"共起ネットワーク (spaCy) 処理中... {percent:.0%}"

        all_words = [word for sublist in doc_words_list for word in sublist]
        if not all_words:
            msg = "抽出可能な単語がありません。"
            logger.warning(f"run_cooccurrence_network: {msg}"); results["summary"] = msg; return results
        top_n_words_limit = 100
        top_n_words_set = set(pd.Series(all_words).value_counts().head(top_n_words_limit).index)
        G = nx.Graph()
        for words_in_text_set in doc_words_list:
            filtered_words = [word for word in words_in_text_set if word in top_n_words_set]
            for word1, word2 in combinations(sorted(list(filtered_words)), 2):
                if G.has_edge(word1, word2):
                    G[word1][word2]['weight'] += 1
                else:
                    G.add_edge(word1, word2, weight=1)
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            msg = "有効なエッジが構築されませんでした。"
            logger.info(f"run_cooccurrence_network: {msg}"); results["summary"] = msg; return results
        edge_list = []
        for u, v, data in G.edges(data=True):
            edge_list.append({"source": u, "target": v, "weight": data['weight']})
        edges_df = pd.DataFrame(edge_list)
        edges_df_sorted = edges_df.sort_values(by="weight", ascending=False).head(500)
        results["data"] = edges_df_sorted
        st.session_state.progress_text = "共起ネットワーク グラフ描画中..."
        results["image_base64"] = generate_graph_image(
            df=edges_df_sorted,
            plot_type='network',
            title="共起ネットワーク (上位100エッジ)"
        )
        results["summary"] = f"'{text_col}' に対する共起ネットワーク分析を実行。{len(G.nodes())}ノード, {len(G.edges())}エッジを検出。グラフを生成しました。"
        return results
    except Exception as e:
        logger.error(f"run_cooccurrence_network error: {e}", exc_info=True)
        results["summary"] = f"エラー: {e}"
        return results

def run_topic_category_summary(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) 話題カテゴリ別に投稿数、サマリ(AI)、上位キーワードを分析する"""
    logger.info("run_topic_category_summary 実行...")
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}
    
    # (★) UIで編集された列を取得
    topic_col = suggestion.get('suitable_cols', ['話題カテゴリ'])[0]
    
    target_categories = ['グルメ', '自然', '歴史・文化', 'アート', 'イベント', '宿泊・温泉']
    if topic_col not in df.columns:
        msg = f"列 '{topic_col}' が見つかりません。StepAで「話題カテゴリ」が生成されているか確認してください。"
        logger.warning(f"run_topic_category_summary: {msg}")
        return {"data": pd.DataFrame([{"error": msg}]), "image_base64": None, "summary": msg}
    
    # (★) --- 4. TOPキーワードのロジック修正 (地域名を除外) ---
    flag_cols = [col for col in df.columns if col.endswith('キーワード')]
    cols_to_use = [col for col in flag_cols if col != '市区町村キーワード']
    logger.info(f"TOPキーワード集計対象 (地域名除外): {cols_to_use}")
    # (★) --- ここまでが修正点 ---
    
    results_list = []
    
    total_cats = len(target_categories)
    if 'progress_text' not in st.session_state:
            st.session_state.progress_text = ""
            
    for i, category in enumerate(target_categories):
        st.session_state.progress_text = f"話題カテゴリ分析中 ({i+1}/{total_cats}): {category}"
        
        df_filtered = df[df[topic_col].astype(str).str.contains(category, na=False)]
        post_count = len(df_filtered)
        
        if post_count == 0:
            results_list.append({
                "category": category,
                "post_count": 0,
                "summary_ai": "N/A (投稿なし)",
                "top_keywords": []
            })
            continue
        
        text_samples = df_filtered['ANALYSIS_TEXT_COLUMN'].dropna().sample(n=min(10, post_count), random_state=1).tolist()
        text_samples_str = "\n".join([f"- {text[:200]}..." for text in text_samples])
        
        ai_suggestion = {
            "description": f"「{category}」カテゴリに関する以下の投稿サンプルを読み、主要な話題を1～2文で要約してください。\nサンプル:\n{text_samples_str}"
        }
        summary_ai = run_ai_summary_batch(df_filtered, ai_suggestion)
        
        # (★) --- 4. 修正: 上位キーワード (Python) ---
        top_keywords = []
        if cols_to_use and not df_filtered.empty:
            all_keywords_series = []
            for kw_col in cols_to_use:
                if kw_col in df_filtered.columns:
                    s = df_filtered[kw_col].astype(str).str.split(', ').explode()
                    s = s[s.str.strip().isin(['', 'nan', 'None', 'N/A', '該当なし']) == False]
                    s = s.str.strip()
                    if not s.empty:
                        all_keywords_series.append(s)
            if all_keywords_series:
                combined_s = pd.concat(all_keywords_series)
                top_keywords = combined_s.value_counts().head(5).index.tolist()
        
        results_list.append({
            "category": category,
            "post_count": post_count,
            "summary_ai": summary_ai,
            "top_keywords": top_keywords
        })
        
        time.sleep(max(TAGGING_SLEEP_TIME / 2, 1.0))

    st.session_state.progress_text = "話題カテゴリ分析 完了。"
    results_df = pd.DataFrame(results_list)
    
    image_base64 = generate_graph_image(
        df=results_df,
        plot_type='bar',
        x_col='category',
        y_col='post_count',
        title=f"「{topic_col}」別 投稿数"
    )
    
    summary = f"「{topic_col}」別の分析を実行。投稿数グラフを生成しました。"
    return {"data": results_df, "image_base64": image_base64, "summary": summary}

def run_topic_engagement_top5(df: pd.DataFrame, suggestion: Dict[str, Any]) -> Dict[str, Any]:
    """(Step B) 話題カテゴリ別にエンゲージメントTOP5投稿と概要(AI)を分析する"""
    logger.info("run_topic_engagement_top5 実行...")
    results = {"data": pd.DataFrame(), "image_base64": None, "summary": ""}

    # (★) UIで編集された列を取得
    cols_dict = suggestion.get('suitable_cols', {})
    if not isinstance(cols_dict, dict) or 'topic' not in cols_dict or 'text' not in cols_dict or 'engagement' not in cols_dict:
        msg = "列情報（topic, text, engagement）が不十分です。"
        logger.warning(f"run_topic_engagement_top5: {msg}")
        return {"data": pd.DataFrame([{"error": msg}]), "image_base64": None, "summary": msg}

    topic_col = cols_dict['topic'][0]
    text_col = cols_dict['text'][0]
    engagement_col = cols_dict['engagement'][0]
    target_categories = ['グルメ', '自然', '歴史・文化', 'アート', 'イベント', '宿泊・温泉'] # (★) これは固定

    if topic_col not in df.columns:
        msg = f"話題カテゴリ列 '{topic_col}' が見つかりません。"
        return {"data": pd.DataFrame([{"error": msg}]), "image_base64": None, "summary": msg}
    if engagement_col not in df.columns or not pd.api.types.is_numeric_dtype(df[engagement_col]):
        msg = f"エンゲージメント列 '{engagement_col}' が数値列として存在しません。"
        return {"data": pd.DataFrame([{"error": msg}]), "image_base64": None, "summary": msg}
    if text_col not in df.columns:
        msg = f"テキスト列 '{text_col}' が見つかりません。"
        return {"data": pd.DataFrame([{"error": msg}]), "image_base64": None, "summary": msg}


    # (★) --- 5. メディアリンク列を特定 ---
    link_col_candidates = ['link', 'url', 'media_url', '投稿URL', 'URL', 'Link', 'Url']
    df_cols_lower = {col.lower(): col for col in df.columns}
    found_link_col = None
    for cand in link_col_candidates:
        if cand in df_cols_lower:
            found_link_col = df_cols_lower[cand]
            break
    if found_link_col:
        logger.info(f"メディアリンク列: '{found_link_col}' を使用します。")
    else:
        logger.warning(f"メディアリンク列 ({link_col_candidates}) が見つかりませんでした。")
    # (★) --- ここまでが修正点 ---

    results_list = []
    
    total_cats = len(target_categories)
    if 'progress_text' not in st.session_state:
            st.session_state.progress_text = ""

    for i, category in enumerate(target_categories):
        st.session_state.progress_text = f"エンゲージメントTOP5分析中 ({i+1}/{total_cats}): {category}"
        
        df_filtered = df[df[topic_col].astype(str).str.contains(category, na=False)]
        post_count = len(df_filtered)
        
        if post_count == 0:
            continue
            
        df_top5 = df_filtered.nlargest(5, engagement_col, keep='first')
        top5_posts_data = []
        
        if df_top5.empty:
                results_list.append({
                "category": category,
                "post_count": post_count,
                "top_posts": []
            })
                continue

        for _, row in df_top5.iterrows():
            post_text = str(row[text_col])
            engagement_value = row[engagement_col]
            
            ai_suggestion = {
                "description": f"以下の投稿テキストを読み、内容を1文で要約してください。\nテキスト: {post_text[:500]}..."
            }
            summary_ai = run_ai_summary_batch(df_filtered, ai_suggestion)
            
            # (★) --- 5. メディアリンクを取得 ---
            link_value = None
            if found_link_col and found_link_col in row and pd.notna(row[found_link_col]):
                link_value = str(row[found_link_col])
            
            top5_posts_data.append({
                "engagement": int(engagement_value),
                "summary_ai": summary_ai,
                "original_text_snippet": post_text[:100],
                "media_link": link_value # (★) 追加
            })
            
            time.sleep(max(TAGGING_SLEEP_TIME / 2, 1.0))

        results_list.append({
            "category": category,
            "post_count": post_count,
            "top_posts": top5_posts_data
        })

    st.session_state.progress_text = "エンゲージメントTOP5分析 完了。"
    results_df = pd.DataFrame(results_list)
    
    summary = f"「{topic_col}」別の高エンゲージメント投稿TOP5を抽出しました。"
    return {"data": results_df, "image_base64": None, "summary": summary}


# --- 8.2. (★) Step B: AI分析ヘルパー (変更なし) ---
# ... (run_ai_summary_batch は変更なし) ...


# --- 8.3. (★) Step B: 分析実行ルーター (変更なし) ---
def execute_analysis(
    analysis_name: str,
    df: pd.DataFrame,
    suggestion: Dict[str, Any]
) -> Dict[str, Any]:
    """
    (Step B) 分析名に基づき、適切なPythonまたはAIの実行関数を呼び出すルーター
    """
    try:
        analysis_type = suggestion.get('type', 'python')
        
        if analysis_type == 'python':
            if analysis_name == "全体のメトリクス":
                return run_overall_metrics(df, suggestion)
            # (★) 3. "単純集計: {col}" のような動的な名前に対応
            elif analysis_name.startswith("単純集計:"):
                return run_simple_count(df, suggestion)
            elif analysis_name in ["クロス集計（キーワード間）", "クロス集計（キーワード×属性）", "話題カテゴリ別 観光地TOP10"]:
                return run_crosstab(df, suggestion)
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
                logger.warning(f"Python分析 '{analysis_name}' の実行ロジックが定義されていません。AI分析にフォールバックします。")
                suggestion['description'] = f"データサンプルを使い、'{analysis_name}' を実行してください。"
                ai_result_str = run_ai_summary_batch(df, suggestion)
                return {"data": ai_result_str, "image_base64": None, "summary": ai_result_str[:100] + "..."}
        
        elif analysis_type == 'ai':
            ai_result_str = run_ai_summary_batch(df, suggestion)
            return {"data": ai_result_str, "image_base64": None, "summary": ai_result_str[:100] + "..."}
            
        else:
            err_msg = f"不明な分析タイプ ('{analysis_type}') です: {analysis_name}"
            return {"data": err_msg, "image_base64": None, "summary": err_msg}
            
    except Exception as e:
        logger.error(f"execute_analysis ('{analysis_name}') 実行エラー: {e}", exc_info=True)
        err_msg = f"分析 '{analysis_name}' の実行中にエラーが発生しました: {e}"
        return {"data": err_msg, "image_base64": None, "summary": err_msg}

# --- 8.4. (★) Step B: JSON出力ヘルパー (変更なし) ---
# ... (convert_results_to_json_string は変更なし) ...


# --- 8.5. (★) Step B: UI描画関数 (★ 新ワークフロー) ---

def render_step_b():
    """(Step B) 分析手法の提案・実行・データ出力UIを描画する"""
    st.title("📊 Step B: 分析の実行とデータ出力")

    # (★) --- セッションステートの初期化 ---
    if 'df_flagged_B' not in st.session_state:
        st.session_state.df_flagged_B = pd.DataFrame()
    if 'suggestions_B' not in st.session_state:
        # (★) 提案を「辞書」で持つ (task_name -> details)
        st.session_state.suggestions_B = {}
    if 'step_b_results' not in st.session_state:
        # (★) プレビュー用の結果 (task_name -> result_data)
        st.session_state.step_b_results = {}
    if 'step_b_json_output' not in st.session_state:
        st.session_state.step_b_json_output = None
    if 'progress_text' not in st.session_state:
         st.session_state.progress_text = ""
    if 'tips_list' not in st.session_state:
        st.session_state.tips_list = []
    if 'current_tip_index' not in st.session_state:
        st.session_state.current_tip_index = 0
    if 'last_tip_time' not in st.session_state:
        st.session_state.last_tip_time = time.time()
    # (★) --- ここまでが修正点 ---

    # --- 1. ファイルアップロード ---
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
            
            # (★) 新しいファイルをアップロードしたら、古い提案と結果をクリア
            st.session_state.suggestions_B = {}
            st.session_state.step_b_results = {}
            st.session_state.step_b_json_output = None
            
            st.success(f"ファイル「{uploaded_flagged_file.name}」読込完了 ({len(df)}行)")
            with st.expander("データプレビュー (先頭5行)"):
                st.dataframe(df.head())
        except Exception as e:
            logger.error(f"Step B ファイル読込エラー: {e}", exc_info=True)
            st.error(f"ファイル読み込み中にエラー: {e}")
            return
    else:
        st.warning("分析を続けるには、Step A で生成したCSVファイルをアップロードしてください。")
        return

    df_B = st.session_state.df_flagged_B
    
    # (★) --- DFの列情報をキャッシュ (Selectbox用) ---
    all_cols = list(df_B.columns)
    text_cols = ['ANALYSIS_TEXT_COLUMN'] + [col for col in all_cols if 'text' in col.lower() or 'content' in col.lower()]
    keyword_cols = [col for col in all_cols if col.endswith('キーワード')]
    date_cols = [col for col in all_cols if col in df_B.select_dtypes(include='datetime64').columns or 'date' in col.lower()]
    numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df_B[col])]
    engagement_cols = [col for col in numeric_cols if any(c in col.lower() for c in ['いいね', 'like', 'エンゲージメント', 'engagement'])]

    # --- 2. 分析手法の提案と一括実行 --- (★) タイトル変更
    st.header("Step 2: 分析手法の提案と一括実行")
    st.markdown(f"（(★) AI提案モデル: `{MODEL_FLASH_LITE}`）")
    
    analysis_prompt_B = st.text_area(
        "（任意）AIに追加で指示したい分析タスクを入力:",
        placeholder="例: 広島市と観光地の相関関係を深掘りしたい。\n例: ポジティブな意見とネガティブな意見の具体例を3つずつ抽出して。",
        key="step_b_prompt"
    )

    if st.button("💡 分析手法を提案させる (Step 2)", key="suggest_button_B", type="primary"):
        if not st.session_state.tips_list:
            with st.spinner("分析TIPSをAIで生成中..."):
                st.session_state.tips_list = get_analysis_tips_list_from_ai()
                if st.session_state.tips_list:
                    st.session_state.current_tip_index = random.randint(0, len(st.session_state.tips_list) - 1)
                    st.session_state.last_tip_time = time.time()
            
        with st.spinner(f"データ構造と指示内容を分析し、手法を提案中 ({MODEL_FLASH_LITE})..."):
            st.session_state.step_b_results = {} # (★) 提案のたびに結果をリセット
            st.session_state.step_b_json_output = None # (★) 出力もリセット
            
            # (★) 堅牢化された関数 (上記 1. で修正) が呼ばれる
            base_suggestions = suggest_analysis_techniques_py(df_B)
            ai_suggestions = []
            if analysis_prompt_B.strip():
                ai_suggestions = suggest_analysis_techniques_ai(
                    analysis_prompt_B, df_B, base_suggestions
                )
            base_names = {s['name'] for s in base_suggestions}
            filtered_ai_suggestions = [s for s in ai_suggestions if s['name'] not in base_names]
            all_suggestions = sorted(base_suggestions + filtered_ai_suggestions, key=lambda x: x['priority'])
            
            # (★) --- BEGIN FIX (Loop Prevention) ---
            if not all_suggestions:
                # (★) 提案が0件だった場合の処理
                st.session_state.suggestions_B = {}
                st.warning(
                    "分析手法の提案が 0件 でした。\n"
                    "アップロードしたCSVファイルに、分析可能な列（`...キーワード`で終わる列や、`ANALYSIS_TEXT_COLUMN`など）が"
                    "正しく含まれているか確認してください。"
                )
                # (★) st.rerun() を *しない* で、このままスクリプトを続行させる
            else:
                # (★) 提案が1件以上あった場合の処理
                st.session_state.suggestions_B = {s['name']: s for s in all_suggestions}
                st.success(f"分析手法の提案が完了しました ({len(all_suggestions)}件)。Step 2.5 で一括実行してください。")
                st.rerun()

    # (★) --- NEW Step 2.5: Bulk Execution ---
    # (★) 提案 (suggestions_B) が存在する場合にのみ表示
    if st.session_state.suggestions_B:
        st.markdown("---")
        st.info("以下のボタンで、提案されたすべての分析をデフォルトパラメータで一括実行できます。")
        
        if st.button("🏃 全分析をデフォルトで一括実行 (Step 2.5)", type="primary", use_container_width=True):
            st.session_state.progress_text = "一括実行を開始します..."
            total_tasks = len(st.session_state.suggestions_B)
            progress_bar = st.progress(0.0, text="一括実行 待機中...")
            
            # (★) Tips表示
            tip_placeholder_b_bulk = st.empty()
            if st.session_state.tips_list:
                try:
                    current_tip = st.session_state.tips_list[st.session_state.current_tip_index]
                    tip_placeholder_b_bulk.info(f"💡 データ分析TIPS: {current_tip}")
                except IndexError:
                    st.session_state.current_tip_index = 0
            
            with st.spinner(f"全 {total_tasks} 件の分析を一括実行中..."):
                for i, (task_name, suggestion_details) in enumerate(st.session_state.suggestions_B.items()):
                    st.session_state.progress_text = f"({i+1}/{total_tasks}) 「{task_name}」を実行中..."
                    progress_bar.progress((i+1)/total_tasks, text=f"実行中: {task_name}")
                    
                    try:
                        result_data = execute_analysis(task_name, df_B, suggestion_details)
                        st.session_state.step_b_results[task_name] = result_data
                    except Exception as e:
                        logger.error(f"一括実行エラー ({task_name}): {e}", exc_info=True)
                        st.session_state.step_b_results[task_name] = {
                            "data": f"一括実行中にエラーが発生しました: {e}",
                            "image_base64": None,
                            "summary": f"エラー: {e}"
                        }
                
                st.session_state.progress_text = "全分析の一括実行が完了しました。"
                progress_bar.progress(1.0, text="一括実行 完了")
                tip_placeholder_b_bulk.empty()
                st.success("全分析の一括実行が完了しました。Step 3 で結果を確認してください。")
                st.rerun() # (★) プレビューを更新するためにリラン


    # --- 3. 分析のプレビューとパラメータ修正 ---
    if not st.session_state.suggestions_B:
        st.info("Step 2 で「分析手法を提案させる」ボタンを押してください。")
        return

    st.header("Step 3: 分析のプレビューとパラメータ修正")
    st.info("各分析項目の「▼」を開き、プレビューを確認してください。パラメータ（分析対象の列など）を修正して、個別に「再実行/更新」も可能です。")
    
    # (★) Tips表示
    tip_placeholder = st.empty()
    if st.session_state.tips_list:
        try:
            current_tip = st.session_state.tips_list[st.session_state.current_tip_index]
            tip_placeholder.info(f"💡 データ分析TIPS: {current_tip}")
        except IndexError:
            st.session_state.current_tip_index = 0

    # (★) UI進捗表示
    progress_text_placeholder = st.empty()
    if st.session_state.progress_text:
         progress_text_placeholder.info(st.session_state.progress_text)
         
    # (★) 提案されたタスクをループ処理
    for task_name, suggestion_details_from_session in st.session_state.suggestions_B.items():
        
        # (★) 状態がリセットされないよう、セッションの辞書を直接操作せず、
        # (★) 描画ループ用のローカルコピーを作成する
        suggestion_details = suggestion_details_from_session.copy()
        
        st.markdown("---")
        
        # (★) 各タスクのプレビューエリア (Expander の *外*)
        # (★) 実行結果がセッションにあれば表示
        if task_name in st.session_state.step_b_results:
            result = st.session_state.step_b_results[task_name]
            st.subheader(f"✅ プレビュー: {task_name}")
            
            # (★) [object Object] 対策: TOP5の辞書リストを正しく表示
            if task_name == "話題カテゴリ別 エンゲージメントTOP5と概要" and isinstance(result['data'], pd.DataFrame):
                st.dataframe(result['data'])
                for _, row in result['data'].iterrows():
                    st.markdown(f"**カテゴリ: {row['category']}** (投稿数: {row['post_count']})")
                    if row['top_posts']:
                         for post in row['top_posts']:
                             st.markdown(f"  - **EG: {post['engagement']}** - {post['summary_ai']}")
                             if post['media_link']:
                                 st.markdown(f"    [Link]({post['media_link']})")
                    st.markdown("---")
            
            # (★) その他のプレビュー
            elif result['image_base64']:
                st.image(base64.b64decode(result['image_base64']))
            
            if isinstance(result['data'], pd.DataFrame) and task_name != "話題カテゴリ別 エンゲージメントTOP5と概要":
                st.dataframe(result['data'].head(10))
            elif isinstance(result['data'], dict):
                st.json(result['data'])
            elif isinstance(result['data'], str):
                st.markdown(result['data'])
                
            st.caption(f"サマリ: {result.get('summary', 'N/A')}")
        else:
            st.subheader(f"⬜️ 未実行: {task_name}")

        
        # (★) 各タスクの編集・実行エリア
        with st.expander(f"「{task_name}」のパラメータ修正・再実行"):
            
            st.markdown(f"**説明:** {suggestion_details['description']}")
            st.markdown("##### (オプション) パラメータの変更")
            
            # (★) タスクごとに編集UIを動的に生成
            try:
                # (★) 3. 単純集計
                if task_name.startswith("単純集計:"):
                    default_col = suggestion_details['suitable_cols'][0]
                    new_col = st.selectbox(f"集計対象の列 ({task_name})", options=keyword_cols, index=keyword_cols.index(default_col) if default_col in keyword_cols else 0, key=f"sel_{task_name}")
                    suggestion_details['suitable_cols'] = [new_col] # (★) ローカルコピーを更新
                
                # (★) クロス集計
                elif task_name.startswith("クロス集計"):
                    default_col1 = suggestion_details['suitable_cols'][0]
                    default_col2 = suggestion_details['suitable_cols'][1]
                    c1, c2 = st.columns(2)
                    new_col1 = c1.selectbox(f"列 1 ({task_name})", options=all_cols, index=all_cols.index(default_col1) if default_col1 in all_cols else 0, key=f"sel_{task_name}_1")
                    new_col2 = c2.selectbox(f"列 2 ({task_name})", options=all_cols, index=all_cols.index(default_col2) if default_col2 in all_cols else 1, key=f"sel_{task_name}_2")
                    suggestion_details['suitable_cols'] = [new_col1, new_col2] # (★) ローカルコピーを更新

                # (★) 時系列
                elif task_name == "時系列キーワード分析":
                    default_dt = suggestion_details['suitable_cols']['datetime'][0]
                    default_kw = suggestion_details['suitable_cols']['keywords'][0]
                    c1, c2 = st.columns(2)
                    new_dt = c1.selectbox(f"日時列 ({task_name})", options=date_cols, index=date_cols.index(default_dt) if default_dt in date_cols else 0, key=f"sel_{task_name}_dt")
                    new_kw = c2.selectbox(f"キーワード列 ({task_name})", options=keyword_cols, index=keyword_cols.index(default_kw) if default_kw in keyword_cols else 0, key=f"sel_{task_name}_kw")
                    suggestion_details['suitable_cols'] = {"datetime": [new_dt], "keywords": [new_kw]} # (★) ローカルコピーを更新
                
                # (★) テキストマイニング / 共起ネットワーク
                elif task_name in ["テキストマイニング（頻出単語）", "共起ネットワーク"]:
                    default_col = suggestion_details['suitable_cols'][0]
                    new_col = st.selectbox(f"テキスト列 ({task_name})", options=text_cols, index=text_cols.index(default_col) if default_col in text_cols else 0, key=f"sel_{task_name}_txt")
                    suggestion_details['suitable_cols'] = [new_col] # (★) ローカルコピーを更新
                
                # (★) 5. エンゲージメントTOP5
                elif task_name == "話題カテゴリ別 エンゲージメントTOP5と概要":
                    defaults = suggestion_details['suitable_cols']
                    c1, c2 = st.columns(2)
                    new_topic = c1.selectbox(f"話題カテゴリ列 ({task_name})", options=keyword_cols, index=keyword_cols.index(defaults['topic'][0]) if defaults['topic'][0] in keyword_cols else 0, key=f"sel_{task_name}_top")
                    new_eng = c2.selectbox(f"エンゲージメント列 ({task_name})", options=engagement_cols, index=engagement_cols.index(defaults['engagement'][0]) if defaults['engagement'][0] in engagement_cols else 0, key=f"sel_{task_name}_eng")
                    # (★) text_col は変更しない
                    suggestion_details['suitable_cols'] = {'topic': [new_topic], 'text': defaults['text'], 'engagement': [new_eng]} # (★) ローカルコピーを更新

            except Exception as e:
                st.error(f"パラメータUIの描画に失敗: {e}")
                logger.error(f"パラメータUI描画エラー ({task_name}): {e}", exc_info=True)


            # (★) 個別実行ボタン
            if st.button(f"「{task_name}」を再実行/更新", key=f"run_{task_name}"): # (★) 文言変更
                st.session_state.progress_text = f"「{task_name}」を個別に実行中..."
                with st.spinner(f"「{task_name}」を実行中..."):
                    try:
                        # (★) 編集されたローカルコピー (suggestion_details) を渡す
                        result_data = execute_analysis(task_name, df_B, suggestion_details)
                        # (★) 実行結果をセッションステートに保存
                        st.session_state.step_b_results[task_name] = result_data
                        
                        # (★) !!! 重要: 修正したパラメータをセッションステートに書き戻す !!!
                        st.session_state.suggestions_B[task_name] = suggestion_details
                        
                        st.session_state.progress_text = f"「{task_name}」の実行が完了しました。"
                        st.rerun() # (★) UIを即時更新してプレビューを表示
                    except Exception as e:
                         st.error(f"分析実行エラー: {e}")
                         logger.error(f"個別実行エラー ({task_name}): {e}", exc_info=True)
                         st.session_state.progress_text = f"「{task_name}」の実行に失敗しました。"


    # (★) --- 4. (NEW) 最終エクスポート ---
    st.markdown("---")
    st.header("Step 4: 最終JSONのエクスポート")
    
    # (★) 実行状況のサマリー
    total_suggestions = len(st.session_state.suggestions_B)
    total_results = len(st.session_state.step_b_results)
    
    if total_results < total_suggestions:
        st.warning(f"まだすべての分析が実行されていません ({total_results} / {total_suggestions} 件)。(Step 2.5 の一括実行を推奨します)")
    else:
        st.success(f"すべての分析 ({total_results} / {total_suggestions} 件) が実行されました。JSONを生成できます。")

    
    if st.button("StepC用 JSONを生成・エクスポート (Step 4)", type="primary", use_container_width=True):
        if total_results == 0:
            st.error("分析が1つも実行されていません。Step 3 で各分析を実行してください。")
        else:
            with st.spinner("最終JSONファイルを生成中..."):
                try:
                    # (★) プレビューされた結果 (`step_b_results`) をJSONLに変換
                    json_output_string = convert_results_to_json_string(st.session_state.step_b_results)
                    st.session_state.step_b_json_output = json_output_string
                    st.success("StepC用のJSONデータが生成されました！")
                except Exception as e:
                    logger.error(f"Step B 最終JSON出力変換エラー: {e}", exc_info=True)
                    st.error(f"分析結果のJSON変換中にエラー: {e}")

    # (★) --- 5. (NEW) ダウンロードセクション ---
    if st.session_state.step_b_json_output:
        st.info(f"以下のJSONファイルには、Step 3 でプレビュー・実行された {len(st.session_state.step_b_results)} 件の分析結果がすべて含まれています。")
        
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
        
        preview_summaries = []
        try:
            for line in st.session_state.step_b_json_output.splitlines():
                line_data = json.loads(line)
                task_name = line_data.get("analysis_task")
                summary = line_data.get("summary", line_data.get("analysis_summaries", "No summary."))
                img_note = line_data.get("image_note", "No image.")
                
                if task_name == "OverallSummary":
                    preview_summaries.append(f"--- Overall Summary ---")
                    if isinstance(summary, dict):
                        for k, v in summary.items():
                             preview_summaries.append(f"  - {k}: {str(v)[:100]}...")
                    continue
                
                preview_summaries.append(f"[{task_name}] (Image: {img_note})")
                
        except Exception as e:
            preview_summaries = ["JSONプレビューの生成に失敗", str(e)]

        st.text_area(
            "JSONL (サマリープレビュー):",
            value="\n".join(preview_summaries),
            height=300,
            key="json_preview_B_summary",
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
    (★) ご要望に基づき、品質向上のためプロンプト生成ロジックを大幅に強化
    """
    logger.info("Step C プロンプト自動生成 (Flash Lite) 実行...")
    
    llm = get_llm(model_name=MODEL_FLASH_LITE, temperature=0.1)
    if llm is None:
        logger.error("generate_step_c_prompt: LLM (Flash Lite) が利用できません。")
        return "# AIモデル(Flash Lite)が利用できませんでした。"

    # (★) JSONLから分析タスク名（"analysis_task"）を抽出
    task_names = []
    summary_data = {}
    try:
        for line in jsonl_data_string.splitlines():
            try:
                task_data = json.loads(line)
                task_name = task_data.get("analysis_task")
                if task_name == "OverallSummary":
                    summary_data = task_data.get("data", {})
                    # analysis_summaries からタスクリストを取得 (フォールバック)
                    if not task_names and "analysis_summaries" in task_data:
                         task_names = list(task_data.get("analysis_summaries", {}).keys())
                elif task_name:
                    task_names.append(task_name)
            except json.JSONDecodeError:
                continue
        
        task_names_str = ", ".join(list(set(task_names))) # 重複削除
        if not task_names_str:
            task_names_str = "（JSONL内の各分析タスク）"
            
        summary_context = json.dumps(summary_data, ensure_ascii=False, indent=2)
        if len(summary_context) > 2000:
             summary_context = summary_context[:2000] + "\n... (サマリ省略)"
        
    except Exception as e:
        logger.error(f"Step B JSONLのパース (下読み) に失敗: {e}")
        task_names_str = "（JSONL内の各分析タスク）"
        summary_context = "{}"

    # (★) --- gemini-2.5-pro への「超」詳細な指示プロンプトを生成 ---
    # (★) ご要望に基づき、1万文字を超える詳細なレポートを出力させるため、
    # (★) 出力JSON形式、役割、各タスクの処理方法を厳密に定義します。
    
    prompt_template = """
あなたは、高名なデータアナリスト兼経営コンサルタントです。
クライアント（例：観光協会、事業会社）に提出するための、高品質な「データ分析レポート」を作成する任務を負っています。

これから、部下がPythonとAI（Flash Lite）で一次分析した結果（グラフ画像Base64、データテーブル、サマリを含むJSONL）が提供されます。
あなたの仕事は、この一次分析結果を【専門家の視点】で解釈し直し、クライアントの意思決定に資する「インサイト」と「戦略的提言」を含む、詳細かつ構造化されたレポート（JSON形式）を生成することです。

# 0. 全体サマリー（参考）
分析対象となったデータセットの全体像は以下の通りです。
{summary_context}

# 1. 実行タスク
提供される「分析データ（JSONL）」の各行（各分析タスク）を【漏れなく】処理し、以下の「出力JSON形式」に従って、スライド構成案を作成してください。

# 2. 出力JSON形式（厳守）
出力は、以下のキーを持つ辞書の【リスト（配列）】形式 `[ {{...}}, {{...}} ]` とします。
JSON以外のテキスト（例：「承知しました」）は【絶対に】含めないでください。

[
  {{
    "slide_title": "（スライドのタイトル）",
    "slide_layout": "（"title_and_content" または "text_and_image" または "title_only"）",
    "slide_content": [
      "（このスライドで伝えるべき【インサイト】や【考察】を箇条書きで詳細に記述）",
      "（データを単純に羅列するのではなく、専門家としての解釈を加えること）",
      "（必要であれば1スライドあたり1000文字以上の詳細な記述を行うこと）"
    ],
    "image_base64": "（"analysis_task" に "image_base64" が存在すれば、そのBase64文字列をここにコピーする。存在しなければ null とする）"
  }}
]

# 3. 必須スライド構成（この順番で構成すること）

### A. 導入スライド (3〜4枚)
1.  **表紙 (layout: "title_only")**: 「SNSデータ分析レポート（仮）」のようなタイトル。
2.  **エグゼクティブ・サマリー (layout: "title_and_content")**:
    * `analysis_task: "OverallSummary"` のデータ（特に `total_posts` や `total_engagement`）を参照し、分析の最も重要な発見（KGI/KPI）を3〜5点の箇条書きで要約します。
    * 【極めて重要】ここでのインサイトは、単なる数字の報告ではなく、「何を意味するのか」を記述してください。
3.  **分析概要 (layout: "title_and_content")**:
    * 分析の目的（例：「SNS投稿から観光客の動向と関心事を特定する」）と、分析対象データ（例：「Instagram投稿 XXX件」）を記述します。
4.  **アジェンダ (layout: "title_and_content")**:
    * この後のレポートの目次（例：「1. 全体傾向」「2. 主要カテゴリ分析」「3. 戦略的提言」）を記述します。

### B. 詳細分析スライド (JSONLの各タスクごとに1枚以上)
`analysis_task` が "OverallSummary" 以外の各タスク（例： {task_names}）について、以下の指示に従ってスライドを生成します。

5.  **タスク名: `run_simple_count` / `run_text_mining` / `run_topic_category_summary` (グラフがあるタスク)**
    * `slide_layout`: "text_and_image"
    * `slide_title`: JSONLの `analysis_task` 名（例：「頻出単語分析」）を、分かりやすい日本語タイトル（例：「主要な頻出キーワードの分析」）に翻訳・修正して記載します。
    * `image_base64`: JSONLの `image_base64` を【必ずコピー】します。
    * `slide_content`:
        * 【最重要】グラフ（`image_base64`）とデータ（`data`）を詳細に分析し、クライアントが理解すべき【インサイト（発見）】を3〜5点以上の詳細な箇条書きで記述します。
        * （悪い例：「グラフの通り、Aが1位でした。」）
        * （良い例：「分析の結果、Aが圧倒的多数を占めており、これはユーザーの関心がAに集中していることを示唆しています。特に2位のBとの差は...」）
        * `data`（JSON配列）から、上位3位までの具体的な数値や項目名を引用し、考察に厚みを持たせてください。

6.  **タスク名: `run_cooccurrence_network` (共起ネットワーク)**
    * `slide_layout`: "text_and_image"
    * `slide_title`: 「共起ネットワーク分析：話題の関連性」
    * `image_base64`: JSONLの `image_base64` を【必ずコピー】します。
    * `slide_content`:
        * ネットワーク図（`image_base64`）を解釈し、どの単語が中心にあるか（中心性）、どの単語同士が強く結びついているか（コミュニティ）を指摘します。
        * （例：「『A』と『B』、また『C』と『D』のグループが形成されており、これは...という2つの異なる文脈で語られていることを示します。」）
        * `data`（エッジリスト）から、`weight`（重み）が特に高い組み合わせを具体的に引用してください。

7.  **タスク名: `run_crosstab` / `run_topic_engagement_top5` (グラフがないタスク)**
    * `slide_layout`: "title_and_content"
    * `slide_title`: タスク名を分かりやすい日本語タイトル（例：「カテゴリ別 エンゲージメントTOP5」）に修正して記載します。
    * `image_base64`: null
    * `slide_content`:
        * `data`（JSON配列）の分析結果から、特筆すべきパターン、相関、または具体例（例：エンゲージメントが最も高かった投稿のAI要約）を引用し、詳細なインサイトを記述します。

8.  **タスク名: `run_ai_summary_batch` (AIによる自由分析)**
    * `slide_layout`: "title_and_content"
    * `slide_title`: AIが実行したタスク名（例：「ポジティブな意見の具体例」）
    * `image_base64`: null
    * `slide_content`:
        * AIの回答（`data`フィールドの文字列）を、クライアント向けに分かりやすく箇条書きで再構成し、提示します。

### C. 結論スライド (2枚)
9.  **結論 (layout: "title_and_content")**:
    * 全ての詳細分析（スライドB群）から導き出される【全体的な結論】を、強固な根拠（分析データへの言及）とともに要約します。
    * （例：「SNS分析の結果、強みは『X』であり、課題は『Y』であることが明らかになった。」）

10. **戦略的提言 (layout: "title_and_content")**:
    * 【最重要】あなたが経営コンサルタントとして、この分析結果に基づき、クライアントが次に取るべき【具体的なアクション（施策）】を3〜5点、提案してください。
    * （例：「1. 強みである『X』をさらに伸ばすため、...なキャンペーンを推奨する。」）
    * （例：「2. 課題である『Y』を克服するため、...なターゲット層へのアプローチを強化する。」）

# 4. 最終確認
* **情報量**: レポート全体で1万文字を超えるような、詳細で充実した内容を期待します。`slide_content` の各項目は、単語ではなく、完全な「文章」または「詳細な箇条書き」にしてください。
* **厳格な形式**: 出力は `[` で始まり `]` で終わる、JSON配列形式のみとします。

この指示に従い、最高の分析レポート（JSON）を作成してください。
"""

    prompt = PromptTemplate.from_template(prompt_template)
    
    try:
        # (★) Flash Lite に Pro への指示書を生成させる
        generated_prompt = prompt.invoke({
            "summary_context": summary_context,
            "task_names": task_names_str
        })
        
        logger.info("Step C プロンプト自動生成 完了。")
        return generated_prompt

    except Exception as e:
        logger.error(f"generate_step_c_prompt error: {e}", exc_info=True)
        # (★) --- フォールバック用の最小限のプロンプト ---
        return f"# (★) プロンプト自動生成失敗: {e}\n\n# 指示:\nあなたは優秀な経営コンサルタントです。提供される「分析データ（JSONL）」を読み、以下のJSON形式でPowerPoint用の分析レポートを作成してください。\n\n[ {{ \"slide_title\": \"...\", \"slide_layout\": \"title_and_content\", \"slide_content\": [\"...\", \"...\"], \"image_base64\": null }} ]"

# --- 9. (★) Step C: AIレポート生成 (ハイブリッド処理) ---
# (★) 廃止: generate_step_c_prompt() は不要になりました。

def run_step_c_analysis(
    jsonl_data_string: str,
    model_name: str,
    progress_bar: st.delta_generator.DeltaGenerator,
    log_placeholder: st.delta_generator.DeltaGenerator
) -> str:
    """
    (★) Step C: AIレポート生成 (ハイブリッド・RateLimit対応版)
    
    [新ロジック] ハングアップ (504 Timeout) を回避するため、AIにBase64画像(巨大トークン)を
    渡すのを *やめ* 、「テキストデータ」のみを渡して考察を生成させる。
    Python側で、AIの考察(テキスト)と、元のBase64画像を「再結合」する。
    """
    logger.info(f"Step C AIレポート生成 (ハイブリッド処理) 開始... (Model: {model_name})")

    # (★) --- 1. モデルのRPM制限とスリープ時間を定義 ---
    if model_name == MODEL_PRO:
        rpm_limit = 2
        tpm_limit = 125000
    else: # (★) デフォルトは Flash
        model_name = MODEL_FLASH
        rpm_limit = 10
        tpm_limit = 250000
        
    sleep_time = (60 / rpm_limit) + 0.5 # (e.g., Pro: 30.5s, Flash: 6.5s)
    
    logger.info(f"モデル: {model_name}, RPM: {rpm_limit}, 待機: {sleep_time:.1f}秒")

    # (★) --- 2. チャンク生成用のAIプロンプトテンプレートを定義 ---
    # (★) 修正: AIは「画像(image_base64)」を見ない前提のプロンプトに変更
    ITERATIVE_SLIDE_PROMPT_TEMPLATE = """
    あなたはシニアデータアナリストであり、クライアント向けレポートの「スライド1枚」の
    【テキスト部分】を作成しています。
    提供される「分析タスクデータ（テキストと数値データのみ）」を読み、このタスク専用の
    スライドタイトルと考察（slide_content）を生成してください。

    # 分析タスクデータ (テキスト・数値データのみ):
    {task_data_text_only}

    # 指示:
    1.  **タイトル**: `task_data_text_only` の `analysis_task` 名に基づき、 professional な「slide_title」を考案してください。
    2.  **考察 (最重要)**: `task_data_text_only` の `summary` と `data`（テーブルデータ）を解釈し、クライアントが知るべき【インサイト（発見）】を「slide_content」として2〜4点の詳細な箇条書き（文字列リスト）で記述してください。
        (注: あなたにはグラフ画像は提供されていません。`data` の数値や `summary` のテキストのみを根拠に考察を記述してください。)

    # 出力形式 (厳守):
    * JSON以外のテキストは絶対に含めず、【単一のJSONオブジェクト】`{{ ... }}` のみを出力してください。
    * 以下の構造を厳格に守ってください。
        {{
          "slide_title": "（指示1で考案したタイトル）",
          "slide_content": [
            "（指示2で記述したインサイト1）",
            "（指示2で記述したインサイト2）"
          ]
        }}

    # 回答 (単一のJSONオブジェクトのみ):
    """
    
    prompt = PromptTemplate.from_template(ITERATIVE_SLIDE_PROMPT_TEMPLATE)

    # (★) --- 修正: タイムアウトを 120秒 (2分) に設定 ---
    # (★) リクエストはテキストのみなので、300秒も不要
    llm = get_llm(model_name=model_name, temperature=0.2, timeout_seconds=120)
    if llm is None:
        st.error(f"AIモデル({model_name})が利用できません。")
        return "[]" # 空のJSONリスト
    
    chain = prompt | llm | StrOutputParser()
    # (★) --- ここまでが修正点 ---

    # (★) --- 3. 逐次処理ループ (チャンキングは廃止) ---
    report_slides_list = []
    log_messages_ui = []
    
    tasks_all = jsonl_data_string.strip().splitlines()
    
    # (★) 3.1. OverallSummaryを抽出し、残りを処理対象タスクとする
    summary_line = "{}"
    tasks_to_process = []
    for line in tasks_all:
        if '"analysis_task": "OverallSummary"' in line:
            summary_line = line
        else:
            tasks_to_process.append(line)
            
    if not tasks_to_process:
        logger.warning("処理対象の分析タスクが0件です。")
        return "[]"

    total_tasks = len(tasks_to_process)
    logger.info(f"全 {total_tasks} タスクを逐次処理します。")
    
    # (★) 3.2. 表紙スライドを追加
    report_slides_list.append({
        "slide_title": "SNSデータ分析レポート",
        "slide_layout": "title_only",
        "slide_content": ["AI-Generated Analysis (Powered by Gemini)"],
        "image_base64": None
    })
    
    # (★) 3.3. 目次スライドを追加 (この時点ではタスク名のみ)
    try:
        agenda_items = []
        for i, task_line in enumerate(tasks_to_process):
             # (★) 巨大タスク分割(JSONパース失敗)のフォールバック
            try:
                task_name = json.loads(task_line).get('analysis_task', f'分析タスク {i+1}')
            except json.JSONDecodeError:
                task_name = f'分析タスク {i+1} (読み込みエラー)'
            agenda_items.append(f"{i+1}. {task_name}")

        agenda_items.append(f"{len(tasks_to_process) + 1}. 結論と戦略的提言")
        report_slides_list.append({
            "slide_title": "本日のアジェンダ",
            "slide_layout": "title_and_content",
            "slide_content": agenda_items,
            "image_base64": None
        })
    except Exception as e:
        logger.error(f"目次スライドの生成に失敗: {e}")

    # (★) 3.4. メインの分析スライドをループ処理
    for i, task_line in enumerate(tasks_to_process):
        
        task_name = f"Task {i+1}/{total_tasks}"
        original_task_json = {}
        
        try:
            # (★) --- 3.4.1. タスクのパースと画像/テキストの分離 ---
            original_task_json = json.loads(task_line)
            task_name = original_task_json.get('analysis_task', task_name)

            # (★) 1. 画像をPython変数に退避
            image_to_pass_through = original_task_json.get("image_base64")
            
            # (★) 2. AIに渡す「テキストのみ」のJSONを作成
            text_only_task_json = original_task_json.copy()
            text_only_task_json["image_base64"] = None # (★) 画像を削除
            # (★) dataが巨大すぎる場合も考慮し、dataも1000文字に制限
            if "data" in text_only_task_json and len(json.dumps(text_only_task_json["data"])) > 1000:
                text_only_task_json["data"] = f"（データプレビュー: {str(text_only_task_json['data'])[:1000]}...）"
            
            task_data_text_only_str = json.dumps(text_only_task_json)
            
        except Exception as e:
            logger.error(f"タスク '{task_name}' のJSONパースに失敗: {e}")
            log_messages_ui.append(f"  -> ERROR: '{task_name}' のJSONパースに失敗。スキップします。")
            continue # (★) このタスクはスキップ

        # (★) --- 3.4.2. UI（進捗バー・ログ）の更新 ---
        progress_percent = (i + 1) / (total_tasks + 1) # (★) +1 は結論スライド分
        progress_bar.progress(progress_percent, text=f"Step C (スライド生成中): {i+1}/{total_tasks} (モデル: {model_name})")
        log_messages_ui.append(f"[{i+1}/{total_tasks}] '{task_name}' の処理を開始 (文字数: {len(task_data_text_only_str):,})...")
        log_placeholder.text_area("実行ログ:", "\n".join(log_messages_ui[::-1]), height=250, key=f"step_c_log_{i}")

        try:
            # (★) --- 3.4.3. AIへのリクエスト (テキストのみ) ---
            log_messages_ui.append(f"  -> AI ({model_name}) にリクエストを送信... (Timeout: 120s)")
            log_placeholder.text_area("実行ログ:", "\n".join(log_messages_ui[::-1]), height=250, key=f"step_c_log_{i}_sending")
            
            response_str = chain.invoke({"task_data_text_only": task_data_text_only_str})
            
            log_messages_ui.append(f"  -> AI が応答しました。レスポンスを解析中...")
            log_placeholder.text_area("実行ログ:", "\n".join(log_messages_ui[::-1]), height=250, key=f"step_c_log_{i}_received")

            # (★) AIの回答 (タイトルとコンテントのみ) をパース
            match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if match:
                ai_response_json = json.loads(match.group(0))
                
                # (★) --- 3.4.4. AIの考察と、退避させた画像を「再結合」 ---
                final_slide_object = {
                    "slide_title": ai_response_json.get("slide_title", task_name),
                    "slide_layout": "text_and_image" if image_to_pass_through else "title_and_content",
                    "slide_content": ai_response_json.get("slide_content", ["AIによる考察の生成に失敗しました。"]),
                    "image_base64": image_to_pass_through # (★) ここで画像を戻す
                }
                report_slides_list.append(final_slide_object)
                log_messages_ui.append(f"  -> SUCCESS: スライド '{final_slide_object.get('slide_title')}' を生成しました。")
            else:
                raise Exception("AIがJSONオブジェクト `{{...}}` を返しませんでした。")
        
        except Exception as e:
            logger.error(f"タスク '{task_name}' の処理に失敗: {e}", exc_info=True)
            log_messages_ui.append(f"  -> ERROR: '{task_name}' の処理に失敗。{e}")
            report_slides_list.append({
                "slide_title": f"エラー: {task_name}",
                "slide_layout": "title_and_content",
                "slide_content": [f"このスライドの生成に失敗しました。", f"エラー: {e}"],
                "image_base64": None
            })
        
        # (★) --- 3.4.5. Rate Limit のための待機 ---
        if i < total_tasks: # (★) 結論スライドのリクエストがまだ残っているため、必ず待機
            log_messages_ui.append(f"  -> Rate Limit (RPM) のため {sleep_time:.1f} 秒待機します...")
            log_placeholder.text_area("実行ログ:", "\n".join(log_messages_ui[::-1]), height=250, key=f"step_c_log_{i}_sleep")
            time.sleep(sleep_time)

    # (★) 4. 結論スライドの生成
    try:
        chunk_name = f"結論スライド"
        progress_percent = 1.0
        progress_bar.progress(progress_percent, text=f"Step C (チャンク処理中): {chunk_name} (モデル: {model_name})")
        log_messages_ui.append(f"[{total_tasks+1}/{total_tasks+1}] {chunk_name} の処理を開始...")
        log_placeholder.text_area("実行ログ:", "\n".join(log_messages_ui[::-1]), height=250, key="step_c_log_final")

        conclusion_llm = get_llm(model_name=model_name, temperature=0.2, timeout_seconds=120)
        if conclusion_llm is None:
            raise Exception("結論スライド用AIモデルの取得に失敗")

        CONCLUSION_PROMPT_TEMPLATE = """
        あなたはシニアデータアナリストです。
        以下の「分析サマリー」と「これまで生成したスライドのタイトルリスト」に基づき、
        レポートの締めくくりとなる【結論と戦略的提言】のスライド1枚分のJSONオブジェクトを生成してください。

        # 分析サマリー (OverallSummary):
        {summary_data_line}
        
        # 生成済みスライドタイトル:
        {slide_titles}

        # 指示:
        1.  タイトルは「結論と戦略的提言」とします。
        2.  レイアウトは「title_and_content」とします。
        3.  内容は、分析全体から導かれる「結論」と、クライアントが次に取るべき「具体的なアクション（提言）」を3〜5点の箇条書きで記述してください。
        4.  画像 (image_base64) は null とします。

        # 出力形式 (厳守):
        * JSON以外のテキストは絶対に含めず、【単一のJSONオブジェクト】`{{ ... }}` のみを出力してください。

        # 回答 (単一のJSONオブジェクトのみ):
        """
        
        conclusion_prompt = PromptTemplate.from_template(CONCLUSION_PROMPT_TEMPLATE)
        conclusion_chain = conclusion_prompt | conclusion_llm | StrOutputParser()
        
        log_messages_ui.append(f"  -> AI ({model_name}) にリクエストを送信... (Timeout: 120s)")
        log_placeholder.text_area("実行ログ:", "\n".join(log_messages_ui[::-1]), height=250, key="step_c_log_final_sending")
        
        response_str = conclusion_chain.invoke({
            "summary_data_line": summary_line,
            "slide_titles": json.dumps([s.get('slide_title') for s in report_slides_list], ensure_ascii=False)
        })
        
        log_messages_ui.append(f"  -> AI が応答しました。レスポンスを解析中...")
        log_placeholder.text_area("実行ログ:", "\n".join(log_messages_ui[::-1]), height=250, key="step_c_log_final_received")

        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if match:
            report_slides_list.append(json.loads(match.group(0)))
            log_messages_ui.append(f"  -> SUCCESS: 結論スライドを生成しました。")
        else:
            raise Exception("AIが結論スライドのJSONを返しませんでした。")
            
    except Exception as e:
         logger.error(f"結論スライドの生成に失敗: {e}")
         log_messages_ui.append(f"  -> ERROR: 結論スライドの生成に失敗。{e}")
         report_slides_list.append({
                "slide_title": "結論と戦略的提言 (生成失敗)",
                "slide_layout": "title_and_content",
                "slide_content": [f"結論スライドの自動生成に失敗しました。", f"エラー: {e}"],
                "image_base64": None
            })

    # (★) 5. 最終的なJSON文字列を返す
    progress_bar.progress(1.0, text="Step C: 完了！")
    log_placeholder.text_area("実行ログ:", "\n".join(log_messages_ui[::-1]), height=250, key="step_c_log_done")
    
    return json.dumps(report_slides_list, ensure_ascii=False, indent=2)

def render_step_c():
    """(Step C) AIレポート生成UIを描画する"""
    st.title(f"🖋️ Step C: AI分析レポート生成") # (★) モデル名をタイトルから削除

    # Step C 固有のセッションステート
    if 'step_c_jsonl_data' not in st.session_state:
        st.session_state.step_c_jsonl_data = None
    if 'step_c_prompt' not in st.session_state:
        st.session_state.step_c_prompt = None # (★) この変数はもう使用しません
    if 'step_c_report_json' not in st.session_state:
        st.session_state.step_c_report_json = None
    if 'step_c_model' not in st.session_state:
        st.session_state.step_c_model = MODEL_FLASH # (★) デフォルトをFlashに

    # --- 1. ファイルアップロード ---
    st.header("Step 1: 分析データ (JSON) のアップロード")
    st.info("Step B でエクスポートした `analysis_data.json` をアップロードしてください。")
    uploaded_report_file = st.file_uploader(
        "分析データファイル (analysis_data.json)",
        type=['json', 'jsonl', 'txt'],
        key="step_c_uploader"
    )

    if uploaded_report_file:
        try:
            # (★) ファイルがアップロードされても、プロンプト生成は *行わない*
            if st.session_state.step_c_jsonl_data is None:
                jsonl_data_string = uploaded_report_file.getvalue().decode('utf-8')
                st.session_state.step_c_jsonl_data = jsonl_data_string
                st.session_state.step_c_report_json = None # (★) 結果をリセット
                st.success(f"ファイル「{uploaded_report_file.name}」読込完了")
        
        except Exception as e:
            logger.error(f"Step C ファイル読込エラー: {e}", exc_info=True)
            st.error(f"ファイル読み込み中にエラー: {e}")
            return
    else:
        st.session_state.step_c_jsonl_data = None
        st.session_state.step_c_prompt = None
        st.session_state.step_c_report_json = None
        st.warning("分析を続けるには、Step B で生成した JSON ファイルをアップロードしてください。")
        return

    # (★) --- 修正: 旧Step 2 (プロンプト編集) を削除 ---
    # (★) 逐次処理モデルに変更したため、巨大なプロンプトの編集は不要になりました。
    # (★) --- ここまでが修正点 ---


    # --- 2. 分析レポートの実行 (★ 旧Step 3) ---
    st.header("Step 2: AI分析レポートの実行") # (★) ステップ番号を 2 に変更

    # (★) --- 修正: モデル選択UI ---
    st.markdown("分析に使用するAIモデルを選択してください。")
    
    model_options = [MODEL_FLASH, MODEL_PRO]
    try:
        default_index = model_options.index(st.session_state.step_c_model)
    except ValueError:
        default_index = 0
        
    selected_model_name = st.radio(
        "使用モデル",
        options=model_options,
        index=default_index,
        key="step_c_model_radio",
        horizontal=True,
    )
    st.session_state.step_c_model = selected_model_name

    if selected_model_name == MODEL_PRO:
        st.warning(
            f"**`{MODEL_PRO}` (無料枠) は 2 RPM (30秒/リクエスト) の制限があります。**\n"
            f"スライド10枚の生成には約5分かかります。ご注意ください。"
        )
    else:
        st.info(
            f"**`{MODEL_FLASH}` (無料枠) は 10 RPM (6秒/リクエスト) の制限があります。**\n"
            f"比較的 高速に生成できます。（推奨）"
        )
    # (★) --- ここまでが修正点 ---

    
    if st.button(f"分析レポートを生成 (Step 2)", key="execute_button_C", type="primary", use_container_width=True):
        if not st.session_state.step_c_jsonl_data:
            st.error("データがありません。Step 1でファイルをアップロードしてください。")
            return
        
        # (★) --- 修正: 進捗バーとログ用のプレースホルダを定義 ---
        progress_bar = st.progress(0.0, text="Step C: 分析待機中...")
        log_placeholder = st.empty()
        # (★) --- ここまでが修正点 ---

        selected_model = st.session_state.step_c_model
        
        try:
            # (★) 修正: 逐次処理を行う run_step_c_analysis に変更
            st.session_state.step_c_report_json = run_step_c_analysis(
                st.session_state.step_c_jsonl_data,
                selected_model,
                progress_bar, # (★) 進捗バーを渡す
                log_placeholder # (★) ログ表示を渡す
            )
            st.success("AIによる分析レポートが生成されました！")
            
        except Exception as e:
            # (★) 実行時エラーのキャッチ
            logger.error(f"Step C 実行中に予期せぬエラー: {e}", exc_info=True)
            st.error(f"分析実行中に予期せぬエラーが発生しました: {e}")
            progress_bar.progress(1.0, text="エラーにより中断")


    # --- 3. 結果のプレビューとエクスポート (★ 旧Step 4) ---
    if st.session_state.step_c_report_json:
        st.header("Step 3: 分析レポート（JSON）の確認とエクスポート") # (★) ステップ番号を 3 に変更
        st.info("以下の構造化JSONは、Step D (PowerPoint生成) で使用します。")

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
            report_data = json.loads(st.session_state.step_c_report_json)
            if isinstance(report_data, list) and all(isinstance(item, dict) for item in report_data):
                st.text_area(
                    "AIが生成した構造化JSON (プレビュー: 先頭5000文字)",
                    value=st.session_state.step_c_report_json[:5000] + "...",
                    height=300,
                    key="json_preview_C",
                    disabled=True
                )
                
                st.markdown("---")
                st.subheader(f"スライド構成 プレビュー ({len(report_data)}枚)")
                for i, slide in enumerate(report_data):
                    title = slide.get('slide_title', '（タイトルなし）')
                    layout = slide.get('slide_layout', 'N/A')
                    
                    slide_content_list = slide.get('slide_content')
                    if isinstance(slide_content_list, list) and slide_content_list:
                        content_preview = str(slide_content_list[0]) if slide_content_list[0] else "（空のコンテンツ）"
                    else:
                        content_preview = "（コンテンツなし）"

                    has_image = "有り" if slide.get("image_base64") else "無し"
                    
                    expander_label = f"**{i+1}: {title}** (Layout: {layout}, Image: {has_image})"
                    with st.expander(expander_label):
                        st.markdown(f"**内容 (抜粋):**\n- {content_preview}...")
            else:
                st.error("AIの回答が期待したスライドのリスト形式ではありません。")
                st.text_area("AIの生回答 (JSON):", value=st.session_state.step_c_report_json, height=200, disabled=True)
                
        except Exception as e:
            st.error(f"レポートのプレビュー中にエラー: {e}")
            st.text_area("AIの生回答 (パース失敗):", value=st.session_state.step_c_report_json, height=200, disabled=True)
            
        st.success("データをダウンロードし、Step D (PowerPoint生成) に進んでください。")


# (★) ---Step D---
try:
    import pptx
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.dml import MSO_THEME_COLOR
except ImportError:
    st.error(
        "PowerPoint生成ライブラリ(python-pptx)が見つかりません。"
        "pip install python-pptx を実行してください。"
    )

try:
    import pptx
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.dml import MSO_THEME_COLOR
except ImportError:
    st.error(
        "PowerPoint生成ライブラリ(python-pptx)が見つかりません。"
        "pip install python-pptx を実行してください。"
    )

def find_layout_by_name(prs: pptx.presentation.Presentation, layout_name: str) -> Optional[pptx.slide.SlideLayout]:
    """
    (★) プレゼンテーションのマスターから、指定されたレイアウト名（完全一致）でスライドレイアウトを探す。
    """
    for layout in prs.slide_layouts:
        if layout.name == layout_name:
            return layout
    logger.warning(f"  -> '{layout_name}' に一致するレイアウトが見つかりません。")
    return None

def create_powerpoint_presentation(
    template_file: Optional[BytesIO],
    report_data: List[Dict[str, Any]],
    layout_map_names: Dict[str, str] # (★) 選択されたレイアウト名を受け取る
) -> BytesIO:
    """
    (Step D) テンプレート(.pptx)とスライド構成(JSON)に基づき、
    python-pptx を使用して最終的なPowerPointファイルを生成する。
    (★) ユーザーが選択したレイアウト名を使用するようロジックを改修
    """
    logger.info("PowerPoint生成処理 開始...")
    
    try:
        # (★) 1. テンプレートの読み込み
        if template_file:
            template_file.seek(0)
            prs = Presentation(template_file)
            logger.info("アップロードされたテンプレートを使用してPPTXを生成します。")
        else:
            prs = Presentation()
            logger.info("デフォルトのテンプレートを使用してPPTXを生成します。")

        # (★) 2. ユーザーが選択したレイアウトをマッピング
        layout_map = {
            "title_only": find_layout_by_name(prs, layout_map_names.get("title")),
            "agenda": find_layout_by_name(prs, layout_map_names.get("agenda")),
            "title_and_content": find_layout_by_name(prs, layout_map_names.get("content_text")),
            "text_and_image": find_layout_by_name(prs, layout_map_names.get("content_image")),
        }
        
        # (★) フォールバックレイアウト (最も汎用的なもの)
        fallback_layout = prs.slide_layouts[1] # 「タイトルとコンテンツ」
        fallback_title_layout = prs.slide_layouts[0] # 「タイトル スライド」
        
        if layout_map["title_only"] is None:
             layout_map["title_only"] = fallback_title_layout
             logger.warning("「表紙」レイアウトが見つからないため、デフォルトの「タイトル スライド」を使用します。")
        if layout_map["agenda"] is None:
             layout_map["agenda"] = fallback_layout
             logger.warning("「目次」レイアウトが見つからないため、デフォルトの「タイトルとコンテンツ」を使用します。")
        if layout_map["title_and_content"] is None:
             layout_map["title_and_content"] = fallback_layout
             logger.warning("「テキスト」レイアウトが見つからないため、デフォルトの「タイトルとコンテンツ」を使用します。")
        if layout_map["text_and_image"] is None:
             layout_map["text_and_image"] = fallback_layout # 画像ありも最悪これで代用
             logger.warning("「画像+テキスト」レイアウトが見つからないため、デフォルトの「タイトルとコンテンツ」を使用します。")

        logger.info(f"使用レイアウトマッピング: {layout_map_names}")

        # (★) 3. スライドの生成 (JSONデータをループ)
        
        # (★) --- 3.1. 表紙スライド ---
        first_slide_data = report_data[0]
        if first_slide_data.get("slide_layout") == "title_only":
            slide = prs.slides.add_slide(layout_map["title_only"]) # (★) 選択されたレイアウト
            try:
                slide.shapes.title.text = first_slide_data.get("slide_title", "分析レポート")
            except: pass
            try:
                # (★) 多くのテンプレートでは Title Layout の Placeholder[1] が Subtitle
                if len(slide.placeholders) > 1 and slide.placeholders[1]:
                     slide.placeholders[1].text = first_slide_data.get("slide_content", [""])[0]
            except: pass
            
            report_data = report_data[1:]
        
        # (★) --- 3.2. 目次(Agenda)スライドの自動生成 ---
        try:
            logger.info("目次スライドを自動生成します...")
            agenda_slide = prs.slides.add_slide(layout_map["agenda"]) # (★) 選択されたレイアウト
            
            # (★) タイトルプレースホルダを探す
            title_shape = None
            try:
                title_shape = agenda_slide.shapes.title
            except AttributeError:
                for shape in agenda_slide.placeholders:
                    # 0 or 100 (Title) or 136 (Center Title)
                    if shape.placeholder_format.idx == 0 or shape.placeholder_format.idx == 100 or shape.placeholder_format.idx == 136:
                        title_shape = shape
                        break
            if title_shape:
                title_shape.text = "本日のアジェンダ"
            
            # (★) 目次用の本文プレースホルダを探す
            agenda_body_shape = None
            for shape in agenda_slide.placeholders:
                 # 1 (Body) or 101 (Content)
                 if shape.placeholder_format.idx == 1 or shape.placeholder_format.idx == 101: 
                     agenda_body_shape = shape
                     break
            
            if agenda_body_shape:
                tf = agenda_body_shape.text_frame
                tf.clear()
                for i, slide_data in enumerate(report_data):
                    p = tf.add_paragraph()
                    p.text = f"{i+1}. {slide_data.get('slide_title', '（無題のスライド）')}"
                    p.level = 0
            else:
                 logger.warning("目次スライドの本文プレースホルダが見つかりませんでした。")
                 
        except Exception as e:
            logger.error(f"目次スライドの自動生成に失敗: {e}")

        # (★) --- 3.3. コンテンツスライド (残り) ---
        for i, slide_data in enumerate(report_data):
            slide_title = slide_data.get("slide_title", f"スライド {i+3}")
            slide_layout_key = slide_data.get("slide_layout", "title_and_content")
            slide_content = slide_data.get("slide_content", ["（コンテンツなし）"])
            image_base64 = slide_data.get("image_base664") # (★) 既存コードの typo を修正 (664 -> 64)
            if image_base64 is None:
                image_base64 = slide_data.get("image_base64") # (★) 正しいキーでも取得

            if image_base64 and slide_layout_key == "title_and_content":
                slide_layout_key = "text_and_image"
            
            # (★) マップからレイアウトを取得
            if image_base64:
                layout_to_use = layout_map["text_and_image"]
            else:
                layout_to_use = layout_map["title_and_content"]
            
            slide = prs.slides.add_slide(layout_to_use)
            
            try:
                slide.shapes.title.text = slide_title
            except Exception as e:
                logger.warning(f"スライド {i+3} のタイトル設定失敗: {e}")

            # (★) --- コンテンツと画像の配置 (ロジックを堅牢化) ---
            try:
                # (★) プレースホルダをタイプ別に分類
                text_placeholders = []
                image_placeholders = []
                
                for shape in slide.placeholders:
                    if shape.placeholder_format.idx == 0: continue # タイトルは除外
                    
                    if shape.has_text_frame:
                        text_placeholders.append(shape)
                    # (★) 画像用プレースホルダ (idx 101-107, 114-118 など) を推測
                    elif shape.placeholder_format.idx > 100: 
                        image_placeholders.append(shape)

                # (★) 画像がある場合の処理 (text_and_image)
                if image_base64:
                    # (★) 1. テキストを挿入 (最初に見つかったテキストプレースホルダに)
                    if text_placeholders:
                        tf = text_placeholders[0].text_frame
                        tf.clear()
                        p = tf.paragraphs[0]
                        p.text = str(slide_content[0])
                        for item in slide_content[1:]:
                            p = tf.add_paragraph()
                            p.text = str(item)
                    
                    # (★) 2. 画像を挿入 (最初に見つかった画像プレースホルダに)
                    image_ph = None
                    if image_placeholders:
                        image_ph = image_placeholders[0]
                    elif len(text_placeholders) > 1:
                        # (★) 画像用がなければ、2番目のテキストプレースホルダを代用
                        image_ph = text_placeholders[1] 

                    if image_ph:
                        try:
                            img_bytes = base64.b64decode(image_base64)
                            img_stream = BytesIO(img_bytes)
                            image_ph.insert_picture(img_stream)
                            logger.info(f"スライド '{slide_title}': グラフ画像の挿入に成功。")
                        except Exception as e:
                            logger.error(f"スライド '{slide_title}': グラフ画像の挿入に失敗: {e}")
                            if image_ph.has_text_frame:
                                image_ph.text_frame.text = f"（画像挿入エラー: {e}）"
                    else:
                         logger.warning(f"スライド '{slide_title}': 画像用プレースホルダが見つかりません。")

                # (★) 画像がない場合の処理 (title_and_content)
                else:
                    if not text_placeholders:
                         logger.warning(f"スライド '{slide_title}': コンテンツプレースホルダが見つかりません。")
                         continue
                    tf = text_placeholders[0].text_frame
                    tf.clear()
                    p = tf.paragraphs[0]
                    p.text = str(slide_content[0])
                    for item in slide_content[1:]:
                        p = tf.add_paragraph()
                        p.text = str(item)

            except Exception as e:
                logger.error(f"スライド {i+3} ('{slide_title}') のコンテンツ/画像設定中にエラー: {e}", exc_info=True)

        logger.info("PowerPoint生成処理 完了。")
        file_stream = BytesIO()
        prs.save(file_stream)
        file_stream.seek(0)
        return file_stream

    except Exception as e:
        logger.error(f"create_powerpoint_presentation 全体でエラー: {e}", exc_info=True)
        st.error(f"PowerPointの生成に失敗しました: {e}")
        return None

def render_step_d():
    """(Step D) PowerPoint生成UIを描画する"""
    st.title(f"プレゼンテーション (PowerPoint) 生成 (Step D)")

    # Step D 固有のセッションステート
    if 'step_d_template_file' not in st.session_state:
        st.session_state.step_d_template_file = None
    if 'step_d_report_data' not in st.session_state:
        st.session_state.step_d_report_data = []
    if 'step_d_generated_pptx' not in st.session_state:
        st.session_state.step_d_generated_pptx = None
    if 'step_d_layout_map' not in st.session_state:
        st.session_state.step_d_layout_map = {} # (★) 選択されたレイアウト名を保持
    if 'tips_list' not in st.session_state:
        st.session_state.tips_list = []
    if 'current_tip_index' not in st.session_state:
        st.session_state.current_tip_index = 0
    if 'last_tip_time' not in st.session_state:
        st.session_state.last_tip_time = time.time()

    # --- 1. テンプレートのアップロード (★) ---
    st.header("Step 1: テンプレート PowerPoint のアップロード")
    st.info("（オプション）使用したい .pptx テンプレートがあればアップロードしてください。なければデフォルトデザインで生成されます。")
    template_file = st.file_uploader(
        "PowerPoint テンプレート (.pptx)",
        type=['pptx'],
        key="step_d_template_uploader"
    )
    
    template_layout_names = []
    default_layouts = {
        "title": "タイトル スライド",
        "agenda": "セクション見出し",
        "content_text": "タイトルとコンテンツ",
        "content_image": "2つのコンテンツ"
    }
    
    if template_file:
        try:
            # (★) アップロード時にレイアウト名を読み込む
            template_file.seek(0)
            template_bytes = template_file.getvalue()
            prs = Presentation(BytesIO(template_bytes))
            template_layout_names = [layout.name for layout in prs.slide_layouts]
            
            # (★) テンプレートが変更されたら、レイアウトマップをリセットするためにNoneをセット
            if (st.session_state.step_d_template_file is None or 
                st.session_state.step_d_template_file.getvalue() != template_bytes):
                
                st.success(f"テンプレート「{template_file.name}」を読み込みました。")
                st.session_state.step_d_template_file = BytesIO(template_bytes)
                st.session_state.step_d_layout_map = {} # (★) マップをリセット
            
        except Exception as e:
            st.error(f"テンプレートの読み込みに失敗: {e}")
            template_layout_names = []
            st.session_state.step_d_template_file = None

    else:
        # (★) ファイルがクリアされたらリセット
        if st.session_state.step_d_template_file is not None:
             st.session_state.step_d_template_file = None
             st.session_state.step_d_layout_map = {}


    # --- 2. Step C 分析結果のアップロード ---
    st.header("Step 2: Step C 分析レポート (JSON) のアップロード")
    st.info("Step C でエクスポートした `report_for_powerpoint.json` をアップロードしてください。")
    report_file = st.file_uploader(
        "分析レポートファイル (report_for_powerpoint.json)",
        type=['json'],
        key="step_d_report_uploader"
    )

    if report_file:
        try:
            # (★) 辞書のリストであることを確認
            report_json_string = report_file.getvalue().decode('utf-8')
            report_data = json.loads(report_json_string)
            
            if isinstance(report_data, list) and all(isinstance(item, dict) for item in report_data):
                # (★) データの参照を更新する
                if st.session_state.step_d_report_data != report_data:
                    st.success(f"分析レポート「{report_file.name}」を読み込みました ({len(report_data)}スライド)。")
                    st.session_state.step_d_report_data = report_data
            else:
                st.error("アップロードされたJSONが期待する形式（スライドのリスト）ではありません。")
                st.session_state.step_d_report_data = []
        except Exception as e:
            logger.error(f"Step D JSONレポート読込エラー: {e}", exc_info=True)
            st.error(f"分析レポートの読み込み中にエラー: {e}")
            st.session_state.step_d_report_data = []
    
    if not st.session_state.step_d_report_data:
        st.session_state.step_d_report_data = []
        st.session_state.step_d_generated_pptx = None
        st.warning("PowerPointを生成するには、Step C で生成した JSON レポートをアップロードしてください。")
        return

    # --- 3. (★) テンプレートのレイアウト割り当て ---
    st.header("Step 3: テンプレートレイアウトの割り当て")
    
    if not st.session_state.step_d_template_file:
        st.info("Step 1 でテンプレートをアップロードすると、レイアウト名を選択できます。（現在デフォルト設定）")
        layout_options = list(default_layouts.values()) # デフォルト名を表示
    else:
        st.info("テンプレートから読み込んだレイアウト名を、各スライドタイプに割り当ててください。")
        layout_options = template_layout_names
        
    if not layout_options:
         st.error("レイアウトの読み込みに失敗しました。デフォルト設定を使用します。")
         layout_options = list(default_layouts.values())

    # (★) デフォルトのインデックスを探すヘルパー
    def get_default_index(default_name_key):
        # (★) 1. セッションステートに保存された値があればそれを優先
        if default_name_key in st.session_state.step_d_layout_map:
            saved_name = st.session_state.step_d_layout_map[default_name_key]
            if saved_name in layout_options:
                return layout_options.index(saved_name)
        
        # (★) 2. なければ、デフォルト名（"タイトル スライド" など）を探す
        target_name = default_layouts[default_name_key]
        if target_name in layout_options:
            return layout_options.index(target_name)
            
        # (★) 3. それもなければ、部分一致（"タイトル"など）で探す
        for i, opt in enumerate(layout_options):
            if default_name_key in opt.lower(): # "title"
                return i
            if target_name.split(' ')[0] in opt: # "タイトル"
                return i

        return 0 # 見つからなければ先頭

    layout_map = {}
    col1, col2 = st.columns(2)
    with col1:
        layout_map["title"] = st.selectbox(
            "1. 表紙 (Title) スライド:", layout_options, 
            index=get_default_index("title"), key="layout_select_title"
        )
        layout_map["agenda"] = st.selectbox(
            "2. 目次 (Agenda) スライド:", layout_options, 
            index=get_default_index("agenda"), key="layout_select_agenda"
        )
    with col2:
        layout_map["content_text"] = st.selectbox(
            "3. 分析 (テキストのみ) スライド:", layout_options, 
            index=get_default_index("content_text"), key="layout_select_text"
        )
        layout_map["content_image"] = st.selectbox(
            "4. 分析 (テキスト+画像) スライド:", layout_options, 
            index=get_default_index("content_image"), key="layout_select_image"
        )
    
    # (★) 選択結果をセッションステートに即時保存
    st.session_state.step_d_layout_map = layout_map


    # --- 4. スライド構成の編集 (★ 旧Step 3) ---
    st.header("Step 4: スライド構成の確認・編集")
    st.info("（(★) マウスのドラッグ＆ドロップでスライドの順番を入れ替えることができます）")

    try:
        headers_list = []
        header_to_item_map = {}
        
        if not st.session_state.step_d_report_data:
             st.warning("JSONデータが空か、正しく読み込まれていません。")
             return

        for i, item in enumerate(st.session_state.step_d_report_data):
            if not isinstance(item, dict):
                st.error(f"データ形式エラー: {item} は辞書ではありません。")
                continue
            
            title = item.get('slide_title', '（タイトルなし）')
            layout = item.get('slide_layout', 'N/A')
            has_image = "🖼️" if (item.get("image_base64") or item.get("image_base664")) else "📄" # (★) Typo修正
            header_str = f"**{i+1}: {title}** (Layout: `{layout}`, {has_image})"
            headers_list.append(header_str)
            header_to_item_map[header_str] = item

        if not all(isinstance(h, str) for h in headers_list):
            st.error("内部エラー: ヘッダーリストの作成に失敗しました。")
            return

        sorted_headers = sort_items(
            items=headers_list,
            key="sortable_slides_v4" # (★) キーを更新
        )
        
        cleaned_sorted_data = []
        for header in sorted_headers:
            if header in header_to_item_map:
                cleaned_sorted_data.append(header_to_item_map[header])
            else:
                logger.error(f"マッピングエラー: ソート後のヘッダー '{header}' が見つかりません。")
            
        st.session_state.step_d_report_data = cleaned_sorted_data
        
    except Exception as e:
        logger.error(f"streamlit-sortables 実行エラー: {e}", exc_info=True)
        st.error(f"スライド編集UIの描画に失敗: {e}。")


    # --- 5. AIによる修正指示 (★ 旧Step 4) ---
    st.header("Step 5: (オプション) AIによる内容の修正指示")
    st.markdown(f"（(★) 使用モデル: `{MODEL_PRO}`）")
    
    with st.expander("AIにスライド内容の修正を指示する"):
        correction_prompt = st.text_area(
            "修正内容を具体的に指示してください:",
            placeholder=(
                "例: 「エグゼクティブ・サマリー」スライドの箇条書きを3点に要約して。\n"
                "例: 「共起ネットワーク」スライドを削除して。"
            ),
            key="step_d_correction_prompt"
        )
        
        if st.button("AIでスライド構成を修正", key="run_ai_correction_D", type="secondary"):
            if correction_prompt.strip():
                with st.spinner(f"AI ({MODEL_PRO}) がスライド構成 (JSON) を修正中..."):
                    current_json_str = json.dumps(st.session_state.step_d_report_data, ensure_ascii=False)
                    corrected_json_str = run_step_d_ai_correction(current_json_str, correction_prompt)
                    
                    try:
                        corrected_data = json.loads(corrected_json_str)
                        if isinstance(corrected_data, list):
                            st.session_state.step_d_report_data = corrected_data
                            st.success("AIによるスライド構成の修正が完了しました。Step 4 の構成が更新されています。")
                            st.rerun() # UIを即時更新
                        else:
                            st.error("AIがリスト形式でないデータを返しました。修正はキャンセルされました。")
                    except Exception as e:
                        st.error(f"AIの回答のパースに失敗: {e}。修正はキャンセルされました。")
            else:
                st.warning("修正指示を入力してください。")

    # --- 6. PowerPoint生成 (★ 旧Step 5) ---
    st.header("Step 6: PowerPointの生成とエクスポート")
    
    tip_placeholder_d = st.empty()
    
    if st.button("PowerPointを生成 (Step 6)", key="generate_pptx_D", type="primary", use_container_width=True):
        st.session_state.step_d_generated_pptx = None
        
        if not st.session_state.tips_list or len(st.session_state.tips_list) <= 1:
            with st.spinner("分析TIPSをAIで生成中..."):
                st.session_state.tips_list = get_analysis_tips_list_from_ai()
                if st.session_state.tips_list:
                    st.session_state.current_tip_index = random.randint(0, len(st.session_state.tips_list) - 1)
                    st.session_state.last_tip_time = time.time()
        
        with st.spinner("PowerPointファイルを生成中..."):
            now = time.time()
            if (now - st.session_state.last_tip_time > 10):
                if len(st.session_state.tips_list) > 1:
                    st.session_state.current_tip_index = (st.session_state.current_tip_index + 1) % len(st.session_state.tips_list)
                st.session_state.last_tip_time = now
            if st.session_state.tips_list:
                current_tip = st.session_state.tips_list[st.session_state.current_tip_index]
                tip_placeholder_d.info(f"💡 データ分析TIPS: {current_tip}")

            # (★) --- 修正: 選択されたレイアウトマップを渡す ---
            generated_file_stream = create_powerpoint_presentation(
                st.session_state.step_d_template_file,
                st.session_state.step_d_report_data,
                st.session_state.step_d_layout_map # (★) ここで渡す
            )
            
            tip_placeholder_d.empty()
            
            if generated_file_stream:
                st.session_state.step_d_generated_pptx = generated_file_stream.getvalue()
                st.success("PowerPointファイルの生成が完了しました。")
            else:
                st.error("PowerPointファイルの生成に失敗しました。")

    if st.session_state.step_d_generated_pptx:
        st.download_button(
            label="生成された PowerPoint をダウンロード",
            data=st.session_state.step_d_generated_pptx,
            file_name="AI_Analysis_Report_v3.pptx", # (★) v3
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True
        )
        st.balloons()


# --- 11. (★) Main関数 (アプリケーション実行) ---
def main():
    """Streamlitアプリケーションのメイン実行関数"""
    
    # (★) --- st.set_page_config() を最初に実行 ---
    st.set_page_config(page_title="AI Data Analysis App", layout="wide")
    
    # (★) --- セッションステートの初期化を *全て* ここに移動 ---
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'A' # 初期ステップ
    if 'tips_list' not in st.session_state:
        st.session_state.tips_list = []
    if 'current_tip_index' not in st.session_state:
        st.session_state.current_tip_index = 0
    if 'last_tip_time' not in st.session_state:
        st.session_state.last_tip_time = time.time()
    # (★) --- ここまでが修正点 ---

    # (★) --- .envファイルから環境変数を読み込む ---
    try:
        load_dotenv()
        logger.info(".env ファイルの読み込み試行完了。")
    except Exception as e:
        logger.warning(f".env ファイルの読み込みに失敗: {e}") 
    
    # --- サイドバー (ナビゲーション) ---
    with st.sidebar:
        st.title("AI レポーティング App")
        st.markdown("---")
        
        st.header("⚙️ AI 設定")
        
        if not os.getenv("GOOGLE_API_KEY"):
            st.warning(
                "Google APIキーが.envファイルに設定されていないか、読み込めませんでした。\n\n"
                "(.envファイルに `GOOGLE_API_KEY='あなたのAPIキー'` と記載してください)"
            )
        else:
            st.success("Google APIキー 読込完了")
            # (★) アプリ起動時にLLMとspaCyのロードを試みる
            if 'llm_checked' not in st.session_state:
                if get_llm(MODEL_FLASH_LITE) is None: 
                    st.error("LLMの初期化に失敗。APIキーが正しいか確認してください。")
                if load_spacy_model() is None:
                    st.error("spaCyモデルのロードに失敗。")
                st.session_state.llm_checked = True # (★) 毎リラン時にチェックしないよう
        
        st.markdown("---")
        
        # (★) --- Step A〜D のナビゲーションボタン ---
        st.header("🔄 ナビゲーション")
        current_step = st.session_state.current_step
        
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

        if st.button(
            "Step C: AIレポート生成", key="nav_C", use_container_width=True,
            type=("primary" if current_step == 'C' else "secondary")
        ):
            if st.session_state.current_step != 'C':
                st.session_state.current_step = 'C'; st.rerun()

        if st.button(
            "Step D: PowerPoint生成", key="nav_D", use_container_width=True,
            type=("primary" if current_step == 'D' else "secondary")
        ):
            if st.session_state.current_step != 'D':
                st.session_state.current_step = 'D'; st.rerun()
                
        # (★) --- 修正: グローバルTipsの生成トリガーを削除 ---
        # st.markdown("---")
        # if st.button("分析TIPSを更新", key="reload_tips", use_container_width=True):
        #      ... (ブロック全体を削除) ...
        # (★) --- ここまでが修正点 ---

    # (★) 既存の main() 関数のロジック (セッションステート初期化) を移動
    if 'llm_checked' not in st.session_state:
        if os.getenv("GOOGLE_API_KEY"):
            if get_llm(MODEL_FLASH_LITE) is None: 
                pass # (★) エラーはサイドバーで表示
            if load_spacy_model() is None:
                pass # (★) エラーはサイドバーで表示
        st.session_state.llm_checked = True


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