import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from io import BytesIO
import urllib.request

# =========================
# 1. アプリ設定 & 認証
# =========================
st.set_page_config(page_title="RCI×スクイーズ狙撃スキャナー", layout="wide")

# パスワード管理（GitHub公開時はst.secretsを推奨）
MY_PASSWORD = st.secrets["auth_password"]

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔒 認証")
    pwd = st.text_input("パスワードを入力してください", type="password")
    if pwd == MY_PASSWORD:
        st.session_state.auth = True
        st.rerun()
    st.stop()

# =========================
# 2. RCI計算ロジック
# =========================
def calculate_rci(series, period):
    rci = np.zeros(len(series))
    for i in range(period - 1, len(series)):
        d = series[i - period + 1 : i + 1]
        price_rank = d.rank(ascending=False)
        time_rank = np.arange(period, 0, -1)
        diff = price_rank - time_rank
        sum_d2 = (diff**2).sum()
        rci[i] = (1 - (6 * sum_d2) / (period * (period**2 - 1))) * 100
    return pd.Series(rci, index=series.index)

# =========================
# 3. 診断ロジック（拡張版）
# =========================
def diagnose_stock(data: pd.DataFrame, threshold: float, filter_down: bool):
    if len(data) < 52: return None
    
    close = data["Close"]
    
    # --- ボリンジャーバンド & スクイーズ判定 ---
    ma20 = close.rolling(20).mean()
    std = close.rolling(20).std()
    bandwidth = ((ma20 + std*2) - (ma20 - std*2)) / ma20
    curr_bw = bandwidth.iloc[-1]
    min_bw = bandwidth.tail(60).min()
    is_squeeze = curr_bw <= (min_bw * threshold)
    
    # --- トレンドフィルター（25MA） ---
    ma25 = close.rolling(25).mean()
    is_up_trend = (close.iloc[-1] > ma25.iloc[-1]) and (ma25.iloc[-1] > ma25.iloc[-5])
    
    if filter_down and not is_up_trend:
        return None

    # --- RCI計算 (9, 26) ---
    rci9 = calculate_rci(close, 9)
    rci26 = calculate_rci(close, 26)
    
    # RCI反転サイン（短期RCIが底圏-80から上向き、かつ中期が底圏にある）
    rci_signal = ""
    if rci9.iloc[-1] > rci9.iloc[-2] and rci9.iloc[-2] < -80:
        rci_signal = "🔥RCI底打ち反転"
    elif rci9.iloc[-1] > 80:
        rci_signal = "⚠️過熱感あり"
    else:
        rci_signal = "静観"

    if is_squeeze or "🔥" in rci_signal:
        return {
            "Squeeze": "💎あり" if is_squeeze else "-",
            "RCIサイン": rci_signal,
            "RCI9": round(rci9.iloc[-1], 1),
            "RCI26": round(rci26.iloc[-1], 1),
            "トレンド": "上昇中" if is_up_trend else "不明"
        }
    return None

# =========================
# 4. メインUI
# =========================
st.sidebar.title("⚙️ 狙撃設定")
target_market = st.sidebar.selectbox("市場", ["プライム", "スタンダード", "グロース"])
min_tv = st.sidebar.slider("最低売買代金 (億円)", 1, 50, 5)
sq_sens = st.sidebar.slider("スクイーズ感度", 1.0, 1.5, 1.15)

GITHUB_CSV_URL = "https://raw.githubusercontent.com/watarai0202-netizen/squeeeeezze/blob/main/data_j.csv"

@st.cache_data(ttl=3600)
def get_master():
    with urllib.request.urlopen(GITHUB_CSV_URL) as resp:
        df = pd.read_csv(BytesIO(resp.read()))
    # 市場フィルタリング
    key_map = {"プライム": "プライム（内国株式）", "スタンダード": "スタンダード（内国株式）", "グロース": "グロース（内国株式）"}
    target = df[(df["市場・商品区分"] == key_map[target_market]) & (df["33業種区分"] != "－")]
    return target

st.title(f"🎯 {target_market}：RCI×スクイーズ狙撃スキャナー")

if st.button("📡 スキャン開始", type="primary"):
    master = get_master()
    tickers = [f"{str(c).strip().replace('.0', '')}.T" for c in master["コード"]]
    results = []
    
    progress = st.progress(0)
    batch_size = 35
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        progress.progress(min(i/len(tickers), 1.0))
        
        try:
            # 常に最新のローソク足を取得
            df = yf.download(batch, period="6mo", interval="1d", progress=False, group_by='ticker')
            
            for t in batch:
                if t not in df.columns.levels[0]: continue
                data = df[t].dropna()
                if len(data) < 52: continue
                
                # 売買代金チェック
                latest = data.iloc[-1]
                val = (latest["Close"] * latest["Volume"]) / 1e8
                if val < min_tv: continue
                
                diag = diagnose_stock(data, sq_sens, True)
                if diag:
                    info = master[master["コード"].astype(str).str.contains(t.replace(".T",""))].iloc[0]
                    diag.update({
                        "コード": t.replace(".T", ""),
                        "銘柄名": info["銘柄名"],
                        "売買代金(億)": round(val, 1),
                        "現在値": latest["Close"]
                    })
                    results.append(diag)
        except: continue

    progress.empty()
    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df[["コード", "銘柄名", "現在値", "売買代金(億)", "Squeeze", "RCIサイン", "RCI9", "RCI26"]], 
                     use_container_width=True, hide_index=True)
    else:
        st.warning("現在、点火直前の銘柄は見つかりませんでした。")
