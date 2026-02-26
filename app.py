import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# ==========================================
# 1. セキュリティ & 認証管理
# ==========================================
st.set_page_config(page_title="RCI×スクイーズ狙撃スキャナー", layout="wide")

# パスワードはStreamlit CloudのSecrets管理を使用
try:
    MY_PASSWORD = st.secrets["auth_password"]
except KeyError:
    st.error("Secretsに 'auth_password' が設定されていません。")
    st.stop()

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔒 認証")
    pwd = st.text_input("合言葉を入力", type="password")
    if pwd == MY_PASSWORD:
        st.session_state.auth = True
        st.rerun()
    st.stop()

# ==========================================
# 2. 計算エンジン（スクイーズ、RCI、RVOL）
# ==========================================
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

def diagnose_strategy(data: pd.DataFrame, sq_threshold: float, min_rvol: float):
    if len(data) < 52: return None
    
    close = data["Close"]
    vol = data["Volume"]
    
    # --- 0. トレンドフィルター（前提条件） ---
    ma25 = close.rolling(25).mean()
    if not (close.iloc[-1] > ma25.iloc[-1] and ma25.iloc[-1] > ma25.iloc[-5]):
        return None

    # --- 1. 【優先度：高】スクイーズ判定（爆発の準備） ---
    ma20 = close.rolling(20).mean()
    std = close.rolling(20).std()
    bw = ((ma20 + std*2) - (ma20 - std*2)) / ma20
    is_squeeze = bw.iloc[-1] <= (bw.tail(60).min() * sq_threshold)
    
    # --- 2. 【優先度：中】RCI反転（点火の合図） ---
    rci9 = calculate_rci(close, 9)
    rci26 = calculate_rci(close, 26)
    is_rci_turn = (rci9.iloc[-1] > rci9.iloc[-2] and rci9.iloc[-2] < -80)
    
    # --- 3. 【優先度：低/確信】RVOL（資金流入の確認） ---
    avg_vol20 = vol.tail(20).mean()
    rvol = vol.iloc[-1] / avg_vol20 if avg_vol20 > 0 else 0
    is_vol_spike = rvol >= min_rvol

    # --- 優先度格付けロジック ---
    priority = "通常"
    score = 0
    if is_squeeze: 
        score += 10
    if is_rci_turn: 
        score += 5
    if is_vol_spike: 
        score += 2
        
    if score >= 15: priority = "🚨 S級：即時狙撃（収束＋点火）"
    elif score >= 10: priority = "💎 A級：準備完了（収束中）"
    elif score >= 5: priority = "🔥 B級：点火初動（反転のみ）"
    else: return None # 優先度が低いものは表示しない

    return {
        "格付け": priority,
        "Squeeze": "💎" if is_squeeze else "-",
        "RCI反転": "🔥" if is_rci_turn else "-",
        "RVOL": f"{rvol:.2f}倍",
        "RCI9": round(rci9.iloc[-1], 1),
        "25MA乖離": f"{((close.iloc[-1]/ma25.iloc[-1])-1)*100:+.1f}%"
    }

# ==========================================
# 3. UI & 実行
# ==========================================
st.sidebar.title("⚙️ 戦略設定")
market_sel = st.sidebar.selectbox("市場", ["プライム", "スタンダード", "グロース"])
min_tv = st.sidebar.slider("最低売買代金 (億円)", 1, 50, 5)
sq_sens = st.sidebar.slider("スクイーズ感度", 1.0, 1.5, 1.15, 0.05)
min_rvol_set = st.sidebar.slider("確信RVOL (倍)", 1.0, 3.0, 1.5, 0.1)

@st.cache_data(ttl=3600)
def load_data(market):
    try:
        df = pd.read_csv("data_j.csv")
        mapping = {"プライム": "プライム（内国株式）", "スタンダード": "スタンダード（内国株式）", "グロース": "グロース（内国株式）"}
        return df[(df["市場・商品区分"] == mapping[market]) & (df["33業種区分"] != "－")]
    except: return pd.DataFrame()

st.title(f"狙撃スキャナー：{market_sel}")
st.info("優先度：スクイーズ（準備）＞ RCI反転（点火）＞ RVOL（確信）")

if st.button("📡 スキャン開始", type="primary"):
    master = load_data(market_sel)
    tickers = [f"{str(c).strip().replace('.0', '')}.T" for c in master["コード"]]
    results = []
    
    bar = st.progress(0)
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i+40]
        bar.progress(min(i/len(tickers), 1.0))
        try:
            df = yf.download(batch, period="6mo", interval="1d", progress=False, group_by='ticker')
            for t in batch:
                if t not in df.columns.levels[0]: continue
                data = df[t].dropna()
                if len(data) < 52: continue
                
                val = (data["Close"].iloc[-1] * data["Volume"].iloc[-1]) / 1e8
                if val >= min_tv:
                    diag = diagnose_strategy(data, sq_sens, min_rvol_set)
                    if diag:
                        info = master[master["コード"].astype(str).str.contains(t.replace(".T",""))].iloc[0]
                        diag.update({"コード": t.replace(".T", ""), "銘柄名": info["銘柄名"], "代金": f"{val:.1f}億"})
                        results.append(diag)
        except: continue
    
    bar.empty()
    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df[["格付け", "コード", "銘柄名", "代金", "Squeeze", "RCI反転", "RVOL", "RCI9", "25MA乖離"]], use_container_width=True, hide_index=True)
    else:
        st.warning("条件に合致する銘柄は見つかりませんでした。")
