import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os

# ==========================================
# 1. セキュリティ & 認証管理
# ==========================================
st.set_page_config(page_title="RCI×スクイーズ狙撃スキャナー", layout="wide")

# パスワードはStreamlit CloudのSecretsから取得（コード内に直接書かない）
try:
    MY_PASSWORD = st.secrets["auth_password"]
except KeyError:
    st.error("Secretsに 'auth_password' が設定されていません。")
    st.stop()

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("🔒 認証")
    pwd = st.text_input("パスワードを入力してください", type="password")
    if pwd == MY_PASSWORD:
        st.session_state.auth = True
        st.rerun()
    st.stop()

# ==========================================
# 2. テクニカル計算エンジン
# ==========================================
def calculate_rci(series, period):
    """RCI（順位相関指数）の計算"""
    rci = np.zeros(len(series))
    for i in range(period - 1, len(series)):
        d = series[i - period + 1 : i + 1]
        price_rank = d.rank(ascending=False)
        time_rank = np.arange(period, 0, -1)
        diff = price_rank - time_rank
        sum_d2 = (diff**2).sum()
        rci[i] = (1 - (6 * sum_d2) / (period * (period**2 - 1))) * 100
    return pd.Series(rci, index=series.index)

def diagnose_stock(data: pd.DataFrame, sq_threshold: float):
    """スクイーズ、トレンド、RCIの複合診断"""
    if len(data) < 52: return None
    
    close = data["Close"]
    
    # --- トレンドフィルター（25MA基準） ---
    # 最新のローソク足の状態を重視
    ma25 = close.rolling(25).mean()
    curr_ma25 = ma25.iloc[-1]
    prev_ma25 = ma25.iloc[-5] # 5日前と比較
    
    # 上昇トレンド条件：株価が25MAより上で、25MA自体が上向き
    is_up_trend = (close.iloc[-1] > curr_ma25) and (curr_ma25 > prev_ma25)
    if not is_up_trend:
        return None

    # --- ボリンジャーバンド & スクイーズ判定 ---
    ma20 = close.rolling(20).mean()
    std = close.rolling(20).std()
    bandwidth = ((ma20 + std*2) - (ma20 - std*2)) / ma20
    curr_bw = bandwidth.iloc[-1]
    min_bw = bandwidth.tail(60).min() # 直近3ヶ月の最小幅と比較
    is_squeeze = curr_bw <= (min_bw * sq_threshold)
    
    # --- RCI計算 (9日, 26日) ---
    rci9 = calculate_rci(close, 9)
    rci26 = calculate_rci(close, 26)
    
    rci_signal = "静観"
    if rci9.iloc[-1] > rci9.iloc[-2] and rci9.iloc[-2] < -80:
        rci_signal = "🔥RCI底打ち反転"
    elif rci9.iloc[-1] > 80:
        rci_signal = "⚠️過熱（利確検討）" # 利確の重要性

    if is_squeeze or "🔥" in rci_signal:
        return {
            "Squeeze": "💎スクイーズ" if is_squeeze else "-",
            "RCIサイン": rci_signal,
            "RCI9": round(rci9.iloc[-1], 1),
            "RCI26": round(rci26.iloc[-1], 1),
            "25MA乖離": f"{((close.iloc[-1]/curr_ma25)-1)*100:+.1f}%"
        }
    return None

# ==========================================
# 3. データ読み込み & UI
# ==========================================
@st.cache_data(ttl=3600)
def get_master(market):
    """同じリポジトリ内のdata_j.csvから銘柄リストを取得"""
    try:
        df = pd.read_csv("data_j.csv")
        key_map = {"プライム": "プライム（内国株式）", "スタンダード": "スタンダード（内国株式）", "グロース": "グロース（内国株式）"}
        target = df[(df["市場・商品区分"] == key_map[market]) & (df["33業種区分"] != "－")]
        return target
    except FileNotFoundError:
        st.error("data_j.csv が見つかりません。リポジトリ内に配置してください。")
        return pd.DataFrame()

# メイン画面
st.sidebar.title("⚙️ 狙撃設定")
market_sel = st.sidebar.selectbox("対象市場", ["プライム", "スタンダード", "グロース"])
min_tv = st.sidebar.slider("最低売買代金 (億円)", 1, 50, 5)
sq_sens = st.sidebar.slider("スクイーズ感度", 1.0, 1.5, 1.15, step=0.05)

st.title(f"🎯 {market_sel}市場：RCI×スクイーズ狙撃スキャナー")
st.caption("上昇トレンド中の『エネルギー充填』と『反転の初動』を検知します。")

if st.button("📡 スキャン開始", type="primary"):
    master = get_master(market_sel)
    if not master.empty:
        tickers = [f"{str(c).strip().replace('.0', '')}.T" for c in master["コード"]]
        results = []
        
        progress = st.progress(0)
        status_text = st.empty()
        
        # 効率化のためバッチ処理
        batch_size = 40
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            progress.progress(min(i/len(tickers), 1.0))
            status_text.text(f"分析中... {i}/{len(tickers)} 銘柄")
            
            try:
                # 株価取得
                df = yf.download(batch, period="6mo", interval="1d", progress=False, group_by='ticker')
                
                for t in batch:
                    if t not in df.columns.levels[0]: continue
                    data = df[t].dropna()
                    if len(data) < 52: continue
                    
                    latest = data.iloc[-1]
                    val = (latest["Close"] * latest["Volume"]) / 1e8 # 売買代金（億円）
                    
                    if val >= min_tv:
                        diag = diagnose_stock(data, sq_sens)
                        if diag:
                            info = master[master["コード"].astype(str).str.contains(t.replace(".T",""))].iloc[0]
                            diag.update({
                                "コード": t.replace(".T", ""),
                                "銘柄名": info["銘柄名"],
                                "現在値": f"{latest['Close']:,.1f}",
                                "売買代金(億)": round(val, 1)
                            })
                            results.append(diag)
            except: continue

        progress.empty()
        status_text.empty()

        if results:
            st.success(f"{len(results)} 銘柄が条件に合致しました。")
            res_df = pd.DataFrame(results)
            # カラム順序の整理
            cols = ["コード", "銘柄名", "現在値", "売買代金(億)", "Squeeze", "RCIサイン", "RCI9", "RCI26", "25MA乖離"]
            st.dataframe(res_df[cols], use_container_width=True, hide_index=True)
        else:
            st.warning("条件に合致する銘柄はありませんでした。")
