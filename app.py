import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ── 頁面設定 ──────────────────────────────────────────────
st.set_page_config(
    page_title="台股 BB × RSI 掃描器",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px;}
.hero {
    padding: 1.35rem 1.5rem;
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 18px;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, rgba(25,118,210,.09), rgba(76,175,80,.06));
}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
.hero p {margin: 0; opacity: .76; font-size: .95rem;}
.notice {
    padding: .8rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(76,175,80,.28);
    background: rgba(76,175,80,.06);
    margin-bottom: 1rem;
}
</style>
<div class="hero">
  <h1>📈 台股 BB × RSI 掃描器</h1>
  <p>BBand + RSI 雙指標策略｜網頁即時掃描｜Yahoo Finance 資料</p>
</div>
<div class="notice">
  🔔 <b>LINE 通知採自動排程</b>：週一～週五 15:05 由 GitHub Actions 執行。網頁按「開始掃股」只顯示結果，不會傳送 LINE。
</div>
""", unsafe_allow_html=True)

# ── 股票名稱對照表 ────────────────────────────────────────
STOCK_NAMES = {
    "2301": "光寶科", "2303": "聯電", "2308": "台達電", "2312": "金寶", "2313": "華通",
    "2317": "鴻海", "2323": "中環", "2324": "仁寶", "2327": "國巨", "2330": "台積電",
    "2337": "旺宏", "2344": "華邦電", "2345": "智邦", "2347": "聯強", "2352": "佳世達",
    "2353": "宏碁", "2354": "鴻準", "2356": "英業達", "2357": "華碩", "2360": "致茂",
    "2368": "金像電", "2376": "技嘉", "2377": "微星", "2379": "瑞昱", "2382": "廣達",
    "2383": "台光電", "2385": "群光", "2393": "億光", "2395": "研華", "2404": "漢唐",
    "2408": "南亞科", "2409": "友達", "2412": "中華電", "2439": "美律", "2449": "京元電子",
    "2454": "聯發科", "2455": "全新", "2458": "義隆", "2474": "可成", "2492": "華新科",
    "2498": "宏達電", "3008": "大立光", "3017": "奇鋐", "3034": "聯詠", "3035": "智原",
    "3037": "欣興", "3044": "健鼎", "3045": "台灣大", "3231": "緯創", "3443": "創意",
    "3481": "群創", "3532": "台勝科", "3533": "嘉澤", "3653": "健策", "3661": "世芯-KY",
    "3711": "日月光投控", "4904": "遠傳", "4938": "和碩", "4958": "臻鼎-KY", "5269": "祥碩",
    "5434": "崇越", "6116": "彩晶", "6205": "詮欣", "6239": "力成", "6271": "同欣電",
    "6415": "矽力*-KY", "6456": "GIS-KY", "6669": "緯穎", "6770": "力積電", "6806": "昇佳電子",
    "8046": "南電",
    "1101": "台泥", "1102": "亞泥", "1301": "台塑", "1303": "南亞", "1503": "士電",
    "1513": "中興電", "1514": "亞力", "1519": "華城", "2002": "中鋼", "2207": "和泰車",
    "6505": "台塑化",
    "2801": "彰銀", "2812": "台中銀", "2834": "臺企銀", "2880": "華南金", "2881": "富邦金",
    "2882": "國泰金", "2883": "凱基金", "2884": "玉山金", "2885": "元大金", "2886": "兆豐金",
    "2887": "台新金", "2889": "國票金", "2890": "永豐金", "2891": "中信金", "2892": "第一金",
    "2897": "王道銀", "5871": "中租-KY", "5876": "上海商銀", "5880": "合庫金"
}

# ── 指標計算 ──────────────────────────────────────────────
def calc_bollinger(closes, period, multiplier):
    upper, middle, lower, pct_b = [], [], [], []
    for i in range(len(closes)):
        if i < period - 1:
            upper.append(np.nan); middle.append(np.nan)
            lower.append(np.nan); pct_b.append(np.nan)
            continue
        sl = closes[i - period + 1 : i + 1]
        mean, std = sl.mean(), sl.std(ddof=0)
        u, l = mean + multiplier * std, mean - multiplier * std
        upper.append(u); middle.append(mean); lower.append(l)
        pct_b.append(0.5 if std == 0 else (closes[i] - l) / (u - l))
    return np.array(upper), np.array(middle), np.array(lower), np.array(pct_b)

def calc_rsi(closes, period):
    rsi = np.full(len(closes), np.nan)
    if len(closes) <= period:
        return rsi
    changes = np.diff(closes)
    gains   = np.where(changes > 0, changes,  0.0)
    losses  = np.where(changes < 0, -changes, 0.0)
    avg_g, avg_l = gains[:period].mean(), losses[:period].mean()
    rsi[period] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    for i in range(period + 1, len(closes)):
        avg_g = (avg_g * (period - 1) + gains[i - 1])  / period
        avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        rsi[i] = 100.0 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi

# ── 抓資料 ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_stock_data(code):
    try:
        ticker = yf.Ticker(code + ".TW")
        df = ticker.history(period="6mo", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 30:
            return None, STOCK_NAMES.get(code, code)
        closes = df["Close"].dropna().to_numpy()
        if code in STOCK_NAMES:
            name = STOCK_NAMES[code]
        else:
            try:
                info = ticker.info
                name = info.get("longName") or info.get("shortName") or code
            except Exception:
                name = code
        return closes, name
    except Exception:
        return None, STOCK_NAMES.get(code, code)

# ── 單股分析 ──────────────────────────────────────────────
def analyze(code, closes, name, params):
    bb_period = params["bb_period"]
    bb_std    = params["bb_std"]
    pct_b_thr = params["pct_b"]
    grace     = params["grace"]
    rsi_s_per = params["rsi_short"]
    rsi_l_per = params["rsi_long"]

    if len(closes) < max(bb_period, rsi_l_per) + grace + 5:
        return None

    upper, middle, lower, pct_b = calc_bollinger(closes, bb_period, bb_std)
    rsi_s = calc_rsi(closes, rsi_s_per)
    rsi_l = calc_rsi(closes, rsi_l_per)
    n = len(closes)

    vals = [pct_b[-1], rsi_s[-1], rsi_l[-1]]
    if any(np.isnan(vals)):
        return None

    pct_b_ok = any(
        not np.isnan(pct_b[i]) and pct_b[i] < pct_b_thr
        for i in range(n - grace, n)
    )

    golden = False
    for i in range(max(1, n - 3), n):
        if not any(np.isnan([rsi_s[i], rsi_l[i], rsi_s[i-1], rsi_l[i-1]])):
            if rsi_s[i-1] <= rsi_l[i-1] and rsi_s[i] > rsi_l[i]:
                golden = True; break

    death = False
    if n >= 2 and not any(np.isnan([rsi_s[-1], rsi_l[-1], rsi_s[-2], rsi_l[-2]])):
        death = rsi_s[-2] >= rsi_l[-2] and rsi_s[-1] < rsi_l[-1]

    overbought   = closes[-1] >= upper[-1] or rsi_s[-1] > 70
    sell_signal  = overbought and death
    buy_signal   = pct_b_ok and golden and not sell_signal
    watch_signal = (
        pct_b_ok and not golden and not sell_signal
        and rsi_s[-1] < rsi_l[-1] and (rsi_l[-1] - rsi_s[-1]) < 5
    )

    if not buy_signal and not watch_signal:
        return None

    risk   = closes[-1] - lower[-1]
    reward = upper[-1]  - closes[-1]
    rrr    = round(reward / risk, 2) if risk > 0 else None

    return {
        "code":   code,   "name":   name,
        "signal": "BUY"   if buy_signal else "WATCH",
        "price":  round(closes[-1],  2), "pct_b":  round(pct_b[-1],  3),
        "rsi_s":  round(rsi_s[-1],   1), "rsi_l":  round(rsi_l[-1],  1),
        "upper":  round(upper[-1],   2), "middle": round(middle[-1], 2),
        "lower":  round(lower[-1],   2), "stop":   round(lower[-1],  2),
        "target": round(upper[-1],   2), "rrr":    rrr,
    }

# ── 側邊欄：策略參數 ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 策略參數")
    bb_period = st.number_input("BB 週期（天）",  min_value=5,   max_value=50,  value=20)
    bb_std    = st.number_input("BB 標準差倍數",  min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    pct_b_thr = st.number_input("%B 超賣門檻",    min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    grace     = st.number_input("寬容期（天）",    min_value=1,   max_value=14,  value=7)
    rsi_short = st.number_input("RSI 短天期",      min_value=3,   max_value=14,  value=6)
    rsi_long  = st.number_input("RSI 長天期",      min_value=7,   max_value=30,  value=12)

    st.divider()

    st.header("🔔 LINE 自動通知")
    st.success("週一～週五 15:05 自動掃描")
    st.caption("LINE 由 GitHub Actions + auto_scan.py 獨立執行。網頁操作不會觸發任何 LINE 訊息。")

    st.divider()
    st.caption("策略邏輯")
    st.info("🟢 **買進**：%B < 門檻 + RSI黃金交叉\n\n🔴 **賣出**：價破上軌或RSI>70 + 死亡交叉")

params = {
    "bb_period": bb_period, "bb_std": bb_std,
    "pct_b": pct_b_thr,    "grace": grace,
    "rsi_short": rsi_short, "rsi_long": rsi_long,
}

# ── 主頁面：股票輸入 ──────────────────────────────────────
st.subheader("輸入股票清單")
raw = st.text_area(
    "台股代號（逗號或換行分隔）",
    value="""2317, 2330, 2454, 2412, 2382, 2308, 3711, 2881, 2882, 2884, 6505, 1301, 1303, 2002, 2886
2301, 2303, 2312, 2313, 2323, 2324, 2327, 2337, 2344, 2345, 2347, 2352, 2353, 2354, 2356,
2357, 2360, 2368, 2376, 2377, 2379, 2383, 2385, 2393, 2395, 2404, 2408, 2409, 2439, 2449,
2455, 2458, 2474, 2492, 2498, 3008, 3017, 3034, 3035, 3037, 3044, 3045, 3231, 3443, 3481,
3532, 3533, 3653, 3661, 4938, 4958, 5269, 5434, 6116, 6205, 6239, 6271, 6415, 6456, 6669,
6770, 8046, 1503, 1513, 1514, 1519, 4904, 6806, 2801, 2812, 2834, 2880, 2883, 2885, 2887,
2889, 2890, 2891, 2892, 2897, 5871, 5876, 5880, 1101, 1102""",
    height=80,
)
codes = [c.strip() for c in raw.replace("\n", ",").split(",") if c.strip()]
st.caption(f"共 {len(codes)} 檔待掃描")

# ── 掃描按鈕 ──────────────────────────────────────────────
# 已經徹底拔除會導致網頁崩潰的進度條 (st.progress)，改用最穩定的 st.spinner 轉圈圈
if st.button("🔍 開始掃股", type="primary", use_container_width=True):
    results, errors = [], []
    
    with st.spinner("🔍 正在全力掃描所有股票，這可能需要幾十秒，請稍候..."):
        for code in codes:
            closes, name = fetch_stock_data(code)
            if closes is None:
                errors.append(code)
            else:
                r = analyze(code, closes, name, params)
                if r:
                    results.append(r)

    results.sort(key=lambda x: (0 if x["signal"] == "BUY" else 1, -(x["rrr"] or 0)))

    st.divider()
    col_l, col_r = st.columns(2)
    col_l.metric("✅ 符合條件", f"{len(results)} 檔")
    col_r.metric("❌ 抓取失敗", f"{len(errors)} 檔")

    if errors:
        st.warning(f"以下代號抓取失敗：{', '.join(errors)}")

    if not results:
        st.info("目前沒有股票符合條件，可嘗試放寬參數或等待更好時機。")
    else:
        df = pd.DataFrame(results)
        df_display = df[["code","name","signal","price","pct_b",
                          "rsi_s","rsi_l","stop","target","rrr"]].copy()
        df_display.columns = ["代號","名稱","訊號","現價","%B",
                               f"RSI{rsi_short}",f"RSI{rsi_long}","停損","目標","風報比"]

        def color_signal(val):
            if val == "BUY":   return "background-color:#1a4a2e; color:#00e5a0; font-weight:bold"
            if val == "WATCH": return "background-color:#3a3000; color:#ffd166; font-weight:bold"
            return ""

        st.dataframe(
            df_display.style.map(color_signal, subset=["訊號"]),
            use_container_width=True, hide_index=True,
        )

        st.subheader("個股詳細資訊")
        for r in results:
            is_buy = r["signal"] == "BUY"
            badge  = "✅ 買進訊號" if is_buy else "👀 觀察中"
            with st.expander(f"{r['code']} {r['name']} {badge}", expanded=is_buy):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價",             f"{r['price']}")
                c2.metric("%B",              f"{r['pct_b']}", delta="超賣" if r["pct_b"] < pct_b_thr else None)
                c3.metric(f"RSI{rsi_short}", f"{r['rsi_s']}")
                c4.metric(f"RSI{rsi_long}",  f"{r['rsi_l']}")
                st.divider()
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("📌 進場價", f"{r['price']}")
                t2.metric("🛑 停損價", f"{r['stop']}")
                t3.metric("🎯 目標價", f"{r['target']}")
                t4.metric("⚖️ 風報比", f"1 : {r['rrr']}" if r["rrr"] else "N/A")
                st.caption(f"BB 軌道：下軌 {r['lower']} ／ 中軌 {r['middle']} ／ 上軌 {r['upper']}")

# ── 免責聲明 ──────────────────────────────────────────────
st.divider()
st.caption("⚠️ 本工具僅供技術分析參考，不構成投資建議。LINE 僅由排程腳本自動發送，網頁操作不會觸發通知。")
