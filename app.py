from datetime import datetime
from html import escape
import re
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import yfinance as yf
import numpy as np

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



st.set_page_config(page_title="台股策略實驗室｜BB × RSI", page_icon="📊", layout="wide")
st.markdown("""<style>
:root { --ink: #16263d; --muted: #66778e; --blue: #2669e0; }
.stApp { background: #f3f6fb; color: var(--ink); }
.block-container { max-width: 1480px; padding: 2rem 3rem 2.5rem; }
[data-testid="stHeader"] { background: transparent; }
.masthead { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #dbe3ee; padding-bottom: 1.25rem; gap: 1rem; }
.brand { display: flex; align-items: center; gap: .75rem; font-size: .9rem; font-weight: 800; letter-spacing: .1em; }
.brand small { display: block; font-size: .75rem; color: var(--muted); letter-spacing: .16em; margin-top: .15rem; font-weight: 500; }
.brand-mark { display: grid; place-items: center; width: 44px; height: 44px; background: #16263d; color: #7ae1bf; font: 800 1.75rem sans-serif; border-radius: 10px; }
.masthead-tag { color: var(--muted); font-size: .875rem; border: 1px solid #dbe3ee; border-radius: 6px; padding: .35rem .65rem; white-space: nowrap; }
.page-heading { display: flex; justify-content: space-between; align-items: flex-end; margin: 2rem 0 1.5rem; gap: 1rem; }
.eyebrow { color: var(--blue); letter-spacing: .15em; font-size: .75rem; font-weight: 750; margin-bottom: .6rem; }
.page-heading h1 { color: var(--ink); font-size: clamp(1.8rem, 3vw, 2.65rem); letter-spacing: -.045em; line-height: 1.25; padding: 0; font-weight: 800; margin: 0; }
.page-heading h1 span { color: #00a37f; }
.intro { margin: .65rem 0 0; color: var(--muted); font-size: 1rem; }
.data-tag { font-size: .875rem; color: var(--muted); padding-bottom: .25rem; white-space: nowrap; }
.st-key-control_panel { border: 1px solid #dce4ef; border-radius: 14px; background: #fff; padding: 1.4rem; box-shadow: 0 4px 18px rgba(29,53,91,.025); }
.section-label { font-size: .75rem; letter-spacing: .12em; color: #78889b; font-weight: 650; }
.panel-title, .desk-title, .detail-heading h2 { color: var(--ink); font-size: 1.25rem; font-weight: 750; padding: .4rem 0 .7rem; margin: 0; }
.desk-title { font-size: 1.5rem; padding-bottom: .1rem; }
[data-testid="stForm"] { border: 0; padding: 0; }
.input-group { font-size: .9rem; font-weight: 750; border-top: 1px solid #edf1f6; padding: 1.1rem 0 .2rem; margin-top: .35rem; }
.input-group span { display: block; font-size: .75rem; color: #8694a5; font-weight: 500; letter-spacing: .04em; margin-top: .2rem; }
[data-testid="stWidgetLabel"] p { font-size: .875rem; color: #42546b; }
[data-testid="stTextArea"] textarea { font-size: .875rem; line-height: 1.55; }
[data-testid="stFormSubmitButton"] button { border-radius: 8px; min-height: 46px; font-size: 1rem; font-weight: 700; margin-top: .7rem; }
[data-testid="stCaptionContainer"] p { font-size: .8rem; line-height: 1.6; color: var(--muted); }
.stat-card { border: 1px solid #dce4ef; border-radius: 10px; padding: 1rem 1.1rem; background: #fff; min-height: 130px; margin: .25rem 0 .6rem; }
.stat-card p { font-size: .9rem; margin: 0 0 .7rem; font-weight: 600; }
.stat-card p span { display: block; color: #8493a6; font-size: .75rem; letter-spacing: .08em; font-weight: 500; margin-top: .15rem; }
.stat-card strong { font-size: 2.15rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.stat-card small { font-size: .8rem; color: var(--muted); margin-left: .45rem; }
.stat-card.buy { border-top: 3px solid #009d7a; }
.stat-card.buy strong { color: #007d61; }
.stat-card.watch { border-top: 3px solid #d29930; }
.stat-card.watch strong { color: #966000; }
.stat-card.total { border-top: 3px solid #2669e0; }
.empty-state { background: #fff; border: 1px dashed #cdd9e8; border-radius: 14px; text-align: center; padding: 4rem 1.5rem 3rem; margin-top: .5rem; }
.empty-symbol { font-size: 2.25rem; color: #c0ccda; font-weight: 700; letter-spacing: -.06em; margin-bottom: 1.3rem; }
.empty-symbol span { color: #6da1f0; padding: 0 .5rem; font-weight: 300; }
.empty-state h3 { font-size: 1.4rem; font-weight: 750; padding: 0; margin: 0 0 .6rem; color: var(--ink); }
.empty-state p { color: var(--muted); font-size: 1rem; line-height: 1.8; }
.empty-steps { display: flex; justify-content: center; gap: 1.5rem; margin-top: 2.25rem; color: #8190a3; font-size: .8rem; }
.detail-heading { border-top: 1px solid #dbe3ee; margin-top: 1.75rem; padding-top: 1.4rem; }
.detail-heading h2 { margin: .15rem 0 0; }
.st-key-detail_panel { background: #fff; border: 1px solid #dce4ef; border-radius: 12px; padding: 1.25rem; }
.stock-heading { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding-bottom: .6rem; }
.stock-code { color: var(--muted); font-size: .85rem; font-variant-numeric: tabular-nums; }
.stock-heading h3 { display: inline; font-size: 1.3rem; margin-left: .6rem; padding: 0; }
.signal-badge { font-size: .8rem; border-radius: 6px; padding: .4rem .65rem; white-space: nowrap; }
.signal-badge.buy { background: #e7f6f0; color: #007d61; }
.signal-badge.watch { background: #fff3da; color: #966000; }
[data-testid="stMetricLabel"] p { font-size: .8rem; color: var(--muted); }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-variant-numeric: tabular-nums; }
.page-footer { border-top: 1px solid #dbe3ee; display: flex; justify-content: space-between; flex-wrap: wrap; gap: .6rem; margin-top: 2.5rem; padding-top: 1.1rem; font-size: .8rem; color: #77879b; }
.page-footer span:first-child { letter-spacing: .08em; font-weight: 650; }
@media (max-width: 900px) {
  .block-container { padding: 1.4rem 1.25rem 2rem; }
  .page-heading { margin-top: 1.5rem; }
  .st-key-control_panel { padding: 1rem; }
  .stat-card { padding: .85rem; }
}
@media (max-width: 640px) {
  .brand { font-size: .8rem; }
  .brand small { font-size: .7rem; }
  .brand-mark { width: 38px; height: 38px; }
  .masthead-tag { font-size: .75rem; padding: .3rem .4rem; }
  .data-tag { display: none; }
  .page-heading h1 { font-size: 1.95rem; }
  .intro { font-size: .9rem; }
  .empty-state { padding: 2.5rem 1rem; }
  .empty-steps { gap: .7rem; flex-wrap: wrap; }
  .stock-heading { align-items: flex-start; flex-direction: column; }
  .page-footer { line-height: 1.7; }
}

.stApp input, .stApp textarea { color: #16263d; background: #f8fafc; }
.stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"] { background: #f8fafc; border-color: #dce4ef; }
[data-testid="stFormSubmitButton"] button { background: #2669e0; color: #fff; border-color: #2669e0; }
[data-testid="stFormSubmitButton"] button:hover { background: #1d55b8; color: #fff; border-color: #1d55b8; }
.stApp [data-testid="stExpander"] { color: #16263d; border-color: #dce4ef; }
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div { color: #16263d; background: #fff; border-color: #dce4ef; }

</style>""", unsafe_allow_html=True)

DEFAULT_CODES = """2317, 2330, 2454, 2412, 2382, 2308, 3711, 2881, 2882, 2884, 6505, 1301, 1303, 2002, 2886,
2301, 2303, 2312, 2313, 2323, 2324, 2327, 2337, 2344, 2345, 2347, 2352, 2353, 2354, 2356,
2357, 2360, 2368, 2376, 2377, 2379, 2383, 2385, 2393, 2395, 2404, 2408, 2409, 2439, 2449,
2455, 2458, 2474, 2492, 2498, 3008, 3017, 3034, 3035, 3037, 3044, 3045, 3231, 3443, 3481,
3532, 3533, 3653, 3661, 4938, 4958, 5269, 5434, 6116, 6205, 6239, 6271, 6415, 6456, 6669,
6770, 8046, 1503, 1513, 1514, 1519, 4904, 6806, 2801, 2812, 2834, 2880, 2883, 2885, 2887,
2889, 2890, 2891, 2892, 2897, 5871, 5876, 5880, 1101, 1102"""


def parse_codes(raw):
    parts = [p for p in re.split(r"[\s,，、;；]+", raw.strip()) if p]
    return list(dict.fromkeys(parts)), [p for p in parts if not re.fullmatch(r"\d{4,6}", p)]


def price_chart(closes, params):
    upper, middle, lower, _ = calc_bollinger(closes, params["bb_period"], params["bb_std"])
    short = calc_rsi(closes, params["rsi_short"])
    long = calc_rsi(closes, params["rsi_long"])
    x = list(range(1, len(closes) + 1))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.70, 0.30], vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=x, y=upper, name="BB 上軌", line=dict(color="#a1b3c7", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=lower, name="BB 下軌", line=dict(color="#a1b3c7", width=1), fill="tonexty", fillcolor="rgba(38,105,224,.06)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=middle, name="BB 中軌", line=dict(color="#94a3b8", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=closes, name="調整後收盤價", line=dict(color="#2669e0", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=short, name=f"RSI {params['rsi_short']}", line=dict(color="#009d7a", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=long, name=f"RSI {params['rsi_long']}", line=dict(color="#ba7a0d", width=2)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#c6ced8", dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#c6ced8", dash="dot"), row=2, col=1)
    fig.update_layout(height=430, margin=dict(l=8, r=8, t=45, b=12), font=dict(family="sans-serif", color="#54657b", size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", y=1.14, x=0, font=dict(size=12)))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(title_text="交易日序號（最新在右）", row=2, col=1)
    fig.update_yaxes(gridcolor="#edf1f6", zeroline=False, title_text="價格", row=1, col=1)
    fig.update_yaxes(gridcolor="#edf1f6", zeroline=False, range=[0, 100], title_text="RSI", row=2, col=1)
    return fig


st.markdown("""
<div class="masthead">
  <div class="brand"><span class="brand-mark">Q</span><span>QUANT LAB <small>TAIWAN EQUITIES</small></span></div>
  <span class="masthead-tag">BB × RSI · 教學版</span>
</div>
<div class="page-heading"><div><p class="eyebrow">EXPLORE THE SIGNAL</p><h1>台股策略實驗室<span>。</span></h1><p class="intro">從價格位置到動能交叉，把選股條件看清楚。</p></div><span class="data-tag">日線 · 近 6 個月</span></div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2.45], gap="large")
with left:
    with st.container(key="control_panel"):
        st.markdown('<div class="section-label">01 / SCAN SETUP</div><h2 class="panel-title">設定這次掃描</h2>', unsafe_allow_html=True)
        with st.form("scan_form"):
            raw = st.text_area("股票清單", value=DEFAULT_CODES, height=130, help="輸入上市股票代號，以逗號、空格或換行分隔。", key="stock_input")
            st.caption(f"預設 {len(parse_codes(DEFAULT_CODES)[0])} 檔上市股票，可刪減後用於課堂練習。")
            st.markdown('<div class="input-group">價格位置 <span>BOLLINGER BANDS</span></div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            bb_period = a.number_input("BB 週期", min_value=5, max_value=50, value=20, key="bb_period")
            bb_std = b.number_input("標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1, key="bb_std")
            a, b = st.columns(2)
            pct_b_thr = a.number_input("%B 門檻", min_value=0.0, max_value=0.5, value=0.2, step=0.05, key="pct_b")
            grace = b.number_input("回看交易日", min_value=1, max_value=14, value=7, key="grace")
            st.markdown('<div class="input-group">動能交叉 <span>RELATIVE STRENGTH</span></div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            rsi_short = a.number_input("短期 RSI", min_value=3, max_value=14, value=6, key="rsi_short")
            rsi_long = b.number_input("長期 RSI", min_value=7, max_value=30, value=12, key="rsi_long")
            submitted = st.form_submit_button("開始掃描", type="primary", width="stretch")
        st.caption("Yahoo Finance 日線資料快取 1 小時；盤中資料可能尚未收盤。")
    with st.expander("如何判讀訊號？"):
        st.markdown("**BUY｜符合買進條件**\n\n回看期間曾出現 %B 低於門檻，且最近 3 個交易日出現短期 RSI 向上穿越長期 RSI。")
        st.markdown("**WATCH｜接近交叉**\n\n價格位置條件成立，但尚無近期黃金交叉；短期 RSI 低於長期 RSI，差距小於 5。")
        st.markdown("兩種訊號皆排除同時符合超買與最新死亡交叉的情況。下軌與上軌為停損、目標的參考位置；風報比不是報酬保證。")

params = dict(bb_period=bb_period, bb_std=bb_std, pct_b=pct_b_thr, grace=grace, rsi_short=rsi_short, rsi_long=rsi_long)
if submitted:
    codes, invalid = parse_codes(raw)
    if not codes:
        st.error("請先輸入至少一個股票代號。")
    elif invalid:
        st.error("以下不是有效的數字代號：" + "、".join(invalid))
    elif rsi_short >= rsi_long:
        st.error("短期 RSI 必須小於長期 RSI，請調整後再掃描。")
    else:
        results, errors, histories = [], [], {}
        with right, st.spinner(f"正在讀取 {len(codes)} 檔股票的日線資料…"):
            for code in codes:
                closes, name = fetch_stock_data(code)
                if closes is None:
                    errors.append(code)
                    continue
                result = analyze(code, closes, name, params)
                if result:
                    results.append(result)
                    histories[code] = closes
        results.sort(key=lambda x: (0 if x["signal"] == "BUY" else 1, -(x["rrr"] or 0)))
        st.session_state["scan"] = dict(results=results, errors=errors, histories=histories, codes=codes, params=params.copy(), time=datetime.now(ZoneInfo("Asia/Taipei")).strftime("%m/%d %H:%M"))
        st.session_state.pop("detail_code", None)

scan = st.session_state.get("scan")
with right:
    st.markdown('<div class="section-label">02 / SIGNAL DESK</div><h2 class="desk-title">掃描結果</h2>', unsafe_allow_html=True)
    if scan:
        p = scan["params"]
        st.caption(f"完成於 {scan['time']}（台灣時間） · {len(scan['codes'])} 檔 · BB {p['bb_period']} / {p['bb_std']:g}σ · %B < {p['pct_b']:g} · RSI {p['rsi_short']} / {p['rsi_long']}")
    else:
        st.caption("設定左側條件，開始你的第一輪掃描。")
    results = scan["results"] if scan else []
    buys = sum(r["signal"] == "BUY" for r in results)
    watches = sum(r["signal"] == "WATCH" for r in results)
    metrics = st.columns(3)
    for col, label, value, note, tone in zip(metrics, ["買進訊號", "觀察訊號", "成功讀取"], [buys, watches, len(scan["codes"])-len(scan["errors"]) if scan else 0], ["BUY", "WATCH", "STOCKS"], ["buy", "watch", "total"]):
        with col:
            st.markdown(f'<div class="stat-card {tone}"><p>{label}<span>{note}</span></p><strong>{value if scan else "—"}</strong><small>{"檔股票" if scan else "等待掃描"}</small></div>', unsafe_allow_html=True)

    if not scan:
        st.markdown('<div class="empty-state"><div class="empty-symbol">%B <span>×</span> RSI</div><h3>先看位置，再看動能。</h3><p>掃描後，在這裡比較符合條件的股票，<br>再對照布林通道與 RSI 圖表。</p><div class="empty-steps"><span>01 設定股票</span><span>02 調整參數</span><span>03 檢視訊號</span></div></div>', unsafe_allow_html=True)
    else:
        if scan["errors"]:
            st.warning("未能讀取：" + "、".join(scan["errors"]) + "。請確認是上市股票代號，或稍後再試。")
        if not results:
            if len(scan["errors"]) == len(scan["codes"]):
                st.error("這次沒有成功取得任何股票資料，暫時無法判斷訊號。")
            else:
                st.info("這次沒有股票符合條件。可修改參數後再掃描，比較結果的變化。")
        else:
            data = pd.DataFrame(results)[["code", "name", "signal", "price", "pct_b", "rsi_s", "rsi_l", "stop", "target", "rrr"]].copy()
            data.columns = ["代號", "名稱", "訊號", "調整後收盤價", "%B", f"RSI {p['rsi_short']}", f"RSI {p['rsi_long']}", "下軌參考", "上軌參考", "風報比"]
            def signal_style(value):
                return "color:#007d61;background-color:#e7f6f0;font-weight:700" if value == "BUY" else "color:#966000;background-color:#fff3da;font-weight:700"
            st.dataframe(data.style.format({"調整後收盤價": "{:.2f}", "%B": "{:.3f}", f"RSI {p['rsi_short']}": "{:.1f}", f"RSI {p['rsi_long']}": "{:.1f}", "下軌參考": "{:.2f}", "上軌參考": "{:.2f}", "風報比": "{:.2f}"}, na_rep="—").map(signal_style, subset=["訊號"]), width="stretch", hide_index=True, height=min(440, 36*len(results)+42))
            st.markdown('<div class="detail-heading"><span class="section-label">03 / STOCK DETAIL</span><h2>拆解個股訊號</h2></div>', unsafe_allow_html=True)
            by_code = {r["code"]: r for r in results}
            selected = st.selectbox("選擇股票", options=list(by_code), format_func=lambda code: f"{code}  {by_code[code]['name']}  ·  {by_code[code]['signal']}", key="detail_code")
            r = by_code[selected]
            with st.container(key="detail_panel"):
                badge = "符合買進條件" if r["signal"] == "BUY" else "接近交叉，持續觀察"
                st.markdown(f'<div class="stock-heading"><div><span class="stock-code">{escape(r["code"])}</span><h3>{escape(r["name"])}</h3></div><span class="signal-badge {r["signal"].lower()}">{badge}</span></div>', unsafe_allow_html=True)
                st.plotly_chart(price_chart(scan["histories"][selected], p), width="stretch", config={"displayModeBar": False})
                rows = st.columns(4)
                for col, label, value in zip(rows, ["進場參考", "停損參考 · 下軌", "目標參考 · 上軌", "風報比"], [f"{r['price']:.2f}", f"{r['stop']:.2f}", f"{r['target']:.2f}", f"1 : {r['rrr']}" if r["rrr"] is not None else "無法計算"]):
                    col.metric(label, value)
                st.caption(f"最新 %B {r['pct_b']:.3f} · RSI {p['rsi_short']} = {r['rsi_s']:.1f} · RSI {p['rsi_long']} = {r['rsi_l']:.1f} · BB 中軌 {r['middle']:.2f}")
                st.caption("圖表使用調整後收盤價；交易日序號並非日期。圖中的通道與 RSI 依本次掃描參數計算。")

st.markdown('<div class="page-footer"><span>QUANT LAB / BB × RSI</span><span>僅供教學與技術分析參考，不構成投資建議。網頁操作不會傳送 LINE。</span></div>', unsafe_allow_html=True)
