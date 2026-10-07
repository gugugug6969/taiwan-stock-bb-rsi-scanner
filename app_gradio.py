import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==================== 股票名稱對照表 ====================
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

# ==================== 指標計算函式 ====================
def calc_bollinger(closes: np.ndarray, period: int, multiplier: float):
    upper, middle, lower, pct_b = [], [], [], []
    for i in range(len(closes)):
        if i < period - 1:
            upper.append(np.nan); middle.append(np.nan)
            lower.append(np.nan); pct_b.append(np.nan)
            continue
        sl = closes[i - period + 1 : i + 1]
        mean = sl.mean()
        std = sl.std(ddof=0)
        u = mean + multiplier * std
        l = mean - multiplier * std
        upper.append(u); middle.append(mean); lower.append(l)
        pct_b.append(0.5 if std == 0 else (closes[i] - l) / (u - l))
    return np.array(upper), np.array(middle), np.array(lower), np.array(pct_b)

def calc_rsi(closes: np.ndarray, period: int):
    rsi = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        changes = np.diff(closes[i - period : i + 1])
        gains = changes[changes > 0].sum()
        losses = -changes[changes < 0].sum()
        avg_g = gains / period
        avg_l = losses / period
        rsi[i] = 100 if avg_l == 0 else 100 - 100 / (1 + avg_g / avg_l)
    return rsi

# ==================== 抓資料 ====================
def fetch_stock_data(code: str):
    ticker = yf.Ticker(code + ".TW")
    df = ticker.history(period="6mo", interval="1d", auto_adjust=True)
    if df.empty or len(df) < 30:
        return None, STOCK_NAMES.get(code, code)
    
    closes = df["Close"].dropna().to_numpy()
    name = STOCK_NAMES.get(code, code)
    return closes, name

# ==================== 單股分析 ====================
def analyze(code, closes, name, params):
    bb_period = params["bb_period"]
    bb_std = params["bb_std"]
    pct_b_thr = params["pct_b"]
    grace = params["grace"]
    rsi_s_per = params["rsi_short"]
    rsi_l_per = params["rsi_long"]

    min_len = max(bb_period, rsi_l_per) + grace + 5
    if len(closes) < min_len:
        return None

    upper, middle, lower, pct_b = calc_bollinger(closes, bb_period, bb_std)
    rsi_s = calc_rsi(closes, rsi_s_per)
    rsi_l = calc_rsi(closes, rsi_l_per)

    n = len(closes)
    last_price = closes[-1]
    last_pct_b = pct_b[-1]
    last_rsi_s = rsi_s[-1]
    last_rsi_l = rsi_l[-1]
    last_upper = upper[-1]
    last_lower = lower[-1]
    last_middle = middle[-1]

    if any(np.isnan([last_pct_b, last_rsi_s, last_rsi_l])):
        return None

    pct_b_ok = any((not np.isnan(pct_b[i]) and pct_b[i] < pct_b_thr) 
                   for i in range(n - grace, n))

    golden = False
    for i in range(max(1, n - 3), n):
        if not any(np.isnan([rsi_s[i], rsi_l[i], rsi_s[i-1], rsi_l[i-1]])):
            if rsi_s[i-1] <= rsi_l[i-1] and rsi_s[i] > rsi_l[i]:
                golden = True
                break

    death = (not any(np.isnan([rsi_s[-1], rsi_l[-1], rsi_s[-2], rsi_l[-2]])) and
             rsi_s[-2] >= rsi_l[-2] and rsi_s[-1] < rsi_l[-1])

    overbought = last_price >= last_upper or last_rsi_s > 70
    sell_signal = overbought and death
    buy_signal = pct_b_ok and golden and not sell_signal
    watch_signal = (pct_b_ok and not golden and not sell_signal and
                    last_rsi_s < last_rsi_l and (last_rsi_l - last_rsi_s) < 5)

    if not buy_signal and not watch_signal:
        return None

    risk = last_price - last_lower
    reward = last_upper - last_price
    rrr = round(reward / risk, 2) if risk > 0 else None

    return {
        "code": code, "name": name, "signal": "BUY" if buy_signal else "WATCH",
        "price": round(last_price, 2), "pct_b": round(last_pct_b, 3),
        "rsi_s": round(last_rsi_s, 1), "rsi_l": round(last_rsi_l, 1),
        "upper": round(last_upper, 2), "middle": round(last_middle, 2),
        "lower": round(last_lower, 2), "stop": round(last_lower, 2),
        "target": round(last_upper, 2), "rrr": rrr,
    }

# ==================== 主掃描函式 ====================
def run_scan(stock_input, bb_period, bb_std, pct_b_thr, grace, rsi_short, rsi_long):
    codes = [c.strip() for c in stock_input.replace("\n", ",").split(",") if c.strip()]
    params = {
        "bb_period": int(bb_period), "bb_std": float(bb_std), "pct_b": float(pct_b_thr),
        "grace": int(grace), "rsi_short": int(rsi_short), "rsi_long": int(rsi_long)
    }
    
    results = []
    errors = []
    
    for i, code in enumerate(codes):
        closes, name = fetch_stock_data(code)
        if closes is None:
            errors.append(code)
        else:
            r = analyze(code, closes, name, params)
            if r:
                results.append(r)
    
    results.sort(key=lambda x: (0 if x["signal"] == "BUY" else 1, -(x["rrr"] or 0)))
    
    if results:
        df = pd.DataFrame(results)
        df_display = df[["code", "name", "signal", "price", "pct_b", "rsi_s", "rsi_l", "stop", "target", "rrr"]].copy()
        df_display.columns = ["代號", "名稱", "訊號", "現價", "%B", f"RSI{rsi_short}", f"RSI{rsi_long}", "停損", "目標", "風報比"]
        summary = f"✅ 找到 {len(results)} 檔符合條件的股票！"
    else:
        df_display = pd.DataFrame()
        summary = "目前沒有股票符合條件"

    error_msg = f"❌ 抓取失敗：{', '.join(errors)}" if errors else ""
    
    return df_display, f"{summary}\n{error_msg}"

# ==================== Gradio 介面 ====================
with gr.Blocks(title="台股選股系統", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 台股選股系統\n**BBand + RSI 雙指標策略**")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 參數設定")
            bb_period = gr.Slider(5, 50, value=20, step=1, label="BB 週期（天）")
            bb_std = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="BB 標準差倍數")
            pct_b_thr = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="%B 超賣門檻")
            grace = gr.Slider(1, 14, value=7, step=1, label="寬容期（天）")
            rsi_short = gr.Slider(3, 14, value=6, step=1, label="RSI 短天期")
            rsi_long = gr.Slider(7, 30, value=12, step=1, label="RSI 長天期")
        
        with gr.Column(scale=2):
            gr.Markdown("### 輸入股票代號")
            stock_input = gr.Textbox(
                value="2317,2330,2454,2412,2382,2308,3711,2881,2882,2884",
                lines=10,
                placeholder="輸入多檔台股代號，用逗號或換行分隔",
                label="股票清單"
            )
            btn = gr.Button("🔍 開始掃股", variant="primary", size="large")
    
    output_table = gr.DataFrame(label="掃描結果")
    output_msg = gr.Markdown(label="執行結果")
    
    btn.click(
        fn=run_scan,
        inputs=[stock_input, bb_period, bb_std, pct_b_thr, grace, rsi_short, rsi_long],
        outputs=[output_table, output_msg]
    )
    
    gr.Markdown("---\n⚠️ 本工具僅供技術分析參考，不構成投資建議。投資有風險，請自行評估。")

# ==================== 啟動 ====================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 讓手機也能連線
        server_port=7860,
        share=False,            # 本地使用（想公開再改 True）
        inbrowser=True          # 自動開瀏覽器
    )
