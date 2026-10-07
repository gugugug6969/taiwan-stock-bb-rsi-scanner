import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime

LINE_CHANNEL_ID = os.environ.get("LINE_CHANNEL_ID")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
LINE_USER_ID = os.environ.get("LINE_USER_ID")

# 固定排程參數
PARAMS = {
    "bb_period": 20, "bb_std": 2.0, "pct_b": 0.2, 
    "grace": 7, "rsi_short": 6, "rsi_long": 12,
}

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

CODES = [
    "2317", "2330", "2454", "2412", "2382", "2308", "3711", "2881", "2882", "2884", "6505", "1301", "1303", "2002", "2886",
    "2301", "2303", "2312", "2313", "2323", "2324", "2327", "2337", "2344", "2345", "2347", "2352", "2353", "2354", "2356",
    "2357", "2360", "2368", "2376", "2377", "2379", "2383", "2385", "2393", "2395", "2404", "2408", "2409", "2439", "2449",
    "2455", "2458", "2474", "2492", "2498", "3008", "3017", "3034", "3035", "3037", "3044", "3045", "3231", "3443", "3481",
    "3532", "3533", "3653", "3661", "4938", "4958", "5269", "5434", "6116", "6205", "6239", "6271", "6415", "6456", "6669",
    "6770", "8046", "1503", "1513", "1514", "1519", "4904", "6806", "2801", "2812", "2834", "2880", "2883", "2885", "2887",
    "2889", "2890", "2891", "2892", "2897", "5871", "5876", "5880", "1101", "1102"
]

def calc_bollinger(closes, period, multiplier):
    upper, middle, lower, pct_b = [], [], [], []
    for i in range(len(closes)):
        if i < period - 1:
            upper.append(np.nan); middle.append(np.nan); lower.append(np.nan); pct_b.append(np.nan)
            continue
        sl = closes[i - period + 1 : i + 1]
        mean, std = sl.mean(), sl.std(ddof=0)
        u, l = mean + multiplier * std, mean - multiplier * std
        upper.append(u); middle.append(mean); lower.append(l)
        pct_b.append(0.5 if std == 0 else (closes[i] - l) / (u - l))
    return np.array(upper), np.array(middle), np.array(lower), np.array(pct_b)

def calc_rsi(closes, period):
    rsi = np.full(len(closes), np.nan)
    if len(closes) <= period: return rsi
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

def analyze(code, closes, name, params):
    bb_period, bb_std, pct_b_thr, grace, rsi_s_per, rsi_l_per = params["bb_period"], params["bb_std"], params["pct_b"], params["grace"], params["rsi_short"], params["rsi_long"]
    if len(closes) < max(bb_period, rsi_l_per) + grace + 5: return None
    upper, middle, lower, pct_b = calc_bollinger(closes, bb_period, bb_std)
    rsi_s = calc_rsi(closes, rsi_s_per)
    rsi_l = calc_rsi(closes, rsi_l_per)
    n = len(closes)
    if any(np.isnan([pct_b[-1], rsi_s[-1], rsi_l[-1]])): return None
    pct_b_ok = any(not np.isnan(pct_b[i]) and pct_b[i] < pct_b_thr for i in range(n - grace, n))
    golden = False
    for i in range(max(1, n - 3), n):
        if not any(np.isnan([rsi_s[i], rsi_l[i], rsi_s[i-1], rsi_l[i-1]])):
            if rsi_s[i-1] <= rsi_l[i-1] and rsi_s[i] > rsi_l[i]: golden = True; break
    death = False
    if n >= 2 and not any(np.isnan([rsi_s[-1], rsi_l[-1], rsi_s[-2], rsi_l[-2]])):
        death = rsi_s[-2] >= rsi_l[-2] and rsi_s[-1] < rsi_l[-1]
    overbought   = closes[-1] >= upper[-1] or rsi_s[-1] > 70
    sell_signal  = overbought and death
    buy_signal   = pct_b_ok and golden and not sell_signal
    watch_signal = pct_b_ok and not golden and not sell_signal and rsi_s[-1] < rsi_l[-1] and (rsi_l[-1] - rsi_s[-1]) < 5
    if not buy_signal and not watch_signal: return None
    risk, reward = closes[-1] - lower[-1], upper[-1]  - closes[-1]
    rrr = round(reward / risk, 2) if risk > 0 else None
    return {
        "code": code, "name": name, "signal": "BUY" if buy_signal else "WATCH",
        "price": round(closes[-1], 2), "pct_b": round(pct_b[-1], 3),
        "rsi_s": round(rsi_s[-1], 1), "rsi_l": round(rsi_l[-1], 1),
        "stop": round(lower[-1], 2), "target": round(upper[-1], 2), "rrr": rrr,
    }

def get_channel_access_token(channel_id, channel_secret):
    try:
        resp = requests.post("https://api.line.me/v2/oauth/accessToken",
                             headers={"Content-Type": "application/x-www-form-urlencoded"},
                             data={"grant_type": "client_credentials", "client_id": channel_id, "client_secret": channel_secret}, timeout=10)
        return resp.json().get("access_token") if resp.status_code == 200 else None
    except Exception: return None

def build_report_message(results, rsi_short, rsi_long, params):
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    buys = [r for r in results if r["signal"] == "BUY"]
    watch = [r for r in results if r["signal"] == "WATCH"]
    header = f"📈 台股自動掃描報表｜{now}\n參數：BB{params['bb_period']}/{params['bb_std']}σ %B<{params['pct_b']} RSI{rsi_short}/{rsi_long}\n{'━'*30}\n"
    
    def section(title, emoji, rows):
        if not rows: return ""
        lines = [f"{emoji} {title}（{len(rows)} 檔）\n", f"{'代號+名稱':<10} {'現價':>6} {'%B':>5} {'RS短':>5} {'RS長':>5} {'風報比':>6}", "─" * 42]
        for r in rows:
            rrr = f"1:{r['rrr']}" if r["rrr"] else " N/A"
            lines.append(f"{r['code']}{r['name'][:4]:<6} {r['price']:>6} {r['pct_b']:>5.2f} {r['rsi_s']:>5.1f} {r['rsi_l']:>5.1f} {rrr:>6}")
        lines.append("\n  進場 / 停損 / 目標")
        for r in rows: lines.append(f"  {r['code']} {r['name'][:4]}：{r['price']} / {r['stop']} / {r['target']}")
        return "\n".join(lines) + "\n"

    body = section("買進訊號", "✅", buys)
    if buys and watch: body += "━" * 30 + "\n"
    body += section("觀察中", "👀", watch)
    if not buys and not watch: body = "本次掃描無符合條件股票。\n"
    return header + body + f"\n{'━'*30}\n⚠️ 自動排程執行"

def line_push(token, user_id, text):
    try:
        resp = requests.post("https://api.line.me/v2/bot/message/push",
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                      json={"to": user_id, "messages": [{"type": "text", "text": text[:4800]}]}, timeout=10)
        print(f"LINE 回應狀態碼: {resp.status_code}, 內容: {resp.text}")
    except Exception as e: print(f"Push error: {e}")

if __name__ == "__main__":
    if not all([LINE_CHANNEL_ID, LINE_CHANNEL_SECRET, LINE_USER_ID]):
        print("錯誤：缺少 LINE 憑證環境變數。")
        exit(1)
        
    results = []
    for code in CODES:
        try:
            ticker = yf.Ticker(code + ".TW")
            df = ticker.history(period="6mo", interval="1d", auto_adjust=True)
            if df.empty or len(df) < 30: continue
            closes = df["Close"].dropna().to_numpy()
            name = STOCK_NAMES.get(code, code)
            r = analyze(code, closes, name, PARAMS)
            if r: results.append(r)
        except Exception: continue

    results.sort(key=lambda x: (0 if x["signal"] == "BUY" else 1, -(x["rrr"] or 0)))
    
    token = get_channel_access_token(LINE_CHANNEL_ID, LINE_CHANNEL_SECRET)
    if token:
        report = build_report_message(results, PARAMS["rsi_short"], PARAMS["rsi_long"], PARAMS)
        line_push(token, LINE_USER_ID, report)
        print("推播程序執行完畢。")
    else:
        print("錯誤：無法換取 LINE Access Token。請檢查 Channel ID 與 Secret。")
