# 台股 BB × RSI 掃描器 — Scheduled LINE Edition

這個版本把「網頁掃描」與「LINE 自動通知」完全分離。

## 網頁版
- Streamlit 即時掃描台股
- BB(20, 2) + %B + RSI 6/12
- BUY / WATCH 訊號
- 顯示進場、停損、目標與風報比
- **使用者按網頁的「開始掃股」不會傳送 LINE**

## LINE 自動通知
- 由 `.github/workflows/daily_scan.yml` 執行
- 週一到週五台灣時間 **15:05**
- 執行 `auto_scan.py`
- 只有排程執行時才會推送 LINE
- GitHub Secrets：`LINE_CHANNEL_ID`、`LINE_CHANNEL_SECRET`、`LINE_USER_ID`

## 程式分工
- `app.py`：網頁手動掃描與顯示
- `auto_scan.py`：排程掃描與 LINE 推播
- `.github/workflows/daily_scan.yml`：固定時間啟動
