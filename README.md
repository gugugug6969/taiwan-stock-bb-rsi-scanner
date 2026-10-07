# 台股策略實驗室｜BB × RSI

供老師展示與學生操作的 Streamlit 台股掃描網站。完整重做版面，使用價格位置與 RSI 動能交叉比較技術面訊號。

## 網站

- 左側股票清單與策略設定；右側訊號摘要、結果表與個股分析。
- BB 通道、調整後收盤價、短長期 RSI 圖表與風報比參考。
- 每次掃描結果保留在使用者自己的 session，切换檢視股票不需重新掃描。
- 網頁不含 LINE 推播或 LINE 憑證讀取。

## 策略與資料

`app.py` 保留來源新版分支的 BB、Wilder RSI、資料讀取與訊號判斷原始函式，並包含完整樣式；部署只需此入口與 requirements.txt。

預設 BB(20, 2σ)、%B < 0.2、回看 7 個交易日、RSI 6/12。BUY 需最近 3 個交易日出現黃金交叉；WATCH 是尚無近期金叉且短 RSI 低於長 RSI、差距小於 5。兩者排除同時超買與最新死亡交叉的情況。

Yahoo Finance 的 `.TW` 上市股票日線，近 6 個月、價格經還原調整；資料快取 1 小時。不是即時行情、不支援以 `.TWO` 查詢上櫃股票。下軌停損、上軌目標只是參考；風報比不是保證報酬。

## LINE 通知

LINE 仍由原專案 `gugugug6969/-` 的排程負責。此 repo 的 `auto_scan.py` 已改為停用入口：既有排程即使執行，也會直接成功結束，不掃描、不讀取憑證、不發送訊息。網頁不會匯入此入口；此 repo 不需要 LINE Secrets。既有 workflow 檔暫時保留，可在 GitHub Actions 的 Daily Stock Scan 選單中 Disable workflow 以停止執行。

## 部署 Streamlit Community Cloud

- Repository：`gugugug6969/taiwan-stock-bb-rsi-scanner`
- Branch：`main`
- Main file path：`app.py`
- Python：3.12
- 使用 Public app 分享給學生；不需設定 LINE secrets。

本地執行：`pip install -r requirements.txt`，接著 `streamlit run app.py`。

僅供教學與技術分析參考，不構成投資建議。
