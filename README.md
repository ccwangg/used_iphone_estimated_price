# AI 二手物價值評估系統

## 專案簡介

本系統透過影像辨識自動帶入品名，並結合爬蟲數據與機器學習算法，提供科學化的定價建議。

## 技術組成

- **網頁框架：** Flask
- **影像辨識：** YOLOv8 (ultralytics)
- **資料獲取：** BeautifulSoup4 (爬蟲) / 模擬資料
- **機器學習：** Scikit-learn
  - K-Means 分群 (Clustering)
  - Decision Tree 決策樹 (回歸預測)
- **資料庫：** SQLite

## 安裝步驟

1. 安裝 Python 依賴套件：
```bash
pip install -r requirements.txt
```

2. 執行應用程式：
```bash
python app.py
```

3. 開啟瀏覽器訪問：`http://localhost:5000`

## 專案結構

```
final_project/
├── app.py                 # Flask 主應用程式
├── predict.py            # YOLOv8 影像辨識模組
├── data_engine.py        # 資料獲取引擎（爬蟲/模擬資料）
├── model.py              # ML 模組（K-Means + Decision Tree）
├── requirements.txt      # Python 依賴套件
├── templates/            # HTML 模板
│   ├── index.html       # 首頁（上傳介面）
│   ├── result.html      # 結果頁面
│   └── history.html     # 歷史記錄頁面
├── uploads/             # 上傳圖片儲存目錄
├── static/              # 靜態檔案（CSS, JS, 圖片）
│   └── results/         # 處理結果圖片
└── app.db              # SQLite 資料庫（自動產生）
```

## 功能說明

### 1. 影像辨識 (YOLOv8)
- 使用者上傳照片，系統自動辨識物品名稱
- 使用預訓練模型 `yolov8n.pt`
- 輸出物品名稱和信心度

### 2. 資料獲取
- 根據辨識出的物品名稱，獲取市場價格資料
- 預設使用模擬資料 (`mock_data.csv`)
- 可切換為實際爬蟲功能（需自行實作）

### 3. 機器學習分析
- **K-Means 分群：** 將價格資料分為「低價損壞區」、「合理二手區」、「高價全新區」
- **Decision Tree：** 根據外觀狀況(1-5分)和保固剩餘月數，預測建議售價

### 4. 結果展示
- YOLO 標註圖片
- 價格分群視覺化圖表
- 市場價格統計資訊
- 建議售價預測

## 使用流程

1. 上傳物品照片
2. 輸入外觀狀況評分（1-5分）
3. 輸入保固剩餘月數
4. 系統自動處理並顯示評估結果

## 資料庫欄位

| 欄位名 | 型態 | 說明 |
|--------|------|------|
| id | Integer | 紀錄編號 |
| item_name | String | YOLO 辨識出的物品名 |
| user_condition | Integer | 使用者輸入的新舊程度 (1-5) |
| warranty_months | Integer | 保固剩餘月數 |
| pred_price | Float | 系統建議售價 |
| confidence | Float | 辨識信心度 |
| created_at | DateTime | 查詢時間 |

## 開發進度

- [x] 環境設定與專案結構
- [x] YOLO 影像辨識模組
- [x] 資料獲取引擎（模擬資料）
- [x] ML 模組（K-Means + Decision Tree）
- [x] Flask 路由與整合
- [x] 前端介面
- [x] 資料庫設計與實作

## 注意事項

- 首次執行時會自動產生 `mock_data.csv` 模擬資料
- YOLOv8 模型會自動下載（首次使用時）
- 上傳圖片大小限制為 16MB
- 支援的圖片格式：PNG, JPG, JPEG, GIF, WEBP

## 授權

本專案為期末專案作業使用。

