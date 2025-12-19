from ultralytics import YOLO

# 1. 載入模型 (因為你剛剛刪除了，它會自動重新下載一份乾淨的)
model = YOLO('yolov8n.pt') 

# 2. 開始訓練
# data='data.yaml' 確保這個檔案在同一個資料夾內
# workers=0 是為了避免 Windows 上的多線程衝突
results = model.train(
    data='data.yaml', 
    epochs=100, 
    imgsz=640, 
    batch=16, 
    workers=0,
    device='cpu'
)