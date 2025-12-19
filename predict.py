"""
YOLOv8 影像辨識模組
用途：使用者上傳照片，自動判定物品名稱
"""
from ultralytics import YOLO
from PIL import Image
import os


class YOLOPredictor:
    """YOLOv8 預測器類別"""
    
    def __init__(self, model_path='C:\Users\User\runs\detect\train\weights\best.pt'):
        """
        初始化 YOLO 模型
        Args:
            model_path: 模型檔案路徑，預設使用預訓練模型
        """
        self.model = YOLO(model_path)
    
    def predict(self, image_path, confidence=0.25):
        """
        預測圖片中的物品
        Args:
            image_path: 圖片路徑
            confidence: 信心度閾值
        Returns:
            dict: 包含物品名稱、信心度、座標等資訊
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"圖片檔案不存在: {image_path}")
        
        # 執行預測
        results = self.model(image_path, conf=confidence)
        
        # 解析結果
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 取得類別名稱和信心度
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                
                # 取得邊界框座標
                xyxy = box.xyxy[0].tolist()
                
                predictions.append({
                    'name': cls_name,
                    'confidence': conf,
                    'bbox': xyxy
                })
        
        # 如果有多個物品，選擇信心度最高的
        if predictions:
            best_pred = max(predictions, key=lambda x: x['confidence'])
            return {
                'item_name': best_pred['name'],
                'confidence': best_pred['confidence'],
                'all_predictions': predictions
            }
        else:
            return {
                'item_name': 'unknown',
                'confidence': 0.0,
                'all_predictions': []
            }
    
    def predict_with_visualization(self, image_path, output_path=None, confidence=0.25):
        """
        預測並產生標註圖片
        Args:
            image_path: 輸入圖片路徑
            output_path: 輸出圖片路徑（可選）
            confidence: 信心度閾值
        Returns:
            str: 標註後的圖片路徑
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"圖片檔案不存在: {image_path}")
        
        # 執行預測並儲存標註結果
        results = self.model(image_path, conf=confidence, save=True)
        
        # 如果沒有指定輸出路徑，使用預設路徑
        if output_path is None:
            output_path = image_path.replace('.', '_predicted.')
        
        # 儲存結果圖片
        for result in results:
            result.save(output_path)
            return output_path
        
        return image_path


# 測試用主程式
if __name__ == '__main__':
    predictor = YOLOPredictor()
    # 測試範例（需要實際圖片路徑）
    # result = predictor.predict('test_image.jpg')
    # print(f"辨識結果: {result['item_name']}, 信心度: {result['confidence']:.2f}")

