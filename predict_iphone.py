"""
iPhone 型號辨識模組
用途：使用者上傳 iPhone 照片，自動判定 iPhone 型號
"""
from ultralytics import YOLO
from PIL import Image
import os
import re


class iPhonePredictor:
    """iPhone 型號辨識器類別"""
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        初始化 YOLO 模型
        Args:
            model_path: 模型檔案路徑，預設使用預訓練模型
        """
        self.model = YOLO(model_path)
        
        # iPhone 型號關鍵字對照表
        self.iphone_models = {
            'cell phone': 'iPhone',  # 通用手機辨識
            'phone': 'iPhone',
            'mobile': 'iPhone',
        }
        
        # 常見 iPhone 型號列表（用於後續型號識別）
        self.iphone_versions = [
            'iPhone 15 Pro Max', 'iPhone 15 Pro', 'iPhone 15 Plus', 'iPhone 15',
            'iPhone 14 Pro Max', 'iPhone 14 Pro', 'iPhone 14 Plus', 'iPhone 14',
            'iPhone 13 Pro Max', 'iPhone 13 Pro', 'iPhone 13 mini', 'iPhone 13',
            'iPhone 12 Pro Max', 'iPhone 12 Pro', 'iPhone 12 mini', 'iPhone 12',
            'iPhone 11 Pro Max', 'iPhone 11 Pro', 'iPhone 11',
            'iPhone XS Max', 'iPhone XS', 'iPhone XR', 'iPhone X',
            'iPhone 8 Plus', 'iPhone 8',
            'iPhone 7 Plus', 'iPhone 7',
            'iPhone 6s Plus', 'iPhone 6s', 'iPhone 6 Plus', 'iPhone 6',
        ]
    
    def predict(self, image_path, confidence=0.25):
        """
        預測圖片中的 iPhone 型號
        Args:
            image_path: 圖片路徑
            confidence: 信心度閾值
        Returns:
            dict: 包含 iPhone 型號、信心度等資訊
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
        
        # 檢查是否為手機
        is_phone = False
        best_pred = None
        
        for pred in predictions:
            pred_name_lower = pred['name'].lower()
            if 'phone' in pred_name_lower or 'cell' in pred_name_lower or 'mobile' in pred_name_lower:
                is_phone = True
                if best_pred is None or pred['confidence'] > best_pred['confidence']:
                    best_pred = pred
        
        if not is_phone:
            return {
                'item_name': 'unknown',
                'confidence': 0.0,
                'iphone_model': None,
                'all_predictions': predictions,
                'is_iphone': False
            }
        
        # 嘗試從圖片檔名或路徑推斷 iPhone 型號
        # 實際應用中，可能需要更進階的模型來辨識具體型號
        iphone_model = self._detect_iphone_model(image_path, best_pred)
        
        return {
            'item_name': 'iPhone',
            'confidence': best_pred['confidence'],
            'iphone_model': iphone_model,
            'all_predictions': predictions,
            'is_iphone': True
        }
    
    def _detect_iphone_model(self, image_path, prediction):
        """
        嘗試從圖片或預測結果推斷 iPhone 型號
        這是一個簡化版本，實際應用中可能需要：
        1. 訓練專門的 iPhone 型號辨識模型
        2. 使用 OCR 讀取圖片中的型號文字
        3. 分析圖片特徵（尺寸、外觀等）
        """
        # 從檔名嘗試提取型號資訊
        filename = os.path.basename(image_path).lower()
        
        # 檢查檔名中是否包含型號資訊
        for model in self.iphone_versions:
            model_lower = model.lower().replace(' ', '')
            if model_lower in filename.replace(' ', '').replace('_', '').replace('-', ''):
                return model
        
        # 如果無法從檔名判斷，回傳通用 iPhone
        # 實際應用中，這裡可以加入更複雜的辨識邏輯
        return 'iPhone (型號待確認)'
    
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
    predictor = iPhonePredictor()
    # 測試範例
    # result = predictor.predict('test_iphone.jpg')
    # print(f"辨識結果: {result['item_name']}, 型號: {result['iphone_model']}, 信心度: {result['confidence']:.2f}")

