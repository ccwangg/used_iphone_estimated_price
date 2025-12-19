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
    
    def __init__(self, model_path=None):
        """
        初始化 YOLO 模型
        Args:
            model_path: 模型檔案路徑，如果為 None 則自動尋找或使用預設模型
        """
        # 如果沒有指定路徑，嘗試使用訓練好的模型
        if model_path is None:
            # 嘗試多個可能的路徑
            possible_paths = [
                r'C:\Users\User\runs\detect\train\weights\best.pt',
                'best.pt',
                'yolov8n.pt'
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            # 如果都找不到，使用預設模型
            if model_path is None:
                print("警告: 找不到訓練模型，使用預設模型 yolov8n.pt")
                model_path = 'yolov8n.pt'
        
        # 載入模型（使用 try-except 處理可能的錯誤）
        try:
            self.model = YOLO(model_path)
            print(f"✓ 已載入模型: {model_path}")
        except Exception as e:
            print(f"錯誤: 無法載入模型 {model_path}: {e}")
            print("嘗試使用預設模型...")
            try:
                self.model = YOLO('yolov8n.pt')
                print("✓ 已載入預設模型: yolov8n.pt")
            except Exception as e2:
                print(f"嚴重錯誤: 無法載入任何模型: {e2}")
                raise
        
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
        
        # 檢查是否為 iPhone（支援新模型的標籤格式：-iphone, -iphone12, -iphone13 等）
        is_phone = False
        best_pred = None
        
        for pred in predictions:
            pred_name = pred['name']
            pred_name_lower = pred_name.lower()
            
            # 檢查是否為 iPhone（支援多種格式）
            # 1. 新模型格式：-iphone, -iphone12, -iphone13 等（開頭有減號）
            # 2. 舊模型格式：cell phone, phone, mobile
            # 3. 標準格式：iphone, iphone 12, iphone 13 等
            is_iphone_label = (
                pred_name_lower.startswith('-iphone') or  # 新模型格式：-iphone, -iphone12
                'iphone' in pred_name_lower or            # 包含 iphone
                'phone' in pred_name_lower or             # 包含 phone
                'cell' in pred_name_lower or              # 包含 cell
                'mobile' in pred_name_lower               # 包含 mobile
            )
            
            if is_iphone_label:
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
        
        # 從辨識結果中提取 iPhone 型號
        # 新模型可能直接辨識出 -iphone12, -iphone13 等
        iphone_model = self._extract_iphone_model(best_pred, image_path)
        
        return {
            'item_name': 'iPhone',
            'confidence': best_pred['confidence'],
            'iphone_model': iphone_model,
            'all_predictions': predictions,
            'is_iphone': True
        }
    
    def _extract_iphone_model(self, prediction, image_path):
        """
        從辨識結果中提取 iPhone 型號
        支援新模型格式：-iphone, -iphone12, -iphone13 等
        """
        if prediction is None:
            return 'iPhone (型號待確認)'
        
        pred_name = prediction['name']
        pred_name_lower = pred_name.lower()
        
        # 處理新模型格式：-iphone12 -> iPhone 12
        if pred_name_lower.startswith('-iphone'):
            # 移除開頭的減號
            model_str = pred_name.replace('-', '').replace('_', ' ')
            
            # 嘗試提取數字（如 -iphone12 -> iPhone 12）
            # 使用正則表達式提取數字
            numbers = re.findall(r'\d+', model_str)
            if numbers:
                # 如果有數字，組合成 "iPhone 12" 格式
                model_number = numbers[0]
                # 檢查是否有其他關鍵字（Pro, Max, mini, Plus）
                if 'pro' in pred_name_lower and 'max' in pred_name_lower:
                    return f'iPhone {model_number} Pro Max'
                elif 'pro' in pred_name_lower:
                    return f'iPhone {model_number} Pro'
                elif 'mini' in pred_name_lower:
                    return f'iPhone {model_number} mini'
                elif 'plus' in pred_name_lower:
                    return f'iPhone {model_number} Plus'
                else:
                    return f'iPhone {model_number}'
            else:
                # 沒有數字，可能是 -iphone（通用）
                return 'iPhone (型號待確認)'
        
        # 處理標準格式：iphone 12, iPhone 13 等
        if 'iphone' in pred_name_lower:
            # 嘗試提取數字和變體
            numbers = re.findall(r'\d+', pred_name)
            if numbers:
                model_number = numbers[0]
                if 'pro' in pred_name_lower and 'max' in pred_name_lower:
                    return f'iPhone {model_number} Pro Max'
                elif 'pro' in pred_name_lower:
                    return f'iPhone {model_number} Pro'
                elif 'mini' in pred_name_lower:
                    return f'iPhone {model_number} mini'
                elif 'plus' in pred_name_lower:
                    return f'iPhone {model_number} Plus'
                else:
                    return f'iPhone {model_number}'
        
        # 如果無法從辨識結果判斷，嘗試從檔名提取
        filename = os.path.basename(image_path).lower()
        for model in self.iphone_versions:
            model_lower = model.lower().replace(' ', '')
            if model_lower in filename.replace(' ', '').replace('_', '').replace('-', ''):
                return model
        
        # 如果都無法判斷，回傳通用 iPhone
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

