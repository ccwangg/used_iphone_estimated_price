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
            model_path: 模型檔案路徑，如果為 None 則自動尋找訓練好的模型
        """
        # 如果沒有指定模型路徑，嘗試尋找訓練好的模型
        if model_path is None:
            # 常見的訓練輸出位置
            possible_paths = [
                'best.pt',  # 根目錄
                'runs/detect/train/weights/best.pt',
                'iphone.v1/runs/detect/train/weights/best.pt',
                'yolov8n.pt'  # 最後回退到預訓練模型
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"✓ 使用模型: {model_path}")
                    break
            
            # 如果都找不到，使用預訓練模型
            if model_path is None:
                model_path = 'yolov8n.pt'
                print(f"⚠ 警告: 找不到訓練好的模型，使用預訓練模型: {model_path}")
        
        print(f"[iPhonePredictor] 正在載入模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 顯示模型資訊
        print(f"[iPhonePredictor] 模型載入成功")
        if hasattr(self.model, 'names'):
            print(f"[iPhonePredictor] 模型類別數量: {len(self.model.names)}")
            print(f"[iPhonePredictor] 模型類別: {list(self.model.names.values())}")
        
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
    
    def predict(self, image_path, confidence=0.05):
        """
        預測圖片中的 iPhone 型號
        Args:
            image_path: 圖片路徑
            confidence: 信心度閾值
        Returns:
            dict: 包含 iPhone 型號、信心度等資訊
        """
        print(f"\n{'='*60}")
        print(f"[iPhonePredictor] 開始預測: {image_path}")
        print(f"[iPhonePredictor] 使用信心度門檻: {confidence}")
        
        if not os.path.exists(image_path):
            error_msg = f"圖片檔案不存在: {image_path}"
            print(f"[ERROR] {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # 執行預測
        print(f"[iPhonePredictor] 執行模型推理...")
        print(f"[iPhonePredictor] 圖片尺寸: {image_path}")
        
        # 嘗試更低的信心度門檻（如果原本的門檻沒有結果）
        results = self.model(image_path, conf=confidence, verbose=False)
        
        # 如果沒有偵測到任何物體，嘗試更低的門檻
        total_detections = sum(len(result.boxes) for result in results)
        if total_detections == 0 and confidence > 0.01:
            print(f"[iPhonePredictor] 使用門檻 {confidence} 沒有偵測到物體，嘗試更低的門檻 0.01...")
            results = self.model(image_path, conf=0.01, verbose=False)
            total_detections = sum(len(result.boxes) for result in results)
            if total_detections > 0:
                print(f"[iPhonePredictor] 使用門檻 0.01 偵測到 {total_detections} 個物體")
        
        # 解析結果
        predictions = []
        for result in results:
            boxes = result.boxes
            print(f"[iPhonePredictor] 偵測到 {len(boxes)} 個邊界框")
            
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
                print(f"[iPhonePredictor] 偵測到: {cls_name} (信心度: {conf:.3f})")
        
        if len(predictions) == 0:
            print(f"[WARNING] 沒有偵測到任何物體！")
            print(f"[DEBUG] 模型可能的原因：")
            print(f"  - 圖片中沒有可辨識的物體")
            print(f"  - 信心度門檻 {confidence} 太高")
            print(f"  - 模型未訓練過此類型的圖片")
            print(f"  - 圖片品質不佳或角度問題")
        
        # 檢查是否為手機
        is_phone = False
        best_pred = None
        
        print(f"[iPhonePredictor] 檢查 {len(predictions)} 個預測結果...")
        for pred in predictions:
            pred_name_lower = pred['name'].lower()
            print(f"[DEBUG] 檢查預測: '{pred['name']}' (lower: '{pred_name_lower}')")
            
            # 檢查是否為 iPhone（包含 -iphone 開頭的標籤）或一般手機標籤
            if ('iphone' in pred_name_lower or 
                'phone' in pred_name_lower or 
                'cell' in pred_name_lower or 
                'mobile' in pred_name_lower):
                is_phone = True
                print(f"[DEBUG] ✓ 匹配到手機類別: '{pred['name']}'")
                # 挑選信心度最高的結果
                if best_pred is None or pred['confidence'] > best_pred['confidence']:
                    best_pred = pred
        
        if not is_phone:
            print(f"[WARNING] 沒有匹配到任何手機類別")
            print(f"[DEBUG] 所有預測結果: {[p['name'] for p in predictions]}")
            return {
                'item_name': 'unknown',
                'confidence': 0.0,
                'iphone_model': None,
                'all_predictions': predictions,
                'is_iphone': False,
                'debug_info': {
                    'total_detections': len(predictions),
                    'detected_classes': [p['name'] for p in predictions],
                    'confidence_threshold': confidence
                }
            }
        
        # 嘗試從圖片檔名或路徑推斷 iPhone 型號
        # 實際應用中，可能需要更進階的模型來辨識具體型號
        print(f"[iPhonePredictor] 最佳預測: {best_pred['name']} (信心度: {best_pred['confidence']:.3f})")
        iphone_model = self._detect_iphone_model(image_path, best_pred)
        print(f"[iPhonePredictor] 推斷的型號: {iphone_model}")
        print(f"{'='*60}\n")
        
        return {
            'item_name': iphone_model, # 將 iPhone 改為 iphone_model (例如 iPhone 12)
            'confidence': best_pred['confidence'],
            'iphone_model': iphone_model,
            'all_predictions': predictions,
            'is_iphone': True,
            'debug_info': {
                'total_detections': len(predictions),
                'best_prediction': best_pred['name'],
                'detected_classes': [p['name'] for p in predictions]
            }
        }
    
    def _detect_iphone_model(self, image_path, prediction):
        """
        嘗試從預測結果或圖片檔名推斷 iPhone 型號
        優先從模型預測結果中提取型號資訊（如 -iphone11, -iphone12 等）
        """
        # 首先從預測結果中提取型號（如果模型標籤包含型號資訊）
        if prediction and 'name' in prediction:
            pred_name = prediction['name'].lower()
            
            # 處理訓練模型的標籤格式（如 -iphone11, -iphone12, -iphone13 等）
            if pred_name.startswith('-iphone') or 'iphone' in pred_name:
                # 提取型號數字和變體
                # 例如：-iphone11 -> iPhone 11, -iphone12 pro -> iPhone 12 Pro
                model_str = pred_name.replace('-', '').replace('iphone', 'iPhone ')
                
                # 處理變體（pro, pro max, mini, plus 等）
                if 'pro max' in pred_name:
                    model_str = model_str.replace('pro max', 'Pro Max')
                elif 'pro' in pred_name and 'max' not in pred_name:
                    model_str = model_str.replace('pro', 'Pro')
                elif 'mini' in pred_name:
                    model_str = model_str.replace('mini', 'mini')
                elif 'plus' in pred_name:
                    model_str = model_str.replace('plus', 'Plus')
                
                # 清理格式
                model_str = model_str.replace('  ', ' ').strip()
                if model_str and model_str != 'iPhone':
                    return model_str
        
        # 如果預測結果沒有型號資訊，從檔名嘗試提取
        filename = os.path.basename(image_path).lower()
        
        # 檢查檔名中是否包含型號資訊
        for model in self.iphone_versions:
            model_lower = model.lower().replace(' ', '')
            if model_lower in filename.replace(' ', '').replace('_', '').replace('-', ''):
                return model
        
        # 如果都無法判斷，回傳通用 iPhone
        return 'iPhone (型號待確認)'
    
    def predict_with_visualization(self, image_path, output_path=None, confidence=0.05):
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

