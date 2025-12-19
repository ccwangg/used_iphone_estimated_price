"""
Flask 主應用程式
AI 二手物價值評估系統
"""
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import sqlite3
from predict_iphone import iPhonePredictor
from data_engine import DataEngine
from model import PriceAnalyzer
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# 初始化模組
print("\n[APP] 正在初始化 iPhonePredictor...")
iphone_predictor = iPhonePredictor()
print("[APP] iPhonePredictor 初始化完成\n")
# 使用新的資料引擎（從 archive 資料夾讀取真實資料）
data_engine = DataEngine(
    data_dir='archive',
    database_file='archive/iphoneFeaturesPriceDataset.csv',  # 使用 iphoneFeaturesPriceDataset.csv
    min_similarity=70  # 模糊匹配最低相似度 70%
)
price_analyzer = PriceAnalyzer(n_clusters=3)

# 確保必要的目錄存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)


def allowed_file(filename):
    """檢查檔案副檔名是否允許"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db():
    """初始化資料庫"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            user_condition INTEGER NOT NULL,
            warranty_months INTEGER NOT NULL,
            pred_price REAL NOT NULL,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def save_evaluation(item_name, user_condition, warranty_months, pred_price, confidence):
    """儲存評估結果到資料庫"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluations (item_name, user_condition, warranty_months, pred_price, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (item_name, user_condition, warranty_months, pred_price, confidence))
    conn.commit()
    conn.close()


@app.route('/')
def index():
    """首頁：上傳圖片"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """處理圖片上傳和處理"""
    if 'file' not in request.files:
        flash('請選擇要上傳的圖片')
        return redirect(url_for('index'))
    
    file = request.files['file']
    user_condition = request.form.get('condition', type=int)
    warranty_months = request.form.get('warranty_months', type=int)
    screen_broken = request.form.get('screen_broken', type=int, default=0)
    camera_ok = request.form.get('camera_ok', type=int, default=1)
    battery_health = request.form.get('battery_health', type=float, default=85.0) / 100.0  # 轉換為 0-1
    storage = request.form.get('storage', type=int, default=256)
    iphone_variant = request.form.get('iphone_variant', default='').strip()
    
    if file.filename == '':
        flash('請選擇要上傳的圖片')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # 儲存上傳的檔案
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. iPhone 影像辨識
            print(f"\n[APP] 開始處理上傳的圖片: {filename}")
            print(f"[APP] 檔案路徑: {filepath}")
            
            iphone_result = iphone_predictor.predict(filepath)
            item_name = iphone_result['item_name']
            iphone_model = iphone_result.get('iphone_model', 'iPhone')
            confidence = iphone_result['confidence']
            is_iphone = iphone_result.get('is_iphone', False)
            debug_info = iphone_result.get('debug_info', {})
            all_predictions = iphone_result.get('all_predictions', [])
            
            print(f"[APP] 辨識結果:")
            print(f"  - item_name: {item_name}")
            print(f"  - iphone_model: {iphone_model}")
            print(f"  - confidence: {confidence}")
            print(f"  - is_iphone: {is_iphone}")
            print(f"  - 總偵測數: {len(all_predictions)}")
            
            # 檢查是否為 iPhone
            # 如果偵測到 iPhone（is_iphone == True），即使信心度低於 0.05 也接受
            # 只有在完全沒有偵測到或結果為 unknown 時才判定為失敗
            if not is_iphone or item_name == 'unknown':
                print(f"[APP] ⚠ 辨識失敗！")
                print(f"[APP] 失敗原因:")
                print(f"  - is_iphone: {is_iphone}")
                print(f"  - item_name: {item_name}")
                print(f"  - confidence: {confidence}")
                print(f"  - 所有預測: {[p['name'] for p in all_predictions]}")
                
                # 準備詳細的錯誤資訊
                error_details = {
                    'is_iphone': is_iphone,
                    'item_name': item_name,
                    'confidence': confidence,
                    'total_detections': len(all_predictions),
                    'detected_classes': [p['name'] for p in all_predictions],
                    'all_predictions': all_predictions
                }
                
                return render_template('error.html', 
                                     error_type='unknown_item',
                                     item_name='非 iPhone 或無法辨識',
                                     confidence=confidence,
                                     error_details=error_details,
                                     debug_info=debug_info)
            
            # 如果信心度低於 0.05，顯示警告但仍繼續處理
            if confidence < 0.05:
                print(f"[APP] ⚠ 警告: 信心度較低 ({confidence:.3f})，但仍使用此結果")
            
            # 2. 產生標註圖片
            annotated_path = os.path.join(
                app.config['RESULT_FOLDER'],
                f"annotated_{filename}"
            )
            iphone_predictor.predict_with_visualization(filepath, annotated_path)
            
            # 3. 使用資料引擎獲取價格資料（會自動執行：資料庫查詢 -> 模糊匹配 -> 爬蟲）
            # 優先使用辨識到的具體型號，如果沒有則使用 'iPhone'
            # 如果使用者選擇了型號變體，則組合搜尋詞以提高精確度
            if iphone_model and iphone_model != 'iPhone (型號待確認)':
                search_term = iphone_model
            else:
                # 如果沒有辨識出具體型號，但有選擇變體，則組合搜尋
                if iphone_variant:
                    # 將變體轉換為搜尋關鍵字
                    variant_keywords = {
                        'mini': 'mini',
                        'standard': '',  # 一般款不需要額外關鍵字
                        'pro': 'Pro',
                        'pro max': 'Pro Max',
                        'plus': 'Plus'
                    }
                    variant_keyword = variant_keywords.get(iphone_variant.lower(), '')
                    if variant_keyword:
                        search_term = f'iPhone {variant_keyword}'
                    else:
                        search_term = 'iPhone'
                else:
                    search_term = 'iPhone'
            
            print(f"使用搜尋詞: {search_term} (變體: {iphone_variant})")
            price_data = data_engine.get_prices(search_term, max_records=200)
            
            # 如果使用變體但沒找到資料，嘗試更寬鬆的搜尋
            if len(price_data) == 0 and iphone_variant:
                print(f"使用變體 '{iphone_variant}' 未找到資料，嘗試更寬鬆的搜尋...")
                price_data = data_engine.get_prices('iPhone', max_records=200)
            
            # 4. 獲取參考價格（蝦皮、露天最新五筆）
            reference_prices = data_engine.get_reference_prices('iPhone', max_items=5)
            
            # 檢查是否有參考價格資料
            has_references = (reference_prices.get('shopee') and len(reference_prices['shopee']) > 0) or \
                           (reference_prices.get('ruten') and len(reference_prices['ruten']) > 0) or \
                           (reference_prices.get('dcard') and len(reference_prices['dcard']) > 0) or \
                           (reference_prices.get('facebook') and len(reference_prices['facebook']) > 0)
            
            if len(price_data) == 0:
                return render_template('error.html',
                                     error_type='no_price_data',
                                     item_name=iphone_model or 'iPhone',
                                     confidence=confidence)
            
            # 5. K-Means 分群
            cluster_result = price_analyzer.cluster_prices(price_data)
            
            # 6. 訓練決策樹模型（使用 iPhone 專用特徵）
            dt_result = price_analyzer.train_decision_tree(price_data, use_iphone_features=True)
            
            # 7. 預測價格（使用 iPhone 專用特徵）
            if user_condition is None:
                user_condition = 3  # 預設值
            if warranty_months is None:
                warranty_months = 0  # 預設值
            if screen_broken is None:
                screen_broken = 0
            if camera_ok is None:
                camera_ok = 1
            if battery_health is None:
                battery_health = 0.85
            if storage is None:
                storage = 256
            
            predicted_price = price_analyzer.predict_price(
                user_condition, 
                warranty_months,
                screen_broken=screen_broken,
                camera_ok=camera_ok,
                battery_health=battery_health,
                storage=storage,
                use_iphone_features=True
            )
            
            # 8. 視覺化分群結果
            cluster_plot_path = os.path.join(
                'static',
                f"cluster_{timestamp}.png"
            )
            price_analyzer.visualize_clusters(price_data, cluster_plot_path)
            
            # 9. 儲存評估結果
            save_evaluation(item_name, user_condition, warranty_months, predicted_price, confidence)
            
            # 準備結果資料
            result_data = {
                'item_name': iphone_model or item_name,
                'iphone_variant': iphone_variant,
                'confidence': confidence,
                'user_condition': user_condition,
                'warranty_months': warranty_months,
                'screen_broken': screen_broken,
                'camera_ok': camera_ok,
                'battery_health': battery_health,
                'storage': storage,
                'predicted_price': round(predicted_price, 0),
                'annotated_image': f"results/annotated_{filename}",
                'cluster_plot': f"cluster_{timestamp}.png",
                'cluster_info': cluster_result['cluster_info'],
                'model_r2': round(dt_result['test_r2'], 3),
                'price_stats': {
                    'mean': round(price_data['price'].mean(), 0),
                    'min': round(price_data['price'].min(), 0),
                    'max': round(price_data['price'].max(), 0),
                    'median': round(price_data['price'].median(), 0)
                },
                'reference_prices': reference_prices if has_references else None  # 只有有資料時才顯示
            }
            
            return render_template('result.html', result=result_data)
        
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            print(f"\n[APP] ❌ 發生未預期的錯誤！")
            print(f"[APP] 錯誤類型: {type(e).__name__}")
            print(f"[APP] 錯誤訊息: {str(e)}")
            print(f"[APP] 完整錯誤追蹤:")
            print(error_traceback)
            
            # 記錄錯誤到檔案（可選）
            try:
                with open('error_log.txt', 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"檔案: {filename}\n")
                    f.write(f"錯誤: {type(e).__name__}: {str(e)}\n")
                    f.write(f"追蹤:\n{error_traceback}\n")
            except:
                pass
            
            flash(f'處理過程中發生錯誤: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('不支援的檔案格式，請上傳 PNG、JPG、JPEG、GIF 或 WEBP 格式的圖片')
        return redirect(url_for('index'))


@app.route('/result')
def result():
    """顯示結果頁面（直接訪問時重定向到首頁）"""
    return redirect(url_for('index'))


@app.route('/history')
def history():
    """顯示歷史評估記錄"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM evaluations 
        ORDER BY created_at DESC 
        LIMIT 50
    ''')
    records = cursor.fetchall()
    conn.close()
    
    # 轉換為字典列表
    history_data = []
    for record in records:
        history_data.append({
            'id': record[0],
            'item_name': record[1],
            'user_condition': record[2],
            'warranty_months': record[3],
            'pred_price': record[4],
            'confidence': record[5],
            'created_at': record[6]
        })
    
    return render_template('history.html', history=history_data)


# 初始化資料庫
init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

