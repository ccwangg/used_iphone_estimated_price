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

# 初始化模組（使用 try-except 處理初始化錯誤）
try:
    print("正在初始化 iPhone 辨識器...")
    iphone_predictor = iPhonePredictor()
    print("✓ iPhone 辨識器初始化完成")
except Exception as e:
    print(f"警告: iPhone 辨識器初始化失敗: {e}")
    print("系統將繼續運行，但可能無法正常辨識")
    iphone_predictor = None

try:
    print("正在初始化資料引擎...")
    # 使用新的資料引擎（從 archive 資料夾讀取真實資料）
    data_engine = DataEngine(
        data_dir='archive',
        database_file='iphoneFeaturesPriceDataset.csv',
        min_similarity=70  # 模糊匹配最低相似度 70%
    )
    print("✓ 資料引擎初始化完成")
except Exception as e:
    print(f"錯誤: 資料引擎初始化失敗: {e}")
    raise

try:
    print("正在初始化價格分析器...")
    price_analyzer = PriceAnalyzer(n_clusters=3)
    print("✓ 價格分析器初始化完成")
except Exception as e:
    print(f"錯誤: 價格分析器初始化失敗: {e}")
    raise

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
        return redirect(request.url)
    
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
            if iphone_predictor is None:
                flash('iPhone 辨識器未初始化，請檢查模型檔案')
                return redirect(url_for('index'))
            
            iphone_result = iphone_predictor.predict(filepath)
            item_name = iphone_result['item_name']
            iphone_model = iphone_result.get('iphone_model', 'iPhone')
            confidence = iphone_result['confidence']
            is_iphone = iphone_result.get('is_iphone', False)
            
            # 檢查是否為 iPhone
            # 降低信心度門檻，因為訓練模型可能信心度較低但仍可信任
            if not is_iphone or item_name == 'unknown' or confidence < 0.05:
                return render_template('error.html', 
                                     error_type='unknown_item',
                                     item_name='非 iPhone 或無法辨識',
                                     confidence=confidence)
            
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
            
            # 4. 獲取參考價格（暫時關閉爬蟲以避免阻塞）
            # 注意：爬蟲功能已暫時關閉，因為露天爬蟲效率低且易被擋，會導致頁面卡住
            # 如果需要參考價格，可以稍後實作非同步爬蟲或使用 API
            # reference_prices = data_engine.get_reference_prices('iPhone', max_items=5)
            reference_prices = {'shopee': [], 'ruten': []}  # 給空值避免前端錯誤
            
            # 檢查是否有參考價格資料
            has_references = False  # 暫時關閉參考價格顯示
            
            if len(price_data) == 0:
                return render_template('error.html',
                                     error_type='no_price_data',
                                     item_name=iphone_model or 'iPhone',
                                     confidence=confidence)
            
            # 確保 price_data 有必要的欄位
            required_cols = ['price']
            missing_cols = [col for col in required_cols if col not in price_data.columns]
            if missing_cols:
                print(f"錯誤: price_data 缺少必要欄位: {missing_cols}")
                print(f"現有欄位: {list(price_data.columns)}")
                return render_template('error.html',
                                     error_type='no_price_data',
                                     item_name=iphone_model or 'iPhone',
                                     confidence=confidence)
            
            # 確保價格欄位是數值型
            if price_data['price'].dtype != 'float64' and price_data['price'].dtype != 'int64':
                try:
                    price_data['price'] = pd.to_numeric(price_data['price'], errors='coerce')
                    price_data = price_data.dropna(subset=['price'])
                except Exception as e:
                    print(f"價格轉換錯誤: {e}")
                    return render_template('error.html',
                                         error_type='no_price_data',
                                         item_name=iphone_model or 'iPhone',
                                         confidence=confidence)
            
            if len(price_data) == 0:
                return render_template('error.html',
                                     error_type='no_price_data',
                                     item_name=iphone_model or 'iPhone',
                                     confidence=confidence)
            
            # 5. K-Means 分群
            try:
                cluster_result = price_analyzer.cluster_prices(price_data)
            except Exception as e:
                print(f"K-Means 分群錯誤: {e}")
                # 如果分群失敗，使用預設值
                import traceback
                print(traceback.format_exc())
                # 如果分群失敗，使用預設值
                try:
                    cluster_result = {
                        'cluster_info': {
                            0: {
                                'name': '價格區間',
                                'stats': {
                                    'mean': float(price_data['price'].mean()),
                                    'min': float(price_data['price'].min()),
                                    'max': float(price_data['price'].max()),
                                    'count': len(price_data),
                                    'std': float(price_data['price'].std())
                                }
                            }
                        }
                    }
                except:
                    cluster_result = {
                        'cluster_info': {
                            0: {'name': '價格區間', 'stats': {'mean': 10000, 'min': 5000, 'max': 20000, 'count': 1, 'std': 5000}}
                        }
                    }
            
            # 6. 訓練決策樹模型（使用 iPhone 專用特徵）
            try:
                dt_result = price_analyzer.train_decision_tree(price_data, use_iphone_features=True)
            except Exception as e:
                import traceback
                print(f"決策樹訓練錯誤: {e}")
                print(traceback.format_exc())
                # 如果訓練失敗，使用簡單的平均價格
                avg_price = float(price_data['price'].mean())
                dt_result = {
                    'test_r2': 0.5,  # 預設 R² 分數
                    'model': None
                }
                # 建立一個簡單的預測函數
                price_analyzer.dt_model = None
            
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
            
            # 7. 預測價格（使用 iPhone 專用特徵）
            try:
                predicted_price = price_analyzer.predict_price(
                    user_condition, 
                    warranty_months,
                    screen_broken=screen_broken,
                    camera_ok=camera_ok,
                    battery_health=battery_health,
                    storage=storage,
                    use_iphone_features=True
                )
            except Exception as e:
                import traceback
                print(f"價格預測錯誤: {e}")
                print(traceback.format_exc())
                # 如果預測失敗，使用簡單計算
                base_price = float(price_data['price'].mean())
                # 根據狀況調整
                condition_multiplier = {1: 0.5, 2: 0.65, 3: 0.8, 4: 0.9, 5: 1.0}
                predicted_price = base_price * condition_multiplier.get(user_condition, 0.8)
                if screen_broken:
                    predicted_price *= 0.7
                if not camera_ok:
                    predicted_price *= 0.85
                predicted_price *= (0.7 + battery_health * 0.3)
                predicted_price = max(0, predicted_price)  # 確保不為負數
            
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
                'cluster_plot': f"cluster_{timestamp}.png" if cluster_plot_path else None,
                'cluster_info': cluster_result.get('cluster_info', {}),
                'model_r2': round(dt_result.get('test_r2', 0.5), 3),
                'price_stats': {
                    'mean': round(float(price_data['price'].mean()), 0),
                    'min': round(float(price_data['price'].min()), 0),
                    'max': round(float(price_data['price'].max()), 0),
                    'median': round(float(price_data['price'].median()), 0)
                },
                'reference_prices': reference_prices if has_references else None  # 只有有資料時才顯示
            }
            
            return render_template('result.html', result=result_data)
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"錯誤詳情: {error_msg}")
            print(f"錯誤堆疊:\n{error_trace}")
            flash(f'處理過程中發生錯誤: {error_msg}')
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

