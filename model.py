"""
機器學習模組
用途：
1. K-Means 分群：將價格資料分群（低價損壞區、合理二手區、高價全新區）
2. Decision Tree：根據物品狀況預測建議售價
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os


class PriceAnalyzer:
    """價格分析器類別"""
    
    def __init__(self, n_clusters=3):
        """
        初始化價格分析器
        Args:
            n_clusters: K-Means 分群數量（預設3：低價、中價、高價）
        """
        self.n_clusters = n_clusters
        self.kmeans_model = None
        self.dt_model = None
        self.cluster_labels = None
        self.price_data = None
    
    def cluster_prices(self, price_data):
        """
        使用 K-Means 對價格進行分群
        Args:
            price_data: pd.DataFrame，包含 'price' 欄位
        Returns:
            dict: 分群結果和統計資訊
        """
        self.price_data = price_data.copy()
        
        # 準備特徵（價格）
        X = price_data[['price']].values
        
        # 執行 K-Means 分群
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(X)
        
        # 將分群結果加入 DataFrame
        price_data['cluster'] = self.cluster_labels
        
        # 計算每個群集的統計資訊
        cluster_stats = {}
        for i in range(self.n_clusters):
            cluster_prices = price_data[price_data['cluster'] == i]['price']
            cluster_stats[i] = {
                'count': len(cluster_prices),
                'mean': cluster_prices.mean(),
                'min': cluster_prices.min(),
                'max': cluster_prices.max(),
                'std': cluster_prices.std()
            }
        
        # 根據平均價格排序群集（低到高）
        sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['mean'])
        
        # 為群集命名
        cluster_names = ['低價損壞區', '合理二手區', '高價全新區']
        if len(sorted_clusters) == 2:
            cluster_names = ['低價區', '高價區']
        elif len(sorted_clusters) > 3:
            cluster_names = [f'價格區間 {i+1}' for i in range(len(sorted_clusters))]
        
        cluster_info = {}
        for idx, (cluster_id, stats) in enumerate(sorted_clusters):
            cluster_info[cluster_id] = {
                'name': cluster_names[idx] if idx < len(cluster_names) else f'群集 {idx+1}',
                'stats': stats
            }
        
        return {
            'cluster_labels': self.cluster_labels,
            'cluster_info': cluster_info,
            'cluster_centers': self.kmeans_model.cluster_centers_.flatten().tolist()
        }
    
    def train_decision_tree(self, price_data, use_iphone_features=True):
        """
        訓練決策樹模型，根據物品狀況預測價格
        Args:
            price_data: pd.DataFrame，包含價格和特徵欄位
            use_iphone_features: 是否使用 iPhone 專用特徵（螢幕、鏡頭、電池等）
        Returns:
            dict: 訓練結果和模型評估指標
        """
        # 準備特徵和目標變數
        if use_iphone_features:
            # iPhone 專用特徵
            feature_cols = ['condition', 'warranty_months', 'screen_broken', 'camera_ok', 'battery_health']
            # 如果資料中有 storage 欄位，也加入
            if 'storage' in price_data.columns:
                feature_cols.append('storage')
        else:
            # 通用特徵
            feature_cols = ['condition', 'warranty_months']
        
        # 只使用存在的欄位
        available_cols = [col for col in feature_cols if col in price_data.columns]
        if not available_cols:
            # 如果沒有可用特徵，至少使用價格的平均值
            print("警告: 沒有可用的特徵欄位，使用簡單模型")
            available_cols = ['condition'] if 'condition' in price_data.columns else []
            if not available_cols:
                # 如果連 condition 都沒有，建立一個常數特徵
                price_data['condition'] = 3
                available_cols = ['condition']
        
        # 儲存使用的特徵順序，供預測時使用
        self._used_features = available_cols.copy()
        self._feature_order = available_cols  # 儲存特徵順序
        
        # 確保價格欄位是數值型
        if price_data['price'].dtype not in ['float64', 'int64']:
            price_data['price'] = pd.to_numeric(price_data['price'], errors='coerce')
        
        X = price_data[available_cols].values
        y = price_data['price'].values
        
        # 確保資料有效
        if len(X) == 0 or len(y) == 0:
            raise ValueError("資料為空，無法訓練模型")
        
        # 處理 NaN 值
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("處理後資料為空，無法訓練模型")
        
        # 確保至少有一些資料點
        if len(X) < 2:
            print("警告: 資料點太少，無法訓練模型")
            raise ValueError("資料點不足，無法訓練模型")
        
        # 分割訓練集和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 訓練決策樹模型
        self.dt_model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.dt_model.fit(X_train, y_train)
        
        # 評估模型
        y_train_pred = self.dt_model.predict(X_train)
        y_test_pred = self.dt_model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'model': self.dt_model
        }
    
    def predict_price(self, condition, warranty_months, screen_broken=0, camera_ok=1, battery_health=1.0, storage=256, use_iphone_features=True):
        """
        根據物品狀況預測價格
        Args:
            condition: 外觀狀況 (1-5分)
            warranty_months: 保固剩餘月數
            screen_broken: 螢幕是否破裂 (0=完好, 1=破裂)
            camera_ok: 鏡頭是否完好 (0=損壞, 1=完好)
            battery_health: 電池健康度 (0.0-1.0)
            storage: 儲存容量 (GB)
            use_iphone_features: 是否使用 iPhone 專用特徵
        Returns:
            float: 預測價格
        """
        if self.dt_model is None:
            # 如果模型未訓練，使用簡單估算
            print("警告: 決策樹模型未訓練，使用簡單估算")
            return 10000.0  # 預設價格
        
        try:
            # 準備輸入特徵（必須按照訓練時的特徵順序）
            if hasattr(self, '_feature_order') and self._feature_order:
                # 使用訓練時的特徵順序
                feature_dict = {
                    'condition': condition,
                    'warranty_months': warranty_months,
                    'screen_broken': screen_broken,
                    'camera_ok': camera_ok,
                    'battery_health': battery_health,
                    'storage': storage
                }
                # 按照訓練時的特徵順序建立輸入
                feature_list = [feature_dict[col] for col in self._feature_order]
                X = np.array([feature_list])
            elif use_iphone_features:
                # 預設使用所有特徵
                X = np.array([[condition, warranty_months, screen_broken, camera_ok, battery_health, storage]])
            else:
                X = np.array([[condition, warranty_months]])
            
            # 預測
            predicted_price = self.dt_model.predict(X)[0]
            
            return max(0, predicted_price)  # 確保價格不為負數
        except Exception as e:
            print(f"價格預測錯誤: {e}")
            import traceback
            print(traceback.format_exc())
            # 如果預測失敗，使用簡單計算
            base_price = 15000.0  # 預設基礎價格
            condition_multiplier = {1: 0.5, 2: 0.65, 3: 0.8, 4: 0.9, 5: 1.0}
            predicted_price = base_price * condition_multiplier.get(condition, 0.8)
            if screen_broken:
                predicted_price *= 0.7
            if not camera_ok:
                predicted_price *= 0.85
            predicted_price *= (0.7 + battery_health * 0.3)
            return max(0, predicted_price)
    
    def visualize_clusters(self, price_data, output_path='static/cluster_plot.png'):
        """
        視覺化價格分群結果
        Args:
            price_data: pd.DataFrame，包含 'price' 和 'cluster' 欄位
            output_path: 輸出圖片路徑
        Returns:
            str: 圖片檔案路徑
        """
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # 繪製每個群集的價格分佈
        colors = ['red', 'orange', 'green']
        cluster_names = ['低價損壞區', '合理二手區', '高價全新區']
        
        for i in range(self.n_clusters):
            cluster_data = price_data[price_data['cluster'] == i]
            color = colors[i] if i < len(colors) else 'blue'
            label = cluster_names[i] if i < len(cluster_names) else f'群集 {i+1}'
            plt.scatter(
                cluster_data.index,
                cluster_data['price'],
                c=color,
                label=label,
                alpha=0.6,
                s=50
            )
        
        plt.xlabel('資料索引')
        plt.ylabel('價格 (NT$)')
        plt.title('價格分群結果 (K-Means)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 儲存圖片
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path


# 測試用主程式
if __name__ == '__main__':
    # 建立測試資料
    test_data = pd.DataFrame({
        'price': np.random.normal(10000, 3000, 100).clip(1000, 20000),
        'condition': np.random.randint(1, 6, 100),
        'warranty_months': np.random.randint(0, 25, 100)
    })
    
    analyzer = PriceAnalyzer(n_clusters=3)
    
    # 測試分群
    cluster_result = analyzer.cluster_prices(test_data)
    print("分群結果:")
    for cluster_id, info in cluster_result['cluster_info'].items():
        print(f"  {info['name']}: {info['stats']}")
    
    # 測試決策樹
    dt_result = analyzer.train_decision_tree(test_data)
    print(f"\n決策樹 R² 分數: {dt_result['test_r2']:.3f}")
    
    # 測試預測
    predicted = analyzer.predict_price(condition=3, warranty_months=12)
    print(f"\n預測價格 (狀況3分, 保固12個月): NT$ {predicted:.0f}")

