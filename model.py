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
import matplotlib
import os

# 設定中文字體
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題


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
        self.feature_columns = None  # 儲存訓練時使用的特徵欄位
    
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
            raise ValueError("沒有可用的特徵欄位")
        
        # 儲存使用的特徵欄位，供預測時使用
        self.feature_columns = available_cols.copy()
        print(f"[PriceAnalyzer] 訓練模型使用的特徵: {self.feature_columns}")
        
        X = price_data[available_cols].values
        y = price_data['price'].values
        
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
            use_iphone_features: 是否使用 iPhone 專用特徵（已棄用，現在使用訓練時的特徵）
        Returns:
            float: 預測價格
        """
        if self.dt_model is None:
            raise ValueError("決策樹模型尚未訓練，請先呼叫 train_decision_tree()")
        
        if self.feature_columns is None:
            raise ValueError("模型特徵資訊遺失，請重新訓練模型")
        
        # 根據訓練時使用的特徵來構建輸入
        # 建立特徵值字典
        feature_values = {
            'condition': condition,
            'warranty_months': warranty_months,
            'screen_broken': screen_broken,
            'camera_ok': camera_ok,
            'battery_health': battery_health,
            'storage': storage
        }
        
        # 按照訓練時的特徵順序構建輸入向量
        X = np.array([[feature_values[col] for col in self.feature_columns]])
        
        print(f"[PriceAnalyzer] 預測使用的特徵: {self.feature_columns}")
        print(f"[PriceAnalyzer] 特徵值: {[feature_values[col] for col in self.feature_columns]}")
        
        # 預測
        predicted_price = self.dt_model.predict(X)[0]
        
        return max(0, predicted_price)  # 確保價格不為負數
    
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
        
        plt.xlabel('資料索引', fontsize=12, fontweight='bold')
        plt.ylabel('價格 (NT$)', fontsize=12, fontweight='bold')
        plt.title('價格分群結果 (K-Means)', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 儲存圖片
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[PriceAnalyzer] 圖表已儲存至: {output_path}")
        
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

