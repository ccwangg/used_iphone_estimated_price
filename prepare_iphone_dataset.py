"""
iPhone 資料集準備腳本
用途：從 Kaggle 資料集準備 iPhone 規格與價格資料
"""
import pandas as pd
import numpy as np
import os


def prepare_iphone_dataset():
    """
    準備 iPhone 資料集
    從 Kaggle 資料集或手動建立的資料中準備訓練資料
    """
    
    # iPhone 型號與規格對照表（基於真實資料）
    iphone_specs = [
        {
            'model': 'iPhone 15 Pro Max',
            'storage': 256,
            'ram': 8,
            'screen_size': 6.7,
            'camera_mp': 48,
            'battery_mah': 4441,
            'release_year': 2023,
            'base_price': 44900,
        },
        {
            'model': 'iPhone 15 Pro',
            'storage': 256,
            'ram': 8,
            'screen_size': 6.1,
            'camera_mp': 48,
            'battery_mah': 3274,
            'release_year': 2023,
            'base_price': 36900,
        },
        {
            'model': 'iPhone 15 Plus',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.7,
            'camera_mp': 48,
            'battery_mah': 4383,
            'release_year': 2023,
            'base_price': 32900,
        },
        {
            'model': 'iPhone 15',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.1,
            'camera_mp': 48,
            'battery_mah': 3349,
            'release_year': 2023,
            'base_price': 29900,
        },
        {
            'model': 'iPhone 14 Pro Max',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.7,
            'camera_mp': 48,
            'battery_mah': 4323,
            'release_year': 2022,
            'base_price': 38900,
        },
        {
            'model': 'iPhone 14 Pro',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.1,
            'camera_mp': 48,
            'battery_mah': 3200,
            'release_year': 2022,
            'base_price': 34900,
        },
        {
            'model': 'iPhone 14',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.1,
            'camera_mp': 12,
            'battery_mah': 3279,
            'release_year': 2022,
            'base_price': 27900,
        },
        {
            'model': 'iPhone 13 Pro Max',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.7,
            'camera_mp': 12,
            'battery_mah': 4352,
            'release_year': 2021,
            'base_price': 36900,
        },
        {
            'model': 'iPhone 13 Pro',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.1,
            'camera_mp': 12,
            'battery_mah': 3095,
            'release_year': 2021,
            'base_price': 32900,
        },
        {
            'model': 'iPhone 13',
            'storage': 256,
            'ram': 4,
            'screen_size': 6.1,
            'camera_mp': 12,
            'battery_mah': 3240,
            'release_year': 2021,
            'base_price': 25900,
        },
        {
            'model': 'iPhone 12 Pro Max',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.7,
            'camera_mp': 12,
            'battery_mah': 3687,
            'release_year': 2020,
            'base_price': 33900,
        },
        {
            'model': 'iPhone 12 Pro',
            'storage': 256,
            'ram': 6,
            'screen_size': 6.1,
            'camera_mp': 12,
            'battery_mah': 2815,
            'release_year': 2020,
            'base_price': 33900,
        },
        {
            'model': 'iPhone 12',
            'storage': 256,
            'ram': 4,
            'screen_size': 6.1,
            'camera_mp': 12,
            'battery_mah': 2815,
            'release_year': 2020,
            'base_price': 26900,
        },
        {
            'model': 'iPhone 11 Pro Max',
            'storage': 256,
            'ram': 4,
            'screen_size': 6.5,
            'camera_mp': 12,
            'battery_mah': 3969,
            'release_year': 2019,
            'base_price': 35900,
        },
        {
            'model': 'iPhone 11 Pro',
            'storage': 256,
            'ram': 4,
            'screen_size': 5.8,
            'camera_mp': 12,
            'battery_mah': 3046,
            'release_year': 2019,
            'base_price': 35900,
        },
        {
            'model': 'iPhone 11',
            'storage': 256,
            'ram': 4,
            'screen_size': 6.1,
            'camera_mp': 12,
            'battery_mah': 3110,
            'release_year': 2019,
            'base_price': 24900,
        },
    ]
    
    # 產生二手價格資料
    records = []
    
    for spec in iphone_specs:
        # 為每個型號產生多筆二手價格記錄
        for _ in range(50):  # 每個型號 50 筆資料
            # 計算折舊（根據發布年份）
            years_old = 2024 - spec['release_year']
            depreciation = max(0.3, 1 - (years_old * 0.15))  # 每年折舊 15%，最低保留 30%
            
            # 基礎價格（考慮折舊）
            base_price = spec['base_price'] * depreciation
            
            # 隨機儲存容量影響（128GB, 256GB, 512GB, 1TB）
            storage_multiplier = {
                128: 0.85,
                256: 1.0,
                512: 1.3,
                1024: 1.6
            }
            storage = np.random.choice([128, 256, 512, 1024], p=[0.2, 0.5, 0.25, 0.05])
            price = base_price * storage_multiplier[storage]
            
            # 外觀狀況影響（1-5分）
            condition = np.random.randint(1, 6)
            condition_multiplier = {
                1: 0.5,   # 嚴重損壞
                2: 0.65,  # 明顯使用痕跡
                3: 0.8,   # 一般使用狀況
                4: 0.9,   # 輕微使用痕跡
                5: 1.0    # 幾乎全新
            }
            price *= condition_multiplier[condition]
            
            # 螢幕狀況影響
            screen_broken = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% 機率破裂
            if screen_broken:
                price *= 0.7  # 螢幕破裂減價 30%
            
            # 鏡頭狀況影響
            camera_ok = np.random.choice([0, 1], p=[0.15, 0.85])  # 15% 機率損壞
            if not camera_ok:
                price *= 0.85  # 鏡頭損壞減價 15%
            
            # 電池健康度影響
            battery_health = np.random.uniform(0.6, 1.0)  # 60%-100%
            price *= (0.7 + battery_health * 0.3)  # 電池影響 30% 的價格範圍
            
            # 保固剩餘月數
            warranty_months = np.random.randint(0, 25)
            if warranty_months > 0:
                price *= (1 + warranty_months * 0.01)  # 每多一個月保固加價 1%
            
            # 加入一些隨機波動
            price *= np.random.uniform(0.95, 1.05)
            
            records.append({
                'model': spec['model'],
                'storage': storage,
                'ram': spec['ram'],
                'screen_size': spec['screen_size'],
                'camera_mp': spec['camera_mp'],
                'battery_mah': spec['battery_mah'],
                'release_year': spec['release_year'],
                'condition': condition,
                'screen_broken': screen_broken,
                'camera_ok': camera_ok,
                'battery_health': round(battery_health, 2),
                'warranty_months': warranty_months,
                'price': round(price, 0)
            })
    
    # 建立 DataFrame
    df = pd.DataFrame(records)
    
    # 儲存為 CSV
    output_file = 'iphone.v1.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已建立 iPhone 資料集: {output_file}")
    print(f"總共 {len(df)} 筆資料")
    print(f"包含 {df['model'].nunique()} 種 iPhone 型號")
    print("\n資料預覽:")
    print(df.head(10))
    print("\n價格統計:")
    print(df['price'].describe())
    
    return df


if __name__ == '__main__':
    prepare_iphone_dataset()

