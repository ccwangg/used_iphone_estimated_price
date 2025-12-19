"""
資料獲取引擎
用途：根據辨識出的物品名稱，獲取拍賣網站的價格資料
邏輯流程：
1. 查詢資料庫（CSV檔案）
2. 模糊匹配（TheFuzz）
3. 觸發爬蟲（後備）
4. 自動儲存爬蟲結果
"""
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import os
import re
from urllib.parse import quote
from thefuzz import fuzz, process


class DataEngine:
    """資料獲取引擎類別"""
    
    def __init__(self, data_dir='gemma-keras-gemma_1.1_instruct_2b_en-v4', 
                 database_file='mock_data.csv', min_similarity=70):
        """
        初始化資料引擎
        Args:
            data_dir: 資料目錄路徑（Kaggle 資料集位置）
            database_file: 資料庫檔案名稱（用於儲存爬蟲結果）
            min_similarity: 模糊匹配最低相似度（0-100）
        """
        self.data_dir = data_dir
        self.database_file = database_file
        self.min_similarity = min_similarity
        
        # 設定請求 headers（模擬瀏覽器）
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # 載入資料庫（如果存在）
        self.database_df = None
        if os.path.exists(self.database_file):
            try:
                self.database_df = pd.read_csv(self.database_file, encoding='utf-8-sig')
                print(f"已載入資料庫: {self.database_file} ({len(self.database_df)} 筆資料)")
            except Exception as e:
                print(f"載入資料庫失敗: {e}")
        
        # 載入資料目錄中的 CSV 檔案
        self.data_files = self._load_data_files()
    
    def _load_data_files(self):
        """
        載入資料目錄中的所有 CSV 檔案
        Returns:
            dict: {檔案名: DataFrame}
        """
        data_files = {}
        
        if not os.path.exists(self.data_dir):
            print(f"警告: 資料目錄不存在: {self.data_dir}")
            return data_files
        
        # 搜尋所有 CSV 檔案
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(self.data_dir, file)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    data_files[file] = df
                    print(f"已載入資料檔案: {file} ({len(df)} 筆資料)")
                except Exception as e:
                    print(f"載入檔案失敗 {file}: {e}")
        
        return data_files
    
    def _search_database(self, item_name, exact_match=True):
        """
        在資料庫中搜尋物品
        Args:
            item_name: 物品名稱
            exact_match: 是否精確匹配
        Returns:
            pd.DataFrame: 匹配的資料，如果找不到則回傳空 DataFrame
        """
        if self.database_df is None or len(self.database_df) == 0:
            return pd.DataFrame()
        
        # 嘗試找到型號欄位（可能是 'model', 'item_name', 'name' 等）
        model_columns = ['model', 'item_name', 'name', 'product', 'phone_model']
        model_col = None
        
        for col in model_columns:
            if col in self.database_df.columns:
                model_col = col
                break
        
        if model_col is None:
            print("警告: 資料庫中找不到型號欄位")
            return pd.DataFrame()
        
        if exact_match:
            # 精確匹配（不區分大小寫）
            matched = self.database_df[
                self.database_df[model_col].str.lower() == item_name.lower()
            ]
        else:
            # 部分匹配
            matched = self.database_df[
                self.database_df[model_col].str.lower().str.contains(item_name.lower(), na=False)
            ]
        
        return matched.copy()
    
    def _fuzzy_match(self, item_name, data_df, model_column='model'):
        """
        使用 TheFuzz 進行模糊匹配
        Args:
            item_name: 要搜尋的物品名稱
            data_df: 資料 DataFrame
            model_column: 型號欄位名稱
        Returns:
            pd.DataFrame: 匹配的資料
        """
        if data_df is None or len(data_df) == 0 or model_column not in data_df.columns:
            return pd.DataFrame()
        
        # 取得所有唯一的型號
        unique_models = data_df[model_column].dropna().unique()
        
        if len(unique_models) == 0:
            return pd.DataFrame()
        
        # 使用 TheFuzz 找到最相似的型號
        # process.extractOne 回傳 (匹配項目, 分數, 索引)
        result = process.extractOne(
            item_name,
            unique_models,
            scorer=fuzz.ratio
        )
        
        if result is None:
            return pd.DataFrame()
        
        matched_model, similarity_score, _ = result
        
        print(f"模糊匹配結果: '{item_name}' -> '{matched_model}' (相似度: {similarity_score}%)")
        
        # 檢查相似度是否達到最低要求
        if similarity_score < self.min_similarity:
            print(f"相似度 {similarity_score}% 低於最低要求 {self.min_similarity}%")
            return pd.DataFrame()
        
        # 回傳匹配的資料
        matched = data_df[data_df[model_column] == matched_model].copy()
        return matched
    
    def _save_to_database(self, new_data):
        """
        將新資料儲存到資料庫
        Args:
            new_data: pd.DataFrame 或 list of dict，要儲存的資料
        """
        if new_data is None or len(new_data) == 0:
            return
        
        # 轉換為 DataFrame
        if isinstance(new_data, list):
            new_df = pd.DataFrame(new_data)
        else:
            new_df = new_data.copy()
        
        # 如果資料庫不存在，建立新的
        if self.database_df is None or len(self.database_df) == 0:
            self.database_df = new_df
        else:
            # 合併資料（避免重複）
            self.database_df = pd.concat([self.database_df, new_df], ignore_index=True)
            # 移除重複資料
            self.database_df = self.database_df.drop_duplicates()
        
        # 儲存到檔案
        try:
            self.database_df.to_csv(self.database_file, index=False, encoding='utf-8-sig')
            print(f"已儲存 {len(new_df)} 筆新資料到 {self.database_file}")
        except Exception as e:
            print(f"儲存資料失敗: {e}")
    
    def get_prices(self, item_name, max_records=200):
        """
        根據物品名稱獲取價格資料
        邏輯流程：
        1. 查詢資料庫（精確匹配）
        2. 查詢資料庫（部分匹配）
        3. 查詢資料目錄中的 CSV 檔案（精確匹配）
        4. 查詢資料目錄中的 CSV 檔案（模糊匹配）
        5. 觸發爬蟲（後備）
        Args:
            item_name: 物品名稱
            max_records: 最大記錄數
        Returns:
            pd.DataFrame: 包含價格等資訊的 DataFrame
        """
        print(f"\n開始查詢: {item_name}")
        
        # 步驟 1: 查詢資料庫（精確匹配）
        result = self._search_database(item_name, exact_match=True)
        if len(result) > 0:
            print(f"✓ 在資料庫中找到 {len(result)} 筆資料（精確匹配）")
            return result.head(max_records).reset_index(drop=True)
        
        # 步驟 2: 查詢資料庫（部分匹配）
        result = self._search_database(item_name, exact_match=False)
        if len(result) > 0:
            print(f"✓ 在資料庫中找到 {len(result)} 筆資料（部分匹配）")
            return result.head(max_records).reset_index(drop=True)
        
        # 步驟 3: 查詢資料目錄中的 CSV 檔案
        for file_name, df in self.data_files.items():
            # 嘗試找到型號欄位
            model_columns = ['model', 'item_name', 'name', 'product', 'phone_model']
            model_col = None
            
            for col in model_columns:
                if col in df.columns:
                    model_col = col
                    break
            
            if model_col is None:
                continue
            
            # 精確匹配
            matched = df[df[model_col].str.lower() == item_name.lower()]
            if len(matched) > 0:
                print(f"✓ 在 {file_name} 中找到 {len(matched)} 筆資料（精確匹配）")
                return matched.head(max_records).reset_index(drop=True)
            
            # 部分匹配
            matched = df[df[model_col].str.lower().str.contains(item_name.lower(), na=False)]
            if len(matched) > 0:
                print(f"✓ 在 {file_name} 中找到 {len(matched)} 筆資料（部分匹配）")
                return matched.head(max_records).reset_index(drop=True)
            
            # 模糊匹配
            matched = self._fuzzy_match(item_name, df, model_col)
            if len(matched) > 0:
                print(f"✓ 在 {file_name} 中找到 {len(matched)} 筆資料（模糊匹配）")
                return matched.head(max_records).reset_index(drop=True)
        
        # 步驟 4: 觸發爬蟲（後備）
        print(f"⚠ 資料庫和資料檔案中找不到 '{item_name}'，啟動爬蟲...")
        scraped_data = self._scrape_prices(item_name, max_records)
        
        if len(scraped_data) > 0:
            # 自動儲存爬蟲結果
            print("正在儲存爬蟲結果到資料庫...")
            self._save_to_database(scraped_data)
            return scraped_data
        else:
            print("❌ 爬蟲也無法取得資料")
            return pd.DataFrame()
    
    def get_reference_prices(self, item_name, max_items=5):
        """
        獲取參考價格資料（用於顯示給使用者）
        Args:
            item_name: 物品名稱
            max_items: 每個平台最多顯示的筆數
        Returns:
            dict: 包含蝦皮和露天的最新價格資料
        """
        references = {
            'shopee': [],
            'ruten': []
        }
        
        # 爬取蝦皮資料
        try:
            shopee_data = self._scrape_shopee(item_name, max_items)
            if shopee_data:
                references['shopee'] = shopee_data
                # 自動儲存
                self._save_to_database(shopee_data)
        except Exception as e:
            print(f"蝦皮爬蟲錯誤: {e}")
        
        # 爬取露天資料
        try:
            ruten_data = self._scrape_ruten(item_name, max_items)
            if ruten_data:
                references['ruten'] = ruten_data
                # 自動儲存
                self._save_to_database(ruten_data)
        except Exception as e:
            print(f"露天爬蟲錯誤: {e}")
        
        return references
    
    def _scrape_prices(self, item_name, max_records):
        """
        從拍賣網站爬取價格（整合蝦皮和露天）
        """
        print(f"正在爬取 {item_name} 的價格資料...")
        
        all_prices = []
        
        # 爬取蝦皮資料
        try:
            shopee_data = self._scrape_shopee(item_name, max_records // 2)
            if shopee_data:
                all_prices.extend(shopee_data)
        except Exception as e:
            print(f"蝦皮爬蟲錯誤: {e}")
        
        # 爬取露天資料
        try:
            ruten_data = self._scrape_ruten(item_name, max_records // 2)
            if ruten_data:
                all_prices.extend(ruten_data)
        except Exception as e:
            print(f"露天爬蟲錯誤: {e}")
        
        # 轉換為 DataFrame
        if len(all_prices) > 0:
            df = pd.DataFrame(all_prices)
            # 確保有必要的欄位
            required_cols = ['item_name', 'price']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = item_name if col == 'item_name' else 0
            
            return df.head(max_records).reset_index(drop=True)
        
        return pd.DataFrame()
    
    def _scrape_shopee(self, item_name, max_items=5):
        """
        爬取蝦皮拍賣價格
        """
        results = []
        
        search_query = quote(item_name)
        url = f"https://shopee.tw/search?keyword={search_query}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 蝦皮使用動態載入，這裡提供基本框架
            # 實際實作時可能需要使用 Selenium 或解析 JSON
            
            # 注意：這裡不產生模擬資料，如果無法取得就回傳空列表
            print("蝦皮爬蟲：無法解析動態內容（需要 Selenium）")
            
            time.sleep(1)  # 避免請求過快
            
        except Exception as e:
            print(f"蝦皮爬蟲發生錯誤: {e}")
        
        return results
    
    def _scrape_ruten(self, item_name, max_items=5):
        """
        爬取露天拍賣價格
        """
        results = []
        
        search_query = quote(item_name)
        url = f"https://www.ruten.com.tw/find/?q={search_query}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 尋找商品項目
            items = soup.find_all('div', class_=re.compile('item|product|goods', re.I))
            
            if not items:
                items = soup.find_all('a', href=re.compile('/item/show', re.I))
            
            count = 0
            for item in items[:max_items]:
                try:
                    # 提取價格
                    price_text = None
                    price_elem = item.find(text=re.compile(r'[\d,]+'))
                    if price_elem:
                        price_text = price_elem.strip()
                    else:
                        price_elem = item.find(class_=re.compile('price|amount', re.I))
                        if price_elem:
                            price_text = price_elem.get_text().strip()
                    
                    if price_text:
                        price_clean = re.sub(r'[^\d]', '', price_text)
                        if price_clean:
                            price = int(price_clean)
                            
                            # 提取標題
                            title_elem = item.find('a') or item.find(class_=re.compile('title|name', re.I))
                            title = title_elem.get_text().strip() if title_elem else f"{item_name} - 商品 {count+1}"
                            
                            # 提取連結
                            link_elem = item.find('a', href=True)
                            link = link_elem['href'] if link_elem else url
                            if link and not link.startswith('http'):
                                link = f"https://www.ruten.com.tw{link}"
                            
                            results.append({
                                'item_name': item_name,
                                'model': item_name,  # 用於資料庫儲存
                                'price': price,
                                'platform': '露天',
                                'title': title[:50],
                                'url': link
                            })
                            count += 1
                except Exception as e:
                    continue
                
                if count >= max_items:
                    break
            
            time.sleep(1)  # 避免請求過快
            
        except Exception as e:
            print(f"露天爬蟲發生錯誤: {e}")
        
        return results


# 測試用主程式
if __name__ == '__main__':
    engine = DataEngine()
    prices = engine.get_prices('iPhone 15 Pro')
    print(f"\n最終結果: 獲取到 {len(prices)} 筆價格資料")
    if len(prices) > 0:
        print(prices.head())
