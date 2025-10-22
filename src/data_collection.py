# src/data_collection.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional

class BinanceDataCollector:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.cache_dir = "data/raw_data"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, symbol: str, interval: str) -> str:
        """Tạo đường dẫn cache file"""
        return os.path.join(self.cache_dir, f"{symbol}_{interval}.csv")
    
    def _load_from_cache(self, cache_path: str, max_age_minutes: int = 10) -> Optional[pd.DataFrame]:
        """Tải dữ liệu từ cache nếu còn mới"""
        if os.path.exists(cache_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if datetime.now() - file_time < timedelta(minutes=max_age_minutes):
                try:
                    df = pd.read_csv(cache_path, parse_dates=['open_time', 'close_time'])
                    print(f"✅ Đã tải dữ liệu từ cache: {cache_path}")
                    return df
                except Exception as e:
                    print(f"❌ Lỗi đọc cache: {e}")
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: str):
        """Lưu dữ liệu vào cache"""
        try:
            df.to_csv(cache_path, index=False)
            print(f"💾 Đã lưu cache: {cache_path}")
        except Exception as e:
            print(f"❌ Lỗi lưu cache: {e}")
    
    def get_klines_data(self, symbol: str = 'BTCUSDT', interval: str = '1d', 
                       limit: int = 1000, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu giá từ Binance API với caching và retry mechanism"""
        
        # Kiểm tra cache
        cache_path = self._get_cache_path(symbol, interval)
        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data.tail(limit) if len(cached_data) > limit else cached_data
        
        # Retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🌐 Lần thử {attempt + 1}: Đang lấy dữ liệu {symbol} từ Binance...")
                url = f"{self.base_url}/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': min(limit, 500)  # Giảm limit để tăng success rate
                }
                
                # Tăng timeout và thêm headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=15,  # Tăng timeout
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                # Chuyển đổi thành DataFrame
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Chuyển đổi kiểu dữ liệu
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                 'quote_asset_volume', 'number_of_trades',
                                 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
                
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                # Chuyển đổi timestamp
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                
                # Thêm các cột bổ sung
                df['symbol'] = symbol
                df['interval'] = interval
                
                # Lưu cache
                if use_cache:
                    self._save_to_cache(df, cache_path)
                
                print(f"✅ THÀNH CÔNG: Đã lấy {len(df)} records THẬT cho {symbol}")
                return df.tail(limit) if len(df) > limit else df
                
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout lần {attempt + 1}, thử lại sau 2 giây...")
                time.sleep(2)
            except requests.exceptions.ConnectionError as e:
                print(f"🔌 Lỗi kết nối lần {attempt + 1}: {e}")
                time.sleep(2)
            except requests.exceptions.HTTPError as e:
                print(f"🌐 Lỗi HTTP {e.response.status_code}: {e}")
                if e.response.status_code == 429:  # Rate limit
                    print("🚫 Rate limit, đợi 5 giây...")
                    time.sleep(5)
                    continue
                break
            except Exception as e:
                print(f"❌ Lỗi không xác định: {e}")
                break
        
        print(f"🚨 KHÔNG THỂ kết nối Binance sau {max_retries} lần thử")
        
        # Thử tải từ cache cũ nếu có
        cached_data = self._load_from_cache(cache_path, max_age_minutes=1440)  # 24h
        if cached_data is not None:
            print("🔄 Sử dụng dữ liệu cache cũ")
            return cached_data.tail(limit) if len(cached_data) > limit else cached_data
        
        return None
    
    def get_multiple_symbols(self, symbols: List[str] = None, 
                           interval: str = '1d', limit: int = 500) -> Dict[str, pd.DataFrame]:
        """Lấy dữ liệu nhiều coin cùng lúc"""
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']  # Giảm số lượng
        
        all_data = {}
        
        for symbol in symbols:
            print(f"📊 Đang lấy dữ liệu THẬT cho {symbol}...")
            df = self.get_klines_data(symbol=symbol, interval=interval, limit=min(limit, 200))
            if df is not None and not df.empty:
                all_data[symbol] = df
                print(f"✅ Đã lấy dữ liệu THẬT thành công cho {symbol}")
            else:
                print(f"⚠️ Không thể lấy dữ liệu THẬT cho {symbol}")
            time.sleep(1)  # Tăng delay để tránh rate limit
            
        return all_data
    
    def get_24h_ticker(self, top_n: int = 50) -> Optional[pd.DataFrame]:
        """Lấy thông tin 24h cho các coin"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Lọc các cặp USDT
            df = pd.DataFrame(data)
            df = df[df['symbol'].str.endswith('USDT')]
            
            # Chuyển đổi kiểu dữ liệu
            numeric_cols = ['volume', 'quoteVolume', 'priceChange', 'priceChangePercent',
                          'weightedAvgPrice', 'prevClosePrice', 'lastPrice', 'lastQty',
                          'bidPrice', 'askPrice', 'openPrice', 'highPrice', 'lowPrice']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sắp xếp theo volume
            df = df.nlargest(top_n, 'quoteVolume')
            df['rank'] = range(1, len(df) + 1)
            
            return df
            
        except Exception as e:
            print(f"❌ Lỗi khi lấy ticker 24h: {e}")
            return None
    
    def get_exchange_info(self) -> Optional[Dict]:
        """Lấy thông tin về các symbol được hỗ trợ"""
        try:
            url = f"{self.base_url}/exchangeInfo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Lỗi khi lấy exchange info: {e}")
            return None