# src/data_processing.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

class DataProcessor:
    def __init__(self):
        self.technical_indicators = []
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính các chỉ số kỹ thuật nâng cao - KHÔNG DÙNG TA-LIB"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Đảm bảo dữ liệu được sắp xếp theo thời gian
        df = df.sort_values('open_time').reset_index(drop=True)
        
        # Chuyển đổi kiểu dữ liệu
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. MOVING AVERAGES (THAY THẾ TA-LIB)
        df['SMA_7'] = df['close'].rolling(window=7).mean()
        df['SMA_25'] = df['close'].rolling(window=25).mean()
        df['SMA_99'] = df['close'].rolling(window=99).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # 2. MOMENTUM INDICATORS (THAY THẾ TA-LIB)
        df['RSI'] = self.calculate_rsi(df['close'])
        df['RSI_7'] = self.calculate_rsi(df['close'], period=7)
        df['STOCH_K'], df['STOCH_D'] = self.calculate_stochastic(df)
        
        # 3. VOLATILITY INDICATORS (THAY THẾ TA-LIB)
        df['ATR'] = self.calculate_atr(df)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # 4. TREND INDICATORS (THAY THẾ TA-LIB)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = self.calculate_macd(df['close'])
        
        # 5. VOLUME INDICATORS (THAY THẾ TA-LIB)
        df['OBV'] = self.calculate_obv(df)
        
        # 6. CUSTOM INDICATORS
        # Price changes và returns
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility measures
        df['volatility_7d'] = df['price_change_pct'].rolling(window=7).std()
        df['volatility_30d'] = df['price_change_pct'].rolling(window=30).std()
        
        # Volume indicators
        df['volume_SMA_7'] = df['volume'].rolling(window=7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_SMA_7']
        df['volume_oscillator'] = (df['volume'] - df['volume_SMA_7']) / df['volume_SMA_7'] * 100
        
        # Support và Resistance
        df['resistance_20'] = df['high'].rolling(window=20).max()
        df['support_20'] = df['low'].rolling(window=20).min()
        df['resistance_50'] = df['high'].rolling(window=50).max()
        df['support_50'] = df['low'].rolling(window=50).min()
        
        # Price position in range
        df['price_in_range'] = (df['close'] - df['support_20']) / (df['resistance_20'] - df['support_20'])
        
        # 7. TRADING SIGNALS
        # Moving Average Crossovers
        df['MA_signal'] = np.where(df['SMA_7'] > df['SMA_25'], 1, -1)
        
        # RSI Signals
        df['RSI_signal'] = 0
        df['RSI_signal'] = np.where(df['RSI'] > 70, -1, df['RSI_signal'])  # Overbought
        df['RSI_signal'] = np.where(df['RSI'] < 30, 1, df['RSI_signal'])   # Oversold
        
        # MACD Signals
        df['MACD_signal_cross'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
        
        # Bollinger Bands Signals
        df['BB_signal'] = 0
        df['BB_signal'] = np.where(df['close'] < df['BB_lower'], 1, df['BB_signal'])   # Oversold
        df['BB_signal'] = np.where(df['close'] > df['BB_upper'], -1, df['BB_signal'])  # Overbought
        
        # Combined Signal Strength
        df['combined_signal'] = (df['MA_signal'] + df['RSI_signal'] + df['MACD_signal_cross'] + df['BB_signal'])
        
        # 8. RISK METRICS
        df['daily_var_95'] = df['log_return'].rolling(window=30).apply(
            lambda x: np.percentile(x.dropna(), 5), raw=False
        )
        
        # Xử lý giá trị NaN - sử dụng multiple methods
        df = self._handle_missing_values(df)
        
        # Ghi log các indicators đã tính
        self.technical_indicators = [col for col in df.columns if col not in [
            'open_time', 'close_time', 'symbol', 'interval', 'ignore'
        ]]
        
        print(f"✅ Đã tính {len(self.technical_indicators)} chỉ số kỹ thuật")
        return df

    # CÁC HÀM THAY THẾ TA-LIB
    def calculate_rsi(self, prices, period=14):
        """Tính RSI indicator - Thay thế TA-LIB"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Tính MACD indicator - Thay thế TA-LIB"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Tính Bollinger Bands - Thay thế TA-LIB"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        middle_band = sma
        return upper_band, middle_band, lower_band

    def calculate_atr(self, df, period=14):
        """Tính Average True Range - Thay thế TA-LIB"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = true_range.rolling(window=period).mean()
        return atr

    def calculate_stochastic(self, df, period=14, smooth_k=3, smooth_d=3):
        """Tính Stochastic Oscillator - Thay thế TA-LIB"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        stoch_k_smooth = stoch_k.rolling(window=smooth_k).mean()
        stoch_d = stoch_k_smooth.rolling(window=smooth_d).mean()
        
        return stoch_k_smooth, stoch_d

    def calculate_obv(self, df):
        """Tính On Balance Volume - Thay thế TA-LIB"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý missing values với nhiều phương pháp"""
        # Forward fill cho price data
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # Backward fill cho các giá trị còn lại
        df = df.fillna(method='bfill')
        
        # Fill remaining NaN với 0
        df = df.fillna(0)
        
        return df
    
    def create_portfolio_data(self, multiple_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Tạo dữ liệu portfolio từ nhiều coin"""
        portfolio_df = pd.DataFrame()
        
        for symbol, df in multiple_data.items():
            if df is not None and not df.empty:
                print(f"🔄 Xử lý dữ liệu cho {symbol}...")
                df_processed = self.calculate_technical_indicators(df)
                df_processed['symbol'] = symbol
                df_processed['coin_name'] = symbol.replace('USDT', '')
                portfolio_df = pd.concat([portfolio_df, df_processed], ignore_index=True)
        
        if portfolio_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Tính correlation matrix
        close_prices = portfolio_df.pivot(index='open_time', columns='symbol', values='close')
        correlation_matrix = close_prices.corr()
        
        # Tính portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(portfolio_df)
        
        return portfolio_df, correlation_matrix, portfolio_metrics
    
    def _calculate_portfolio_metrics(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Tính các chỉ số portfolio"""
        metrics = []
        
        for symbol in portfolio_df['symbol'].unique():
            symbol_data = portfolio_df[portfolio_df['symbol'] == symbol].copy()
            if len(symbol_data) < 2:
                continue
                
            returns = symbol_data['price_change_pct'].dropna()
            if len(returns) == 0:
                continue
                
            metrics.append({
                'symbol': symbol,
                'coin_name': symbol.replace('USDT', ''),
                'current_price': symbol_data['close'].iloc[-1],
                'total_return': symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1,
                'volatility': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() != 0 else 0,
                'max_drawdown': (symbol_data['close'] / symbol_data['close'].cummax() - 1).min(),
                'avg_volume': symbol_data['volume'].mean(),
                'rsi_current': symbol_data['RSI'].iloc[-1] if 'RSI' in symbol_data.columns else 0,
                'signal_strength': symbol_data['combined_signal'].iloc[-1] if 'combined_signal' in symbol_data.columns else 0
            })
        
        return pd.DataFrame(metrics)
    
    def detect_anomalies(self, df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """Phát hiện anomalies sử dụng Z-score"""
        if df.empty:
            return df
            
        df = df.copy()
        z_score = np.abs((df[column] - df[column].mean()) / df[column].std())
        df['is_anomaly'] = z_score > 3
        df['anomaly_score'] = z_score
        
        return df