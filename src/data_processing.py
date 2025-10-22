# src/data_processing.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
from typing import Dict, Tuple, List

class DataProcessor:
    def __init__(self):
        self.technical_indicators = []
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính các chỉ số kỹ thuật nâng cao"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Đảm bảo dữ liệu được sắp xếp theo thời gian
        df = df.sort_values('open_time').reset_index(drop=True)
        
        # Chuyển đổi kiểu dữ liệu
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. MOVING AVERAGES
        df['SMA_7'] = talib.SMA(df['close'], timeperiod=7)
        df['SMA_25'] = talib.SMA(df['close'], timeperiod=25)
        df['SMA_99'] = talib.SMA(df['close'], timeperiod=99)
        df['EMA_12'] = talib.EMA(df['close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # 2. MOMENTUM INDICATORS
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['RSI_7'] = talib.RSI(df['close'], timeperiod=7)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 3. VOLATILITY INDICATORS
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # 4. TREND INDICATORS
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 5. VOLUME INDICATORS
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
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