# utils/helpers.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def format_currency(value, currency="$"):
    """Định dạng tiền tệ thông minh"""
    if value is None or np.isnan(value):
        return "N/A"
    
    abs_value = abs(value)
    if abs_value >= 1e9:
        return f"{currency}{value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"{currency}{value/1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"{currency}{value/1e3:.2f}K"
    else:
        return f"{currency}{value:.2f}"

def format_percentage(value, decimals=2):
    """Định dạng phần trăm"""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}%"

def calculate_performance_metrics(df):
    """Tính các chỉ số hiệu suất toàn diện"""
    if df.empty or len(df) < 2:
        return {}
    
    returns = df['price_change_pct'].dropna()
    if len(returns) == 0:
        return {}
    
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    volatility = returns.std()
    sharpe_ratio = returns.mean() / volatility if volatility != 0 else 0
    
    # Max drawdown
    cumulative_returns = (1 + returns / 100).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    # Risk-adjusted return
    risk_adjusted_return = total_return / max_drawdown if max_drawdown != 0 else 0
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'risk_adjusted_return': risk_adjusted_return,
        'total_trades': len(returns),
        'avg_daily_return': returns.mean()
    }

def detect_market_regime(df, lookback=20):
    """Phát hiện market regime (Trending/Ranging)"""
    if len(df) < lookback:
        return "UNKNOWN"
    
    recent_data = df.tail(lookback)
    price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1) * 100
    volatility = recent_data['price_change_pct'].std()
    
    if abs(price_change) > 10:  # Trend threshold
        return "TRENDING_UP" if price_change > 0 else "TRENDING_DOWN"
    elif volatility < 2:  # Low volatility threshold
        return "RANGING_LOW_VOL"
    else:
        return "RANGING_HIGH_VOL"

def generate_trading_signals(df):
    """Tạo tín hiệu giao dịch tổng hợp"""
    if len(df) < 2:
        return "NEUTRAL"
    
    current = df.iloc[-1]
    signals = []
    
    # RSI Signal
    if 'RSI' in df.columns:
        if current['RSI'] > 70:
            signals.append('RSI_OVERBOUGHT')
        elif current['RSI'] < 30:
            signals.append('RSI_OVERSOLD')
    
    # MACD Signal
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        if current['MACD'] > current['MACD_signal']:
            signals.append('MACD_BULLISH')
        else:
            signals.append('MACD_BEARISH')
    
    # Moving Average Signal
    if 'SMA_7' in df.columns and 'SMA_25' in df.columns:
        if current['SMA_7'] > current['SMA_25']:
            signals.append('MA_BULLISH')
        else:
            signals.append('MA_BEARISH')
    
    # Volume Signal
    if 'volume_ratio' in df.columns:
        if current['volume_ratio'] > 1.5:
            signals.append('HIGH_VOLUME')
        elif current['volume_ratio'] < 0.5:
            signals.append('LOW_VOLUME')
    
    # Determine overall signal
    bull_signals = len([s for s in signals if 'BULLISH' in s or 'OVERSOLD' in s])
    bear_signals = len([s for s in signals if 'BEARISH' in s or 'OVERBOUGHT' in s])
    
    if bull_signals > bear_signals + 1:
        return "STRONG_BUY"
    elif bull_signals > bear_signals:
        return "BUY"
    elif bear_signals > bull_signals + 1:
        return "STRONG_SELL"
    elif bear_signals > bull_signals:
        return "SELL"
    else:
        return "NEUTRAL"

def safe_divide(numerator, denominator, default=0):
    """Phép chia an toàn, tránh division by zero"""
    if denominator == 0 or denominator is None or numerator is None:
        return default
    return numerator / denominator

def clean_column_names(df):
    """Làm sạch tên cột trong DataFrame"""
    df = df.copy()
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    return df