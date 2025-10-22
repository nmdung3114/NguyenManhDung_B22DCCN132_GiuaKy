# models/price_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_features(self, df, target_col='close', lookback=10):
        """Chuẩn bị features cho prediction"""
        if df.empty or len(df) < lookback + 1:
            return None, None, None
            
        df = df.copy()
        
        # Tạo lag features
        for i in range(1, lookback + 1):
            df[f'close_lag_{i}'] = df[target_col].shift(i)
            df[f'volume_lag_{i}'] = df['volume'].shift(i) if 'volume' in df.columns else 0
        
        # Technical indicators as features
        feature_cols = []
        if 'RSI' in df.columns:
            feature_cols.append('RSI')
            df['RSI_lag_1'] = df['RSI'].shift(1)
            
        if 'MACD' in df.columns:
            feature_cols.append('MACD')
            
        if 'volatility_7d' in df.columns:
            feature_cols.append('volatility_7d')
        
        # Price movement features
        df['price_trend'] = df[target_col].pct_change(5)  # 5-period trend
        df['volume_trend'] = df['volume'].pct_change(5) if 'volume' in df.columns else 0
        
        # Target variable (next period price)
        df['target'] = df[target_col].shift(-1)
        
        # Feature columns
        lag_cols = [f'close_lag_{i}' for i in range(1, lookback + 1)]
        all_features = lag_cols + feature_cols + ['price_trend', 'volume_trend']
        
        # Remove rows with NaN
        df_clean = df.dropna()
        
        if df_clean.empty:
            return None, None, None
            
        X = df_clean[all_features]
        y = df_clean['target']
        
        return X, y, all_features
    
    def train(self, df, model_type='random_forest'):
        """Train prediction model"""
        X, y, features = self.prepare_features(df)
        
        if X is None or len(X) < 10:
            print("⚠️ Không đủ dữ liệu để train model")
            return False
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        else:
            self.model = LinearRegression()
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"✅ Đã train model {model_type} với {len(X)} samples")
        return True
    
    def predict(self, df, future_periods=5):
        """Dự đoán giá future"""
        if self.model is None:
            print("❌ Chưa train model")
            return None
            
        predictions = []
        current_data = df.copy()
        
        for _ in range(future_periods):
            X, _, features = self.prepare_features(current_data)
            if X is None:
                break
                
            X_scaled = self.scaler.transform(X.tail(1))
            pred_price = self.model.predict(X_scaled)[0]
            predictions.append(pred_price)
            
            # Update data với prediction
            new_row = current_data.iloc[-1:].copy()
            new_row['close'] = pred_price
            new_row['open_time'] = new_row['open_time'] + pd.Timedelta(days=1)
            current_data = pd.concat([current_data, new_row])
        
        return predictions
    
    def evaluate(self, df):
        """Đánh giá model"""
        X, y, _ = self.prepare_features(df)
        
        if X is None:
            return None
            
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate accuracy within 2%
        accuracy = np.mean(np.abs((y - y_pred) / y) < 0.02) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'Accuracy_2%': accuracy,
            'Predictions': y_pred,
            'Actual': y.values
        }