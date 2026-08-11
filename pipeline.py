import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def create_features(df):
    """Extracts temporal, cyclical, lag, and rolling features."""
    data = df.copy()
    
    # Time properties
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['quarter'] = data.index.quarter
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical Encoding
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12.0)
    
    # Lags & Rolling Moving Averages
    data['lag_1h'] = data['kWh'].shift(1)
    data['lag_24h'] = data['kWh'].shift(24)
    data['lag_168h'] = data['kWh'].shift(168)
    data['rolling_mean_6h'] = data['kWh'].shift(1).rolling(window=6).mean()
    data['rolling_mean_24h'] = data['kWh'].shift(1).rolling(window=24).mean()
    data['rolling_std_24h'] = data['kWh'].shift(1).rolling(window=24).std()
    
    return data.dropna()

def train_and_forecast(df, n_estimators=500, learning_rate=0.03, max_depth=6):
    """Trains XGBoost model and calculates metrics."""
    full_df = create_features(df)
    
    feature_cols = [
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
        'dayofweek', 'quarter', 'is_weekend',
        'lag_1h', 'lag_24h', 'lag_168h', 
        'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h'
    ]
    target_col = 'kWh'

    # Chronological Split (80% Train, 20% Test)
    split_date = full_df.index[int(len(full_df) * 0.8)]
    train = full_df.loc[full_df.index < split_date]
    test = full_df.loc[full_df.index >= split_date].copy()

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    
    test['Predicted_kWh'] = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, test['Predicted_kWh']))
    mae = mean_absolute_error(y_test, test['Predicted_kWh'])
    r2 = r2_score(y_test, test['Predicted_kWh'])
    
    metrics = {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'R2': round(r2, 4)
    }
    
    return model, test, metrics