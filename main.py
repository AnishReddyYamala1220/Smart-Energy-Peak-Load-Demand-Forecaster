import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

print("--- 1. LOADING DATA ---")
df = pd.read_csv('PJME_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)
df = df.drop_duplicates(subset=['Datetime'])

# Convert MW to kWh
df['kWh'] = df['PJME_MW'] * 1000
df.set_index('Datetime', inplace=True)
df = df.asfreq('h')
df['kWh'] = df['kWh'].interpolate(method='linear')

print(f"Dataset loaded. Total hourly records: {len(df)}")


print("\n--- 2. FEATURE ENGINEERING ---")
def create_time_features(data):
    df_feat = data.copy()
    
    # Calendar properties
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['dayofyear'] = df_feat.index.dayofyear
    df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
    
    # Cyclical Encoding (Sine/Cosine)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12.0)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12.0)
    
    # Lag & Rolling Features
    df_feat['lag_1h'] = df_feat['kWh'].shift(1)
    df_feat['lag_24h'] = df_feat['kWh'].shift(24)
    df_feat['lag_168h'] = df_feat['kWh'].shift(168) # 1 week lag
    
    df_feat['rolling_mean_6h'] = df_feat['kWh'].shift(1).rolling(window=6).mean()
    df_feat['rolling_mean_24h'] = df_feat['kWh'].shift(1).rolling(window=24).mean()
    df_feat['rolling_std_24h'] = df_feat['kWh'].shift(1).rolling(window=24).std()
    
    return df_feat.dropna()

full_df = create_time_features(df)


print("\n--- 3. CHRONOLOGICAL TRAIN/TEST SPLIT ---")
feature_cols = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
    'dayofweek', 'quarter', 'is_weekend',
    'lag_1h', 'lag_24h', 'lag_168h', 
    'rolling_mean_6h', 'rolling_mean_24h', 'rolling_std_24h'
]
target_col = 'kWh'

split_date = full_df.index[int(len(full_df) * 0.8)]
train = full_df.loc[full_df.index < split_date]
test = full_df.loc[full_df.index >= split_date]

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]

print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")


print("\n--- 4. TRAINING XGBOOST MODEL ---")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)


print("\n--- 5. EVALUATION AND PEAK DEMAND ANALYSIS ---")
test['pred_kWh'] = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, test['pred_kWh']))
mae = mean_absolute_error(y_test, test['pred_kWh'])
mape = np.mean(np.abs((y_test - test['pred_kWh']) / y_test)) * 100
r2 = r2_score(y_test, test['pred_kWh'])

print(f"RMSE : {rmse:,.2f} kWh")
print(f"MAE  : {mae:,.2f} kWh")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# Peak Threshold Identification (Top 5% highest energy usage)
peak_threshold = test['kWh'].quantile(0.95)
actual_peaks = test[test['kWh'] >= peak_threshold]
predicted_peaks = test[test['pred_kWh'] >= peak_threshold]

print(f"\nPeak Threshold (95th Percentile): {peak_threshold:,.2f} kWh")
print(f"Actual Peak Hours    : {len(actual_peaks)}")
print(f"Predicted Peak Hours : {len(predicted_peaks)}")


print("\n--- 6. VISUALIZATION AND SAVING MODEL ---")
plt.figure(figsize=(14, 6))
sample = test.iloc[:336] # 2-week window (14 * 24 hrs)
plt.plot(sample.index, sample['kWh'], label='Actual kWh Demand', color='#1f77b4', alpha=0.85)
plt.plot(sample.index, sample['pred_kWh'], label='Forecasted kWh Demand', color='#ff7f0e', linestyle='--')
plt.axhline(y=peak_threshold, color='red', linestyle=':', label='Peak Load Threshold')
plt.title('Hourly Electricity Peak Load Demand Forecast (VS Code Execution)')
plt.xlabel('Date')
plt.ylabel('Demand (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot to folder
plt.savefig('peak_load_forecast.png', dpi=300)
print("Plot saved as 'peak_load_forecast.png'")
plt.show()

# Save trained model
joblib.dump(model, 'peak_load_xgb_model.pkl')
print("Model saved as 'peak_load_xgb_model.pkl'")