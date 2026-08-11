import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pipeline import train_and_forecast
# Streamlit Page Setup
st.set_page_config(
    page_title="Smart Energy Peak Demand Forecaster",
    page_icon="⚡",
    layout="wide"
)
# Title & Description
st.title("⚡ Smart Energy Peak Load Forecaster & Anomaly Detector")
st.markdown("Forecast hourly electricity demand, detect peak load spikes, and identify grid load anomalies using Isolation Forests.")
# Sidebar Configuration
st.sidebar.header("⚙️ Model Settings")
n_estimators = st.sidebar.slider("XGBoost Estimators", 100, 1000, 500, step=100)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.1], value=0.03)
max_depth = st.sidebar.slider("Tree Max Depth", 3, 10, 6)
peak_percentile = st.sidebar.slider("Peak Load Threshold Percentile", 80, 99, 95)
st.sidebar.markdown("---")
st.sidebar.header("🚨 Anomaly Detection (Isolation Forest)")
contamination = st.sidebar.slider(
    "Anomaly Contamination Rate (%)", 
    min_value=1, 
    max_value=10, 
    value=3, 
    step=1,
    help="Expected percentage of anomalous consumption hours in dataset."
) / 100.0
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Hourly Energy CSV", type=["csv"])
# Data Loading Logic
@st.cache_data
def load_data(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer)
    datetime_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
    value_col = [c for c in df.columns if c != datetime_col][0]
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).drop_duplicates(subset=[datetime_col])
    df['kWh'] = df[value_col] * (1000 if 'mw' in value_col.lower() else 1)
    df.set_index(datetime_col, inplace=True)
    df = df.asfreq('h').interpolate(method='linear')
    return df
if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
else:
    dates = pd.date_range("2023-01-01", "2025-01-01", freq="h")
    np.random.seed(42)
    # Use .values to extract raw NumPy arrays so they support mutable operations
    base = 30000 + 5000 * np.sin(2 * np.pi * dates.dayofyear.values / 365)
    daily = 8000 * np.sin(2 * np.pi * (dates.hour.values - 6) / 24)
    noise = np.random.normal(0, 1000, size=len(dates))
    # Calculate initial values
    kWh_vals = base + daily + noise
    # Inject artificial anomalous spikes for demo (now works cleanly)
    kWh_vals[50] += 18000   # Extreme surge
    kWh_vals[120] -= 15000  # Unusual drop
    raw_df = pd.DataFrame({'kWh': kWh_vals}, index=dates)
# Run Pipeline
with st.spinner("Training models & detecting grid anomalies..."):
    model, test_results, metrics = train_and_forecast(
        raw_df, n_estimators, learning_rate, max_depth, contamination
    )
threshold_val = test_results['kWh'].quantile(peak_percentile / 100.0)
total_anomalies = test_results['Is_Anomaly'].sum()
# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model RMSE", f"{metrics['RMSE']:,} kWh")
col2.metric("Model R² Score", f"{metrics['R2']}")
col3.metric("Peak Threshold Value", f"{threshold_val:,.0f} kWh")
col4.metric("Grid Anomalies Detected", f"{total_anomalies} hrs", delta="Isolation Forest", delta_color="off")
st.markdown("---")
# Chart Visualization
st.subheader("📈 Load Forecast with Highlighted Anomaly Spikes")
days_to_view = st.slider("Select Horizon Window (Days):", 3, 30, 14)
sample_window = test_results.iloc[: days_to_view * 24]
anomalies_in_window = sample_window[sample_window['Is_Anomaly']]
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(sample_window.index, sample_window['kWh'], label='Actual Demand (kWh)', color='#1f77b4', linewidth=1.2, alpha=0.8)
ax.plot(sample_window.index, sample_window['Predicted_kWh'], label='Forecasted Demand', color='#ff7f0e', linestyle='--', linewidth=1.5)
ax.axhline(y=threshold_val, color='grey', linestyle=':', label=f'Peak Threshold ({threshold_val:,.0f} kWh)')
# Scatter Red Dots on Anomaly Points
if not anomalies_in_window.empty:
    ax.scatter(
        anomalies_in_window.index, 
        anomalies_in_window['kWh'], 
        color='red', 
        s=70, 
        zorder=5, 
        label=f'Detected Grid Anomaly ({len(anomalies_in_window)})'
    )
ax.set_ylabel("Demand (kWh)")
ax.set_xlabel("Date & Hour")
# Position legend outside the plot area so no lines/peaks are blocked
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
# Table showing anomalies
st.subheader("🚨 Detected Anomaly Log")
anomaly_df = test_results[test_results['Is_Anomaly']][['kWh', 'Predicted_kWh']]
st.dataframe(anomaly_df.style.highlight_max(axis=0, color='#ff9999'), height=250)