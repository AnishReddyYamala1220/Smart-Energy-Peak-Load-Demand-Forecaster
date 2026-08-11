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
st.title("⚡ Smart Energy Peak Load Demand Forecaster")
st.markdown("Forecast hourly electricity consumption and detect critical peak load spikes to prevent grid blackouts.")

# Sidebar Configuration
st.sidebar.header("⚙️ Model Settings")

n_estimators = st.sidebar.slider("XGBoost Estimators", 100, 1000, 500, step=100)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.1], value=0.03)
max_depth = st.sidebar.slider("Tree Max Depth", 3, 10, 6)
peak_percentile = st.sidebar.slider("Peak Load Threshold Percentile", 80, 99, 95)

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
else:
    st.info("ℹ️ No file uploaded. Generating synthetic 2-year hourly dataset for demo...")
    dates = pd.date_range("2023-01-01", "2025-01-01", freq="h")
    np.random.seed(42)
    base = 30000 + 5000 * np.sin(2 * np.pi * dates.dayofyear / 365)
    daily = 8000 * np.sin(2 * np.pi * (dates.hour - 6) / 24)
    noise = np.random.normal(0, 1000, size=len(dates))
    raw_df = pd.DataFrame({'kWh': base + daily + noise}, index=dates)

# Run Forecast
with st.spinner("Training model and forecasting peak loads..."):
    model, test_results, metrics = train_and_forecast(raw_df, n_estimators, learning_rate, max_depth)

# Top Key Performance Indicators (KPIs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Root Mean Sq. Error (RMSE)", f"{metrics['RMSE']:,} kWh")
col2.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:,} kWh")
col3.metric("R² Variance Score", f"{metrics['R2']}")

threshold_val = test_results['kWh'].quantile(peak_percentile / 100.0)
peak_hours_detected = len(test_results[test_results['Predicted_kWh'] >= threshold_val])
col4.metric(f"Peak Hours (>{peak_percentile}th %ile)", f"{peak_hours_detected} hrs")

st.markdown("---")

# Main Interactive Plot
st.subheader("📈 Hourly Electricity Demand: Actual vs. Forecasted Peak Load")

days_to_view = st.slider("Select Horizon Window to View (Days):", 3, 30, 14)
sample_window = test_results.iloc[: days_to_view * 24]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(sample_window.index, sample_window['kWh'], label='Actual Demand (kWh)', color='#1f77b4', linewidth=1.5)
ax.plot(sample_window.index, sample_window['Predicted_kWh'], label='Forecasted Demand (kWh)', color='#ff7f0e', linestyle='--', linewidth=1.5)
ax.axhline(y=threshold_val, color='red', linestyle=':', label=f'Peak Alert Cutoff ({threshold_val:,.0f} kWh)')

ax.set_ylabel("Electricity Demand (kWh)")
ax.set_xlabel("Date & Hour")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig)

# Data Table & Export
st.subheader("📋 Predictions & Peak Load Alerts Table")

alerts_only = st.checkbox("Show Peak Hours Only (Demand Above Alert Cutoff)")
display_df = test_results[['kWh', 'Predicted_kWh']]

if alerts_only:
    display_df = display_df[display_df['Predicted_kWh'] >= threshold_val]

st.dataframe(display_df.style.highlight_max(axis=0, color='#ff9999'), height=300)

# CSV Download
csv_data = test_results[['kWh', 'Predicted_kWh']].to_csv().encode('utf-8')
st.download_button(
    label="📥 Download Full Forecast Report (CSV)",
    data=csv_data,
    file_name="peak_load_forecast_report.csv",
    mime="text/csv"
)