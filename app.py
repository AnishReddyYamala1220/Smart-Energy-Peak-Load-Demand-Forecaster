import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
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
st.sidebar.header("🕹️ What-If Demand Response Simulator")
enable_dr = st.sidebar.checkbox("Activate Demand Response (DR)", value=True)
dr_reduction_pct = st.sidebar.slider(
    "Peak Hour Load Reduction (%)", 
    min_value=0, 
    max_value=30, 
    value=10, 
    step=1,
    help="Simulates curtailing consumer demand during predicted high-load hours."
)
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
else:
    dates = pd.date_range("2023-01-01", "2025-01-01", freq="h")
    np.random.seed(42)
    base = 30000 + 5000 * np.sin(2 * np.pi * dates.dayofyear.values / 365)
    daily = 8000 * np.sin(2 * np.pi * (dates.hour.values - 6) / 24)
    noise = np.random.normal(0, 1000, size=len(dates))
    kWh_vals = base + daily + noise
    kWh_vals[50] += 18000   # Extreme surge
    kWh_vals[120] -= 15000  # Unusual drop
    raw_df = pd.DataFrame({'kWh': kWh_vals}, index=dates)
# Run Pipeline
with st.spinner("Training models & detecting grid anomalies..."):
    model, test_results, metrics = train_and_forecast(
        raw_df, n_estimators, learning_rate, max_depth, contamination
    )
threshold_val = test_results['kWh'].quantile(peak_percentile / 100.0)
# Demand Response Simulation Logic
test_results['DR_Simulated_kWh'] = test_results['Predicted_kWh'].copy()
if enable_dr and dr_reduction_pct > 0:
    is_peak = test_results['Predicted_kWh'] >= threshold_val
    reduction_factor = 1.0 - (dr_reduction_pct / 100.0)
    test_results.loc[is_peak, 'DR_Simulated_kWh'] = test_results.loc[is_peak, 'Predicted_kWh'] * reduction_factor
original_peak_hours = len(test_results[test_results['Predicted_kWh'] >= threshold_val])
new_peak_hours = len(test_results[test_results['DR_Simulated_kWh'] >= threshold_val])
kwh_saved = (test_results['Predicted_kWh'] - test_results['DR_Simulated_kWh']).sum()
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
if enable_dr and dr_reduction_pct > 0:
    ax.plot(sample_window.index, sample_window['DR_Simulated_kWh'], label=f'Simulated Load ({dr_reduction_pct}% DR Reduction)', color='#2ca02c', linewidth=2.0)
ax.axhline(y=threshold_val, color='grey', linestyle=':', label=f'Peak Threshold ({threshold_val:,.0f} kWh)')
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
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
# PDF Report Generator Function
def generate_pdf_report(metrics_dict, threshold, peak_hrs, saved_kwh, anomaly_cnt, df_sample):
    pdf = FPDF()
    pdf.add_page()
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Smart Energy Demand & Risk Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "Automated Executive Summary & Forecast Analytics", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    # Section 1: Executive Metrics
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Executive Summary & Performance Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f" * Model RMSE: {metrics_dict['RMSE']} kWh", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f" * Model R2 Score: {metrics_dict['R2']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f" * Peak Cutoff Threshold: {threshold:,.0f} kWh", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f" * High-Risk Peak Hours Detected: {peak_hrs} hours", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f" * Energy Saved via Demand Response: {saved_kwh:,.0f} kWh", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f" * Grid Anomaly Events Flagged: {anomaly_cnt} hours", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    # Section 2: Data Sample Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Peak Load Forecast Sample Data", new_x="LMARGIN", new_y="NEXT")
    # Table Header
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 7, "Timestamp", border=1)
    pdf.cell(40, 7, "Actual (kWh)", border=1)
    pdf.cell(40, 7, "Forecast (kWh)", border=1)
    pdf.cell(40, 7, "DR Sim (kWh)", border=1, new_x="LMARGIN", new_y="NEXT")
    # Table Rows
    pdf.set_font("Helvetica", "", 8)
    for idx, row in df_sample.head(12).iterrows():
        pdf.cell(50, 6, str(idx)[:16], border=1)
        pdf.cell(40, 6, f"{row['kWh']:,.1f}", border=1)
        pdf.cell(40, 6, f"{row['Predicted_kWh']:,.1f}", border=1)
        pdf.cell(40, 6, f"{row['DR_Simulated_kWh']:,.1f}", border=1, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
# Data Table & Export Options
st.subheader("📋 Predictions & Export Options")
display_cols = ['kWh', 'Predicted_kWh', 'DR_Simulated_kWh']
display_df = test_results[display_cols]
# Action Buttons Side-by-Side
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    csv_data = display_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Data Report (CSV)",
        data=csv_data,
        file_name="energy_forecast_data.csv",
        mime="text/csv",
        use_container_width=True
    )
with btn_col2:
    pdf_bytes = generate_pdf_report(
        metrics, threshold_val, new_peak_hours, kwh_saved, total_anomalies, display_df
    )
    st.download_button(
        label="📄 Download Executive Summary (PDF)",
        data=pdf_bytes,
        file_name="energy_forecast_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
st.dataframe(display_df.style.highlight_max(subset=['Predicted_kWh'], axis=0, color='#ff9999'), height=300)