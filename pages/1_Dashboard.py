import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dashboard")

if 'df' not in st.session_state:
    st.info("Please upload a CSV file in the sidebar to view dashboard metrics.")
    st.stop()

df = st.session_state.df

# --- KPIs ---
st.subheader("Key Performance Indicators")

total_transactions = len(df)
fraud_df = df[df['Fraud_Prediction'] == 1]
total_fraud = len(fraud_df)
fraud_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0
amount_at_risk = fraud_df['Amount'].sum() if 'Amount' in fraud_df.columns else 0

# Warn if fraud rate exceeds thresholds
if fraud_rate > 15:
    st.error(f"CRITICAL ALERT: Fraud rate is exceptionally high at {fraud_rate:.2f}%!")
elif fraud_rate > 5:
    st.warning(f"WARNING: Fraud rate is elevated at {fraud_rate:.2f}%.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total_transactions:,}", help="Total number of transactions analyzed in the uploaded batch.")
col2.metric("Total Fraud Detected", f"{total_fraud:,}", delta=f"{total_fraud} frauds", delta_color="inverse", help="Number of transactions classified as fraudulent.")
col3.metric("Fraud Rate %", f"{fraud_rate:.2f}%", help="Percentage of total transactions classified as fraud.")
col4.metric("Amount at Risk", f"${amount_at_risk:,.2f}", delta=f"-${amount_at_risk:,.2f}", delta_color="inverse", help="Total monetary value of the flagged fraudulent transactions.")

st.divider()

# --- Charts ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Fraud vs Legitimate Split")
    fig_pie = px.pie(
        names=['Legitimate', 'Fraud'], 
        values=[total_transactions - total_fraud, total_fraud],
        hole=0.4,
        color_discrete_sequence=["#00FF7F", "#FF4B4B"]
    )
    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Transaction Amount Distribution")
    if 'Amount' in df.columns:
        fig_hist = px.histogram(
            df, 
            x='Amount', 
            color='Fraud_Prediction',
            barmode='overlay',
            nbins=50,
            color_discrete_map={0: "#00FF7F", 1: "#FF4B4B"},
            labels={'Fraud_Prediction': 'Prediction (1=Fraud)'}
        )
        fig_hist.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Amount column missing.")

st.divider()

st.subheader("Fraud Count Over Time")
if 'Time' in df.columns:
    # Bin into hours
    df_temp = df.copy()
    df_temp['Hour'] = (df_temp['Time'] // 3600).astype(int)
    
    # Group by hour and count frauds
    time_series = df_temp[df_temp['Fraud_Prediction'] == 1].groupby('Hour').size().reset_index(name='Fraud Count')
    
    if not time_series.empty:
        fig_line = px.line(time_series, x='Hour', y='Fraud Count', markers=True, 
                           title="Fraudulent Transactions per Elapsed Hour",
                           color_discrete_sequence=["#FF4B4B"])
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No fraudulent transactions found to plot over time.")
else:
    st.warning("Time column missing.")
