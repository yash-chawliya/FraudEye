import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Analyze Transactions")

if 'df' not in st.session_state:
    st.info("Please upload a CSV file in the sidebar to analyze transactions.")
    st.stop()

df = st.session_state.df

tab1, tab2 = st.tabs(["Data Table & Filters", "Single Manual Input"])

with tab1:
    st.subheader("Filter and View Transactions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fraud_only = st.checkbox("Show only Frauds", value=False)
    with col2:
        if 'Amount' in df.columns:
            max_amt = float(df['Amount'].max()) if not df['Amount'].empty else 1000.0
            amt_range = st.slider("Amount Range", 0.0, max_amt, (0.0, max_amt))
        else:
            amt_range = (0.0, 1000.0)
    with col3:
        prob_thresh = st.slider("Min Probability % Threshold", 0.0, 1.0, 0.0)
        
    filtered_df = df.copy()
    if fraud_only:
        filtered_df = filtered_df[filtered_df['Fraud_Prediction'] == 1]
    if 'Amount' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['Amount'] >= amt_range[0]) & (filtered_df['Amount'] <= amt_range[1])]
    filtered_df = filtered_df[filtered_df['Fraud_Probability'] >= prob_thresh]
    
    st.write(f"Showing **{len(filtered_df)}** transactions.")
    
    # Styled dataframe
    def color_fraud_rows(row):
        color = '#FF4B4B' if row['Fraud_Prediction'] == 1 else '#00FF7F'
        # Apply color but keep text readable in dark mode
        return [f'color: {color}'] * len(row)
        
    st.dataframe(filtered_df.style.apply(color_fraud_rows, axis=1), use_container_width=True)
    
    # CSV Download
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Results as CSV",
        data=csv,
        file_name='filtered_fraud_predictions.csv',
        mime='text/csv',
    )

with tab2:
    st.subheader("Manual Transaction Input")
    st.write("Enter values manually to check probability of fraud.")
    
    with st.form("manual_input_form"):
        col_t, col_a = st.columns(2)
        time_val = col_t.number_input("Time (Seconds)", value=0.0)
        amount_val = col_a.number_input("Amount", value=0.0)
        
        st.write("PCA Features (V1 to V28)")
        v_cols = st.columns(4)
        v_vals = {}
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            v_vals[f'V{i}'] = v_cols[col_idx].number_input(f"V{i}", value=0.0)
            
        submitted = st.form_submit_button("Predict")
        
    if submitted:
        input_data = {'Time': [time_val]}
        for i in range(1, 29):
            input_data[f'V{i}'] = [v_vals[f'V{i}']]
        input_data['Amount'] = [amount_val]
        
        input_df = pd.DataFrame(input_data)
        from src.predict import predict
        
        try:
            preds, probs = predict(input_df, model_path='model.pkl', scaler_path='scaler.pkl')
            pred_class = preds[0]
            pred_prob = probs[0]
            
            st.markdown(f"### Result: {'Fraudulent' if pred_class == 1 else 'Legitimate'}")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#FF4B4B" if pred_prob > 0.5 else "#00FF7F"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(0, 255, 127, 0.2)"},
                        {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                    ]
                }
            ))
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
