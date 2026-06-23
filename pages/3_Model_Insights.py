import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score

try:
    import shap
    from streamlit_shap import st_shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.title("Model Insights")

if 'df' not in st.session_state:
    st.info("Please upload a CSV file in the sidebar to view insights.")
    st.stop()

df = st.session_state.df

has_labels = 'Class' in df.columns

if has_labels:
    st.subheader("Performance Metrics")
    
    threshold = st.slider("Operating Threshold", 0.0, 1.0, 0.5, 0.05)
    
    y_true = df['Class']
    y_prob = df['Fraud_Probability']
    y_pred = (y_prob >= threshold).astype(int)
    
    col_cm, col_curves = st.columns([1, 2])
    
    with col_cm:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Predicted Label", y="True Label", color="Count"),
                           x=['Legitimate', 'Fraud'], y=['Legitimate', 'Fraud'])
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_curves:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC = {roc_auc:.3f})", line=dict(color='orange')))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='gray'), name='Random'))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

else:
    st.info("Your uploaded dataset does not contain a 'Class' column (true labels). Therefore, performance metrics (Confusion Matrix, ROC) cannot be computed.")

st.divider()

st.subheader("Explainability (SHAP)")

if not SHAP_AVAILABLE:
    st.warning("SHAP library is not installed. Please `pip install shap streamlit-shap` to view explainability plots.")
else:
    with st.spinner("Calculating SHAP values... (this may take a moment)"):
        model = st.session_state.model
        scaler = st.session_state.scaler
        
        # Take a sample to compute SHAP to avoid freezing the app
        sample_size = min(len(df), 500)
        explain_df = df.sample(sample_size, random_state=42)
        
        # Prepare data for explainer
        features_to_drop = ['Class', 'Fraud_Prediction', 'Fraud_Probability']
        X_explain = explain_df.drop(columns=[col for col in features_to_drop if col in explain_df.columns])
        
        # Preprocess using scaler logic from predict
        if 'Amount' in X_explain.columns and 'Time' in X_explain.columns:
            from src.preprocess import preprocess_data
            X_explain_proc = preprocess_data(X_explain.copy(), fit_scaler=False, scaler_path='scaler.pkl')
            if 'Class' in X_explain_proc.columns:
                X_explain_proc = X_explain_proc.drop(columns=['Class'])
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain_proc)
            
            st.markdown("**Top Features Driving Predictions (Global)**")
            with st.expander("Show SHAP Summary Plot", expanded=True):
                # SHAP summary plot requires matplotlib under the hood for streamlit_shap, or we can use st_shap
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_values, X_explain_proc, show=False)
                # Apply dark theme to matplotlib figure if needed, but keeping it simple
                st.pyplot(fig)
            
            st.markdown("**Explain Individual Transaction**")
            tx_index = st.number_input("Select Transaction Index to Explain", 0, len(X_explain_proc)-1, 0)
            st.write(f"Explanation for transaction #{tx_index} (Fraud Prob: {explain_df.iloc[tx_index]['Fraud_Probability']:.2%}):")
            st_shap(shap.force_plot(explainer.expected_value, shap_values[tx_index,:], X_explain_proc.iloc[tx_index,:]))
        else:
            st.warning("Cannot compute SHAP values because required features are missing.")
