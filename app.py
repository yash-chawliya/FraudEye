import streamlit as st
import pandas as pd
import joblib
import os
from src.predict import predict

# Ensure Streamlit caching is used for model and data
@st.cache_resource
def load_global_model_and_scaler(model_path='model.pkl', scaler_path='scaler.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_data
def load_and_predict_data(file_obj, _model, _scaler):
    df = pd.read_csv(file_obj)
    
    # Validate 30 columns
    expected_cols = 30
    # The uploaded file might or might not have 'Class'
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) < expected_cols:
        st.error(f"Invalid dataset. Expected at least {expected_cols} numeric features. Found {len(numeric_cols)}.")
        return None
        
    predictions, probabilities = predict(df, model_path='model.pkl', scaler_path='scaler.pkl')
    df['Fraud_Prediction'] = predictions
    df['Fraud_Probability'] = probabilities
    return df

st.set_page_config(page_title="FraudEye", layout="wide")

# Inject CSS for boxy UI
st.markdown("""
<style>
* {
    border-radius: 0px !important;
}
div[data-testid="stButton"] > button {
    border-radius: 0px !important;
}
div[data-baseweb="input"] > div {
    border-radius: 0px !important;
}
div[data-baseweb="select"] > div {
    border-radius: 0px !important;
}
div[data-testid="stMetric"] {
    border-radius: 0px !important;
}
div[data-testid="stExpander"] {
    border-radius: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar setup for global actions
st.sidebar.title("Global Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV to analyze", type=["csv"])

if uploaded_file:
    model, scaler = load_global_model_and_scaler()
    if model is None:
        st.error("Model or Scaler not found! Please run the training script first.")
        st.stop()
        
    with st.spinner("Analyzing entire dataset in the background..."):
        df = load_and_predict_data(uploaded_file, model, scaler)
        if df is not None:
            st.session_state.df = df
            st.session_state.model = model
            st.session_state.scaler = scaler

with st.sidebar.expander("About This Model"):
    st.markdown("""
    **Model Type:** XGBoost Classifier
    **Training Dataset:** Kaggle ULB Credit Card Fraud (Anonymized PCA features)
    **Disclaimer:** This is a demonstration project.
    **GitHub:** [Link placeholder]
    """)

# Define multi-page navigation
pages = {
    "FraudEye Dashboard": [
        st.Page("pages/1_Dashboard.py", title="Dashboard"),
        st.Page("pages/2_Analyze_Transactions.py", title="Analyze Transactions"),
        st.Page("pages/3_Model_Insights.py", title="Model Insights"),
        st.Page("pages/4_Reports.py", title="Reports")
    ]
}

pg = st.navigation(pages)
pg.run()
