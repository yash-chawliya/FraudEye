import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile
import os

st.title("Reports & Alerts")

if 'df' not in st.session_state:
    st.info("Please upload a CSV file in the sidebar to generate reports.")
    st.stop()

df = st.session_state.df

st.subheader("High-Risk Alert Log")
alerts_df = df[df['Fraud_Probability'] > 0.8].copy()

if not alerts_df.empty:
    st.error(f"Found {len(alerts_df)} high-risk transactions (Probability > 80%).")
    # Style alerts
    st.dataframe(alerts_df.style.set_properties(**{'background-color': '#4A0000', 'color': 'white'}), use_container_width=True)
else:
    st.success("No high-risk transactions detected.")

st.divider()

st.subheader("Downloadable Summary Report")

def generate_pdf_report(dataframe):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Fraud Detection Summary Report", ln=True, align='C')
    pdf.ln(10)
    
    # Stats
    total = len(dataframe)
    fraud = len(dataframe[dataframe['Fraud_Prediction'] == 1])
    rate = (fraud/total)*100 if total > 0 else 0
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Total Transactions Analyzed: {total}", ln=True)
    pdf.cell(0, 8, f"Total Fraudulent Transactions: {fraud}", ln=True)
    pdf.cell(0, 8, f"Overall Fraud Rate: {rate:.2f}%", ln=True)
    pdf.ln(10)
    
    # Top Frauds Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top High-Risk Transactions", ln=True)
    pdf.set_font("Arial", '', 10)
    
    high_risk = dataframe[dataframe['Fraud_Probability'] > 0.8].head(10)
    if high_risk.empty:
        pdf.cell(0, 8, "No high-risk transactions found.", ln=True)
    else:
        # Just writing out simple lines for the PDF for simplicity
        for idx, row in high_risk.iterrows():
            amt = row.get('Amount', 'N/A')
            prob = row.get('Fraud_Probability', 0)
            pdf.cell(0, 8, f"ID: {idx} | Amount: ${amt} | Probability: {prob:.2%}", ln=True)
            
    # Save to temp
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    pdf.output(path)
    return path

if st.button("Generate PDF Report"):
    with st.spinner("Generating PDF..."):
        pdf_path = generate_pdf_report(df)
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
            
        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name="Fraud_Summary_Report.pdf",
            mime="application/pdf"
        )
