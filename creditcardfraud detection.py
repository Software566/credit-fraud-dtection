import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model_path = "model/fraud_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Trained model not found! Please run train.py first.")
        return None

model = load_model()

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
Upload credit card transaction data as a CSV file, and the system will predict whether each transaction is fraudulent or not using a machine learning model.
""")

uploaded_file = st.file_uploader("📂 Upload CSV file (no 'Class' column)", type="csv")

if uploaded_file is not None:
    # Load uploaded CSV
    data = pd.read_csv("/Users/ashishkumar/Downloads/CROME DOWNLOAD/creditcard 2.csv")
    st.subheader("📋 Uploaded Data")
    st.dataframe(data.head())

    if model:
        # Prediction
        if st.button("🚨 Detect Fraudulent Transactions"):
            predictions = model.predict(data)
            data['Fraud'] = predictions

            fraud_count = int(data['Fraud'].sum())
            st.success(f"✅ Detected {fraud_count} fraudulent transactions.")

            st.subheader("🔍 Prediction Results")
            st.dataframe(data)

            # Download option
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv',
            )
else:
    st.warning("👈 Please upload a file to begin.")