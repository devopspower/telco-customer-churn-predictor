import streamlit as st
import torch
import pandas as pd
from dataset import TelcoDataset
from model import ChurnModel

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

@st.cache_resource
def load_resources():
    """Load model and dataset once to save memory."""
    dataset_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    model_path = 'best_churn_model.pth'
    
    # Load dataset for encoders/scaler reference
    ds = TelcoDataset(dataset_path, mode='train')
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChurnModel(ds.get_input_dims()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return ds, model, device

# Initialize resources
ds, model, device = load_resources()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Customer Information")

def user_input_features():
    data = {
        'gender': st.sidebar.selectbox('Gender', ('Female', 'Male')),
        'SeniorCitizen': st.sidebar.selectbox('Senior Citizen', (0, 1)),
        'Partner': st.sidebar.selectbox('Partner', ('Yes', 'No')),
        'Dependents': st.sidebar.selectbox('Dependents', ('Yes', 'No')),
        'tenure': st.sidebar.slider('Tenure (Months)', 1, 72, 12),
        'PhoneService': st.sidebar.selectbox('Phone Service', ('Yes', 'No')),
        'MultipleLines': st.sidebar.selectbox('Multiple Lines', ('No', 'Yes', 'No phone service')),
        'InternetService': st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No')),
        'OnlineSecurity': st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service')),
        'OnlineBackup': st.sidebar.selectbox('Online Backup', ('No', 'Yes', 'No internet service')),
        'DeviceProtection': st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service')),
        'TechSupport': st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service')),
        'StreamingTV': st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service')),
        'StreamingMovies': st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service')),
        'Contract': st.sidebar.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year')),
        'PaperlessBilling': st.sidebar.selectbox('Paperless Billing', ('Yes', 'No')),
        'PaymentMethod': st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')),
        'MonthlyCharges': st.sidebar.number_input('Monthly Charges', 0.0, 150.0, 70.0),
        'TotalCharges': st.sidebar.number_input('Total Charges', 0.0, 9000.0, 70.0)
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- MAIN PAGE ---
st.title("🛡️ Customer Churn Risk Assessment")
st.write("Adjust customer attributes in the sidebar to predict the probability of churn.")

# Preprocessing
processed_df = input_df.copy()
for col, le in ds.label_encoders.items():
    if col in processed_df.columns:
        processed_df[col] = le.transform(processed_df[col])

# Prediction
numerical_data = ds.scaler.transform(processed_df.values)
input_tensor = torch.tensor(numerical_data, dtype=torch.float32).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    risk_score = prob * 100

# --- DISPLAY RESULTS ---
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Churn Risk Score", value=f"{risk_score:.2f}%")
    if risk_score > 70:
        st.error("⚠️ HIGH RISK: This customer is very likely to churn.")
    elif risk_score > 40:
        st.warning("🟡 MEDIUM RISK: Monitoring recommended.")
    else:
        st.success("✅ LOW RISK: This customer is likely to stay.")

with col2:
    # Optional visual indicator
    st.progress(prob)
    st.write("Detailed Parameters:")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))