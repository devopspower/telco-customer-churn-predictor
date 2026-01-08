import torch
import pandas as pd
import numpy as np
from dataset import TelcoDataset
from model import ChurnModel

def get_risk_score(customer_data, model_path, dataset_path):
    """
    Predicts the churn risk for a single customer.
    Args:
        customer_data (dict): Dictionary containing customer features.
        model_path (str): Path to 'best_churn_model.pth'.
        dataset_path (str): Path to the original CSV for scaling/encoding reference.
    """
    # 1. Ground Objectively: Load the reference dataset to match scaling and encoding
    # In a production environment, you would save the scaler and encoders as .pkl files
    ref_ds = TelcoDataset(dataset_path, mode='train')
    input_dim = ref_ds.get_input_dims()

    # 2. Analyze Logically: Initialize and Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChurnModel(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Preprocess Input Data
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # Apply the same encoding logic used in dataset.py
    # This manually maps common categories to the integers expected by the model
    # Note: Ensure these match the LabelEncoder mappings in your dataset.py
    for col, le in ref_ds.label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                # Handle unseen categories by picking a default (0)
                input_df[col] = 0

    # 4. Scale and Convert to Tensor
    numerical_data = ref_ds.scaler.transform(input_df.values)
    input_tensor = torch.tensor(numerical_data, dtype=torch.float32).to(device)

    # 5. Predict Risk Score
    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
    
    risk_score = round(probability * 100, 2)
    return risk_score

if __name__ == "__main__":
    # Example customer data (matching the Telco dataset columns)
    new_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,  # New customer
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }

    score = get_risk_score(
        customer_data=new_customer,
        model_path='best_churn_model.pth',
        dataset_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )

    print(f"\n--- Customer Risk Assessment ---")
    print(f"Churn Risk Score: {score}/100")
    
    if score > 70:
        print("Action: HIGH RISK - Immediate retention offer recommended.")
    elif score > 40:
        print("Action: MEDIUM RISK - Follow up with satisfaction survey.")
    else:
        print("Action: LOW RISK - Continue standard engagement.")