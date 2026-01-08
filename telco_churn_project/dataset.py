import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class TelcoDataset(Dataset):
    def __init__(self, csv_path, mode='train', random_state=42):
        """
        Args:
            csv_path (str): Path to the Kaggle CSV file.
            mode (str): 'train' or 'test' to return the appropriate split.
            random_state (int): Seed for reproducibility.
        """
        # 1. Load Data
        df = pd.read_csv(csv_path)

        # 2. Ground Objectively: Data Cleaning
        # Convert TotalCharges to numeric, turning ' ' into NaN, then dropping rows
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        
        # Remove unique ID which provides no predictive value
        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)

        # 3. Analyze Logically: Encoding Categorical Features
        # We store encoders to ensure consistency across train/test splits
        self.label_encoders = {}
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # 4. Feature/Target Separation
        X = df.drop('Churn', axis=1).values
        y = df['Churn'].values

        # 5. Systematically Split the Data
        # Stratify=y ensures both splits have the same ratio of churned customers
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        # 6. Scaling Numerical Features
        # Deep learning models converge faster when features are mean=0, std=1
        self.scaler = StandardScaler()
        
        if mode == 'train':
            self.X = torch.tensor(self.scaler.fit_transform(X_train), dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        else:
            # Re-apply the training scale to the test data to prevent leakage
            self.scaler.fit(X_train) 
            self.X = torch.tensor(self.scaler.transform(X_test), dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Returns one sample (feature vector and target label)."""
        return self.X[idx], self.y[idx]

    def get_input_dims(self):
        """Utility to help initialize the model input layer."""
        return self.X.shape[1]

# Quick verification block
if __name__ == "__main__":
    try:
        # Assuming the file is in a 'data' folder
        ds = TelcoDataset('data/WA_Fn-UseC_-Telco-Customer-Churn.csv', mode='train')
        print(f"Dataset loaded successfully!")
        print(f"Input features: {ds.get_input_dims()}")
        print(f"Sample features: {ds[0][0]}")
        print(f"Sample label: {ds[0][1]}")
    except FileNotFoundError:
        print("CSV file not found. Please ensure the file is in the 'data/' directory.")