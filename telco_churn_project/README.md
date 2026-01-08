# Telco Customer Churn Predictor

This repository contains an end-to-end deep learning pipeline designed to predict customer churn using **PyTorch**. By analyzing customer demographics, account information, and service usage, the model identifies high-risk customers, allowing businesses to act precisely with retention strategies through a live interactive dashboard.

## 🚀 Project Overview

- **Objective:** Predict binary churn (Yes/No) with high precision.
- **Model:** Multilayer Perceptron (MLP) with Batch Normalization and Dropout.
- **Performance:** Achieved **~79.74% accuracy** on the validation set.
- **Interactive Interface:** Live Streamlit web application for real-time risk scoring.

## 🛠️ Step-by-Step Process

1. **Step 1:** We defined churn as the target variable (Binary Classification) to help the sales team prioritize retention efforts.

2. **Step 2 (Data):** \* Cleaned the Kaggle Telco dataset, handling "empty string" errors in `TotalCharges`.

- Encoded categorical features using `LabelEncoder`.

- Applied `StandardScaler` to ensure numerical features (Tenure, Charges) had a mean of 0 and variance of 1.

3. **Step 3 (Architecture):** \* Designed a 3-layer MLP architecture (`64 -> 32 -> 1`).

- Implemented `BatchNorm1d` for training stability and `Dropout` to prevent overfitting.

4. **Step 4 (Training):**

- Used `BCEWithLogitsLoss` for numerical stability.

- Trained for 30 epochs, reaching a final training loss of **0.4060**.

5. **Step 5:** Monitored accuracy across epochs, starting at **78.54%** and peaking at **79.74%**.

6. **Step 6:** \* Developed an `inference.py` script for individual customer scoring.

- Deployed a **Streamlit** dashboard for non-technical stakeholders to perform "What-If" analysis.

## 📊 Business Intelligence Visualizations

The following visualizations provide context for why the model makes specific predictions:

### 1. Churn Rate by Contract Type and Gender

- **Observation:** Customers on **Month-to-Month** contracts exhibit drastically higher churn rates compared to One-year or Two-year contracts.

- **Insight:** Contract flexibility is the primary driver of attrition; gender plays a minimal role.

### 2. Customer Churn Rate by Tenure Group

- **Observation:** Churn is highest in the **0-1 Year** group and decreases significantly as tenure increases.

- **Insight:** The first 12 months are the "critical period" requiring focused onboarding.

### 3. Churn Rate by Internet Service and Online Security

- **Observation:** **Fiber Optic** users without **Online Security** are high-risk.

- **Insight:** Bundling security services with high-speed internet is a strategic retention necessity.

### 4. Most Commonly Used Payment Methods

- **Observation:** **Electronic Check** is the most frequent payment method but is historically linked to higher churn.

- **Insight:** Incentivizing a move toward **Automatic Credit Card** or **Bank Transfer** could reduce involuntary churn.

## 📁 Repository Structure

- `data/`: Contains the `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- `dataset.py`: Preprocessing and PyTorch `Dataset` class.
- `model.py`: Neural Network architecture.
- `train.py`: Logic for training and validation loops.
- `main.py`: Entry point for the full training pipeline.
- **`inference.py`**: Script to generate a risk score (0-100) for a single customer.
- **`app.py`**: Streamlit web interface for interactive predictions.
- `requirements.txt`: Environment dependencies.

## 💻 How to Run

1. **Clone the repo** and navigate to the directory.
2. **Install dependencies:**

```bash
pip install -r requirements.txt

```

3. **Train the model:**

```bash
python main.py

```

4. **Launch the Interactive Dashboard:**

```bash
streamlit run app.py

```
