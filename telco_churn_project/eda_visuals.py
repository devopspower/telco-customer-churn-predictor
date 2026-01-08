import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional business reports
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 1. Churn Rate by Contract Type and Gender
plt.figure()
sns.barplot(data=df, x='Contract', y=(df['Churn'] == 'Yes').astype(int), hue='gender', ci=None)
plt.title('Churn Rate by Contract Type and Gender', fontsize=15)
plt.ylabel('Churn Rate (Probability)')
plt.xlabel('Contract Type')
plt.show()

# 2. Customer Churn Rate by Tenure Group
# Define logical bins for tenure
def tenure_group(months):
    if months <= 12: return '0-1 Year'
    elif months <= 24: return '1-2 Years'
    elif months <= 48: return '2-4 Years'
    else: return 'Over 4 Years'

df['TenureGroup'] = df['tenure'].apply(tenure_group)

plt.figure()
order = ['0-1 Year', '1-2 Years', '2-4 Years', 'Over 4 Years']
sns.barplot(data=df, x='TenureGroup', y=(df['Churn'] == 'Yes').astype(int), order=order, palette='viridis')
plt.title('Churn Rate by Tenure Duration', fontsize=15)
plt.ylabel('Churn Rate')
plt.show()

# 3. Churn Rate by Internet Service and Online Security
plt.figure()
sns.barplot(data=df, x='InternetService', y=(df['Churn'] == 'Yes').astype(int), hue='OnlineSecurity')
plt.title('Impact of Online Security on Churn across Internet Services', fontsize=15)
plt.ylabel('Churn Rate')
plt.show()

# 4. Most Commonly Used Payment Methods
plt.figure()
sns.countplot(data=df, y='PaymentMethod', order=df['PaymentMethod'].value_counts().index, palette='magma')
plt.title('Distribution of Customer Payment Methods', fontsize=15)
plt.xlabel('Number of Customers')
plt.show()