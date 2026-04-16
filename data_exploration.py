import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('bank_dataset.csv')

# Display basic info
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df['deposit'].value_counts())

# Visualize target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='deposit', data=df)
plt.title('Deposit Subscription Distribution')
plt.savefig('deposit_distribution.png')
plt.show()

# Correlation heatmap for numerical features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(10,8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()