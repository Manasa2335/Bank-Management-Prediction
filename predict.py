import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
raw_df = pd.read_csv('bank_dataset.csv')

# Define categorical columns and preprocessing helpers
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
label_encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    raw_df[col] = encoder.fit_transform(raw_df[col])
    label_encoders[col] = encoder

raw_df['deposit'] = raw_df['deposit'].map({'yes': 1, 'no': 0})

# Split features and target
X = raw_df.drop('deposit', axis=1)
y = raw_df['deposit']

# Scale numeric features
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with class balancing
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, max_depth=15),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=15)
}

metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, pred),
        'confusion_matrix': confusion_matrix(y_test, pred),
        'classification_report': classification_report(y_test, pred, digits=4)
    }

# Choose the best model by accuracy
best_name = max(metrics, key=lambda k: metrics[k]['accuracy'])
best_model = metrics[best_name]['model']
best_metrics = metrics[best_name]

# User-friendly mapping helpers
job_aliases = {
    'admin': 'admin.', 'blue collar': 'blue-collar', 'blue-collar': 'blue-collar',
    'entrepreneur': 'entrepreneur', 'housemaid': 'housemaid', 'management': 'management',
    'retired': 'retired', 'self employed': 'self-employed', 'self-employed': 'self-employed',
    'services': 'services', 'student': 'student', 'technician': 'technician',
    'unemployed': 'unemployed', 'unknown': 'unknown'
}

# Default customer inputs for prediction
default_customer = {
    'age': 35,
    'job': 'admin.',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 50000,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'may',
    'duration': 200,
    'campaign': 1,
    'pdays': -1,
    'previous': 0,
    'poutcome': 'unknown'
}

print('-' * 28)
print('   Bank Marketing Predictor')
print('-' * 28)
print('Enter Customer Details:')
print()

age_input = input('Age (default 35): ').strip()
job_input = input('Job (default Admin): ').strip()
balance_input = input('Balance (default 50000): ').strip()
print()
print('[Predict]')
print()

input_row = default_customer.copy()

if age_input:
    try:
        input_row['age'] = int(age_input)
    except ValueError:
        print('Invalid age entered. Using default age 35.')

if job_input:
    cleaned_job = job_input.strip().lower().replace(' ', ' ')
    input_row['job'] = job_aliases.get(cleaned_job, cleaned_job)

if balance_input:
    try:
        input_row['balance'] = int(balance_input)
    except ValueError:
        print('Invalid balance entered. Using default 50000.')

print(f"Age: {input_row['age']}")
print(f"Job: {input_row['job'].replace('.', '').title()}")
print(f"Balance: {input_row['balance']}")
print()

encoded_row = {}
for col in X.columns:
    if col in categorical_cols:
        encoded_row[col] = label_encoders[col].transform([input_row[col]])[0]
    else:
        encoded_row[col] = input_row[col]

input_df = pd.DataFrame([encoded_row])
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

prediction = best_model.predict(input_df)[0]
confidence = best_model.predict_proba(input_df)[0][prediction]

print('Result:')
if prediction == 1:
    print('[YES] Customer will SUBSCRIBE')
else:
    print('[NO] Customer will NOT subscribe')
print(f'Confidence: {confidence:.0%}')
print()
print('For evaluation (behind the scenes)')
print(f'Model used: {best_name}')
print(f"Accuracy: {best_metrics['accuracy']:.2%}")
print()
print('Confusion Matrix:')
print(best_metrics['confusion_matrix'])
print()
print('Classification Report:')
print(best_metrics['classification_report'])
