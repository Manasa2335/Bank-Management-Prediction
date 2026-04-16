import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# Load and preprocess data
raw_df = pd.read_csv('bank_dataset.csv')

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_cols:
    encoder = LabelEncoder()
    raw_df[col] = encoder.fit_transform(raw_df[col])

raw_df['deposit'] = raw_df['deposit'].map({'yes': 1, 'no': 0})

X = raw_df.drop('deposit', axis=1)
y = raw_df['deposit']

numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build tuned base learners
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    scale_pos_weight=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

hgb_model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

# Stacking ensemble with logistic regression meta learner
stack_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('hgb', hgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    cv=5,
    passthrough=True,
    n_jobs=-1
)

print('Training stacked model...')
stack_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = stack_model.predict(X_test)
y_pred_proba = stack_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('Test Accuracy:', round(accuracy, 4))
print('Test AUC:', round(auc, 4))
print('\nClassification Report:')
print(report)
print('\nConfusion Matrix:')
print(cm)

# Save model and scaler
joblib.dump(stack_model, 'best_final_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print('\nSaved best_final_model.pkl and scaler.pkl')
