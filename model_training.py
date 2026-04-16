import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib

# Load processed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Initialize models with optimized hyperparameters
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=20, min_samples_split=5, min_samples_leaf=2),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=20, min_samples_split=5, min_samples_leaf=2),
    'Extra Trees': ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=20, min_samples_split=5, min_samples_leaf=2),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=7, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, scale_pos_weight=1.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate models
best_accuracy = 0
best_model_name = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        y_pred_proba = None
        auc_score = None

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4f}")
    if auc_score is not None:
        print(f"AUC Score: {auc_score:.4f}")
    else:
        print("AUC Score: Not available for this estimator")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 60)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
    
    # Save model
    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
print("All models trained and saved.")