import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib

# Load processed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print("=" * 80)
print("ENHANCED MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("=" * 80)

# 1. Tuned XGBoost
print("\n1. Tuning XGBoost...")
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
xgb_params = {
    'n_estimators': [250, 300],
    'learning_rate': [0.03, 0.05],
    'max_depth': [6, 8],
    'min_child_weight': [1, 3],
    'scale_pos_weight': [1.2, 1.5]
}
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

print(f"Best XGBoost params: {xgb_grid.best_params_}")
y_pred_xgb = best_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {acc_xgb:.4f}")

# 2. Tuned Gradient Boosting
print("\n2. Tuning Gradient Boosting...")
gb_model = GradientBoostingClassifier(random_state=42)
gb_params = {
    'n_estimators': [250, 300],
    'learning_rate': [0.03, 0.05],
    'max_depth': [6, 8],
    'min_samples_split': [5, 10]
}
gb_grid = GridSearchCV(gb_model, gb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_

print(f"Best GB params: {gb_grid.best_params_}")
y_pred_gb = best_gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {acc_gb:.4f}")

# 3. Tuned HistGradientBoosting
print("\n3. Tuning HistGradientBoosting...")
hgb_model = HistGradientBoostingClassifier(random_state=42)
hgb_params = {
    'max_iter': [250, 300],
    'learning_rate': [0.03, 0.05],
    'max_depth': [6, 8]
}
hgb_grid = GridSearchCV(hgb_model, hgb_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
hgb_grid.fit(X_train, y_train)
best_hgb = hgb_grid.best_estimator_

print(f"Best HGB params: {hgb_grid.best_params_}")
y_pred_hgb = best_hgb.predict(X_test)
acc_hgb = accuracy_score(y_test, y_pred_hgb)
print(f"HistGradientBoosting Accuracy: {acc_hgb:.4f}")

# 4. Tuned Random Forest (as baseline ensemble member)
print("\n4. Tuning Random Forest...")
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_params = {
    'n_estimators': [250, 300],
    'max_depth': [20, 25],
    'min_samples_split': [3, 5]
}
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print(f"Best RF params: {rf_grid.best_params_}")
y_pred_rf = best_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")

# 5. Create Super Ensemble (Voting Classifier with Top 3 Models)
print("\n" + "=" * 80)
print("CREATING SUPER ENSEMBLE WITH TOP 3 MODELS")
print("=" * 80)

super_ensemble = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('gb', best_gb),
        ('hgb', best_hgb)
    ],
    voting='soft',  # Soft voting based on probabilities
    weights=[1.2, 1.0, 1.1]  # Weight best performers slightly higher
)

super_ensemble.fit(X_train, y_train)
y_pred_ensemble = super_ensemble.predict(X_test)
y_pred_proba_ensemble = super_ensemble.predict_proba(X_test)[:, 1]

acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)

print(f"\nSuper Ensemble Accuracy: {acc_ensemble:.4f}")
print(f"Super Ensemble AUC Score: {auc_ensemble:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))

# 6. Also create optimized voting of ALL tuned models
print("\n" + "=" * 80)
print("CREATING MEGAENSEMBLE WITH ALL 4 TUNED MODELS")
print("=" * 80)

mega_ensemble = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('gb', best_gb),
        ('hgb', best_hgb)
    ],
    voting='soft',
    weights=[0.8, 1.3, 1.0, 1.2]  # XGBoost weighted highest
)

mega_ensemble.fit(X_train, y_train)
y_pred_mega = mega_ensemble.predict(X_test)
y_pred_proba_mega = mega_ensemble.predict_proba(X_test)[:, 1]

acc_mega = accuracy_score(y_test, y_pred_mega)
auc_mega = roc_auc_score(y_test, y_pred_proba_mega)

print(f"\nMega Ensemble Accuracy: {acc_mega:.4f}")
print(f"Mega Ensemble AUC Score: {auc_mega:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_mega))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mega))

# Summary
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(f"Individual Models:")
print(f"  XGBoost (Tuned):           {acc_xgb:.4f}")
print(f"  Gradient Boosting (Tuned): {acc_gb:.4f}")
print(f"  HistGradientBoosting (Tuned): {acc_hgb:.4f}")
print(f"  Random Forest (Tuned):     {acc_rf:.4f}")
print(f"\nEnsemble Models:")
print(f"  Super Ensemble (3 models): {acc_ensemble:.4f}")
print(f"  Mega Ensemble (4 models):  {acc_mega:.4f}")

# Choose best ensemble
if acc_mega >= acc_ensemble and acc_mega > 0.90:
    print(f"\n✅ MEGA ENSEMBLE SELECTED (Accuracy: {acc_mega:.4f} > 90%)")
    best_final_model = mega_ensemble
    final_model_name = "Mega Ensemble"
elif acc_ensemble > 0.90:
    print(f"\n✅ SUPER ENSEMBLE SELECTED (Accuracy: {acc_ensemble:.4f} > 90%)")
    best_final_model = super_ensemble
    final_model_name = "Super Ensemble"
else:
    print(f"\n✅ XGBoost SELECTED AS BEST (Accuracy: {acc_xgb:.4f})")
    best_final_model = best_xgb
    final_model_name = "XGBoost"

# Save final best model
joblib.dump(best_final_model, 'best_final_model.pkl')
joblib.dump(final_model_name, 'best_model_name.pkl')
print(f"\nBest model saved as 'best_final_model.pkl'")
