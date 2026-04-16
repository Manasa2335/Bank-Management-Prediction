import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# load dataset
file_path = 'bank_dataset.csv'
df = pd.read_csv(file_path)
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df['deposit'] = df['deposit'].map({'yes':1,'no':0})
X = df.drop('deposit', axis=1)
y = df['deposit']
num_cols = ['age','balance','day','duration','campaign','pdays','previous']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('Train', len(X_train), 'Test', len(X_test))

params = {
    'n_estimators': [300, 400],
    'learning_rate': [0.02, 0.03, 0.05],
    'max_depth': [5, 6, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3]
}
model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
search.fit(X_train, y_train)
print('best params', search.best_params_)
print('cv best', search.best_score_)
best = search.best_estimator_
y_pred = best.predict(X_test)
print('test acc', accuracy_score(y_test, y_pred))
print('test auc', roc_auc_score(y_test, best.predict_proba(X_test)[:,1]))
