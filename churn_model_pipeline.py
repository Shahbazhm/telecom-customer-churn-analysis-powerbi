"""
Telco Customer Churn - Logistic Regression Pipeline
Companion model to the Power BI churn dashboard.
Predicts churn probability per customer using scikit-learn.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

# ---------- 1. LOAD & CLEAN ----------
original = pd.read_csv('Telco-Customer-Churn.csv')   # keep original with customerID for later
df = original.copy()
df['TotalCharges'] = df['TotalCharges'].replace(' ', '0').astype(float)
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ---------- 2. ENCODE CATEGORICALS ----------
cat_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# ---------- 3. TRAIN/TEST SPLIT + SCALE ----------
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 4. TRAIN MODEL ----------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ---------- 5. EVALUATE ----------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("=== Model Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.1f}%")
print(f"Recall:    {recall_score(y_test, y_pred)*100:.1f}%")
print(f"F1 score:  {f1_score(y_test, y_pred)*100:.1f}%")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- 6. FEATURE IMPORTANCE ----------
coefs = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_[0]})
coefs['abs_coefficient'] = coefs['coefficient'].abs()
coefs = coefs.sort_values('abs_coefficient', ascending=False)
coefs.to_csv('feature_importance.csv', index=False)

# ---------- 7. EXPORT PREDICTIONS FOR POWER BI ----------
# X_test.index still matches row positions in the original file - use it to bring CustomerID back
customer_ids = original.loc[X_test.index, 'customerID']

output = X_test.copy()
output.insert(0, 'CustomerID', customer_ids)
output['Actual_Churn'] = y_test.values
output['Churn_Probability_%'] = (y_prob * 100).round(1)
output['Risk_Tier'] = pd.cut(
    output['Churn_Probability_%'],
    bins=[-1, 40, 70, 101], labels=['Low Risk', 'Medium Risk', 'High Risk']
)
output = output.sort_values('Churn_Probability_%', ascending=False)
output.to_csv('customer_churn_predictions.csv', index=False)

print("\nExported: feature_importance.csv, customer_churn_predictions.csv")
