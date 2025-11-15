#Classification on Loan approval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#lets create random data first and make it dataframe
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Married': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
    'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
    'ApplicantIncome': np.random.randint(1500, 20000, n_samples),
    'CoapplicantIncome': np.random.randint(0, 10000, n_samples),
    'LoanAmount': np.random.randint(50, 700, n_samples),
    'Credit_History': np.random.choice([1.0, 0.0], n_samples, p=[0.8, 0.2]),
    'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
    'Loan_Status': np.random.choice(['Y', 'N'], n_samples, p=[0.7, 0.3])
})
print(data.head())
print(data)
print(data.columns)
print(data.columns.tolist())
print(data.info())
print(data.describe().T)
print(data.isnull().sum())
print(data.dtypes)
print(data.shape)
print(data.nunique())

#lets draw some graphs to get representation of the data
# Distribution of Loan_Status
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=data, palette='pastel')
plt.title('Loan Status Distribution (Approved vs Rejected)')
plt.show()

# Loan approval by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Loan_Status', data=data, palette='Set2')
plt.title('Loan Status by Gender')
plt.show()

#lets do feature engineering now
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['Loan_to_Income_Ratio'] = data['LoanAmount'] / data['Total_Income']

# Encode target variable
label_encoder = LabelEncoder()
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])  # Y=1, N=0

# Categorical columns to encode
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

# ColumnTransformer for OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False), cat_cols)],
    remainder='passthrough'
)

# Features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Transform categorical features
X_encoded = preprocessor.fit_transform(X)

# Get new feature names after encoding
encoded_feature_names = preprocessor.named_transformers_['encoder'].get_feature_names_out(cat_cols)
numeric_cols = X.drop(columns=cat_cols).columns
all_features = list(encoded_feature_names) + list(numeric_cols)

X_encoded_df = pd.DataFrame(X_encoded, columns=all_features)

# Feature scaling
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income', 'Loan_to_Income_Ratio']
scaler = StandardScaler()
X_encoded_df[numeric_features] = scaler.fit_transform(X_encoded_df[numeric_features])

#lets split

x_train, x_test, y_train, y_test = train_test_split(X_encoded_df, y, train_size=0.8, random_state=42)

#models we will use
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

#evaluation function to evaluate which model perform best
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Cohen Kappa': cohen_kappa_score(y_test, y_pred)
    }
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
    return metrics, cm


#training checking cross validation and evaluating
#what is stratifiedkfold
#StratifiedKFold:

#It's a method of splitting your dataset into K folds (here 5) for cross-validation.

# The "Stratified" part ensures that each fold has roughly the same proportion of each class as the original dataset.

# Example: If your dataset has 70% Approved (1) and 30% Rejected (0), each fold will also have about 70%-30%.

# This is very important for imbalanced classification problems (like loans, where most loans might be approved).

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(x_train, y_train)
    cv_scores = cross_val_score(model, X_encoded_df, y, cv=cv, scoring='accuracy')
    metrics, cm = evaluate_model(model, x_test, y_test)
    
    # Save results for comparison
    results.append({
        'Model': name,
        'CV Accuracy Mean': cv_scores.mean(),
        **metrics
    })
    
    print(f"{name} CV Accuracy scores: {np.round(cv_scores,4)} | Mean: {cv_scores.mean():.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nMetrics:")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

#summary in dataframe form of the results
results_df = pd.DataFrame(results)
print("\n\n===== Model Comparison =====")
print(results_df)

#plotting important feature we used in the models for each model
for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X_encoded_df.columns, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title(f"Top 10 Feature Importances: {name}")
        plt.show()
