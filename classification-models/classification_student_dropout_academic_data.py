import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
import seaborn as sns

df = pd.read_csv('dataset/classification_datasets/student_dropout_academic_data.csv', delimiter=';')
print(df)
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df ['Target_Encoded']= le.fit_transform(df['Target'])
print("Class labels mapping:", dict(zip(le.classes_, le.transform(le.classes_)))) 
print(df[['Target', 'Target_Encoded']].head)

# Visualize class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Target', data=df, palette='Set2')
plt.title('Distribution of Student Status (Dropout / Enrolled / Graduate)')
plt.xlabel('Target (Student Status)')
plt.ylabel('Count')
plt.show()

# Optional: Pie chart for better visual impression
target_counts = df['Target'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
plt.title('Percentage of Dropout / Enrolled / Graduate Students')
plt.show()

X = df.iloc[:, 0:36]
print(X)
y = df['Target_Encoded']
print(y)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

x_test, x_train, y_test, y_train = train_test_split(X, y, train_size=.80, random_state=42)

#randomforestclassifier

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
y_pred_rf_model = rf_model.predict(x_test)

# Convert numeric predictions back to original labels
y_test_labels = le.inverse_transform(y_test)
print(y_test_labels)
y_pred_labels = le.inverse_transform(y_pred_rf_model)
print(y_pred_labels)

# Create a comparison DataFrame
results_df = pd.DataFrame({'Actual': y_test_labels, 'Predicted': y_pred_labels})

# Plot comparison counts
plt.figure(figsize=(8,5))
sns.countplot(x='Actual', data=results_df, palette='Set2', alpha=0.7, label='Actual')
sns.countplot(x='Predicted', data=results_df, palette='Set1', alpha=0.5, label='Predicted')
plt.title('Model Prediction Results: Actual vs Predicted (Random Forest)')
plt.xlabel('Student Status')
plt.ylabel('Count')
plt.legend()
plt.show()

# Also, show how many were predicted correctly
correct_preds = (results_df['Actual'] == results_df['Predicted']).sum()
total_preds = len(results_df)
accuracy_visual = correct_preds / total_preds * 100
print(f"\nModel correctly predicted {correct_preds} out of {total_preds} students ({accuracy_visual:.2f}% accuracy).")


print("Random Forest classifier:\n", classification_report(y_test, y_pred_rf_model))

report = classification_report(y_test, y_pred_rf_model, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot classification report heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Classification Report - Random Forest Classifier")
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.show()

#logistic regression

scalar = StandardScaler()
scalar.fit(X)
X_scaled = scalar.transform(X.values)
print("These are x scaled values:\n",X_scaled)

#let make another train split for this

x_train_scaled, x_test_scaled, y_train_reg, y_test_reg = train_test_split(X_scaled, y, test_size= .2, random_state= 42)


print(f"Train size: {round(len(x_train_scaled) / len(X) * 100)}% \n\
Test size: {round(len(x_test_scaled) / len(X) * 100)}%")

logistic_regressor = LogisticRegression()

logistic_regressor.fit(x_train_scaled, y_train_reg)
logistic_regressor_predict = logistic_regressor.predict(x_test_scaled)
print(logistic_regressor_predict)
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test_reg, logistic_regressor_predict)
print("Confusion Matric report:\n",conf_mat)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_test_reg, logistic_regressor_predict)
print(acc_score)

