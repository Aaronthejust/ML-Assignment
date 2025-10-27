import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings as wr
wr.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, recall_score,f1_score,precision_score,confusion_matrix,roc_auc_score

df = pd.read_csv('dataset/breast_cancer_dataset.csv')
print(df.head())
print(df)
print(df.columns)
print(df.columns.tolist())
print(df.info())
print(df.describe().T)
print(df.isnull().sum())
print(df.dtypes)
print(df.shape)
print(df.nunique())

#we have two column that need to drop and do need in the data
#one is id and other unnamed: 32

#one column which is our target i.e diagnosis need to be in 
#numeric form because it is object type 
#so we will give Malignant 1 it is cancerious and Benign 0 which is non cancerious

df = df.drop(['id', 'Unnamed: 32'], axis=1)
print(df.columns.tolist())
print(df.shape)
#let see how our target column look like
print(df['diagnosis'])
#our target column is object lets do encoding through mapping
#we are giving Malignant binary number 1 and to Benign number 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print("After mapping or encoding Diagnosis column become:\n",df['diagnosis'])

#lets draw it
show_graphs = False
if show_graphs:
    diagnosis_type = df['diagnosis'].value_counts()
    plt.figure(figsize=(6,4))
    plt.bar(diagnosis_type.index, diagnosis_type, color = 'purple')
    plt.title("Data Count of Malignant vs Benign")
    plt.xlabel("Diagnosis Type")
    plt.ylabel("Malignant and Benign Count")
    plt.xticks(ticks=[1,0], labels=['M', 'B'])
    plt.show()

    #lets do it by seaborn

    sns.countplot(x = df['diagnosis'].map({0: 'B', 1: 'M'}))
    plt.title("Count of Malignant and Benign")
    plt.xlabel("Malignant and Benign")
    plt.ylabel('Count')
    plt.show()

#let take our feature and target column now

X = df.drop('diagnosis', axis=1)
print(X.columns.tolist())
print(X)

y = df['diagnosis']
print("Target Column:\n",y)

#splitting data

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=42)
print(f'Train Size: {round(len(x_train) / len(X) *100 )}% \n\
Test Size: {round(len(x_test) / len(X) * 100)}%')

#standarizing or doing normalization on data

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train)

#training of model and taking object for all model classes

rf_clf = RandomForestClassifier()
dt_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier()
nb_clf = GaussianNB()
lr_clf = LogisticRegression()
svm_clf = SVC()
xgb_clf = XGBClassifier()

#lets train all models

rf_clf.fit(x_train, y_train)
dt_clf.fit(x_train,y_train)
knn_clf.fit(x_train, y_train)
nb_clf.fit(x_train,y_train)
lr_clf.fit(x_train, y_train)
svm_clf.fit(x_train,y_train)
xgb_clf.fit(x_train,y_train)

#now after training lets predict and see what actual data and 
# predicted data look like
rf_clf_predict = rf_clf.predict(x_test)
dt_clf_predict = dt_clf.predict(x_test)
knn_clf_predict = knn_clf.predict(x_test)
nb_clf_predict = nb_clf.predict(x_test)
lr_clf_predict = lr_clf.predict(x_test)
svm_clf_predict = svm_clf.predict(x_test)

result_df_rf_clf = pd.DataFrame({"Actual":  y_test, "Predicted": rf_clf_predict })
result_df_dt_clf = pd.DataFrame({"Actual": y_test, "Predicted": dt_clf_predict})
result_df_knn_clf = pd.DataFrame({"Actual": y_test, "Predicted": knn_clf_predict})
result_df_nb_clf =  pd.DataFrame({"Actual" : y_test, "Predicted" : nb_clf_predict})
result_df_lr_clf =  pd.DataFrame({"Actual": y_test, "Predicted" : lr_clf_predict})
result_df_svm_clf = pd.DataFrame({"Actual" : y_test, "Predicted" : svm_clf_predict})

#lets print all dataframes

print("Random Forest Result: \n",result_df_rf_clf)
print("\nDecesion Tree Result:\n",result_df_dt_clf)
print("\nk-Nearest Neighbors (KNN) Result:\n",result_df_knn_clf)
print("\nNaive Bayes Result:\n",result_df_nb_clf)
print("\nLogistic Regression Result:\n",result_df_lr_clf)
print("\nSupport Vector Machine:\n",result_df_svm_clf)

#lets apply metrics now

models = {
    "Random Forest Classifier" : rf_clf_predict,
    "Decision Tree Classifier" : dt_clf_predict,
    "K-Nearest Neighbour Classifier" : knn_clf_predict,
    "Naive Bayes Classifier" : nb_clf_predict,
    "Logistic Regression Classifier" : lr_clf_predict,
    "Support Vector Machine Classifier" : svm_clf_predict
}

for name, pred in models.items():
    acc_score = accuracy_score(y_test, pred)
    print(f'Accuracy Score Metrics:\n{name} : {acc_score}')
    rec_score = recall_score(y_test, pred)
    print(f'Recall Score Metrics:\n{name} : {rec_score}')
    f1_sc = f1_score(y_test, pred)
    print(f'F1 Score Metrics:\n{name} : {f1_sc}')
    prec_score = precision_score(y_test, pred)
    print(f'Precision Score Metrics:\n{name} : {prec_score}')
    conf_mat = confusion_matrix(y_test, pred)
    print(f'Confusion Matrix Metrics:\n{name} : {conf_mat}')
    roc_auc_scr = roc_auc_score(y_test, pred)
    print(f'Area Under the ROC Curve (AUC) (roc_auc_score):\n{name} : {roc_auc_scr}')

# Model Evaluation Summary
# Logistic Regression and Support Vector Machine (SVM) gave the 
# best performance
# with 97.6% accuracy, 1.0 precision, and 0.9687 AUC score.
# These models show excellent ability to distinguish between 
# benign and malignant tumors
# while maintaining high reliability and minimal false positives.
# Random Forest and Naive Bayes also performed strongly 
# (~95% accuracy).
# Overall, Logistic Regression or SVM can be considered the 
# best model for this dataset.

#Extra Work fro Visualization and better understanding

#we can also plot graph and convert this data into dataframe

# Collect results for each model
results = []

for name, pred in models.items():
    acc_score = accuracy_score(y_test, pred)
    rec_score = recall_score(y_test, pred)
    f1_sc = f1_score(y_test, pred)
    prec_score = precision_score(y_test, pred)
    roc_auc_scr = roc_auc_score(y_test, pred)

    # Store all metrics in a list
    results.append([name, acc_score, prec_score, rec_score, f1_sc, roc_auc_scr])

    # Your print statements stay the same
    print(f'Accuracy Score Metrics:\n{name} : {acc_score}')
    print(f'Recall Score Metrics:\n{name} : {rec_score}')
    print(f'F1 Score Metrics:\n{name} : {f1_sc}')
    print(f'Precision Score Metrics:\n{name} : {prec_score}')
    print(f'Area Under the ROC Curve (AUC):\n{name} : {roc_auc_scr}')

# Convert the list into a DataFrame after the loop
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
print("\nOverall Model Comparison:\n")
print(results_df)

#now using this dataframe we can plot it 

results_df['Model'] = results_df['Model'].str.replace(' Classifier', '', regex=False)

results_df.plot(x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], kind='bar', figsize=(12,7))
plt.title('Model Performance Comparison')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

