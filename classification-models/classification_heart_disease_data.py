import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, accuracy_score


df = pd.read_csv('dataset/heart_disease_dataset.csv')
print(df)

#performing EDA techniques on df

#first five entries
print(df.head())

#last 5 entries of dataframe
print("\n",df.tail())

#print no of rows and column of df
print("\n",df.shape,"\n")

#print the number of records in each column, type of data, whether 
# any values are missing and how much memory the dataset uses.
print("\n",df.info())

# statistical summary of the DataFrame (Transpose)
print("\n",df.describe().T)

#printing all column names as a list
print("\n",df.columns.to_list(),"\n")

#checking if there are any null values in the column
print(df.isnull().sum,"\n")

#checking for duplicate values
#Note: If a column has 1 unique value better to drop it 
# because it has no variation for ml model
print(df.nunique(),"\n")

#returning unique values for specific column as an array
print(df['age'].unique(),"\n")

#universal analysis or analysis through chart and graphs

#lets check how many males and females we have

#As i draw graphs and i do not want to show it again and now
#so i put all graphs inside if statement and made that variable
#false because now i am training ML model and do not want that
#all graph appear on each run so to avoid code taking extra time for graphs

#Selecting all the plotting lines i wanted inside the 
# if show_graphs: block.
# Press Tab → to indent everything once
# Shift + Tab → to unindent if you added too much
#if want to show graphs just make it True

show_graphs = True
if show_graphs:
    sex_count = df['sex'].value_counts()

    plt.figure(figsize=(6,4))
    plt.bar(sex_count.index, sex_count, color = 'green')
    plt.title("Count plot of No of Male and Female")
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

    #Above figure does not show male and female on x-axis rather 
    #it shoes ranges of number for male and female lets fix this by
    #adding a line to above code and it will show 0 and 1 but still
    #we do not know what 0 is for and what 1 shows but it is because
    #our data has just 0 and 1 

    plt.figure(figsize=(6,4))
    plt.bar(sex_count.index, sex_count, color = 'red')
    plt.title("Count plot of No of Male and Female")
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.xticks(ticks=range(len(sex_count.index)), labels=sex_count.index)
    plt.show()

    #let set manually and take 0 for male and 1 for female

    plt.figure(figsize=(6,5))
    plt.bar(sex_count.index, sex_count, color = 'purple')
    plt.title("Male and Female plot chart")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.xticks(ticks=[0,1], labels=['Male', 'Female'])
    plt.show()

    #We can also do it by using seaborn
    #taking male as 0 and female as 1

    sns.countplot(x = 'sex', data=df, palette='mako')
    plt.title("Count plot of Male and Female")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()

    #now doing by showing male and female
    sns.countplot(x = df['sex'].map({0: 'Male', 1: 'Female'}))
    plt.title("Seaborn Chart of Male and Female")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()

    #lets draw kernel density plot for understanding variance in dataset
    sns.set_style("darkgrid") #Alternatives: "whitegrid", "dark", "white", "ticks"

    #num_columns = df.select_dtypes(include=["float64", "int64"]).columns

    #Selecting only numeric columns (integers and floats) from our 
    # DataFrame.Returns a list/Index of all numeric column 
    # names that are stored in num_columns

    num_columns = df.select_dtypes(include=["float64", "int64"]).columns

    #Creates one big figure that will hold all the subplots.
    # The height dynamically scales with the number of numeric columns.
    # Each feature gets roughly 3.5 inches of height.
    #import math also so that it draw the graphs accordingly with correct
    #padding and height size
    #for exact calculation of grid import math
    import math
    plt.figure(figsize=(15, math.ceil(len(num_columns) /4) * 2.5))
    for idx, feature in enumerate(num_columns, 1):
        plt.subplot(math.ceil(len(num_columns) / 4) , 4, idx)
        sns.histplot(df[feature], kde= True, color= 'purple')
        plt.title(f"{feature} | Skewness: {round(df[feature].skew(),2)}", fontsize = 10)
        plt.xlabel("")  # remove x-labels for clarity
        plt.ylabel("Count")
    plt.tight_layout(pad=4.0, h_pad=3.0)
    plt.show()

#After doing EDA lets move to apply our ML model techniques

#let first take our input(Feature) columns and target columns
#so we can take our x through iloc indexing method as columns
#are 14 so leave the last which is target and take other 13

#x = df.iloc[: , 0:13]
x = df.drop('target', axis=1)
print("Feature Columns",x.columns)

#y will be target let take it by label

y = df['target']
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
print(f"Train Size: {round(len(x_train) / len(x) * 100)}% \n\
Test Size: {round(len(x_test) / len(x) * 100)}%")

#let standardize our input data by applying z-score normalization
# transforms the data such that each feature has a mean of 
# 0 and a standard deviation of 1.
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled)  #a numpy array of x_scaled values


#let train our different models first

#lets make objects of all our model classes first

rf_clf = RandomForestClassifier() #rf for random forest and clf for classifier and vice versa
lr_clf = LogisticRegression()
xgb_clf = XGBClassifier()
dt_clf = DecisionTreeClassifier()



#training all models

rf_clf.fit(x_train_scaled, y_train)
rf_clf_pred = rf_clf.predict(x_test_scaled)

lr_clf.fit(x_train_scaled, y_train)
lr_clf_pred = lr_clf.predict(x_test_scaled)

xgb_clf.fit(x_train_scaled, y_train)
xgb_clf_pred = xgb_clf.predict(x_test_scaled)

dt_clf.fit(x_train_scaled, y_train)
dt_clf_pred = dt_clf.predict(x_test_scaled)

result_df_rf_clf = pd.DataFrame({"Actual": y_test, "Predicted": rf_clf_pred})
result_df_lr_clf = pd.DataFrame({"Actual": y_test, "Predicted": lr_clf_pred})
result_df_xgb_clf = pd.DataFrame({"Actual": y_test, "Predicted": xgb_clf_pred})
result_df_dt_clf = pd.DataFrame({"Actual": y_test, "Predicted": dt_clf_pred})

print(result_df_rf_clf)
print("\n\n", result_df_lr_clf)
print("\n\n", result_df_xgb_clf)
print("\n\n", result_df_dt_clf)

#applying matrics of sklearn classification metrics

models = {
    "Random Forest": rf_clf_pred,
    "Logistic Regression": lr_clf_pred,
    "XGBoost": xgb_clf_pred,
    "Decision Tree": dt_clf_pred
}

for name, pred in models.items():
    clas_report = classification_report(y_test, pred)
    print(f'{name}: {clas_report}')

for name, pred in models.items():
    conf_matrix = confusion_matrix(y_test, pred)
    print(f'{name}: {conf_matrix}')

for name, pred in models.items():
    prec_score = precision_score(y_test,pred)
    print(f'{name}: {prec_score}')

for name,pred in models.items():
    acc_score = accuracy_score(y_test, pred)
    print(f'{name}: {acc_score}')

#lets draw graph for each metrics

# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.title("Confusion Matrix - Logistic Regression")
# plt.xlabel("Predicted Labels")
# plt.ylabel("Actual Labels")
# plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# Assuming 'y_test' and the 'models' dictionary with predictions are already defined.
# For example:
# models = {
#     "Random Forest": rf_clf_pred,
#     "Logistic Regression": lr_clf_pred,
#     "XGBoost": xgb_clf_pred,
#     "Decision Tree": dt_clf_pred
# }
# where rf_clf_pred, lr_clf_pred, etc., are the prediction arrays.

# Determine the number of models to plot
# num_models = len(models)
# fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
# cmaps = ['Blues', 'Greens', 'Reds', 'Purples']

# # Loop through the models dictionary and plot each confusion matrix
# for i, (name, pred) in enumerate(models.items()):
    
#     # Plot the heatmap
    
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
#     axes[i].set_title(f'{name} Confusion Matrix')
#     axes[i].set_xlabel('Predicted Label')
#     if i == 0:
#         axes[i].set_ylabel('True Label')
#     else:
#         axes[i].set_ylabel('')

# plt.tight_layout()
# plt.show()






    






