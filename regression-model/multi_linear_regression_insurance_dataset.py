import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
#Think of ColumnTransformer as a manager that tells 
# which preprocessing technique to apply on which columns.
from sklearn.compose import ColumnTransformer
from sklearn.metrics import max_error, explained_variance_score,mean_absolute_error,root_mean_squared_error, r2_score

df = pd.read_csv('dataset/Insurance_Charges_Dataset.csv')
print(df)
print(df.head())
print(df.describe().T)
print(df.info())
print(df.dtypes)
#print no of null values in dataframe
print(df.isnull().sum())
#print column names
print(df.columns)
#check if there is any duplicate column names
print(df.columns.duplicated())
#let print the column as list to know more details
print(df.columns.tolist())

#This line tells how many unique categories each object 
# (categorical) column has.
# It’s super helpful for deciding: Which columns to encode 
# (label or one-hot).Whether a column has too many categories 
# (which might need special handling).
print(df.select_dtypes(include='object').nunique())

#taking our x and y i.e input/feature columns and target

X = df.drop('charges', axis= 1)
y = df['charges']


# --------------------------------------------------------
# We have 3 categorical columns,
# sex (male/female) which is binary
# smoker (yes/no)   also binary
# region having 4 type values i.e multi-class
# --------------------------------------------------------

cat_cols = X.select_dtypes(include='object').columns
print("Our categorical(object) Columns are: ",cat_cols)

# --------------------------------------------------------
# Apply OneHotEncoder only to these categorical columns.
# drop='first' removes one dummy column to avoid multicollinearity
# sparse_output=False ensures output is a normal NumPy array
# remainder='passthrough' keeps numeric columns as they are
# --------------------------------------------------------

cols_transformer = ColumnTransformer(transformers= 
                                     [('encoder', OneHotEncoder
                                        (drop='first', sparse_output=False),
                                            cat_cols)], remainder='passthrough')

#In Above line
#it tells to creates an object (cols_transformer) that tells 
# scikit-learn:Apply OneHotEncoder on these specific 
# categorical columns (cat_cols), and just pass through 
# all other columns without changing them.

#now lets transform the encoded columns

X_encoded = cols_transformer.fit_transform(X)
print(X_encoded)

#----------------------------------------------------
#lets print encoded dataframe now but as X_encoded is numpy array now
# we need to convert it back to a DataFrame with proper column 
# names so we can:
# visualize correlations (heatmap)
# train ML models easily
#but first we need to get our feature name columns 
# so extrascting our feature names of column
#----------------------------------------------------

encoded_feature_names = cols_transformer.named_transformers_['encoder'].get_feature_names_out(cat_cols)

#Get numeric (non-categorical) columns that were passed through
numeric_cols = X.drop(columns= cat_cols).columns

#combine both into one numeric and categoric
all_feature_names = list(encoded_feature_names) + list(numeric_cols)

#convert back into dataframe

#--------------------------------------------------------------
# NOTE:
# You will notice that some categories like 'female' or 'no' 
# are missing from the encoded DataFrame.
# This happens because we used `drop='first'` in OneHotEncoder, 
# which intentionally drops the first
# category from each categorical column. This prevents the 
# "dummy variable trap"a situation where
# one encoded column can be perfectly predicted from the others 
# (causing multicollinearity in regression).
# 
# Example:
#   If we have 'sex_male' only, then whenever 'sex_male' = 0, it 
# implies the person is female.
#   Similarly, if 'smoker_yes' = 0, that means 'smoker_no' = 1.
# 
# So even though those dropped categories are not visible as 
# columns, their information is still
# fully represented in the encoded data. If you ever want to 
# see *all* categories (for analysis or
# visualization), use `drop=None` instead of `drop='first'`.
#---------------------------------------------------------------

X_encoded_df = pd.DataFrame(X_encoded, columns= all_feature_names)
print("My encoded dataframe is: \n",X_encoded_df)

#printing encoded dataframe column names now
print(X_encoded_df.columns.to_list())

#lets combine encoded dataframe with y so to draw a graph of
#correaltion

final_df = pd.concat([X_encoded_df, y], axis= 1)
print("Encoded DataFrame with Target columns (I.e charges) now:\n", final_df)
#lets print all column names in a list
print(final_df.columns.tolist())
correlation = final_df.corr()
print("Correlation between all columns including target:\n",correlation)

#lets draw it

plt.figure(figsize=(12,8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap incluidng Target(charges)")
plt.show()


#lets split the data now as we have cleaned data now all are numbers

x_train, x_test, y_train, y_test = train_test_split(X_encoded, y, train_size=.75, random_state= 42)

print(f"Train Size: {round(len(x_train) / len(X_encoded) * 100)}% \n\
Test Size: {round(len(x_test) / len(X_encoded) * 100)}%")

#lets scale our data
scaler = StandardScaler()
scaler.fit(x_train)
x_scaled_train = scaler.transform(x_train)
x_scaled_test = scaler.transform(x_test)
print("X_Scaled_Train data: ", x_scaled_train)
print("X_Scaled_Test data: ", x_scaled_test)

#let train our model now

multi_linear_reg = LinearRegression()
multi_linear_reg.fit(x_scaled_train, y_train)

#lets print intercept and coefficient of the feature

print("\nMultilinear Regression Intercept: ", multi_linear_reg.intercept_)
print("\nMultiLinear Regression Coefficient:\n",multi_linear_reg.coef_)

#lets put these into dataframes

feature_name = X_encoded_df.columns
model_coefficient = multi_linear_reg.coef_
coefficient_df = pd.DataFrame(data=model_coefficient, index=feature_name, columns=['Coefficient Values'])
print(coefficient_df)

#lets show this visually

#first we need to sort this dataframe by magnitude
#so large values will print first then small
coefficient_df_sorted = coefficient_df.sort_values(by='Coefficient Values', ascending=False)
print(coefficient_df_sorted) 

plt.figure(figsize=(10,6))

#--------------------------------------------------------------
#barh stands for bar horizontal
#Horizontal bars are often better when you have long or many 
# feature names (like in regression coefficients).
#coefficient_df_sorted.index these are the labels for the Y-axis
#coefficient_df_sorted['Coefficient Values'] This is the height 
# (or length) of each bar, how much each feature influences
#  the output (insurance charges).
#---------------------------------------------------------------

plt.barh(coefficient_df_sorted.index, coefficient_df_sorted['Coefficient Values'], color = 'skyblue')
plt.xlabel('Coefficinet Values')
plt.title('Feature Importance(Multi-Linear Regression Coefficients)')

#---------------------------------------------------------------
#plt.gca()
#Stands for Get Current Axes — it gives you access to the 
# current plot’s axis object (the part that holds labels, ticks, etc.)
#.invert_yaxis()
# By default, Matplotlib places the first item (top of DataFrame) 
# at the bottom of the plot.
# Calling invert_yaxis() flips the Y-axis, so your largest 
# coefficient appears on top.
# This makes interpretation more natural — biggest influence at 
# the top, smallest at the bottom.
#---------------------------------------------------------------

plt.gca().invert_yaxis #it will print largest coefficient on top
#Adds soft horizontal grid lines
#Makes comparison easier
#plt.grid Adds a grid to your chart for better readability.
#add grid on x axis with -- style not solid line and alpha is 
# Controls transparency (0 is fully transparent, 1 is fully 
# opaque).
#A lower alpha makes the grid less distracting
plt.grid(axis='x', linestyle = '--', alpha = 0.6)
plt.show()

#so the above graph shows that for one unit increase in each
#coefficient or feature how much the charges increases

#lets predict now how well our model performs

multi_linear_reg_predict = multi_linear_reg.predict(x_scaled_test)
result_df = pd.DataFrame({"Actual": y_test, "Predicted": multi_linear_reg_predict})
print(result_df)

#lets check result from metrics so we know how well our model performed
max_err = max_error(y_test, multi_linear_reg_predict)
exp_var_score = explained_variance_score(y_test, multi_linear_reg_predict)
mae = mean_absolute_error(y_test, multi_linear_reg_predict)
rmse = root_mean_squared_error(y_test,multi_linear_reg_predict)
r2 = r2_score(y_test, multi_linear_reg_predict)

# max_error shows largest single prediction mistake
# explained_variance_score shows how much of target variance model explains
# mean_absolute_error tells average absolute difference between 
# predicted and actual values
# root_mean_squared_error tells average squared error giving 
# more weight to large errors 
# r2_score tells overall goodness of fit (1 = perfect, 0 = poor)

print("Max_Error: ",max_err)
print("Explained_variance_Score: ",exp_var_score)
print("Mean_Absolute_Error: ",mae)
print("Root_Mean_Squared_Error: ",rmse)
print("R2_Score: ",r2)

# Model Evaluation Summary:
# The model performs reasonably well, explaining about 76% 
# (R² ≈ 0.77) of the variance in insurance charges.
# The average prediction error is around $4,200 (MAE), 
# with larger errors penalized up to about $5,900 (RMSE).
# Although not perfect, the model captures most patterns 
# effectively and could be improved with feature scaling, 
# polynomial terms, or interaction effects.


