import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

df = pd.read_csv('dataset/car_fuel_consumption_dataset.csv')
print(df)
print(df.head())
print(df.tail())
print(df.describe().T)
print(df.shape)
print(df.info())
print(df.dtypes) #we have two object classes in which horsepower is basically numeric but with values like ? we will have to convert into numbers for better calculation
print("No of Null values in dataframe:\n\n",df.isnull().sum())
print("\nHorsepower column datatype before cleaning:\n",df['horsepower'].dtype)
#cleaning dataset
#as we have some ? values and also car name is not helpful for
#because it is string

#Replacing ? with nan in hosrsepower column

df['horsepower'] = df['horsepower'].replace('?', np.nan)

#let convert horsepower to numeric now
df['horsepower']= pd.to_numeric(df['horsepower'])

#best practice
#Filling horsepower nan values with mean
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())

#lets check now 

print("\nAfter cleaning now HorsePower columns data type is:\n",df['horsepower'].dtype)
print(df.dtypes)
print(df.describe())
print(df.info())

#Now we have one object column let drop it because we do not need
#and it is irrelevant here also 

df = df.drop('car name', axis=1)
print(df)
print(df.info())
co_relation = df.corr()
print(co_relation)
#lets plot the correlation between data
graph = sns.heatmap(co_relation, annot=True).set(title = "Heatmap of Fuel consumption data")
plt.show()

# The above heatmap shows correlation between features. mpg is 
# strongly negatively correlated with weight, displacement, 
# and horsepower (heavier cars → lower fuel efficiency).


#now our data is clean and ready for regression or training

x = df.drop('mpg', axis= 1)
y = df['mpg']

print("My Feature columns (input):", x.columns)
print("My Target Column is:\n", y)

#splitting the data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.10, random_state= 42)
print(f"Train Size: {round(len(x_train)/ len(x) * 100)}% \n\
Test Size: {round(len(x_test)/ len(x) * 100)}% ")

#lets do normalization now

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print("Mean (Train): ", x_train_scaled.mean())
print("Mean (Test): ", x_test_scaled.mean())
print(x_test_scaled)

# Always fit the scaler ONLY on x_train (to avoid data leakage); 
# x_test is transformed using the same scaler, so its mean 
# won't be exactly 0 — that's expected and correct.

#now we have perform normalization now forward to training model

ml_reg = LinearRegression()

#lets fit the model or train the model through fit

ml_reg.fit(x_train_scaled, y_train)

#showing intercept and coefficient of the feature

print("ml_reg (Multi_Linear Regression Intercept)", ml_reg.intercept_)
print("ml_reg (Multi_Linear Regression Co-efficinet)", ml_reg.coef_)

#lets put these into dataframe
feature_name = x.columns
model_coefficient = ml_reg.coef_
coefficient_df = pd.DataFrame(data = model_coefficient, index= feature_name, columns= ['coefficient Value'])

# Coefficients show how much mpg changes with a one-unit change 
# in each feature (positive = increase, negative = decrease).
print(coefficient_df)


#lets predict now that how well our model has trained
ml_reg_pred = ml_reg.predict(x_test_scaled)

result_df_ml_reg = pd.DataFrame({"Actual": y_test, "Predicted": ml_reg_pred})
print(result_df_ml_reg)



#Now lets perform different meric to analyze how well our model
#performed

mae = mean_absolute_error(y_test, ml_reg_pred)
mse = mean_squared_error(y_test, ml_reg_pred)
r2 = r2_score(y_test, ml_reg_pred)

# MAE shows average prediction error (lower is better)
# MSE penalizes larger errors more strongly (lower is better)
# R2 shows how much variance in mpg is explained 
# by our model (closer to 1 = better fit)

print(f"Mean Absolute Error: {mae : .4f}")
print(f'Mean Squared Error: {mse: .4f}')
print(f"R2 Score: {r2: .4f}")

# The multiple linear regression model achieved an R² of 0.8253, 
# indicating that about 82.5% of the variance in fuel efficiency 
# (MPG) is explained by the model. With an MAE of 2.6, the model 
# predicts MPG values with reasonable accuracy, demonstrating a 
# good overall fit.


#let see if our data has outliers just trying to see relation 
#between these two and the scatter plot show the corelation is positive
# plt.figure(figsize=(6,5))
# df.plot.scatter(x = 'horsepower', y= 'model year')
# plt.show()
# print(df[['horsepower', 'model year']].corr())


