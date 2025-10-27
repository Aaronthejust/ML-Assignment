import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dataset/regression_datasets/Real_Estate_Sales_2001-2022_GL-Short.csv', index_col='Serial Number')
print(df)
print("Calling different methods for quick analysis on Data Frame.")
print(df.info())
print("Data type of each column is:\n",df.dtypes)
print("Describe method for quick statistical analysis:\n",df.describe())
print("\n\nRows and columns detail through shape attribute:\n",df.shape)
print(df.head())
print(df.tail())

# Convert the 'Assessed Value' column from a 1D Series to a 2D NumPy array,
# which is the required format for scikit-learn's estimator models.
#.Values convert pandas into numpy array
#-1 takes all the possible rows data have while 1 mean take this single column which is assessed value
#reshape is important
#Most scikit-learn estimators and transformers are designed 
# to work with two-dimensional (2D) data, 
# which represents a collection of "samples" and "features".
#The expected shape is (n_samples, n_features).
#lets take assessed value and sale amount column for applying
#  linear regression 

X = df['Assessed Value'].values.reshape(-1,1)
y = df['Sale Amount'].values.reshape(-1,1)

print("X Column of our data is:\n",X)
print("Y column of our data is:\n",y)
print(X.shape)
print(y.shape)

print(df["Assessed Value"].values)
print(df["Sale Amount"].values)

#let import the model from sk learn to train our data and then test
from sklearn.model_selection import train_test_split

#taking x and y train size and random state
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.90, random_state= 72)

print(X_train)
print(y_test)
#applying linear regression on x train and y train
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

#Now training our data through fit 
regressor.fit(X_train, y_train)

#If no errors are thrown - the regressor found the best 
# fitting line! The line is defined by our features and the 
# intercept/slope. In fact, we can inspect the intercept and 
# slope by printing the regressor.intecept_ and 
# regressor.coef_ attributes, respectively:

print(regressor.intercept_)

#retrieving slope which is basically coefficient of X

print(regressor.coef_)

#as regression for x and y work on line slope formula 
# where we use slope intercept formula which y = mx + c
#here is m is the coefficient while c is the intercept

#lets define function for it

def calc (coefficient, intercept, assessed_value):
    return coefficient*assessed_value + intercept
#this is just a method where we passed one x value by ourself 
# and we got the result through the define function
score = calc(regressor.coef_, regressor.intercept_, 110500.)
print(score)  #result will be 212115.86475143

#let do it through predict method if our model work best for it
#now let find our values through predict method to predict the possible values for our test data

score = regressor.predict([[110500.]])
print(score) #212115.86475143 got same result 

score = regressor.predict([[150500.]])
print(score) 

score = regressor.predict([[217640.]])
print(score) 
#predicting y by passing our train x data set for which it will predict y
y_predict = regressor.predict(X_test)

#The y_pred variable now contains all the predicted values 
# for the input values in the X_test. We can now compare the 
# actual output values for X_test with the predicted values, 
# by arranging them side by side in a dataframe structure:


print(f"Length of y_test: {len(y_test.ravel())}")
print(f"Length of y_predict: {len(y_predict.ravel())}")

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_predict.squeeze()})
print(df_preds)

#Now lets apply some metrics from sk learn
#Here we will apply MAE MSE and MSE

#importing the required metrics from sklearn

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = root_mean_squared_error(y_test, y_predict)
mse_square_root = np.sqrt(mse)

#lets print the result for these calculation

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse: .2f}')
print(f'Square root of MSE: {mse_square_root:.2f}')

#so taking square root of mse and root mean squared error both are same








