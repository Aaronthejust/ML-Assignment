#======================= Sales Prediction Model =======================#
#we will build a regression model to predict Sales based on ad spending
#using pandas, numpy, seaborn, and scikit-learn

#======================= Importing Libraries ==========================#
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as wr
wr.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#======================= Load Dataset from datasets folder =================================#
#using the advertising dataset 
df = pd.read_csv("dataset/regression_datasets/advertising.csv")

print(df.head())
print(df.info())
print(df.describe().T)
print(df.isnull().sum())
print(df.shape)
print(df.dtypes)

#======================= Data Understanding ===========================#
#our dataset contains TV, Radio, Newspaper advertising budgets and Sales
#we will predict Sales using these spending variables

#======================= Visualization ================================#
show_graphs = True
if show_graphs:
    sns.pairplot(df, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', kind='scatter')
    plt.suptitle("Relationship Between Ad Spend & Sales", y=1.02)
    plt.show()

#======================= Feature & Target Split =======================#
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

print("Features:\n", X.columns.tolist())
print("Target: Sales")

#======================= Train Test Split =============================#
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train Size: {round(len(x_train)/len(X)*100)}%  Test Size: {round(len(x_test)/len(X)*100)}%')

#======================= Feature Scaling ==============================#
#Scaling helps models like Linear, Ridge, Lasso perform better
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#======================= Model Training ===============================#
lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.01)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(x_train, y_train)
ridge.fit(x_train, y_train)
lasso.fit(x_train, y_train)
rf.fit(x_train, y_train)

#======================= Predictions ==================================#
lr_pred = lr.predict(x_test)
ridge_pred = ridge.predict(x_test)
lasso_pred = lasso.predict(x_test)
rf_pred = rf.predict(x_test)

#======================= Evaluation Metrics ===========================#
models = {
    "Linear Regression": lr_pred,
    "Ridge Regression": ridge_pred,
    "Lasso Regression": lasso_pred,
    "Random Forest Regressor": rf_pred
}

for name, pred in models.items():
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"\nModel: {name}")
    print(f"R² Score: {r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

#======================= Summary & Visualization ======================#
#collecting results in dataframe for easy comparison
results = []
for name, pred in models.items():
    results.append([name,
                    r2_score(y_test, pred),
                    mean_squared_error(y_test, pred),
                    mean_absolute_error(y_test, pred)])

results_df = pd.DataFrame(results, columns=['Model', 'R2', 'MSE', 'MAE'])
print("\nOverall Model Comparison:\n")
print(results_df)

#plot results
results_df.plot(x='Model', y=['R2', 'MSE', 'MAE'], kind='bar', figsize=(10,6))
plt.title("Model Performance Comparison - Sales Prediction")
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#======================= Conclusion ===================================#
# Random Forest performed best because it captures complex relations.
# Linear, Ridge, and Lasso are great for understanding variable importance.
# this model successfully predicts sales based on ad spending features.
