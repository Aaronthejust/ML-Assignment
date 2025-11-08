# California Housing Regression Exercise (15+ Features)

#lets import libraries and sk learn 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error


#Lets load the data as it is sklearn built in dataset

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()   # features + target

print(df.head())
print(df.describe().T)
print(df.info())
print(df.dtypes)

#lets create raw features
# Assignment mentions raw features: MedInc, HouseAge, TotalRooms, TotalBedrooms,
# Population, Households, Latitude, Longitude raw and ocean_proximity which is categorical so 8 raw 1 categ.
# sklearn fetch provides AveRooms, AveBedrms, AveOccup, Population, MedInc, HouseAge, Latitude, Longitude.
# We'll compute TotalRooms, TotalBedrooms, Households from available averages
# Households = Population / AveOccup
# TotalRooms = AveRooms * Households
# TotalBedrooms = AveBedrms * Households

# to avoid divide by zero 
#as data has not total rooms or bedroom it has average
#so first we will find total
print("\nMinimum AveOccup:", df['AveOccup'].min())
# Compute Households and totals
df['Households'] = df['Population'] / df['AveOccup']
df['TotalRooms'] = df['AveRooms'] * df['Households']
df['TotalBedrooms'] = df['AveBedrms'] * df['Households']

# created columns now
print("\nCreated totals and households (head):\n", df[['Households','TotalRooms','TotalBedrooms']].head())


#lets handle missing values now

#In word file we are given to Impute small number of missing values in total_bedrooms
# here this dataset typically has no missing values, but we'll check and impute median.
print("\nNull counts before imputation:\n", df[['TotalBedrooms']].isnull().sum())

# If any missing, fill with median as assignment suggested
if df['TotalBedrooms'].isnull().sum() > 0:
    median_tb = df['TotalBedrooms'].median()
    df['TotalBedrooms'] = df['TotalBedrooms'].fillna(median_tb)
    print("Filled TotalBedrooms NaNs with median:", median_tb)
else:
    print("No missing values in TotalBedrooms. No imputation required.")


#lets do feature engineering and derving 3 featured column now
#As this formula is given in assignment using that
# Rooms_per_Household = TotalRooms / Households
# Bedrooms_per_Room = TotalBedrooms / TotalRooms
# Population_per_Household = Population / Households

#also doing the techinique to avoid 0 division 
df['Rooms_per_Household'] = np.where(df['Households'] == 0, 0, df['TotalRooms'] / df['Households'])
df['Bedrooms_per_Room'] = np.where(df['TotalRooms'] == 0, 0, df['TotalBedrooms'] / df['TotalRooms'])
df['Population_per_Household'] = np.where(df['Households'] == 0, 0, df['Population'] / df['Households'])

print("\nEngineered features head:\n", df[['Rooms_per_Household','Bedrooms_per_Room','Population_per_Household']].head())


# now creating ocean proximity and performing one hot encoding

# Note: sklearn's fetch_california_housing does NOT include ocean_proximity.
# Assignment requires ocean_proximity one-hot encoding 5 categories.
# We'll create a proxy 'ocean_proximity' by clustering Latitude & Longitude into 5 clusters.
#this step has been done by seaching and analysig from net
coords = df[['Latitude','Longitude']].copy()
kmeans = KMeans(n_clusters=5, random_state=42)
df['ocean_proximity'] = kmeans.fit_predict(coords).astype(str)  # categorical string labels

print("\nProxy ocean_proximity categories (clusters):", df['ocean_proximity'].unique())

# One-hot encode ocean_proximity into 5 binary columns drop first=True to get 4 dummies
ocean_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean_prox', drop_first=True)
df = pd.concat([df, ocean_dummies], axis=1)

print("\nOne-hot columns created:", ocean_dummies.columns.tolist())


#so final list of 15 features

# Raw numerical 8: MedInc, HouseAge, TotalRooms, TotalBedrooms, Population, Households, Latitude, Longitude
# Engineered 3: Rooms_per_Household, Bedrooms_per_Room, Population_per_Household
# Encoded 4: ocean_prox_* (due to drop_first True)
raw_numerical = ['MedInc','HouseAge','TotalRooms','TotalBedrooms','Population','Households','Latitude','Longitude']
engineered = ['Rooms_per_Household','Bedrooms_per_Room','Population_per_Household']
encoded = ocean_dummies.columns.tolist()   # ~4 columns because drop_first=True

#adding all to get 15
features = raw_numerical + engineered + encoded
print("\nTotal feature count should be 15:", len(features))
print("Feature list:\n", features)

# target
target = 'MedHouseVal'   # sklearn naming
print("\nTarget column:", target)


#checking for null if any in features as given in the assignment

print("\nNulls in selected features:\n", df[features].isnull().sum())
# If any numeric NaNs appear, fill with median for that column
for col in features:
    if df[col].isnull().sum() > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"Filled NaN in {col} with median: {med}")


#standardizing or scaling the feature column apart from target
# Scaling all approx 14 numerical features encoded dummies are binary and can be left as is
numerical_to_scale = raw_numerical + engineered   #11 numeric columns
print("\nNumerical columns to scale:", numerical_to_scale)

scaler = StandardScaler()
df_scaled_numeric = pd.DataFrame(scaler.fit_transform(df[numerical_to_scale]), columns=numerical_to_scale, index=df.index)

# combine scaled numeric + encoded dummies
df_model = pd.concat([df_scaled_numeric, df[encoded], df[target]], axis=1)
print("\nPrepared dataset for modeling (head):\n", df_model.head())
print("\nPrepared dataset shape:", df_model.shape)


#splitting the data now
X = df_model.drop(columns=[target])
y = df_model[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
print(f"\nTrain Size: {round(len(x_train) / len(X) * 100)}%")
print(f"Test Size: {round(len(x_test) / len(X) * 100)}%")


#Training Models now
# Models: Random Forest, XGBoost, SVR given in the assignment
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# Fit models
print("\nTraining Random Forest...")
rf.fit(x_train, y_train)
print("Random Forest trained.")

print("\nTraining XGBoost...")
xgb.fit(x_train, y_train)
print("XGBoost trained.")

print("\nTraining SVR...")
svr.fit(x_train, y_train)
print("SVR trained.")


#doing Cross Validation i.e CV which is
# CV means: split training data into multiple parts (folds), train on some parts and validate on the hold-out part,
# repeat so each part gets to be validation once. This checks how stable the model is across different subsets.
# We'll use 5-fold CV and compute RMSE for each fold (lower RMSE is better).
def rmse_cv(model, X, y, cv=5):
    # cross_val_score with neg_mean_squared_error returns negative MSE scores
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

print("\nPerforming 5-Fold CV for Random Forest...")
rf_cv_rmse = rmse_cv(rf, x_train, y_train, cv=5)
print("Random Forest CV RMSE scores:", np.round(rf_cv_rmse,4))
print("Mean CV RMSE:", np.round(rf_cv_rmse.mean(),4))

print("\nPerforming 5-Fold CV for XGBoost...")
xgb_cv_rmse = rmse_cv(xgb, x_train, y_train, cv=5)
print("XGBoost CV RMSE scores:", np.round(xgb_cv_rmse,4))
print("Mean CV RMSE:", np.round(xgb_cv_rmse.mean(),4))

print("\nPerforming 5-Fold CV for SVR...")
svr_cv_rmse = rmse_cv(svr, x_train, y_train, cv=5)
print("SVR CV RMSE scores:", np.round(svr_cv_rmse,4))
print("Mean CV RMSE:", np.round(svr_cv_rmse.mean(),4))


# Evaluation as mentioned in assignment RMSE primary, R2 and 3 additional metrics

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)                              # primary metric
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    maxerr = max_error(y_test, y_pred)
    exp_var = explained_variance_score(y_test, y_pred)
    # MAPE - Mean Absolute Percentage Error (handle zeros defensively)
    nonzero = y_test != 0
    if nonzero.sum() == 0:
        mape = np.nan
    else:
        mape = (np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])).mean() * 100

    print(f"\n{name} Test Evaluation:")
    print("RMSE:", np.round(rmse,5))
    print("MAE:", np.round(mae,5))
    print("R2:", np.round(r2,5))
    print("MAPE (%):", np.round(mape,4))
    print("Max Error:", np.round(maxerr,5))
    print("Explained Variance:", np.round(exp_var,5))
    return y_pred, {'rmse':rmse, 'mae':mae, 'r2':r2, 'mape':mape, 'maxerr':maxerr, 'exp_var':exp_var}

rf_pred, rf_metrics = evaluate("Random Forest", rf, x_test, y_test)
xgb_pred, xgb_metrics = evaluate("XGBoost", xgb, x_test, y_test)
svr_pred, svr_metrics = evaluate("SVR", svr, x_test, y_test)


#lets plot important features from the model

# Random Forest feature importances
feat_imp_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Random Forest feature importances:\n", feat_imp_rf.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp_rf.head(10).values, y=feat_imp_rf.head(10).index)
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# XGBoost feature importances
feat_imp_xgb = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 XGBoost feature importances:\n", feat_imp_xgb.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp_xgb.head(10).values, y=feat_imp_xgb.head(10).index)
plt.title("Top 10 Feature Importances - XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# ACTUAL vs PREDICTED PLOTS for each model

def plot_actual_vs_predicted(y_true, y_pred, title):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)  # diagonal
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title(title)
    plt.show()

plot_actual_vs_predicted(y_test, rf_pred, "Random Forest - Actual vs Predicted")
plot_actual_vs_predicted(y_test, xgb_pred, "XGBoost - Actual vs Predicted")
plot_actual_vs_predicted(y_test, svr_pred, "SVR - Actual vs Predicted")


#Comparison table fro all models
results_df = pd.DataFrame({
    'Model': ['RandomForest','XGBoost','SVR'],
    'RMSE': [rf_metrics['rmse'], xgb_metrics['rmse'], svr_metrics['rmse']],
    'MAE': [rf_metrics['mae'], xgb_metrics['mae'], svr_metrics['mae']],
    'R2': [rf_metrics['r2'], xgb_metrics['r2'], svr_metrics['r2']],
    'MAPE(%)': [rf_metrics['mape'], xgb_metrics['mape'], svr_metrics['mape']],
    'MaxError': [rf_metrics['maxerr'], xgb_metrics['maxerr'], svr_metrics['maxerr']],
    'ExplainedVar': [rf_metrics['exp_var'], xgb_metrics['exp_var'], svr_metrics['exp_var']]
})

print("\nModel Comparison on Test Set:\n", results_df)

# Plot RMSE comparison
plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='RMSE', data=results_df)
plt.title("Test RMSE Comparison")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Final Summary
print("\nFINAL SUMMARY:")
print("Features used 15:", len(features))
print("Raw features 8:", raw_numerical)
print("Engineered 3:", engineered)
print("Encoded which are dummies:", encoded)
print("Primary metric: RMSE (lower is better).")
print("Other metrics reported: MAE, R2, MAPE, Max Error, Explained Variance.")
print("Models trained: RandomForest, XGBoost, SVR.")
print("Check printed CV RMSE means above to understand model stability across folds.")

