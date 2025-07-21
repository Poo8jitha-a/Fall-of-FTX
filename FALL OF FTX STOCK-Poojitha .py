#!/usr/bin/env python
# coding: utf-8

# In[56]:


#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')


# In[57]:


# Loading the dataset
df=pd.read_csv(r'C:\Users\pooji\Downloads\The ftx Dataset.csv')

# Displaying basic information about the dataset
df.head()


# In[58]:


df.describe()


# In[59]:


df.shape 


# In[60]:


df.info()


# In[61]:


df.isna().sum()


# In[62]:


# Renaming columns
df.columns = ["date", "opening_amount", "highest_amount", "lowest_amount", "closing_amount", "volume", "market_cap"]


# In[63]:


# Converting date column to datetime format
df["date"] = pd.to_datetime(df["date"], errors="coerce")


# In[64]:


# Define a function to clean numeric columns by removing "$" and ","
def clean_currency(value):
    return float(value.replace("$", "").replace(",", ""))

# Apply the function to numeric columns
numeric_columns = ["opening_amount", "highest_amount", "lowest_amount", "closing_amount", "volume", "market_cap"]
df[numeric_columns] = df[numeric_columns].applymap(clean_currency)

# Verify data types after conversion
df.dtypes


# In[65]:


df.info()


# In[66]:


#Outlier Treatment by InterQuartileRange(IQR)
numeric_columns=["date", "opening_amount", "highest_amount", "lowest_amount", "closing_amount", "volume", "market_cap"]
# Outlier Detection using IQR for Price & Volume
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# In[67]:


# Create box plots for numerical columns to detect outliers
plt.figure(figsize=(15, 8))
df[numeric_columns].boxplot()
plt.xticks(rotation=45)
plt.title("Box Plot for Outlier Detection in FTX Dataset")
plt.show()


# In[68]:


# Applying IQR outlier removal for Volume and Market Cap
df_cleaned = remove_outliers(df, "volume")
df_cleaned = remove_outliers(df_cleaned, "market_cap")


# In[69]:


# Display the number of rows removed
rows_removed = len(df) - len(df_cleaned)
rows_removed


# In[70]:


# Visualize outliers before removal
plt.figure(figsize=(12, 5))
sns.boxplot(data=df_cleaned[numeric_columns])
plt.title("Boxplot of Price & Volume Columns After Outlier Removal")
plt.show()


# In[71]:


df_cleaned.shape


# TIME-SERIES ANALYSIS

# Predicting next 30 day closing price

# In[72]:


from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model (Auto ARIMA selection can be done later)
model = ARIMA(df_cleaned["closing_amount"].dropna(), order=(1, 1, 1))  # ARIMA(p,d,q)
model_fit = model.fit()

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)

# Create future date range
future_dates = pd.date_range(start=df_cleaned["date"].max(), periods=30, freq="D")

# Create forecast DataFrame
forecast_df = pd.DataFrame({"date": future_dates, "Predicted Closing Amount": forecast})

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned["date"], df_cleaned["closing_amount"], label="Actual Closing Price", color="blue")
plt.plot(forecast_df["date"], forecast_df["Predicted Closing Amount"], label="Predicted (30 Days)", color="red", linestyle="dashed")

plt.xlabel("Date")
plt.ylabel("Closing Amount")
plt.title("30-Day Forecast of Closing Prices (ARIMA)")
plt.legend()
plt.show()

# Display forecasted values
forecast_df


# In[73]:


# Calculate 7-day and 30-day Moving Averages for Closing Amount
df_cleaned["MA_7"] = df_cleaned["closing_amount"].rolling(window=7).mean()
df_cleaned["MA_30"] = df_cleaned["closing_amount"].rolling(window=30).mean()

# Plot Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned["date"], df_cleaned["closing_amount"], label="Closing Price", color="blue", alpha=0.6)
plt.plot(df_cleaned["date"], df_cleaned["MA_7"], label="7-Day MA", color="red")
plt.plot(df_cleaned["date"], df_cleaned["MA_30"], label="30-Day MA", color="green")

plt.title("FTX Closing Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# When the 7-day MA crosses above the 30-day MA, it signals a potential uptrend (buy signal).
# 
# When the 7-day MA crosses below the 30-day MA, it indicates a possible downtrend (sell signal).
# 
# Large deviations between the two moving averages suggest high volatility, while close proximity indicates stability.

# seasonal trend detection

# In[74]:


from statsmodels.tsa.seasonal import seasonal_decompose

# Perform time-series decomposition (Additive Model)
decomposition = seasonal_decompose(df_cleaned.set_index("date")["closing_amount"], model="additive", period=30)

# Plot decomposition results
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(decomposition.observed, label='Observed', color='blue')
plt.legend()

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='red')
plt.legend()

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.legend()

plt.subplot(414)
plt.plot(decomposition.resid, label='Residual', color='black')
plt.legend()

plt.tight_layout()
plt.show()


# Trend (Red Line): Shows long-term price movement.
# 
# Seasonality (Green Line): Repeating patterns in price changes.
# 
# Residuals (Black Line): Random fluctuations (noise).

# STOCK VOLITALITY PREDICTION

# In[75]:


# Calculate Daily Price Fluctuation (High - Low)
df_cleaned["Daily Fluctuation"] = df_cleaned["highest_amount"] - df_cleaned["lowest_amount"]

# Grouping by Month-Year and calculate average fluctuation per month
df_cleaned["Year-Month"] = df_cleaned["date"].dt.to_period("M")
monthly_volatility = df_cleaned.groupby("Year-Month")["Daily Fluctuation"].mean()


# In[76]:


# Plotting Monthly Volatility
plt.figure(figsize=(12, 6))
monthly_volatility.plot(kind="bar", color="red", alpha=0.7)
plt.xlabel("Year-Month")
plt.ylabel("Average Daily Fluctuation")
plt.title("Monthly Stock Volatility (Based on Price Fluctuation)")
plt.xticks(rotation=45)
plt.show()


# In[77]:


# Identifying the most volatile months
most_volatile_months = monthly_volatility.sort_values(ascending=False).head(5)
print("Top 5 Most Volatile Months:")
print(most_volatile_months)


# In[78]:


from sklearn.cluster import KMeans

# Select the feature for clustering (Trading Volume Trends)
volume_data = df_cleaned[["volume"]]

# Determine optimal number of clusters using Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(volume_data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal Clusters in Trading Volume")
plt.show()


# In[79]:


# Apply KMeans with optimal clusters (k=3 based on elbow method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_cleaned["Volume Cluster"] = kmeans.fit_predict(volume_data)
# Plot Clustered Trading Volume Trends
plt.figure(figsize=(12, 6))
plt.scatter(df_cleaned["date"], df_cleaned["volume"], c=df_cleaned["Volume Cluster"], cmap="viridis", alpha=0.6)
plt.xlabel("Date")
plt.ylabel("Trading Volume")
plt.title("Trading Volume Clustering (Low, Medium, High Activity)")
plt.colorbar(label="Cluster")
plt.show()

# Display count of days per cluster
df_cleaned["Volume Cluster"].value_counts()


# 3 Clusters Identified:
#  Cluster 0 (137 days) - Low Activity
#  Cluster 1 (630 days) - Medium Activity
#  Cluster 2 (340 days) - High Activity
# Key Insight: Majority of trading days had medium volume, while only 137 days saw very low trading activity.
# 
# 

# In[80]:


#IDENTIFYING PRICE ANOMALIES

# Calculating Price Deviation from 7-day MA
df_cleaned["Price Deviation"] = abs(df_cleaned["closing_amount"] - df_cleaned["MA_7"])

# Defining anomaly threshold (e.g., 2 standard deviations from the mean deviation)
threshold = df_cleaned["Price Deviation"].mean() + 2 * df_cleaned["Price Deviation"].std()

# Identifying the anomaly days
df_cleaned["Anomaly"] = df_cleaned["Price Deviation"] > threshold
anomalies = df_cleaned[df_cleaned["Anomaly"]]


# In[81]:


# Plot Closing Price and Highlight Anomalies
plt.figure(figsize=(12, 6))
plt.plot(df_cleaned["date"], df_cleaned["closing_amount"], label="Closing Price", color="blue", alpha=0.6)
plt.scatter(anomalies["date"], anomalies["closing_amount"], color="red", label="Anomalies", marker="o", s=50)
plt.xlabel("Date")
plt.ylabel("Closing Amount")
plt.title("Price Anomalies Detection (Based on 7-Day Moving Average)")
plt.legend()
plt.show()


# In[27]:


# Display detected anomalies
anomalies[["date", "closing_amount", "MA_7", "Price Deviation"]]


# MODEL BUILDING

# In[82]:


df_cleaned


# In[87]:


df_cleaned.info()



# In[93]:


df_cleaned = df_cleaned.dropna(subset=['MA_7', 'MA_30','Price Deviation'])



# In[94]:


df_cleaned.info()


# In[95]:


df = df_cleaned
df


# In[96]:


df['Anomaly'] = df['Anomaly'].astype(int)


# In[97]:


from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
# Normalize Market Cap
scaler = MinMaxScaler()
df["market_cap_normalized"] = scaler.fit_transform(df[["market_cap"]])


# In[99]:


#STOCK PRICE PREDICTION SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve

# Preparing data
X_price = df[["opening_amount", "highest_amount", "lowest_amount", "market_cap_normalized","MA_7","MA_30","Daily Fluctuation"]]
y_price = df["closing_amount"]
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

# Training model
lr_model = LinearRegression()
lr_model.fit(X_train_price, y_train_price)
y_pred_price = lr_model.predict(X_test_price)



# Evaluating model
mae = mean_absolute_error(y_test_price, y_pred_price)
mse = mean_squared_error(y_test_price, y_pred_price)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_price, y_pred_price)

print("Stock Price Prediction - Linear Regression")
print("MAE:", mae, "MSE:", mse, "RMSE:", rmse,"R²:", r2)




# In[100]:


#MARKET CRASH PREDICTION
X = df[['opening_amount', 'highest_amount', 'lowest_amount', 'volume', 'market_cap_normalized',"MA_7", "MA_30"]]
y = df['Anomaly'].dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Random Forest Crash Prediction Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))


# In[101]:


from sklearn.metrics import classification_report  
print(classification_report(y_test, y_pred))  


# In[102]:


#Plotting Feature importance
import matplotlib.pyplot as plt  
feature_importance = rf_model.feature_importances_  
plt.barh(X.columns, feature_importance)  
plt.xlabel("Feature Importance")  
plt.title("Random Forest Feature Importance")  
plt.show()


# In[103]:


# Drop 'lowest_amount','highest_amount','volume' from features and retrain
features_reduced = ['opening_amount', 'MA_30', 'MA_7', 'market_cap_normalized']
X_reduced = df_cleaned[features_reduced]

# Train-test split
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost again
xgb_model.fit(X_train_red, y_train_red)
y_pred_xgb_red = xgb_model.predict(X_test_red)

print("New XGBoost Accuracy:", accuracy_score(y_test_red, y_pred_xgb_red))


# In[104]:


# Trading Volume Spike Prediction
df['Volume Spike'] = (df['volume'].pct_change() > 1.5).astype(int)
y_vol = df['Volume Spike'].dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y_vol, test_size=0.2, random_state=42, stratify=y_vol)
xgb_vol = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, gamma=0.1, random_state=42)
xgb_vol.fit(X_train, y_train)
vol_preds = xgb_vol.predict(X_test)
print("XGBoost Volume Spike Prediction Accuracy:", accuracy_score(y_test, vol_preds))


# MODEL TESTING AND EVALUATION

# In[105]:


# Plot Actual vs Predicted Prices(linear Regression)
plt.figure(figsize=(12, 6))
plt.plot(y_test_price.values, label='Actual Prices', color='blue')
plt.plot(y_pred_price, label='Predicted Prices', color='red', linestyle='dashed')
plt.xlabel('Test Data Index')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
plt.show()


# In[106]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_probs = xgb_model.predict_proba(X_test_red)[:, 1]  # Get probabilities for class 1 (crash)
fpr, tpr, _ = roc_curve(y_test_red, y_probs)
roc_auc = auc(fpr, tpr)


# In[107]:


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal Line (Random Model)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# AUC = 0.91: Hencemodel has 91% probability of ranking a randomly chosen positive instance higher than a negative instance.

# In[108]:


from sklearn.model_selection import RandomizedSearchCV
# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform Randomized Search
rf_tuned = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=10, cv=3, n_jobs=-1)
rf_tuned.fit(X_train, y_train)

# Best parameters and new accuracy
best_rf = rf_tuned.best_estimator_
y_pred_best = best_rf.predict(X_test)

print("Best Parameters:", rf_tuned.best_params_)
print("New Accuracy:", accuracy_score(y_test, y_pred_best))


# In[109]:


from xgboost import XGBClassifier
xgb_tuned = XGBClassifier(n_estimators=300, min_samples_split= 5, min_samples_leaf=4,  max_depth= 30, bootstrap= True)
xgb_tuned.fit(X_train, y_train)

y_pred_xgb_tuned = xgb_tuned.predict(X_test)
print("Tuned XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb_tuned))


# In[110]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
# Model Evaluation
cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
print("Cross-validation Accuracy Scores:", scores.mean())


# In[ ]:




