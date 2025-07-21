📉 Fall of FTX Stock – Analysis and Machine Learning Prediction
📌 Project Overview
This project investigates the collapse of the FTX cryptocurrency exchange by performing time-series forecasting, volatility analysis, and machine learning-based crash prediction using historical market data. The goal is to uncover hidden patterns, detect anomalies, and predict future behavior in volatile crypto markets.

📂 Dataset Description
File Name: The FTX Dataset.csv

Fields Included:

Date: Trading date (converted to datetime)

Opening, High, Low, Closing Amount: Price data for each day

Volume: Daily traded volume

Market Cap: Daily market capitalization

All monetary values were cleaned to remove symbols like $ and , before analysis.

🧹 Data Preprocessing
Renamed columns for consistency

Removed currency symbols and converted all prices to float

Outliers were treated using the Interquartile Range (IQR) method

Created new features such as:

Moving Averages (7-day, 30-day)

Daily Fluctuation (High - Low)

Price Deviation from MA

Anomaly Flag (for market crash detection)

📈 Time-Series Forecasting
✅ ARIMA Model
Trained on daily closing prices

Forecasted next 30 days of closing prices

Visualized actual vs. predicted values

✅ Trend Analysis
Moving Average Crossover (7-day vs. 30-day)

Helps detect potential uptrends or downtrends

✅ Seasonal Decomposition
Decomposed closing prices into:

Trend

Seasonal Component

Residuals

📊 Volatility and Pattern Detection
🔥 Volatility Analysis
Monthly average Daily Fluctuation calculated

Identified Top 5 most volatile months in FTX history

🔍 Clustering of Volume Activity
Used K-Means Clustering to segment days by trading volume:

Low (Cluster 0)

Medium (Cluster 1)

High (Cluster 2)

⚠️ Anomaly Detection
Defined "price anomaly" as significant deviation from 7-day MA

Set threshold using 2 standard deviations

Identified and plotted outlier days (crash signals)

🤖 Machine Learning Models
📌 Stock Price Prediction (Regression)
Model: Linear Regression

Target: Closing Price

Features: Open, High, Low, MA values, Market Cap, etc.

Metrics:

MAE, MSE, RMSE

R² Score

📌 Market Crash Detection (Classification)
Model: Random Forest Classifier

Target: Binary Anomaly flag

Evaluation:

Accuracy: 91%

ROC AUC Score: 0.91

Classification Report

Feature Importance Visualization

📌 Volume Spike Prediction
Model: XGBoost Classifier

Target: Volume Spike (defined as >150% change)

Metrics: Accuracy, ROC AUC

📌 Hyperparameter Tuning
Used RandomizedSearchCV to optimize Random Forest and XGBoost

Improved model performance with fine-tuned settings

📊 Visualizations
Line plots for:

Actual vs. Predicted Prices

Moving Averages

ROC Curves

Bar plots:

Monthly volatility

Feature importances

Scatter plot:

Clustering trading volume

Anomalies in price

🧠 Key Insights
FTX showed extreme volatility in specific months, possibly due to news or scandals

Most trading days had moderate volume, with a few spikes suggesting institutional dumps or panic

Crash prediction model achieved high accuracy and can be extended to other cryptocurrencies

Volume anomalies and moving average deviations were strong signals of instability
