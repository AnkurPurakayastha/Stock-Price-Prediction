import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = yf.download("AAPL", start="2020-01-01", end="2024-12-31", auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.reset_index()

for lag in range(1, 6):
    data[f"lag_{lag}"] = data["Close"].shift(lag)

data["ma_5"] = data["Close"].rolling(window=5).mean()
data["ma_10"] = data["Close"].rolling(window=10).mean()
data = data.dropna()

feature_cols = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "ma_5", "ma_10"]

X = data[feature_cols]
y = data["Close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

accuracy = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test.values - predictions) / y_test.values)) * 100

print("Model Accuracy (R2 Score):", accuracy)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape, "%")

plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.savefig("stock_prediction_plot.png")
print("Plot saved as stock_prediction_plot.png")
