import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== Load and Preprocess Walmart Sales Dataset ==========
df = pd.read_csv("Walmart.csv", parse_dates=["Date"])

# Ensure 'Date' column is set as index properly
if "Date" in df.columns:
    df.set_index("Date", inplace=True)

# Convert index to DatetimeIndex
df.index = pd.to_datetime(df.index, errors="coerce")

# Ensure the index is sorted
df = df.sort_index()

# Ensure Weekly_Sales column is numeric
df["Weekly_Sales"] = pd.to_numeric(df["Weekly_Sales"], errors="coerce")

# Drop rows after index 4500
df = df.iloc[:4500]

# Handle missing values
df.fillna(method="ffill", inplace=True)

# ========== Feature Engineering for XGBoost ==========
df["Lag_7"] = df["Weekly_Sales"].shift(7)
df["Lag_30"] = df["Weekly_Sales"].shift(30)
df["MA_7"] = df["Weekly_Sales"].rolling(window=7).mean()
df["MA_30"] = df["Weekly_Sales"].rolling(window=30).mean()

# Drop NaN values (ensures X and y are the same length)
df.dropna(inplace=True)

# Define Target Variable (Weekly_Sales)
y = df["Weekly_Sales"]
X = df.drop(columns=["Weekly_Sales"])  # Features

# ========== Function to Evaluate Models ==========
def evaluate_model(y_actual, y_pred, model_name):
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100  # Mean Absolute Percentage Error

    print(f"\n{model_name} Performance Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return r2, mape  # Store R² and MAPE

# ========== Run LSTM & XGBoost 10 Times ==========
lstm_scores = []
xgb_scores = []
lstm_mape_scores = []
xgb_mape_scores = []

# Scale Weekly_Sales for LSTM
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

# Prepare LSTM sequences
def create_lstm_sequences(data, time_steps=10):
    X_seq, y_seq = [], []
    for i in range(len(data) - time_steps):
        X_seq.append(data[i : i + time_steps])
        y_seq.append(data[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# Create sequences
time_steps = 10
X_lstm, y_lstm = create_lstm_sequences(y_scaled, time_steps)

# Split data
split = int(0.8 * len(X_lstm))
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

for i in range(10):
    print(f"\nIteration {i+1}")

    # ------ XGBoost Model ------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, shuffle=False)

    xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=1000, learning_rate=0.05, random_state=i)
    xgb_model.fit(X_train, y_train)

    # Predict sales
    xgb_predictions = xgb_model.predict(X_test)

    # Evaluate XGBoost
    xgb_r2, xgb_mape = evaluate_model(y_test, xgb_predictions, "XGBoost")
    xgb_scores.append(xgb_r2)
    xgb_mape_scores.append(xgb_mape)

    # ------ LSTM Model ------
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    # Compile the model
    lstm_model.compile(optimizer="adam", loss="mse")

    # Train the model
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=16, verbose=0)

    # Predict using LSTM
    lstm_predictions = lstm_model.predict(X_test_lstm)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    # Evaluate LSTM
    lstm_r2, lstm_mape = evaluate_model(y_test.iloc[-len(lstm_predictions):], lstm_predictions.flatten(), "LSTM")
    lstm_scores.append(lstm_r2)
    lstm_mape_scores.append(lstm_mape)

# ========== Print Average Scores ==========
print(f"\nAverage LSTM R² Score: {np.mean(lstm_scores):.2f}")
print(f"Average XGBoost R² Score: {np.mean(xgb_scores):.2f}")
print(f"Average LSTM MAPE: {np.mean(lstm_mape_scores):.2f}%")
print(f"Average XGBoost MAPE: {np.mean(xgb_mape_scores):.2f}%")

# ========== Residual Analysis ==========
lstm_residuals = y_test.iloc[-len(lstm_predictions):] - lstm_predictions.flatten()
xgb_residuals = y_test - xgb_predictions

# Plot residual distribution
plt.figure(figsize=(12, 5))
sns.histplot(lstm_residuals, bins=50, color="blue", kde=True, label="LSTM Residuals")
sns.histplot(xgb_residuals, bins=50, color="red", kde=True, label="XGBoost Residuals")
plt.legend()
plt.title("Residual Distribution (LSTM vs XGBoost)")
plt.show()
