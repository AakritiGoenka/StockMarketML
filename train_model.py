import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import joblib  # For saving the scaler
import yfinance as yf  # Fetch stock data

# Fetch stock data (Example: Tesla 'TSLA')
df = yf.download("TSLA", start="2020-01-01", end="2024-01-01")
df = df[['Close']]  # Use only closing prices

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Save scaler for later use
joblib.dump(scaler, "scaler.pkl")

# Prepare dataset for LSTM
sequence_length = 50
X, y = [], []
for i in range(len(df_scaled) - sequence_length):
    X.append(df_scaled[i:i + sequence_length])
    y.append(df_scaled[i + sequence_length])

X, y = np.array(X), np.array(y)

# Split into training and testing sets (80-20 split)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile model with fixed loss function
model.compile(optimizer="adam", loss=MeanSquaredError())

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("stock_prediction_model.h5")
print("✅ Model saved as stock_prediction_model.h5")
