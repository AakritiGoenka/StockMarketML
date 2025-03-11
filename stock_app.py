import streamlit as st
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model and scaler
model = load_model("stock_prediction_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# Streamlit Page Layout
st.set_page_config(page_title="Stock Market Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Predict"])

# Function to Fetch Stock Data
def get_stock_data(symbol):
    try:
        df = yf.download(symbol, period="1y")  # Last 1 year data
        return df[['Close']]
    except:
        return None

# 🏠 Home Page
if page == "Home":
    st.title("📈 Welcome to Stock Market Predictor")
    st.markdown("""
        - 🔮 **Predict future stock prices using AI**
        - 📊 **View past stock trends and graphs**
        - 🔍 **Enter stock symbols like TSLA, AAPL, AMZN**
        - 🚀 **Built using TensorFlow, Streamlit & Yahoo Finance API**
    """)

# ℹ️ About Page
elif page == "About":
    st.title("ℹ️ About This App")
    st.write("""
        - This app uses a **Recurrent Neural Network (RNN)** with LSTM layers to predict future stock prices.
        - **Yahoo Finance API** is used to fetch real-time stock data.
        - Built using **Python, TensorFlow, Keras, and Streamlit**.
    """)

# 🔮 Prediction Page
elif page == "Predict":
    st.title("🔮 Stock Price Prediction")

    # User input for stock symbol
    stock_symbol = st.text_input("🔍 Enter Stock Symbol (e.g., TSLA, AAPL, AMZN)", "TSLA")

    # Fetch stock data
    df = get_stock_data(stock_symbol)

    if df is not None and not df.empty:
        # Display Stock Price Chart
        st.subheader(f"📊 {stock_symbol} - Last 1 Year Stock Prices")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df['Close'], label="Closing Price", color='blue')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)

        # Prepare Data for Prediction
        last_50_days = df[-50:].values  # Last 50 days closing prices
        last_50_days_scaled = scaler.transform(last_50_days)
        X_input = np.array([last_50_days_scaled])

        # Make Prediction
        predicted_price = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

        # Display Predicted Price
        st.subheader("🔮 Predicted Next Closing Price")
        st.write(f"💰 **{predicted_price[0][0]:.2f} USD**")

    else:
        st.warning("⚠️ No data found! Please enter a valid stock symbol.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("💡 Developed by Supp 🚀")
