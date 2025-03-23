import pandas as pd
import numpy as np
import yfinance as yf
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
import requests
import json

# Replace with your NewsAPI key
NEWS_API_KEY = "ed7f28369c0d4440bb581eb0605d5a5b"

# Load FinBERT for sentiment analysis
MODEL_NAME = "ProsusAI/finbert"
tokenizer_bert = BertTokenizer.from_pretrained(MODEL_NAME)
model_bert = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to fetch historical stock prices
def fetch_stock_data(company, period="1y"):
    stock = yf.Ticker(company)
    df = stock.history(period=period)
    return df[['Close']]

# Function to fetch news headlines
def fetch_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()["articles"]
        return [article["title"] for article in articles if article["title"]]
    return []

# Function to analyze sentiment using FinBERT
def analyze_sentiment(headlines):
    sentiments = []
    for text in headlines:
        inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model_bert(**inputs)
        
        probabilities = softmax(outputs.logits, dim=1)[0]
        sentiment_class = torch.argmax(probabilities).item()
        sentiment_score = probabilities[2].item() - probabilities[0].item()  # Positive - Negative
        
        sentiments.append(sentiment_score)
    
    return np.mean(sentiments) if sentiments else 0

# Function to prepare dataset
def prepare_data(df):
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    # Convert stock returns to sequence data
    X, y = [], []
    sequence_length = 10
    for i in range(len(df) - sequence_length):
        X.append(df['Return'].iloc[i:i+sequence_length].values)
        y.append(df['Return'].iloc[i+sequence_length])
    
    return np.array(X), np.array(y)

# Train LSTM Model
def train_model(company):
    df = fetch_stock_data(company)
    news_headlines = fetch_news(company)
    sentiment_score = analyze_sentiment(news_headlines)
    
    X, y = prepare_data(df)
    
    # Reshape for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)

    # Save the model
    model.save("stock_model.h5")
    print("Model trained and saved as stock_model.h5")

# Run training for a sample company (e.g., Apple)
if __name__ == "__main__":
    train_model("AAPL")
