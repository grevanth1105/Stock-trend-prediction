import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import datetime

# Constants
RSI_WINDOW = 14
STOP_LOSS_PERCENTAGE = 0.05
MIN_DATA_POINTS = 200

# Function to fetch data
def fetch_data(ticker, start_date, end_date):
    with st.spinner('Fetching data...'):
        data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data fetched. Check the ticker symbol or date range.")
        return None
    return data

# Function to calculate technical indicators
def calculate_indicators(data, short_ma, long_ma):
    data['Date'] = data.index
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    data[f'{short_ma}_MA'] = data['Close'].rolling(window=short_ma).mean()
    data[f'{long_ma}_MA'] = data['Close'].rolling(window=long_ma).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=RSI_WINDOW).mean()
    avg_loss = loss.rolling(window=RSI_WINDOW).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data.dropna()

# Function to prepare data for LSTM
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Function to train LSTM model
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    return model

# Streamlit App
st.title('Stock Trend Prediction')

# User Inputs
ticker = st.text_input('Enter Stock Ticker (e.g., GME):', 'GME')
if not ticker:
    st.error("Please enter a valid ticker symbol.")
    st.stop()

time_frame = st.radio("Select Time Frame for Analysis", ('Short-Term', 'Mid-Term', 'Long-Term'))
short_ma, long_ma = (5, 20) if time_frame == 'Short-Term' else (50, 100) if time_frame == 'Mid-Term' else (100, 200)

default_end = datetime.datetime.now()
default_start = default_end - datetime.timedelta(days=365)
start_date = st.date_input('Start Date:', value=default_start)
end_date = st.date_input('End Date:', value=default_end)

look_back = st.slider("LSTM Look-Back Period (days)", 10, 100, 60)

# Fetch and Process Data
data = fetch_data(ticker, start_date, end_date)
if data is not None:
    st.success("✅ Data fetched successfully!")
    st.write(data.tail())

    data = calculate_indicators(data, short_ma, long_ma)
    if len(data) < MIN_DATA_POINTS:
        st.error(f"Not enough data after processing. Need at least {MIN_DATA_POINTS} days.")
        st.stop()

    # Latest values for analysis
    latest_rsi = data['RSI'].iloc[-1].item()
    latest_macd = data['MACD'].iloc[-1].item()
    latest_signal = data['Signal_Line'].iloc[-1].item()
    latest_close = data['Close'].iloc[-1].item()
    latest_short_ma = data[f'{short_ma}_MA'].iloc[-1].item()
    latest_long_ma = data[f'{long_ma}_MA'].iloc[-1].item()

    # Train LSTM Model
    with st.spinner('Training LSTM model...'):
        X, y, scaler = prepare_lstm_data(data, look_back)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_lstm_model(X_train, y_train)

        # Predictions
        y_pred_scaled = model.predict(X_test)
        y_pred_full_scaled = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_pred_full = scaler.inverse_transform(y_pred_full_scaled.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Latest predicted price (for tiebreaker)
        latest_predicted_price = y_pred_full[-1][0]

        # Model Evaluation
        mse = mean_squared_error(y_test_inv, y_pred)
        r2 = r2_score(y_test_inv, y_pred)

    # Recommendation Logic (2/3 conditions + LSTM tiebreaker)
    recommendation = "Hold"
    trigger_price = latest_close
    stop_loss = None
    holding_period = "Monitor Trends"

    buy_conditions = [latest_rsi < 40, latest_macd > latest_signal, latest_short_ma > latest_long_ma]
    sell_conditions = [latest_rsi > 60, latest_macd < latest_signal, latest_short_ma < latest_long_ma]

    buy_count = sum(buy_conditions)
    sell_count = sum(sell_conditions)

    if buy_count >= 2 or (buy_count == 1 and sell_count == 0 and latest_predicted_price > latest_close):
        recommendation = "✅ Buy"
        stop_loss = latest_close * (1 - STOP_LOSS_PERCENTAGE)
        holding_period = "Short to Mid-Term (1-6 months)"
    elif sell_count >= 2 or (sell_count == 1 and buy_count == 0 and latest_predicted_price < latest_close):
        recommendation = "❌ Sell"
        stop_loss = latest_close * (1 + STOP_LOSS_PERCENTAGE)
        holding_period = "Exit Immediately"

    # Display Recommendation
    st.write("## Stock Recommendation")
    st.write(f"**Stock:** {ticker}")
    st.write(f"**Current Price:** ${latest_close:.2f}")
    st.write(f"**Latest Predicted Price (LSTM):** ${latest_predicted_price:.2f}")
    st.write(f"**Recommendation:** {recommendation}")
    st.write(f"**Trigger Price:** ${trigger_price:.2f}")
    if stop_loss:
        st.write(f"**Stop Loss:** ${stop_loss:.2f}")
    st.write(f"**Suggested Holding Period:** {holding_period}")

    # Visualization Section
    st.write("### Comprehensive Stock Analysis")

    # Moving Averages Plot
    st.write(f"#### {ticker} Moving Averages ({short_ma}-Day vs {long_ma}-Day)")
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data[f'{short_ma}_MA'], color='green', label=f'{short_ma}-Day MA')
    plt.plot(data['Date'], data[f'{long_ma}_MA'], color='orange', label=f'{long_ma}-Day MA')
    plt.title(f"{ticker} Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    st.write("""
    #### Moving Averages Explanation
    - **What it shows**: Moving Averages smooth price data to identify trends.
    - **Short MA above Long MA**: Bullish trend (price may rise).
    - **Short MA below Long MA**: Bearish trend (price may fall).
    """)

    # RSI Plot
    st.write("#### Relative Strength Index (RSI)")
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['RSI'], color='purple', label='RSI')
    plt.axhline(60, color='red', linestyle='--', label="Overbought (60)")
    plt.axhline(40, color='green', linestyle='--', label="Oversold (40)")
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    st.write("""
    #### RSI Explanation
    - **What it shows**: RSI measures momentum (0-100 scale).
    - **RSI > 60**: Overbought (potential sell signal).
    - **RSI < 40**: Oversold (potential buy signal).
    """)

    # MACD Plot
    st.write("#### Moving Average Convergence Divergence (MACD)")
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['MACD'], color='blue', label='MACD')
    plt.plot(data['Date'], data['Signal_Line'], color='red', label='Signal Line')
    plt.title("MACD")
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    st.write("""
    #### MACD Explanation
    - **What it shows**: MACD identifies trend direction and reversals.
    - **MACD above Signal**: Bullish (price may rise).
    - **MACD below Signal**: Bearish (price may fall).
    """)

    # Actual vs Predicted Prices Plot
    st.write(f"#### {ticker} Actual vs Predicted Prices")
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], color='blue', label='Actual Prices')
    plt.plot(data['Date'][look_back:], y_pred_full, color='red', label='Predicted Prices')
    plt.title(f"{ticker} Actual vs Predicted Prices")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    st.pyplot(plt)
    plt.clf()

    st.write("""
    #### Actual vs Predicted Prices Explanation
    - **What it shows**: Compares the actual stock prices with LSTM model predictions.
    - **Blue Line**: Historical closing prices.
    - **Red Line**: Predicted prices based on the LSTM model.
    - **Interpretation**: If the red line closely follows the blue line, the model is accurate; deviations indicate prediction errors.
    """)

    # Model Accuracy at the End
    st.write("### Model Accuracy")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")