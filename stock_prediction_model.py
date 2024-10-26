import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

def load_data(file_path):
    """Load stock data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, feature_col='Close', time_step=60):
    """Preprocess the data for LSTM input."""
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        y.append(scaled_data[i, df.columns.get_loc(feature_col)])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def create_model(input_shape):
    """Create and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train):
    """Train the LSTM model."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate the model and return predictions."""
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(np.mean((predicted_prices - y_test_actual) ** 2))
    mae = np.mean(np.abs(predicted_prices - y_test_actual))

    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return predicted_prices, y_test_actual

def plot_results(y_test, predicted_prices):
    """Plot the actual vs predicted stock prices."""
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, color='red', label='Actual Prices')
    plt.plot(predicted_prices, color='blue', label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def save_model(model, model_path='stock_model.h5'):
    """Save the trained model to a file."""
    model.save(model_path)
    print(f"Model saved to {model_path}")

def main():
    # Step 1: Load Data
    file_path = 'path_to_your_stock_data.csv'  # Replace with your data file
    df = load_data(file_path)

    if df is None:
        return

    # Step 2: Preprocess Data
    X, y, scaler = preprocess_data(df)

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Create Model
    model = create_model((X_train.shape[1], X_train.shape[2]))

    # Step 5: Train Model
    train_model(model, X_train, y_train)

    # Step 6: Evaluate Model
    predicted_prices, y_test_actual = evaluate_model(model, X_test, y_test, scaler)

    # Step 7: Plot Results
    plot_results(y_test_actual, predicted_prices)

    # Step 8: Save Model
    save_model(model)

if __name__ == '__main__':
    main()
