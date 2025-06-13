import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Fetch real-time weather data from OpenWeatherMap
def fetch_weather_data(city="Delhi", api_key="e597f0454b011ac1ad8a410141ca2ff6"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'weather_temp': data['main']['temp'],
            'rainfall': data.get('rain', {}).get('1h', 0),
            'date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
    return {'weather_temp': 20, 'rainfall': 10, 'date': '2025-04-10'}

# Simulate real-time Agmarknet data (mock API)
def fetch_agmarknet_data():
    mock_data = [
        {'date': '2025-04-01', 'commodity': 'Lentils', 'region': 'Delhi', 'price': 50},
        {'date': '2025-04-02', 'commodity': 'Lentils', 'region': 'Delhi', 'price': 52},
        {'date': '2025-04-03', 'commodity': 'Lentils', 'region': 'Delhi', 'price': 48},
        {'date': '2025-04-04', 'commodity': 'Lentils', 'region': 'Delhi', 'price': 51},
        {'date': '2025-04-05', 'commodity': 'Lentils', 'region': 'Delhi', 'price': 53}
    ]
    return pd.DataFrame(mock_data)

# Load and preprocess data
def load_and_preprocess_data(api_key):
    price_df = fetch_agmarknet_data()
    weather_data = fetch_weather_data(api_key=api_key)
    
    price_df['weather_temp'] = weather_data['weather_temp']
    price_df['rainfall'] = weather_data['rainfall']
    price_df['festival'] = 'No'  # Placeholder
    
    le_commodity = LabelEncoder()
    le_region = LabelEncoder()
    le_festival = LabelEncoder()
    price_df['commodity'] = le_commodity.fit_transform(price_df['commodity'])
    price_df['region'] = le_region.fit_transform(price_df['region'])
    price_df['festival'] = le_festival.fit_transform(price_df['festival'])
    
    return price_df, le_commodity, le_region, le_festival

# Prepare data for Random Forest
def prepare_rf_data(df):
    X = df[['commodity', 'region', 'weather_temp', 'rainfall', 'festival']]
    y = df['price']
    return X, y

# Train Random Forest
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict with Random Forest
def predict_rf(model, input_data, le_commodity, le_region, le_festival):
    input_df = pd.DataFrame([input_data], columns=['commodity', 'region', 'weather_temp', 'rainfall', 'festival'])
    input_df['commodity'] = le_commodity.transform([input_data['commodity']])[0]
    input_df['region'] = le_region.transform([input_data['region']])[0]
    input_df['festival'] = le_festival.transform([input_data['festival']])[0]
    return model.predict(input_df)[0]

# Plot predictions
def plot_predictions(actual, predicted, filename='static/prediction_plot.png'):
    plt.figure(figsize=(8, 5))
    plt.plot(actual, label='Actual Price')
    plt.plot(predicted, label='Predicted Price', linestyle='--')
    plt.legend()
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (₹)')
    plt.savefig(filename)
    plt.close()