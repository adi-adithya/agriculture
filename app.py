import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import logging
import os
import re
import threading
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create directories
os.makedirs('cache', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)  # Dedicated directory for plot images
CACHE_EXPIRY_MINUTES = 5  # Cache expires after 5 minutes

# Fetch temperature using OpenWeatherMap
def fetch_temperature(lat, lon):
    try:
        api_key = "e597f0454b011ac1ad8a410141ca2ff6"  # Replace with your API key
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['main']['temp']
        logger.warning(f"Failed to fetch temperature: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Error fetching temperature: {str(e)}")
        return None

# Fetch prices from BigBasket with better error handling
def fetch_bigbasket_prices(product_query=None):
    try:
        # We'll query multiple categories to get comprehensive results
        categories = [
            "fruits-vegetables",
            "foodgrains-oil-masala",
            "bakery-cakes-dairy",
            "snacks-branded-foods",
            "kitchen-garden"
        ]
        
        all_products = []
        for category in categories:
            url = f"https://www.bigbasket.com/pc/{category}/"
            if product_query:
                url += f"?q={product_query}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    products = soup.find_all('div', {'class': 'product-item'})
                    
                    for product in products:
                        name_elem = product.find('div', {'class': 'product-name'})
                        price_elem = product.find('div', {'class': 'product-price'})
                        if name_elem and price_elem:
                            name = name_elem.text.strip()
                            price_text = price_elem.text.strip()
                            price_match = re.search(r'₹\s*(\d+(?:\.\d+)?)', price_text)
                            if price_match:
                                price = float(price_match.group(1))
                                # Categorize products appropriately
                                commodity_type = 'Fruits' if any(x in name.lower() for x in ['apple', 'banana', 'orange', 'fruit']) else \
                                             'Vegetables' if any(x in name.lower() for x in ['potato', 'onion', 'tomato', 'vegetable']) else \
                                             'Pulses' if any(x in name.lower() for x in ['dal', 'pulse', 'lentil', 'beans']) else \
                                             'Groceries'
                                all_products.append({
                                    'commodity': name,
                                    'commodity_type': commodity_type,
                                    'price': price,
                                    'market': 'BigBasket',
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'state': 'Online',
                                    'district': 'Online'
                                })
            except requests.exceptions.RequestException as req_error:
                logger.error(f"Request error for {category}: {str(req_error)}")
                continue
        
        logger.info(f"Scraped {len(all_products)} products from BigBasket")
        return pd.DataFrame(all_products)
    except Exception as e:
        logger.error(f"Error fetching BigBasket prices: {str(e)}")
    return pd.DataFrame()

# Fetch historical data with more comprehensive product list
def fetch_historical_data(product_query=None):
    cache_file = f"cache/historical_{product_query.replace(' ', '_') if product_query else 'all'}.csv"
    if os.path.exists(cache_file):
        cache_data = pd.read_csv(cache_file)
        last_update = pd.to_datetime(cache_data['date'].iloc[0]) if not cache_data.empty else None
        if last_update and (datetime.now() - last_update) < timedelta(days=7):
            logger.info(f"Using cached historical data for {product_query}")
            return cache_data
    
    today = datetime.now()
    # Generate more data points for better historical visualization (60 days instead of 30)
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(60)]
    
    # Expanded product list with more diverse categories
    base_products = {
        # Cereals
        'Rice': 60, 
        'Basmati Rice': 120,
        'Brown Rice': 90,
        'Wheat Flour': 40,
        'Wheat': 35,
        'Maida': 42,
        'Rava/Suji': 45,
        'Oats': 75,
        
        # Pulses
        'Toor Dal': 120,
        'Moong Dal': 110,
        'Chana Dal': 90,
        'Masoor Dal': 95,
        'Urad Dal': 105,
        'Rajma': 130,
        'Black Chickpeas': 85,
        'Green Peas': 75,
        
        # Vegetables
        'Potato': 35, 
        'Onion': 30, 
        'Tomato': 20,
        'Carrot': 35,
        'Cabbage': 28,
        'Cauliflower': 32,
        'Cucumber': 22,
        'Capsicum': 50,
        'Green Beans': 55,
        'Bitter Gourd': 60,
        'Lady Finger': 45,
        'Eggplant': 40,
        'Pumpkin': 30,
        'Green Chilli': 65,
        'Ginger': 100,
        'Garlic': 140,
        'Spinach': 30,
        'Coriander': 20,
        'Mint': 25,
        'Broccoli': 90,
        
        # Fruits
        'Apple': 150,
        'Red Delicious Apple': 180,
        'Green Apple': 200,
        'Banana': 60,
        'Orange': 100,
        'Mango': 180,
        'Alphonso Mango': 300,
        'Grapes': 120,
        'Black Grapes': 140,
        'Watermelon': 35,
        'Papaya': 70,
        'Pineapple': 80,
        'Pomegranate': 130,
        'Sweet Lime': 90,
        'Guava': 60,
        'Strawberry': 250,
        'Kiwi': 180,
        
        # Groceries
        'Vegetable Oil': 110,
        'Sunflower Oil': 130,
        'Groundnut Oil': 160,
        'Olive Oil': 450,
        'Mustard Oil': 140,
        'Ghee': 550,
        'Sugar': 45,
        'Brown Sugar': 65,
        'Jaggery': 70,
        'Salt': 20,
        'Iodized Salt': 25,
        'Black Pepper': 700,
        'Red Chilli Powder': 350,
        'Turmeric Powder': 280,
        'Coriander Powder': 240,
        'Cumin Seeds': 400,
        'Mustard Seeds': 200,
        'Tea': 180,
        'Green Tea': 320,
        'Coffee': 220,
        'Milk': 60,
        'Butter': 500,
        'Cheese': 400,
        'Paneer': 350,
        'Curd/Yogurt': 80,
        'Honey': 300,
        'Cashew Nuts': 800,
        'Almonds': 900,
        'Raisins': 400
    }
    
    if product_query:
        filtered_products = {k: v for k, v in base_products.items() if product_query.lower() in k.lower()}
        if filtered_products:
            base_products = filtered_products
        else:
            # Check if query is a category name
            category_matches = {
                'cereals': ['Rice', 'Basmati Rice', 'Brown Rice', 'Wheat Flour', 'Wheat', 'Maida', 'Rava/Suji', 'Oats'],
                'pulses': ['Toor Dal', 'Moong Dal', 'Chana Dal', 'Masoor Dal', 'Urad Dal', 'Rajma', 'Black Chickpeas', 'Green Peas'],
                'vegetables': ['Potato', 'Onion', 'Tomato', 'Carrot', 'Cabbage', 'Cauliflower', 'Cucumber', 'Capsicum', 
                               'Green Beans', 'Bitter Gourd', 'Lady Finger', 'Eggplant', 'Pumpkin', 'Green Chilli', 
                               'Ginger', 'Garlic', 'Spinach', 'Coriander', 'Mint', 'Broccoli'],
                'fruits': ['Apple', 'Red Delicious Apple', 'Green Apple', 'Banana', 'Orange', 'Mango', 'Alphonso Mango', 
                          'Grapes', 'Black Grapes', 'Watermelon', 'Papaya', 'Pineapple', 'Pomegranate', 'Sweet Lime', 
                          'Guava', 'Strawberry', 'Kiwi'],
                'groceries': ['Vegetable Oil', 'Sunflower Oil', 'Groundnut Oil', 'Olive Oil', 'Mustard Oil', 'Ghee', 
                             'Sugar', 'Brown Sugar', 'Jaggery', 'Salt', 'Iodized Salt', 'Black Pepper', 'Red Chilli Powder', 
                             'Turmeric Powder', 'Coriander Powder', 'Cumin Seeds', 'Mustard Seeds', 'Tea', 'Green Tea', 
                             'Coffee', 'Milk', 'Butter', 'Cheese', 'Paneer', 'Curd/Yogurt', 'Honey', 'Cashew Nuts', 
                             'Almonds', 'Raisins']
            }
            
            for category, items in category_matches.items():
                if product_query.lower() in category:
                    filtered_items = {k: base_products[k] for k in items if k in base_products}
                    base_products = filtered_items
                    break
    
    data = []
    for product, base_price in base_products.items():
        # Categorize products
        if product in ['Rice', 'Basmati Rice', 'Brown Rice', 'Wheat Flour', 'Wheat', 'Maida', 'Rava/Suji', 'Oats']:
            commodity_type = 'Cereals'
        elif product in ['Toor Dal', 'Moong Dal', 'Chana Dal', 'Masoor Dal', 'Urad Dal', 'Rajma', 'Black Chickpeas', 'Green Peas']:
            commodity_type = 'Pulses'
        elif product in ['Apple', 'Red Delicious Apple', 'Green Apple', 'Banana', 'Orange', 'Mango', 'Alphonso Mango', 
                         'Grapes', 'Black Grapes', 'Watermelon', 'Papaya', 'Pineapple', 'Pomegranate', 'Sweet Lime', 
                         'Guava', 'Strawberry', 'Kiwi']:
            commodity_type = 'Fruits'
        elif product in ['Potato', 'Onion', 'Tomato', 'Carrot', 'Cabbage', 'Cauliflower', 'Cucumber', 'Capsicum', 
                         'Green Beans', 'Bitter Gourd', 'Lady Finger', 'Eggplant', 'Pumpkin', 'Green Chilli', 
                         'Ginger', 'Garlic', 'Spinach', 'Coriander', 'Mint', 'Broccoli']:
            commodity_type = 'Vegetables'
        else:
            commodity_type = 'Groceries'
        
        # Create more realistic price fluctuations with seasonal patterns
        # Generate a seasonal component with a 30-day cycle
        seasonal_component = np.sin(np.linspace(0, 4*np.pi, len(dates))) * (base_price * 0.1)
        
        # Generate a trend component
        trend = np.random.choice([-1, 0, 1]) * 0.5  # -0.5, 0, or 0.5
        trend_component = np.array([trend * i * (base_price * 0.005) for i in range(len(dates))])
        
        # Generate random noise
        noise = np.random.normal(0, base_price * 0.02, len(dates))
        
        # Generate market-specific variation
        markets = ['Local Market', 'Wholesale Market', 'Government Market']
        market_factors = {'Local Market': 1.1, 'Wholesale Market': 0.9, 'Government Market': 1.0}
        
        for date_idx, date in enumerate(dates):
            price_base = base_price + seasonal_component[date_idx] + trend_component[date_idx] + noise[date_idx]
            
            for market in markets:
                # Add market-specific adjustment
                market_price = price_base * market_factors[market]
                # Add daily fluctuation
                daily_adj = np.random.normal(0, market_price * 0.01)
                final_price = max(market_price + daily_adj, base_price * 0.6)  # Ensure price doesn't go too low
                
                data.append({
                    'commodity': product,
                    'commodity_type': commodity_type,
                    'price': round(final_price, 2),
                    'market': market,
                    'state': 'National Average',
                    'district': 'All',
                    'date': date
                })
    
    df = pd.DataFrame(data)
    df.to_csv(cache_file, index=False)
    logger.info(f"Generated historical data for {len(base_products)} products across {len(markets)} markets")
    return df

# Modified fetch_all_prices with cache expiry
def fetch_all_prices(product_query=None):
    query_slug = product_query.replace(' ', '_').lower() if product_query else 'all'
    cache_file = f"cache/prices_{query_slug}.csv"
    
    should_refresh_cache = True
    
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        # Check if cache is fresher than 5 minutes
        if datetime.now() - cache_time < timedelta(minutes=CACHE_EXPIRY_MINUTES):
            should_refresh_cache = False
            logger.info(f"Using cached price data for {product_query}")
            cache_data = pd.read_csv(cache_file)
            if product_query:
                # Use regex for more flexible matching
                pattern = re.compile(f".*{re.escape(product_query)}.*", re.IGNORECASE)
                filtered_data = cache_data[cache_data['commodity'].str.contains(pattern, na=False) | 
                                          cache_data['commodity_type'].str.contains(pattern, na=False)]
                if not filtered_data.empty:
                    return filtered_data
            return cache_data
    
    # If cache doesn't exist or is expired, fetch new data
    if should_refresh_cache:
        logger.info(f"Cache expired or not found for {product_query}. Fetching fresh data...")
        
        # Start a background task to refresh the cache
        refresh_cache_in_background(product_query)
        
        # Fetch new data immediately for the current request
        bigbasket_df = fetch_bigbasket_prices(product_query)
        historical_df = fetch_historical_data(product_query)
        dfs = [df for df in [bigbasket_df, historical_df] if not df.empty]
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Save to cache
            combined_df.to_csv(cache_file, index=False)
            logger.info(f"Combined {len(combined_df)} price records and updated cache")
            return combined_df
    
    logger.warning("No price data found")
    return pd.DataFrame()

# Background cache refresh
def refresh_cache_in_background(product_query=None):
    """Refresh cache in a background thread"""
    def background_task():
        try:
            logger.info(f"Background task: Refreshing cache for {product_query}")
            query_slug = product_query.replace(' ', '_').lower() if product_query else 'all'
            cache_file = f"cache/prices_{query_slug}.csv"
            
            # Fetch new data
            bigbasket_df = fetch_bigbasket_prices(product_query)
            historical_df = fetch_historical_data(product_query)
            dfs = [df for df in [bigbasket_df, historical_df] if not df.empty]
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                # Save to cache
                combined_df.to_csv(cache_file, index=False)
                logger.info(f"Background task completed: Updated cache with {len(combined_df)} price records")
            else:
                logger.warning("Background task: No price data found to update cache")
        except Exception as e:
            logger.error(f"Error in background cache refresh task: {str(e)}")
    
    # Start the background thread
    thread = threading.Thread(target=background_task)
    thread.daemon = True  # Thread will exit when main program exits
    thread.start()

# Check for price updates
def check_for_price_updates(old_df, new_df):
    """
    Check if prices have changed between old and new dataframes
    Returns a dataframe with price changes
    """
    if old_df.empty or new_df.empty:
        return pd.DataFrame()
    
    # Ensure date columns are datetime
    old_df['date'] = pd.to_datetime(old_df['date'])
    new_df['date'] = pd.to_datetime(new_df['date'])
    
    # Get the latest data for each product in both dataframes
    old_latest = old_df.sort_values('date', ascending=False).drop_duplicates(subset=['commodity', 'market'])
    new_latest = new_df.sort_values('date', ascending=False).drop_duplicates(subset=['commodity', 'market'])
    
    # Merge to find differences
    merged = pd.merge(
        old_latest, new_latest, 
        on=['commodity', 'market'], 
        suffixes=('_old', '_new')
    )
    
    # Calculate price changes
    merged['price_diff'] = merged['price_new'] - merged['price_old']
    merged['price_diff_pct'] = (merged['price_diff'] / merged['price_old']) * 100
    
    # Filter only where price changed
    changed = merged[merged['price_diff'] != 0]
    
    if not changed.empty:
        logger.info(f"Found {len(changed)} price updates")
        # Log significant changes
        significant_changes = changed[abs(changed['price_diff_pct']) > 5]
        for _, row in significant_changes.iterrows():
            logger.info(f"Significant price change for {row['commodity']} at {row['market']}: "
                       f"₹{row['price_old']:.2f} → ₹{row['price_new']:.2f} "
                       f"({row['price_diff_pct']:.2f}%)")
    
    return changed

# Clean expired cache
def clean_expired_cache():
    """Remove cache files older than CACHE_EXPIRY_MINUTES"""
    cache_dir = 'cache'
    now = datetime.now()
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        if os.path.isfile(file_path):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_mtime > timedelta(minutes=CACHE_EXPIRY_MINUTES):
                try:
                    os.remove(file_path)
                    logger.info(f"Removed expired cache file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing cache file {filename}: {str(e)}")

# Initialize scheduler
def initialize_scheduler():
    """Initialize the scheduler for periodic tasks"""
    scheduler = BackgroundScheduler()
    
    # Schedule cache refresh every 5 minutes
    scheduler.add_job(
        lambda: refresh_cache_in_background(None),  # Refresh all products
        'interval', 
        minutes=CACHE_EXPIRY_MINUTES
    )
    
    # Schedule cache cleanup every 30 minutes
    scheduler.add_job(clean_expired_cache, 'interval', minutes=30)
    
    # Start the scheduler
    scheduler.start()
    
    # Register shutdown
    import atexit
    atexit.register(lambda: scheduler.shutdown())
    
    return scheduler

# Train prediction model with improved features
def train_prediction_model():
    try:
        logger.info("Training price prediction model...")
        df = fetch_historical_data()
        if df.empty:
            logger.warning("No data for training")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract more meaningful temporal features
        df['weekday'] = df['date'].dt.weekday
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Add simulated temperature data - would be replaced with real data in production
        df['temperature'] = np.random.uniform(10, 35, len(df))
        # Make temperatures seasonally appropriate
        df.loc[df['month'].isin([12, 1, 2]), 'temperature'] = np.random.uniform(5, 20, len(df[df['month'].isin([12, 1, 2])]))
        df.loc[df['month'].isin([5, 6, 7]), 'temperature'] = np.random.uniform(25, 40, len(df[df['month'].isin([5, 6, 7])]))
        
        df = df.sort_values(['commodity', 'date'])
        
        # Add more lag features for better time series modeling
        df['price_lag_1'] = df.groupby('commodity')['price'].shift(1)
        df['price_lag_3'] = df.groupby('commodity')['price'].shift(3)
        df['price_lag_7'] = df.groupby('commodity')['price'].shift(7)
        df['price_lag_14'] = df.groupby('commodity')['price'].shift(14)
        
        # Add rolling mean features
        df['price_rolling_3'] = df.groupby('commodity')['price'].rolling(window=3).mean().reset_index(level=0, drop=True)
        df['price_rolling_7'] = df.groupby('commodity')['price'].rolling(window=7).mean().reset_index(level=0, drop=True)
        
        # Clean missing values
        df = df.dropna()
        
        # Encode categorical variables
        label_encoders = {}
        for cat_col in ['commodity', 'commodity_type', 'market', 'state', 'district']:
            le = LabelEncoder()
            df[f'{cat_col}_encoded'] = le.fit_transform(df[cat_col])
            label_encoders[cat_col] = le
        
        # Select features for model training
        features = [
            'weekday', 'month', 'day', 'day_of_year', 'week_of_year',
            'temperature', 'price_lag_1', 'price_lag_3', 'price_lag_7', 
            'price_lag_14', 'price_rolling_3', 'price_rolling_7',
            'commodity_encoded', 'commodity_type_encoded', 'market_encoded'
        ]
        
        X = df[features]
        y = df['price']
        
        # Split data with stratification by commodity type
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=df['commodity_type_encoded'] if len(df['commodity_type_encoded'].unique()) < 10 else None
        )
        
        # Train a more robust model
        model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train, y_train)
        
        logger.info(f"Model R² - Train: {model.score(X_train, y_train):.4f}, Test: {model.score(X_test, y_test):.4f}")
        return {'model': model, 'label_encoders': label_encoders, 'features': features}
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None

# Get price history with improved filtering
def get_price_history(product, days=30):
    df = fetch_historical_data(product)
    if df.empty:
        return []
    
    # Improved product filtering with case-insensitive exact matching
    filtered_df = df[df['commodity'].str.lower() == product.lower()]
    
    # If exact match returns nothing, try partial matching
    if filtered_df.empty:
        filtered_df = df[df['commodity'].str.contains(product, case=False)]
        
    if filtered_df.empty:
        return []
    
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    filtered_df = filtered_df.sort_values('date')
    
    # Filter to the requested time period
    cutoff_date = datetime.now() - timedelta(days=days)
    filtered_df = filtered_df[filtered_df['date'] >= cutoff_date]
    
    # Format date for output
    filtered_df['date'] = filtered_df['date'].dt.strftime('%Y-%m-%d')
    
    # Return more detailed price history including market information
    return filtered_df[['date', 'price', 'market', 'commodity_type']].to_dict('records')

# Analyze price trends with more detailed factors
def analyze_price_trends(product, history, temperature=None):
    if not history:
        return {
            'trend': 'Unknown',
            'trend_description': 'No data available',
            'factors': []
        }
    
    # Group prices by date and calculate average if multiple entries per day
    price_df = pd.DataFrame(history)
    if 'date' not in price_df.columns or price_df.empty:
        return {
            'trend': 'Unknown',
            'trend_description': 'Insufficient data',
            'factors': []
        }
    
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df['price'] = price_df['price'].astype(float)
    
    # Get daily average prices
    daily_prices = price_df.groupby('date')['price'].mean().reset_index()
    daily_prices = daily_prices.sort_values('date')
    
    prices = daily_prices['price'].tolist()
    dates = daily_prices['date'].tolist()
    
    if len(prices) < 2:
        return {
            'trend': 'Unknown',
            'trend_description': 'Insufficient price history',
            'factors': []
        }
    
    current_price = prices[-1]
    yesterday_price = prices[-2]
    price_diff = current_price - yesterday_price
    percent_change = (price_diff / yesterday_price * 100) if yesterday_price != 0 else 0
    
    # Calculate weekly and monthly changes if enough data
    weekly_change = None
    monthly_change = None
    
    if len(prices) >= 7:
        week_ago_price = prices[-7]
        weekly_change = ((current_price - week_ago_price) / week_ago_price * 100) if week_ago_price != 0 else 0
    
    if len(prices) >= 30:
        month_ago_price = prices[-30]
        monthly_change = ((current_price - month_ago_price) / month_ago_price * 100) if month_ago_price != 0 else 0
    
    # Determine trend with more detailed analysis
    if abs(percent_change) < 1:
        trend = "Stable →"
        trend_description = "Price remained stable over the last day"
    elif price_diff > 0:
        if percent_change > 5:
            trend = "Sharp Rise ↑↑"
            trend_description = f"Price increased significantly by {abs(percent_change):.2f}% (₹{abs(price_diff):.2f}) in the last day"
        else:
            trend = "Rising ↑"
            trend_description = f"Price increased by {abs(percent_change):.2f}% (₹{abs(price_diff):.2f}) in the last day"
    else:
        if percent_change < -5:
            trend = "Sharp Fall ↓↓"
            trend_description = f"Price decreased significantly by {abs(percent_change):.2f}% (₹{abs(price_diff):.2f}) in the last day"
        else:
            trend = "Falling ↓"
            trend_description = f"Price decreased by {abs(percent_change):.2f}% (₹{abs(price_diff):.2f}) in the last day"
    
    # Add weekly and monthly context to the trend description
    if weekly_change is not None:
        trend_description += f" | {abs(weekly_change):.2f}% {'higher' if weekly_change > 0 else 'lower'} than last week"
    
    if monthly_change is not None:
        trend_description += f" | {abs(monthly_change):.2f}% {'higher' if monthly_change > 0 else 'lower'} than last month"
    
    # Identify factors affecting prices
    factors = []
    
    # Temperature factor
    if temperature is not None:
        if temperature > 30 and any(word in product.lower() for word in ['vegetable', 'fruit', 'tomato', 'onion']):
            factors.append({
                'factor': 'High Temperature',
                'impact': 'Hot weather may affect supply and freshness, increasing prices',
                'importance': 'high'
            })
        elif temperature < 10 and any(word in product.lower() for word in ['vegetable', 'fruit']):
            factors.append({
                'factor': 'Low Temperature',
                'impact': 'Cold weather may affect yields, impacting supply',
                'importance': 'high'
            })
        trend_description += f". Current temperature: {temperature}°C."
    
    # Seasonal factors
    current_month = datetime.now().month
    if current_month in [5, 6, 7]:  # Summer
        if any(word in product.lower() for word in ['mango', 'watermelon']):
            factors.append({
                'factor': 'Peak Season',
                'impact': 'Summer fruits are in season, potentially reducing prices',
                'importance': 'high'
            })
        elif any(word in product.lower() for word in ['apple', 'orange']):
            factors.append({
                'factor': 'Off Season',
                'impact': 'Some fruits are off-season, potentially increasing prices',
                'importance': 'medium'
            })
    elif current_month in [11, 12, 1]:  # Winter
        if any(word in product.lower() for word in ['carrot', 'cauliflower', 'cabbage']):
            factors.append({
                'factor': 'Peak Season',
                'impact': 'Winter vegetables are in season, potentially reducing prices',
                'importance': 'high'
            })
    
    # Market demand factors
    if any(word in product.lower() for word in ['rice', 'wheat', 'flour', 'dal']):
        factors.append({
            'factor': 'Staple Food',
            'impact': 'Steady demand as an essential food item',
            'importance': 'medium'
        })
    
    # Price volatility analysis
    if len(prices) >= 14:
        recent_prices = prices[-14:]
        volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
        if volatility > 10:
            factors.append({
                'factor': 'High Price Volatility',
                'impact': f'Prices have been fluctuating (±{volatility:.1f}% variation)',
                'importance': 'high'
            })
    
    return {
        'trend': trend,
        'trend_description': trend_description,
        'factors': factors
    }

# Generate detailed price plot with improved visualization
def generate_detailed_price_plot(product, history, filename):
    if not history:
        return None
    
    # Convert history to DataFrame for easier manipulation
    df = pd.DataFrame(history)
    if df.empty:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = df['price'].astype(float)
    
    # Group by date and get average prices
    daily_avg = df.groupby('date')['price'].mean().reset_index()
    daily_avg = daily_avg.sort_values('date')
    
    # Calculate rolling average for smoother trend line
    daily_avg['rolling_avg'] = daily_avg['price'].rolling(window=7, min_periods=1).mean()
    
    # Calculate min and max for each day
    daily_min = df.groupby('date')['price'].min().reset_index()
    daily_max = df.groupby('date')['price'].max().reset_index()
    
    # Create a more professional looking plot
    plt.figure(figsize=(12, 7))
    plt.plot(daily_avg['date'], daily_avg['price'], marker='o', markersize=5, 
             linestyle='-', linewidth=2, color='#3498db', label='Daily Average Price')
    plt.plot(daily_avg['date'], daily_avg['rolling_avg'], linestyle='-', 
             linewidth=3, color='#e74c3c', label='7-Day Moving Average')
    
    # Add price range as a shaded area (min to max)
    plt.fill_between(daily_min['date'], daily_min['price'], daily_max['price'],
                     color='#3498db', alpha=0.2, label='Price Range')
    
    # Add trend line
    x = np.arange(len(daily_avg))
    z = np.polyfit(x, daily_avg['price'], 1)
    p = np.poly1d(z)
    plt.plot(daily_avg['date'], p(x), "k--", alpha=0.8, linewidth=2,
             label=f"Trend: {'Rising' if z[0] > 0 else 'Falling'} at ₹{abs(z[0]):.2f}/day")
    
    # Enhance the plot appearance
    plt.title(f'Price History for {product}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Annotate current price
    latest_price = daily_avg.iloc[-1]['price']
    latest_date = daily_avg.iloc[-1]['date']
    plt.annotate(f'Current: ₹{latest_price:.2f}', 
                 xy=(latest_date, latest_price),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Save with high resolution
    plt.savefig(filename, dpi=120)
    plt.close()
    
    return filename

# Generate market comparison plot to show price differences across markets
def generate_market_comparison_plot(product, history, filename):
    if not history:
        return None
    
    df = pd.DataFrame(history)
    if df.empty or 'market' not in df.columns:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = df['price'].astype(float)
    
    # Get the most recent data points for each market
    markets = df['market'].unique()
    if len(markets) <= 1:
        return None  # No comparison needed if only one market
    
    # Create figure with market comparison
    plt.figure(figsize=(12, 7))
    
    # Use color palette for different markets
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#1abc9c', '#34495e']
    
    for i, market in enumerate(markets):
        market_data = df[df['market'] == market].sort_values('date')
        if len(market_data) > 0:
            plt.plot(market_data['date'], market_data['price'], 
                     marker='o', markersize=4, 
                     linestyle='-', linewidth=2, 
                     color=colors[i % len(colors)], 
                     label=market)
    
    # Enhance the plot appearance
    plt.title(f'Market Price Comparison for {product}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    
    # Add current prices annotation
    for i, market in enumerate(markets):
        market_data = df[df['market'] == market].sort_values('date')
        if not market_data.empty:
            latest = market_data.iloc[-1]
            plt.annotate(f'{market}: ₹{latest["price"]:.2f}',
                xy=(latest['date'], latest['price']),
                xytext=(10, 10 + i*20), textcoords='offset points',
                color=colors[i % len(colors)],
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2',
                                color=colors[i % len(colors)]))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    
    return filename

# Updated index route
@app.route('/')
def index():
    # Load available products and categories for the search form
    df = fetch_all_prices()
    products = df['commodity'].unique().tolist() if not df.empty else []
    categories = df['commodity_type'].unique().tolist() if not df.empty else []
    
    return render_template('index.html', products=products, categories=categories)

# Updated search route to show only searched items
@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query') if request.method == 'POST' else request.args.get('query', '')
    location = request.form.get('location') if request.method == 'POST' else request.args.get('location', '')
    time_period = request.form.get('period') if request.method == 'POST' else request.args.get('period', '30')
    
    try:
        time_period = int(time_period)
    except ValueError:
        time_period = 30  # Default to 30 days if invalid
    
    if not query:
        return jsonify({'status': 'error', 'message': 'No search query provided'})
    
    temperature = None
    lat, lon = None, None
    if location:
        try:
            lat, lon = map(float, location.split(','))
            temperature = fetch_temperature(lat, lon)
        except:
            logger.warning("Invalid location format")
            
    # Get price data for specific query only
    df = fetch_all_prices(query)
    
    # If no results, try a more specific search within category
    if df.empty:
        # Try searching by category if the query matches a category name
        categories = ['Vegetables', 'Fruits', 'Pulses', 'Cereals', 'Groceries']
        if query.lower() in [c.lower() for c in categories]:
            df = fetch_all_prices()
            df = df[df['commodity_type'].str.contains(query, case=False, na=False)]
            logger.info(f"Showing all items in category: {query}")
        else:
            return jsonify({'status': 'error', 'message': f'No products found matching "{query}"'})
    
    # Enhanced search logic for better matches
    if query.lower() in ['vegetables', 'fruits', 'pulses', 'groceries', 'cereals']:
        # If query is a category, show all products in that category
        matches = df[df['commodity_type'].str.contains(query, case=False, na=False)]
    else:
        # Search for partial matches in commodity name or type
        matches = df[df['commodity'].str.contains(query, case=False, na=False) | 
                    df['commodity_type'].str.contains(query, case=False, na=False)]
    
    if matches.empty:
        return jsonify({'status': 'error', 'message': f'No products found matching "{query}"'})
    
    matches['date'] = pd.to_datetime(matches['date'])
    latest_prices = matches.sort_values('date', ascending=False).drop_duplicates(subset=['commodity', 'market'])
    
    # Ensure we have a prediction model
    if not hasattr(app, 'prediction_model') or app.prediction_model is None:
        app.prediction_model = train_prediction_model()
    
    results = []
    for _, row in latest_prices.iterrows():
        product = row['commodity']
        
        # Get comprehensive price history with the requested time period
        history = get_price_history(product, time_period)
        
        # Generate improved price history visualization
        detailed_plot = generate_detailed_price_plot(
            product, 
            history, 
            f"static/plots/history_{product.replace(' ', '_')}.png"
        )
        
        trend_analysis = analyze_price_trends(product, history, temperature)
        
        # Generate comparative market plot showing all markets for this product
        market_comparison_plot = generate_market_comparison_plot(
            product, 
            history, 
            f"static/plots/markets_{product.replace(' ', '_')}.png"
        )
        
        # Get prediction for 7 days ahead
        prediction_data = get_price_prediction(product, 7, lat, lon, row)
        
        results.append({
            'product': product,
            'current_price': float(row['price']),
            'market': row['market'],
            'category': row['commodity_type'],
            'date': row['date'].strftime('%Y-%m-%d'),
            'price_history': history,
            'trend': trend_analysis['trend'],
            'trend_description': trend_analysis['trend_description'],
            'factors': trend_analysis['factors'],
            'history_plot': detailed_plot,
            'market_comparison_plot': market_comparison_plot,
            'temperature': temperature,
            'predicted_prices': prediction_data.get('predictions', []),
            'prediction_plot': prediction_data.get('plot_url')
        })
    
    return jsonify({
        'status': 'success',
        'query': query,
        'results': results,
        'period': time_period
    })

# Updated get_price_prediction to handle null location and Series input
def get_price_prediction(product, days_ahead=7, lat=None, lon=None, current_data=None):
    try:
        if current_data is None:
            df = fetch_historical_data(product)
            if df.empty:
                return {'status': 'error', 'message': f'No historical data for {product}'}
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            most_recent = df.iloc[0].to_dict()
        else:
            most_recent = current_data.to_dict() if isinstance(current_data, pd.Series) else current_data
        
        # Get temperature with better error handling
        temperature = None
        if lat is not None and lon is not None:
            temperature = fetch_temperature(lat, lon)
        if temperature is None:
            temperature = 28.0  # Default temperature if unavailable
        
        if not hasattr(app, 'prediction_model') or app.prediction_model is None:
            app.prediction_model = train_prediction_model()
        
        if app.prediction_model is None:
            return {'status': 'error', 'message': 'Prediction model unavailable'}
        
        model = app.prediction_model['model']
        label_encoders = app.prediction_model['label_encoders']
        
        # Get historical data for lags with better error handling
        df = fetch_historical_data(product)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False)
        
        # Ensure date is in datetime format
        if isinstance(most_recent['date'], str):
            today = datetime.strptime(most_recent['date'], '%Y-%m-%d')
        else:
            today = most_recent['date']
            
        future_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_ahead + 1)]
        
        predictions = []
        current_price = most_recent['price'] if isinstance(most_recent['price'], (int, float)) else float(most_recent['price'])
        price_lag_1 = current_price
        price_lag_7 = df.iloc[min(6, len(df)-1)]['price'] if len(df) > 6 else current_price
        
        for future_date in future_dates:
            date_obj = datetime.strptime(future_date, '%Y-%m-%d')
            X_pred = pd.DataFrame({
                'weekday': [date_obj.weekday()],
                'month': [date_obj.month],
                'day': [date_obj.day],
                'temperature': [temperature],
                'price_lag_1': [price_lag_1],
                'price_lag_7': [price_lag_7],
                'commodity_encoded': [label_encoders['commodity'].transform([most_recent['commodity']])[0]],
                'commodity_type_encoded': [label_encoders['commodity_type'].transform([most_recent['commodity_type']])[0]],
                'market_encoded': [label_encoders['market'].transform([most_recent['market']])[0]]
            })
            
            predicted_price = model.predict(X_pred)[0]
            price_lag_7 = price_lag_1 if len(predictions) >= 6 else price_lag_7
            price_lag_1 = predicted_price
            
            predictions.append({
                'date': future_date,
                'predicted_price': round(predicted_price, 2)
            })
        
        # Create improved prediction plot
        plt.figure(figsize=(12, 7))
        plot_dates = [today.strftime('%Y-%m-%d')] + future_dates
        plot_prices = [current_price] + [p['predicted_price'] for p in predictions]
        
        # Plot current price and predictions with better styling
        plt.plot(plot_dates[:-days_ahead], plot_prices[:1], marker='o', markersize=8, 
                 linestyle='-', linewidth=2.5, color='#3498db', label='Current')
        plt.plot(plot_dates[-days_ahead-1:], plot_prices[-days_ahead-1:], marker='o', 
                 markersize=8, linestyle='--', linewidth=2.5, color='#e74c3c', 
                 label=f'Predicted (Temp: {temperature}°C)')
        
        # Add confidence interval (simple implementation)
        upper_bound = [price * 1.1 for price in plot_prices[-days_ahead-1:]]
        lower_bound = [price * 0.9 for price in plot_prices[-days_ahead-1:]]
        plt.fill_between(plot_dates[-days_ahead-1:], lower_bound, upper_bound, 
                          color='#e74c3c', alpha=0.15, label='Confidence Interval')
        
        plt.title(f'Price Prediction for {product}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (₹)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save plot with high quality
        plot_filename = f"static/plots/prediction_{product.replace(' ', '_')}.png"
        plt.savefig(plot_filename, dpi=120)
        plt.close()
        
        return {
            'status': 'success',
            'product': product,
            'current_price': current_price,
            'temperature': temperature,
            'predictions': predictions,
            'plot_url': plot_filename
        }
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {'status': 'error', 'message': f'Error: {str(e)}'}

if __name__ == '__main__':
    # Clean expired cache files
    clean_expired_cache()
    
    # Initialize prediction model on startup
    app.prediction_model = train_prediction_model()
    
    # Initialize scheduler for background tasks
    scheduler = initialize_scheduler()
    
    # Start the application
    app.run(debug=True, host='0.0.0.0', port=5000)