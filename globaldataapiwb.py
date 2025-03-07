import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sqlite3
import numpy as np
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# SQLite Database Setup
conn = sqlite3.connect("world_bank_data.db", check_same_thread=False)
cursor = conn.cursor()

# Ensure table exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS world_bank_data (
        country TEXT,
        country_code TEXT,
        indicator TEXT,
        year INTEGER,
        value REAL
    )
""")
conn.commit()

# Define the API URL
WORLD_BANK_API = "http://api.worldbank.org/v2/country/{}/indicator/{}?format=json&per_page=100"

# Define the indicators
INDICATORS = {
    "Health Expenditure (% GDP)": "SH.XPD.CHEX.GD.ZS",
    "GDP per Capita": "NY.GDP.PCAP.CD",
    "Poverty Headcount": "SI.POV.DDAY",
    "Human Capital Index": "HD.HCI.OVRL",
    "Life Expectancy": "SP.DYN.LE00.IN",
    "Unemployment Rate": "SL.UEM.TOTL.ZS",
    "Inflation Rate": "FP.CPI.TOTL.ZG",
    "Government Health Spending": "SH.XPD.GHED.GD.ZS"
}

# Fetch country list
@st.cache_data
def get_countries():
    url = "http://api.worldbank.org/v2/country?format=json&per_page=300"
    response = requests.get(url)
    data = response.json()
    countries = {item["name"]: item["id"] for item in data[1] if item["region"]["id"] != "NA"}
    return countries

COUNTRIES = get_countries()

# Fetch stored data from SQLite
def get_stored_data(country_code, indicator):
    query = """
        SELECT country, country_code, year, AVG(value) as value 
        FROM world_bank_data 
        WHERE country_code = ? AND indicator = ? 
        GROUP BY country, country_code, year
        ORDER BY year
    """
    df = pd.read_sql_query(query, conn, params=(country_code, indicator))
    return df

# Fetch data from API and store in SQLite
def fetch_data(country_code, country_name, indicator):
    url = WORLD_BANK_API.format(country_code, indicator)
    response = requests.get(url)
    data = response.json()

    if len(data) > 1 and "value" in data[1][0]:
        records = [{"year": int(entry["date"]), "value": entry["value"]} for entry in data[1] if entry["value"] is not None]

        for record in records:
            cursor.execute("""
                INSERT INTO world_bank_data (country, country_code, indicator, year, value)
                VALUES (?, ?, ?, ?, ?)
            """, (country_name, country_code, indicator, record["year"], record["value"]))
        conn.commit()
        return pd.DataFrame(records)

    return pd.DataFrame(columns=["year", "value"])

# Fetch country coordinates
@st.cache_data
def get_country_coordinates(country_name):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(country_name)
    if location:
        return location.latitude, location.longitude
    return None, None

# Display all selected countries on a single map
def display_countries_map(selected_countries):
    m = folium.Map(location=[0, 0], zoom_start=2)
    for country_name in selected_countries:
        lat, lon = get_country_coordinates(country_name)
        if lat and lon:
            folium.Marker([lat, lon], popup=country_name).add_to(m)
    folium_static(m)

# ARIMA Forecasting
def arima_forecast(data, future_years):
    data = data.set_index("year")["value"].dropna()
    if len(data) < 5:
        return None

    model = ARIMA(data, order=(2, 1, 2))  
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(future_years))

    return pd.DataFrame({"year": future_years, "ARIMA Prediction": forecast.values})

# LSTM Forecasting
def lstm_forecast(data, future_years):
    if len(data) < 5:
        return None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = data["value"].values.reshape(-1, 1)
    values_scaled = scaler.fit_transform(values)
    
    X_train, y_train = [], []
    for i in range(len(values_scaled) - 3):
        X_train.append(values_scaled[i:i+3])
        y_train.append(values_scaled[i+3])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(3, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    last_values = values_scaled[-3:].reshape(1, 3, 1)
    lstm_preds = []
    for _ in range(len(future_years)):
        pred = model.predict(last_values)[0]
        lstm_preds.append(pred)
        last_values = np.append(last_values[:, 1:, :], [[pred]], axis=1)
    
    lstm_preds = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1))
    return pd.DataFrame({"year": future_years, "LSTM Prediction": lstm_preds.flatten()})

# Streamlit UI
def main():
    st.title(" World Economy & Health Dashboard and Forecasting")

    st.sidebar.header("Select Options")
    selected_countries = st.sidebar.multiselect("Select Countries", list(COUNTRIES.keys()), default=["Burkina Faso", "Zimbabwe"])
    selected_indicator = st.sidebar.selectbox("Select Indicator", list(INDICATORS.keys()))
    
    st.sidebar.subheader("Forecasting Options")
    forecast_years = st.sidebar.slider("Select Forecasting Years", 5, 20, 10)
    future_years = np.arange(2025, 2025 + forecast_years)

    if not selected_countries:
        st.warning("Please select at least one country.")
        return

    st.subheader("Map of Selected Countries")
    display_countries_map(selected_countries)

    st.subheader("Indicator Definitions")
    indicator_info = pd.DataFrame({
        "Indicator": list(INDICATORS.keys()),
        "Description": [
            "Total health expenditure as a percentage of GDP",
            "Gross Domestic Product per capita (USD)",
            "Percentage of population living below national poverty line",
            "Human Capital Index score",
            "Average life expectancy at birth (years)",
            "Total unemployment rate (%)",
            "Annual percentage change in consumer prices",
            "Government expenditure on health as a percentage of GDP"
        ]
    })
    st.table(indicator_info)

    

    for country_name in selected_countries:
        country_code = COUNTRIES.get(country_name)
        
        if not country_code:
            st.warning(f"Country code not found for {country_name}")
            continue

        st.subheader(f"📈 Forecast for {selected_indicator} in {country_name}")
        data = get_stored_data(country_code, INDICATORS[selected_indicator])

        if data.empty:
            data = fetch_data(country_code, country_name, INDICATORS[selected_indicator])

        if not data.empty:
            arima_result = arima_forecast(data, future_years)
            lstm_result = lstm_forecast(data, future_years)
            
            fig = px.line(data, x="year", y="value", markers=True, title=f"{selected_indicator} Trend in {country_name}", labels={"value": f"{selected_indicator} (units)"})
            if arima_result is not None:
                fig.add_scatter(x=arima_result["year"], y=arima_result["ARIMA Prediction"], mode='lines+markers', name="ARIMA Prediction")
            if lstm_result is not None:
                fig.add_scatter(x=lstm_result["year"], y=lstm_result["LSTM Prediction"], mode='lines+markers', name="LSTM Prediction")
            st.plotly_chart(fig)
    

# Run the app
if __name__ == "__main__":
    main()
