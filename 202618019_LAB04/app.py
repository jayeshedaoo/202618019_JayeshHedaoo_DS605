import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("airbnb_price_model.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Airbnb Nightly Price Predictor")

st.write(
    "Enter the details of an Airbnb listing to estimate "
    "its nightly price in New York City."
)


# --------------------------------------------------
# User inputs
# --------------------------------------------------

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    [
        "Manhattan",
        "Brooklyn",
        "Queens",
        "Bronx",
        "Staten Island"
    ]
)

neighbourhood = st.text_input(
    "Neighbourhood",
    "Midtown"
)

room_type = st.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ]
)

latitude = st.number_input(
    "Latitude",
    min_value=40.4,
    max_value=40.95,
    value=40.75,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    min_value=-74.3,
    max_value=-73.6,
    value=-73.98,
    format="%.6f"
)

minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=3
)

number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    value=20
)

reviews_per_month = st.number_input(
    "Reviews per Month",
    min_value=0.0,
    value=1.5
)

calculated_host_listings_count = st.number_input(
    "Host Listings Count",
    min_value=1,
    value=1
)

availability_365 = st.number_input(
    "Availability per Year",
    min_value=0,
    max_value=365,
    value=200
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Price"):

    availability_ratio = availability_365 / 365

    reviews_log = np.log1p(number_of_reviews)

    minimum_nights_log = np.log1p(minimum_nights)

    input_data = pd.DataFrame({
        "neighbourhood_group": [neighbourhood_group],
        "neighbourhood": [neighbourhood],
        "room_type": [room_type],
        "latitude": [latitude],
        "longitude": [longitude],
        "minimum_nights": [minimum_nights],
        "number_of_reviews": [number_of_reviews],
        "reviews_per_month": [reviews_per_month],
        "calculated_host_listings_count": [
            calculated_host_listings_count
        ],
        "availability_365": [availability_365],
        "availability_ratio": [availability_ratio],
        "reviews_log": [reviews_log],
        "minimum_nights_log": [minimum_nights_log]
    })

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    st.success(
        f"Estimated Nightly Price: ${prediction:.2f}"
    )