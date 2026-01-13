import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# Page configuration (minimal & professional)
# -------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -------------------------------------------------
# Load trained artifacts
# -------------------------------------------------
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl")

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("🏠 House Price Prediction")
st.caption(
    "A machine learning application that estimates house prices "
    "and provides simple buying guidance."
)

st.divider()

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
st.subheader("House Details")

area = st.number_input(
    "Area (sq.ft)",
    min_value=200,
    max_value=10000,
    step=50
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    step=1
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    step=1
)

floors = st.number_input(
    "Floors",
    min_value=1,
    max_value=5,
    step=1
)

year_built = st.number_input(
    "Year Built",
    min_value=1950,
    max_value=2025,
    step=1
)

location = st.selectbox(
    "Location",
    ["Urban", "Semi-Urban", "Rural"]
)

condition = st.selectbox(
    "Condition",
    ["Poor", "Average", "Good", "Excellent"]
)

garage = st.selectbox(
    "Garage",
    ["Yes", "No"]
)

# -------------------------------------------------
# Create input DataFrame 
# -------------------------------------------------
input_df = pd.DataFrame([{
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Floors": floors,
    "YearBuilt": year_built,
    "Location": location,
    "Condition": condition,
    "Garage": garage
}])

# -------------------------------------------------
# Preprocessing 
# -------------------------------------------------
num_cols = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
cat_cols = ["Location", "Condition", "Garage"]

xnum = scaler.transform(input_df[num_cols])
xcat = ohe.transform(input_df[cat_cols])

df1 = np.hstack([xnum, xcat])

# -------------------------------------------------
# Prediction & Recommendation
# -------------------------------------------------
st.divider()

if st.button("Predict Price"):
    price = model.predict(df1)[0]

    st.success(f"Estimated House Price: ₹ {price:,.0f}")

    st.subheader("Recommendation")

    if price < 300000:
        st.info(
            "This property is **budget-friendly** and may be suitable for "
            "first-time buyers or rental investment."
        )
    elif price <= 700000:
        st.success(
            "This property is **fairly priced** and offers a good balance "
            "between cost and features."
        )
    else:
        st.warning(
            "This is a **premium-priced** property. Consider comparing "
            "similar listings or negotiating."
        )
