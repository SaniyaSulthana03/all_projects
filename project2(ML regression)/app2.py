import streamlit as st
import pandas as pd            # ✅ THIS WAS MISSING
import joblib
from xgboost import XGBRegressor

st.set_page_config(page_title="Inventory Stress Predictor")

# ===========================
# UI INPUTS
# ===========================
st.title("Inventory Stress & Demand Mismatch Early Warning System")

st.sidebar.header("SKU Input Parameters")

Inventory_Level = st.sidebar.number_input("Inventory Level", min_value=0, value=100)
Units_Ordered = st.sidebar.number_input("Units Ordered", min_value=0, value=50)
Price = st.sidebar.number_input("Price", value=20.0)
Discount = st.sidebar.number_input("Discount (%)", 0, 100, 0)
Promotion = st.sidebar.selectbox("Promotion", [0, 1])
Epidemic = st.sidebar.selectbox("Epidemic", [0, 1])
Price_Gap = st.sidebar.number_input("Price Gap vs Competitor", value=0.0)

Seasonality = st.sidebar.selectbox(
    "Seasonality", ["Winter", "Spring", "Summer", "Autumn"]
)
Category = st.sidebar.selectbox(
    "Category", ["Electronics", "Clothing", "Groceries", "Furniture", "Toys"]
)
Region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
Weather_Condition = st.sidebar.selectbox(
    "Weather Condition", ["Sunny", "Rainy", "Snowy", "Cloudy"]
)

# ===========================
# LOAD MODEL
# ===========================
@st.cache_resource
def load_model():
    preprocessor = joblib.load("preprocessor.pkl")
    model = XGBRegressor()
    model.load_model("xgb_model.json")
    return preprocessor, model

preprocessor, model = load_model()

# ===========================
# INPUT DATAFRAME
# ===========================
input_df = pd.DataFrame(
    [[
        Inventory_Level,
        Units_Ordered,
        Price,
        Discount,
        Promotion,
        Price_Gap,
        Seasonality,
        Category,
        Region,
        Weather_Condition,
        Epidemic
    ]],
    columns=[
        "Inventory_Level",
        "Units_Ordered",
        "Price",
        "Discount",
        "Promotion",
        "Price_Gap",
        "Seasonality",
        "Category",
        "Region",
        "Weather_Condition",
        "Epidemic"
    ]
)

# ===========================
# PREDICTION
# ===========================
X_input_processed = preprocessor.transform(input_df)
prediction = float(model.predict(X_input_processed)[0])

# ===========================
# RISK LOGIC
# ===========================
if prediction <= 0:
    risk = "Overstocked"
elif prediction <= 20:
    risk = "Safe"
elif prediction <= 50:
    risk = "Moderate Risk"
else:
    risk = "High Risk"

# ===========================
# OUTPUT
# ===========================
st.subheader("Predicted Inventory Stress Score")
st.metric("Inventory Stress", round(prediction, 2))

st.subheader("Risk Category")
if risk == "High Risk":
    st.error(risk)
elif risk == "Moderate Risk":
    st.warning(risk)
else:
    st.success(risk)
