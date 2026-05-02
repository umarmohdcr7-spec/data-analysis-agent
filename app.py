import streamlit as st #Imports Streamlit for building the web app UI
import pandas as pd
import numpy as np
import joblib #Loads saved ML models and feature files.

# Load model and feature list
model = joblib.load("outputs/final_car_price_model.pkl")
model_features = joblib.load("outputs/model_features.pkl")

# Load cleaned dataset for dropdown options
df = pd.read_csv("outputs/cleaned_cars_dataset.csv")

st.title("AI-Powered Used Car Valuation Assistant")
st.title("By Mohammad Umar")

st.write("Estimate a fair market price and check whether a car listing is overpriced or underpriced.")

# Dynamic dropdowns
make_options = sorted(df["make"].dropna().unique().tolist())
fuel_options = sorted(df["fuel"].dropna().unique().tolist())
offer_options = sorted(df["offerType"].dropna().unique().tolist())

st.sidebar.header("Car Details")

mileage = st.sidebar.number_input("Mileage", min_value=0, value=50000)
hp = st.sidebar.number_input("Horsepower", min_value=1, value=120)
year = st.sidebar.number_input("Year", min_value=1990, max_value=2026, value=2018)

make = st.sidebar.selectbox("Make", make_options)
fuel = st.sidebar.selectbox("Fuel", fuel_options)
offer_type = st.sidebar.selectbox("Offer Type", offer_options)

listed_price = st.sidebar.number_input("Listed Price (€)", min_value=0, value=15000)

analyze_button = st.sidebar.button("Analyze Listing")

input_data = pd.DataFrame([{
    "mileage": mileage,
    "make": make,
    "fuel": fuel,
    "offerType": offer_type,
    "hp": hp,
    "year": year
}])

input_encoded = pd.get_dummies(input_data)

# Match training columns exactly
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

if analyze_button:
    log_prediction = model.predict(input_encoded)[0]
    predicted_price = np.expm1(log_prediction)
    lower_bound = predicted_price / np.exp(0.18)
    upper_bound = predicted_price * np.exp(0.18)

    difference = listed_price - predicted_price
    percentage_difference = (difference / predicted_price) * 100
    score = 50 + percentage_difference
    score = max(0, min(100, score))

    st.subheader("Valuation Score")
    st.progress(int(score))
    st.write(f"Score: **{score:.0f}/100**")

    if score > 60:
     st.write("Higher score means the listing is more expensive than the model’s fair value estimate.")
    elif score < 40:
     st.write("Lower score means the listing may be cheaper than the model’s fair value estimate.")
    else:
     st.write("The listing is close to the model’s fair value estimate.")

    st.subheader("Valuation Result")
    comparison_df = pd.DataFrame({
    "Type": ["Predicted Fair Price", "Listed Price"],
    "Price": [predicted_price, listed_price]
})

    st.subheader("Price Comparison")

    st.bar_chart(
    comparison_df.set_index("Type")
    )
    st.write(f"Estimated Fair Price: **€{predicted_price:,.0f}**")
    st.write(f"Expected Price Range: **€{lower_bound:,.0f} – €{upper_bound:,.0f}**")
    st.write(f"Listed Price: **€{listed_price:,.0f}**")

    if percentage_difference > 10:
        st.error(f"This listing appears overpriced by approximately €{difference:,.0f} ({percentage_difference:.1f}%).")
        #the .errror in the line above gives red box on the app.
    elif percentage_difference < -10:
        st.success(f"This listing appears underpriced by approximately €{abs(difference):,.0f} ({abs(percentage_difference):.1f}%).")
    else:
        st.info("This listing appears fairly priced based on the model prediction.")

    st.caption("Note: This is an estimated valuation based on historical listing data and should be used as decision support, not as an exact market price.")
    st.subheader("Why this result?")

explanations = []

if mileage > df["mileage"].median():
    explanations.append("The mileage is above the dataset median, which usually reduces the expected price.")
else:
    explanations.append("The mileage is below the dataset median, which usually supports a higher expected price.")

if hp > df["hp"].median():
    explanations.append("The horsepower is above average, which can increase the estimated value.")
else:
    explanations.append("The horsepower is below average, which may limit the estimated value.")

if year > df["year"].median():
    explanations.append("The car is newer than the dataset median year, which usually increases price.")
else:
    explanations.append("The car is older than the dataset median year, which usually reduces price.")

for explanation in explanations:
    st.write(f"- {explanation}")

st.subheader("Valuation Summary")

if percentage_difference > 10:
    summary = (
        f"Based on the model estimate, this car appears overpriced. "
        f"The predicted fair market value is around €{predicted_price:,.0f}, "
        f"while the listed price is €{listed_price:,.0f}. "
        f"The difference of approximately €{difference:,.0f} suggests that the listing "
        f"price may be too high for the given mileage, horsepower, year, fuel type, offer type, and make."
    )

elif percentage_difference < -10:
    summary = (
        f"Based on the model estimate, this car appears underpriced. "
        f"The predicted fair market value is around €{predicted_price:,.0f}, "
        f"while the listed price is €{listed_price:,.0f}. "
        f"This could indicate a potentially good deal, but the vehicle should still be "
        f"checked carefully for condition, history, and hidden issues."
    )

else:
    summary = (
        f"Based on the model estimate, this car appears fairly priced. "
        f"The listed price of €{listed_price:,.0f} is close to the predicted fair value "
        f"of €{predicted_price:,.0f}. "
        f"This suggests the listing is reasonably aligned with similar vehicles in the dataset."
    )

st.write(summary)