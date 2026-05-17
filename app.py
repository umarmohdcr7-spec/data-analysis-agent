import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Used Car Valuation Assistant",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.hero {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 2.2rem;
    border-radius: 22px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 44px;
    margin-bottom: 0.4rem;
}

.hero p {
    font-size: 18px;
    color: #dbeafe;
}

.card {
    background-color: #111827;
    padding: 1.4rem;
    border-radius: 18px;
    border: 1px solid #263244;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    margin-bottom: 1.2rem;
}

.small-muted {
    color: #6b7280;
    font-size: 14px;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# Load model and feature list
model = joblib.load("outputs/deploy_car_price_model.pkl")
model_features = joblib.load("outputs/model_features.pkl")

# Load cleaned dataset for dropdown options
df = pd.read_csv("outputs/cleaned_cars_dataset.csv")

st.markdown("""
<div class="hero">
    <h1>Used Car Valuation Assistant</h1>
    <p>Estimate fair market value, compare listed prices, and identify overpriced or underpriced used cars.</p>
</div>
""", unsafe_allow_html=True)

# Dynamic dropdowns
make_options = sorted(df["make"].dropna().unique().tolist())
fuel_options = sorted(df["fuel"].dropna().unique().tolist())
offer_options = sorted(df["offerType"].dropna().unique().tolist())

# Sidebar inputs
st.sidebar.header("Car Details")

st.sidebar.markdown("### Enter Vehicle Specifications")
st.sidebar.caption("Adjust vehicle parameters to generate an AI-powered valuation.")

mileage = st.sidebar.number_input("Mileage", min_value=0, value=50000)

year = st.sidebar.number_input("Year", min_value=1990, max_value=2026, value=2018)

make = st.sidebar.selectbox("Make", make_options)
model_options = sorted(
    df[df["make"] == make]["model"].dropna().unique().tolist()
)

model_selected = st.sidebar.selectbox("Model", model_options)
fuel = st.sidebar.selectbox("Fuel", fuel_options)
offer_type = st.sidebar.selectbox("Offer Type", offer_options)
model_hp_values = df[
    (df["make"] == make) &
    (df["model"] == model_selected)
]["hp"].dropna()

if len(model_hp_values) > 0:
    hp = int(model_hp_values.median())
else:
    hp = int(df["hp"].median())

listed_price = st.sidebar.number_input("Listed Price (€)", min_value=0, value=15000)

analyze_button = st.sidebar.button("Analyze Listing")

# Create input dataframe
input_data = pd.DataFrame([{
    "mileage": mileage,
    "make": make,
    "model": model_selected,
    "fuel": fuel,
    "offerType": offer_type,
    "hp": hp,
    "year": year
}])

# One-hot encode input
input_encoded = pd.get_dummies(input_data)

# Match training columns exactly
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

if analyze_button:
    # Prediction
    log_prediction = model.predict(input_encoded)[0]
    predicted_price = np.expm1(log_prediction)

    # Confidence range using approximate model error
    lower_bound = predicted_price / np.exp(0.18)
    upper_bound = predicted_price * np.exp(0.18)

    # Price comparison
    difference = listed_price - predicted_price
    percentage_difference = (difference / predicted_price) * 100

    # Valuation score
    score = 50 + percentage_difference
    score = max(0, min(100, score))
   
    st.subheader("Valuation Score")

score_col1, score_col2 = st.columns([3, 1])

with score_col1:
    st.progress(int(score))

with score_col2:
    st.metric("Score", f"{score:.0f}/100")
    
    if score > 60:
        st.write("Higher score means the listing is more expensive than the model’s fair value estimate.")
    elif score < 40:
        st.write("Lower score means the listing may be cheaper than the model’s fair value estimate.")
    else:
        st.write("The listing is close to the model’s fair value estimate.")

    # Valuation result
    st.subheader("Valuation Result")

    comparison_df = pd.DataFrame({
        "Type": ["Predicted Fair Price", "Listed Price"],
        "Price": [predicted_price, listed_price]
    })

    st.subheader("Price Comparison")

    
    col1, col2 = st.columns(2)

    with col1:
     st.metric("Estimated Fair Price", f"€{predicted_price:,.0f}")

    with col2:
     st.metric(
        "Listed Price",
        f"€{listed_price:,.0f}",
        delta=f"{percentage_difference:.1f}% vs fair value"
    )

    st.caption(
    f"Expected market range: €{lower_bound:,.0f} – €{upper_bound:,.0f}"
)

    if percentage_difference > 10:
        st.error(
            f"This listing appears overpriced by approximately "
            f"€{difference:,.0f} ({percentage_difference:.1f}%)."
        )
    elif percentage_difference < -10:
        st.success(
            f"This listing appears underpriced by approximately "
            f"€{abs(difference):,.0f} ({abs(percentage_difference):.1f}%)."
        )
    else:
        st.info("This listing appears fairly priced based on the model prediction.")

    st.caption(
        "Note: This is an estimated valuation based on historical listing data "
        "and should be used as decision support, not as an exact market price."
    )

    # Explanation section
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

    st.subheader("Key Value Drivers")

    drivers = []

    if mileage > df["mileage"].median():
        drivers.append(("High mileage", "Negative"))
    else:
        drivers.append(("Low mileage", "Positive"))

    if hp > df["hp"].median():
        drivers.append(("Above-average horsepower", "Positive"))
    else:
        drivers.append(("Below-average horsepower", "Negative"))

    if year > df["year"].median():
        drivers.append(("Newer vehicle age", "Positive"))
    else:
        drivers.append(("Older vehicle age", "Negative"))
    if percentage_difference > 10:
        drivers.append(("Listed price is far above predicted fair value", "Negative"))
    elif percentage_difference < -10:
        drivers.append(("Listed price is below predicted fair value", "Positive"))
    else:
        drivers.append(("Listed price is close to predicted fair value", "Neutral"))    

    for driver, impact in drivers:
        if impact == "Positive":
            st.success(f"{driver}: Positive impact")
        elif impact == "Negative":
            st.error(f"{driver}: Negative impact")
        else:
            st.info(f"{driver}: Neutral impact")

    # Summary
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
    report_text = f"""
    Used Car Valuation Report

    Selected Vehicle:
    - Make: {make}
    - Fuel: {fuel}
    - Offer Type: {offer_type}
    - Year: {year}
    - Horsepower: {hp}
    - Mileage: {mileage:,} km

    Valuation:
    - Estimated Fair Price: €{predicted_price:,.0f}
    - Listed Price: €{listed_price:,.0f}
    - Expected Market Range: €{lower_bound:,.0f} – €{upper_bound:,.0f}
    - Difference: €{difference:,.0f}
    - Percentage Difference: {percentage_difference:.1f}%

    Summary:
    {summary}
    """

    st.download_button(
    label="Download Valuation Report",
    data=report_text,
    file_name="used_car_valuation_report.txt",
    mime="text/plain"
)
    with st.expander("Model Details"):

        st.write("**Model Type:** Random Forest Regressor")
        st.write("**R² Accuracy:** ≈ 0.94")
        st.write("**RMSE:** ≈ 0.167")
        st.caption(
    "Model trained on cleaned real-world automotive marketplace data with feature engineering, log-price transformation, and predictive performance optimization."
)
    st.subheader("Market Insights")

    brand_prices = (
    df.groupby("make")["price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

    st.write("Top 10 brands by average listing price:")
    st.bar_chart(brand_prices)

else:
    st.info("Enter car details in the sidebar and click **Analyze Listing** to generate a valuation.")