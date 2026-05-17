import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI

st.set_page_config(
    page_title="Used Car Valuation Assistant",
    page_icon="🚗",
    layout="wide"
)

# Load model and feature list
model = joblib.load("outputs/deploy_car_price_model.pkl")
model_features = joblib.load("outputs/model_features.pkl")
df = pd.read_csv("outputs/cleaned_cars_dataset.csv")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.markdown("""
<div style="background: linear-gradient(135deg, #0f172a, #1e3a8a);
padding: 2rem; border-radius: 18px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1>Used Car Valuation Assistant</h1>
    <p>Estimate fair market value, compare listed prices, and identify overpriced or underpriced used cars.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
make_options = sorted(df["make"].dropna().unique().tolist())
fuel_options = sorted(df["fuel"].dropna().unique().tolist())
offer_options = sorted(df["offerType"].dropna().unique().tolist())

st.sidebar.header("Car Details")
st.sidebar.markdown("### Enter Vehicle Specifications")

mileage = st.sidebar.number_input("Mileage", min_value=0, value=50000)
year = st.sidebar.number_input("Year", min_value=1990, max_value=2026, value=2018)

make = st.sidebar.selectbox("Make", make_options)

model_options = sorted(df[df["make"] == make]["model"].dropna().unique().tolist())
model_selected = st.sidebar.selectbox("Model", model_options)

fuel = st.sidebar.selectbox("Fuel", fuel_options)
offer_type = st.sidebar.selectbox("Offer Type", offer_options)

model_hp_values = df[
    (df["make"] == make) &
    (df["model"] == model_selected)
]["hp"].dropna()

hp = int(model_hp_values.median()) if len(model_hp_values) > 0 else int(df["hp"].median())

listed_price = st.sidebar.number_input("Listed Price (€)", min_value=0, value=15000)

analyze_button = st.sidebar.button("Analyze Listing")


def generate_ai_recommendation(make, model_selected, fuel, offer_type, year, mileage, hp,
                               listed_price, predicted_price, lower_bound, upper_bound,
                               percentage_difference):
    prompt = f"""
    You are a car valuation assistant. Write a concise, practical buyer recommendation.

    Vehicle:
    - Make: {make}
    - Model: {model_selected}
    - Fuel: {fuel}
    - Offer type: {offer_type}
    - Year: {year}
    - Mileage: {mileage}
    - Estimated horsepower: {hp}

    Valuation:
    - Listed price: €{listed_price:,.0f}
    - Predicted fair price: €{predicted_price:,.0f}
    - Expected market range: €{lower_bound:,.0f} – €{upper_bound:,.0f}
    - Difference vs fair value: {percentage_difference:.1f}%

    Give:
    1. A short verdict
    2. What the buyer should check
    3. Whether the price looks attractive or risky

    Keep it under 120 words.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


if analyze_button:
    input_data = pd.DataFrame([{
        "mileage": mileage,
        "make": make,
        "model": model_selected,
        "fuel": fuel,
        "offerType": offer_type,
        "hp": hp,
        "year": year
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    log_prediction = model.predict(input_encoded)[0]
    predicted_price = np.expm1(log_prediction)

    lower_bound = predicted_price / np.exp(0.18)
    upper_bound = predicted_price * np.exp(0.18)

    difference = listed_price - predicted_price
    percentage_difference = (difference / predicted_price) * 100

    score = 50 + percentage_difference
    score = max(0, min(100, score))

    if percentage_difference > 10:
        summary = (
            f"Based on the model estimate, this car appears overpriced. "
            f"The predicted fair market value is around €{predicted_price:,.0f}, "
            f"while the listed price is €{listed_price:,.0f}. "
            f"The difference of approximately €{difference:,.0f} suggests that the listing "
            f"price may be too high."
        )
    elif percentage_difference < -10:
        summary = (
            f"Based on the model estimate, this car appears underpriced. "
            f"The predicted fair market value is around €{predicted_price:,.0f}, "
            f"while the listed price is €{listed_price:,.0f}. "
            f"This could indicate a potentially good deal, but the vehicle should still be checked carefully."
        )
    else:
        summary = (
            f"Based on the model estimate, this car appears fairly priced. "
            f"The listed price of €{listed_price:,.0f} is close to the predicted fair value "
            f"of €{predicted_price:,.0f}."
        )

    st.session_state["valuation_done"] = True
    st.session_state["valuation_data"] = {
        "make": make,
        "model_selected": model_selected,
        "fuel": fuel,
        "offer_type": offer_type,
        "year": year,
        "mileage": mileage,
        "hp": hp,
        "listed_price": listed_price,
        "predicted_price": predicted_price,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "difference": difference,
        "percentage_difference": percentage_difference,
        "score": score,
        "summary": summary
    }

if "valuation_done" not in st.session_state:
    st.info("Enter car details in the sidebar and click **Analyze Listing** to generate a valuation.")
else:
    data = st.session_state["valuation_data"]

    st.subheader("Valuation Score")

    score_col1, score_col2 = st.columns([3, 1])

    with score_col1:
        st.progress(int(data["score"]))

    with score_col2:
        st.metric("Score", f"{data['score']:.0f}/100")

    if data["score"] > 60:
        st.write("Higher score means the listing is more expensive than the model’s fair value estimate.")
    elif data["score"] < 40:
        st.write("Lower score means the listing may be cheaper than the model’s fair value estimate.")
    else:
        st.write("The listing is close to the model’s fair value estimate.")

    st.divider()

    st.subheader("Price Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Estimated Fair Price", f"€{data['predicted_price']:,.0f}")

    with col2:
        st.metric(
            "Listed Price",
            f"€{data['listed_price']:,.0f}",
            delta=f"{data['percentage_difference']:.1f}% vs fair value"
        )

    st.caption(
        f"Expected market range: €{data['lower_bound']:,.0f} – €{data['upper_bound']:,.0f}"
    )

    if data["percentage_difference"] > 10:
        st.error(
            f"This listing appears overpriced by approximately "
            f"€{data['difference']:,.0f} ({data['percentage_difference']:.1f}%)."
        )
    elif data["percentage_difference"] < -10:
        st.success(
            f"This listing appears underpriced by approximately "
            f"€{abs(data['difference']):,.0f} ({abs(data['percentage_difference']):.1f}%)."
        )
    else:
        st.info("This listing appears fairly priced based on the model prediction.")

    st.caption(
        "Note: This is an estimated valuation based on historical listing data "
        "and should be used as decision support, not as an exact market price."
    )

    st.divider()

    st.subheader("Why this result?")

    explanations = []

    if data["mileage"] > df["mileage"].median():
        explanations.append("The mileage is above the dataset median, which usually reduces the expected price.")
    else:
        explanations.append("The mileage is below the dataset median, which usually supports a higher expected price.")

    if data["hp"] > df["hp"].median():
        explanations.append("The automatically estimated horsepower is above average, which can increase value.")
    else:
        explanations.append("The automatically estimated horsepower is below average, which may limit value.")

    if data["year"] > df["year"].median():
        explanations.append("The car is newer than the dataset median year, which usually increases price.")
    else:
        explanations.append("The car is older than the dataset median year, which usually reduces price.")

    for explanation in explanations:
        st.write(f"- {explanation}")

    st.divider()

    st.subheader("Key Value Drivers")

    drivers = []

    if data["mileage"] > df["mileage"].median():
        drivers.append(("High mileage", "Negative"))
    else:
        drivers.append(("Low mileage", "Positive"))

    if data["hp"] > df["hp"].median():
        drivers.append(("Above-average estimated horsepower", "Positive"))
    else:
        drivers.append(("Below-average estimated horsepower", "Negative"))

    if data["year"] > df["year"].median():
        drivers.append(("Newer vehicle age", "Positive"))
    else:
        drivers.append(("Older vehicle age", "Negative"))

    if data["percentage_difference"] > 10:
        drivers.append(("Listed price is far above predicted fair value", "Negative"))
    elif data["percentage_difference"] < -10:
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

    st.divider()

    st.subheader("Valuation Summary")
    st.write(data["summary"])

    report_text = f"""
Used Car Valuation Report

Selected Vehicle:
- Make: {data["make"]}
- Model: {data["model_selected"]}
- Fuel: {data["fuel"]}
- Offer Type: {data["offer_type"]}
- Year: {data["year"]}
- Estimated Horsepower: {data["hp"]}
- Mileage: {data["mileage"]:,} km

Valuation:
- Estimated Fair Price: €{data["predicted_price"]:,.0f}
- Listed Price: €{data["listed_price"]:,.0f}
- Expected Market Range: €{data["lower_bound"]:,.0f} – €{data["upper_bound"]:,.0f}
- Difference: €{data["difference"]:,.0f}
- Percentage Difference: {data["percentage_difference"]:.1f}%

Summary:
{data["summary"]}
"""

    st.download_button(
        label="Download Valuation Report",
        data=report_text,
        file_name="used_car_valuation_report.txt",
        mime="text/plain"
    )

    st.divider()

    st.subheader("AI Buyer Recommendation")

    if st.button("Generate AI Recommendation"):
        with st.spinner("Generating AI recommendation..."):
            ai_text = generate_ai_recommendation(
                data["make"], data["model_selected"], data["fuel"], data["offer_type"],
                data["year"], data["mileage"], data["hp"], data["listed_price"],
                data["predicted_price"], data["lower_bound"], data["upper_bound"],
                data["percentage_difference"]
            )

        st.write(ai_text)
    else:
        st.caption("Click the button to generate an AI-powered buyer recommendation.")

    with st.expander("Model Details"):
        st.write("**Model Type:** Random Forest Regressor")
        st.write("**R² Accuracy:** ≈ 0.94")
        st.write("**RMSE:** ≈ 0.167")
        st.caption(
            "Model trained on cleaned real-world automotive marketplace data with feature engineering, "
            "log-price transformation, and predictive performance optimization."
        )

    with st.expander("Market Insights"):
        brand_prices = (
            df.groupby("make")["price"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        st.write("Top 10 brands by average listing price:")
        st.bar_chart(brand_prices)