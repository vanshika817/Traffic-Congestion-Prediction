
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    saved = joblib.load("traffic_model.joblib")
    return saved["model"], saved["features"]


try:
    model, features = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please run "
        "'python3 train_model.py' first."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()


# --------------------------------------------------
# LOAD HISTORICAL TRAFFIC DATA
# --------------------------------------------------

@st.cache_data
def load_historical_data():
    url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00492/"
        "Metro_Interstate_Traffic_Volume.csv.gz"
    )

    history = pd.read_csv(url, compression="gzip")

    history["date_time"] = pd.to_datetime(
        history["date_time"],
        errors="coerce"
    )

    history = history.dropna(
        subset=["date_time", "traffic_volume"]
    )

    history["hour"] = history["date_time"].dt.hour
    history["day_name"] = history["date_time"].dt.day_name()

    return history


# --------------------------------------------------
# DASHBOARD TITLE
# --------------------------------------------------

st.title("🚦 Traffic Congestion Prediction System")

st.write(
    "Estimate traffic volume using weather conditions "
    "and time-related information."
)

st.info(
    "Dataset: Metro Interstate Traffic Volume (USA). "
    "This demonstration uses historical data, not live traffic."
)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.header("Enter Traffic Conditions")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider(
        "Hour of the day",
        min_value=0,
        max_value=23,
        value=8
    )

    day_names = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    day = st.selectbox(
        "Day of the week",
        options=list(range(7)),
        format_func=lambda x: day_names[x]
    )

    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=5
    )

    temp = st.number_input(
        "Temperature (Kelvin)",
        min_value=200.0,
        max_value=330.0,
        value=288.0,
        step=1.0
    )

    clouds = st.slider(
        "Cloud cover (%)",
        min_value=0,
        max_value=100,
        value=40
    )

with col2:
    rain = st.number_input(
        "Rain in last hour (mm)",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    snow = st.number_input(
        "Snow in last hour (mm)",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    weather_main = st.selectbox(
        "Main weather",
        [
            "Clear", "Clouds", "Rain", "Drizzle",
            "Mist", "Snow", "Fog", "Haze", "Thunderstorm"
        ]
    )

    # Keep descriptions associated with the selected
    # main weather category.
    weather_options = {
        "Clear": ["clear sky"],
        "Clouds": [
            "few clouds", "scattered clouds",
            "broken clouds", "overcast clouds"
        ],
        "Rain": [
            "light rain", "moderate rain",
            "heavy intensity rain", "very heavy rain",
            "extreme rain", "freezing rain"
        ],
        "Drizzle": [
            "light intensity drizzle", "drizzle",
            "heavy intensity drizzle"
        ],
        "Mist": ["mist"],
        "Snow": [
            "light snow", "snow", "heavy snow",
            "sleet", "light shower snow"
        ],
        "Fog": ["fog"],
        "Haze": ["haze"],
        "Thunderstorm": [
            "thunderstorm", "proximity thunderstorm"
        ]
    }

    weather_description = st.selectbox(
        "Weather description",
        options=weather_options[weather_main]
    )

    holiday = st.selectbox(
        "Holiday",
        [
            "None", "Columbus Day", "Veterans Day",
            "Thanksgiving Day", "Christmas Day",
            "New Year's Day", "Washington's Birthday",
            "Memorial Day", "Independence Day",
            "Labor Day", "Martin Luther King Jr Day"
        ]
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Traffic Volume", type="primary"):

    input_values = {
        "hour": hour,
        "day_of_week": day,
        "month": month,
        "is_weekend": int(day >= 5),
        "temp": temp,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "weather_main": weather_main,
        "weather_description": weather_description,
        "holiday": holiday
    }

    input_data = pd.DataFrame([input_values])

    # Use the exact feature order expected by the model.
    missing_features = [
        feature for feature in features
        if feature not in input_data.columns
    ]

    if missing_features:
        st.error(
            "The model expects additional features: "
            + ", ".join(missing_features)
        )
    else:
        input_data = input_data[features]

        try:
            prediction = float(model.predict(input_data)[0])
            prediction = max(0, prediction)

            st.divider()
            st.subheader("Prediction Result")

            st.metric(
                "Estimated traffic volume",
                f"{prediction:,.0f} vehicles/hour"
            )

            if prediction < 2000:
                st.success("Estimated traffic level: LOW")
            elif prediction < 4500:
                st.warning("Estimated traffic level: MODERATE")
            else:
                st.error("Estimated traffic level: HIGH")

            st.caption(
                "Traffic-level thresholds are illustrative, not official "
                "congestion standards. Predictions are estimates based on "
                "the trained model and the selected input conditions."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()
st.header("📊 Model Performance")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Mean Absolute Error (MAE)", "293.51")
metric2.metric("Root Mean Squared Error (RMSE)", "529.44")
metric3.metric("R² Score", "0.9276")

st.caption(
    "These are the reported test-set results from the trained model. "
    "MAE and RMSE measure prediction error; an R² score closer to 1 "
    "indicates a better fit on the evaluated test set."
)


# --------------------------------------------------
# HISTORICAL TRAFFIC ANALYSIS
# --------------------------------------------------

st.divider()
st.header("🚗 Historical Traffic Analysis")

try:
    history = load_historical_data()

    # Chart 1: Average traffic by hour
    st.subheader("Average Traffic by Hour")

    hourly_traffic = (
        history.groupby("hour", as_index=False)["traffic_volume"]
        .mean()
        .sort_values("hour")
    )

    hourly_traffic = hourly_traffic.rename(
        columns={
            "traffic_volume": "Average Vehicles per Hour"
        }
    )

    st.line_chart(
        hourly_traffic,
        x="hour",
        y="Average Vehicles per Hour"
    )

    # Chart 2: Average traffic by weekday
    st.subheader("Average Traffic by Day")

    day_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    daily_traffic = (
        history.groupby("day_name", as_index=False)["traffic_volume"]
        .mean()
    )

    daily_traffic["day_name"] = pd.Categorical(
        daily_traffic["day_name"],
        categories=day_order,
        ordered=True
    )

    daily_traffic = daily_traffic.sort_values("day_name")

    daily_traffic = daily_traffic.rename(
        columns={
            "day_name": "Day",
            "traffic_volume": "Average Vehicles per Hour"
        }
    )

    st.bar_chart(
        daily_traffic,
        x="Day",
        y="Average Vehicles per Hour"
    )

    st.caption(
        "These charts show historical patterns from a US highway dataset. "
        "They do not represent current or live traffic conditions."
    )

except Exception as e:
    st.warning(
        f"Could not load historical charts: {e}"
    )
