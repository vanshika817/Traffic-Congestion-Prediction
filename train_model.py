
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the traffic dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"

print("Loading traffic dataset...")
df = pd.read_csv(url, compression="gzip")

# 2. Prepare the data
df = df.drop_duplicates().copy()
df["date_time"] = pd.to_datetime(df["date_time"])

df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df = df.sort_values("date_time").reset_index(drop=True)

# 3. Select input features and target
features = [
    "hour", "day_of_week", "month", "is_weekend",
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "holiday"
]

X = df[features]
y = df["traffic_volume"]

# 4. Split chronologically: first 80% for training, last 20% for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Handle numerical and categorical features
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 6. Train the Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training model. This may take a few minutes...")
model.fit(X_train, y_train)

# 7. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("----------------")
print(f"MAE:  {mae:.2f} vehicles/hour")
print(f"RMSE: {rmse:.2f}")
print(f"R² score: {r2:.4f}")

# 8. Save the trained model and feature names
joblib.dump(
    {"model": model, "features": features},
    "traffic_model.joblib"
)

print("\nModel saved successfully as traffic_model.joblib!")
