import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("dress_data.csv")

# Train Model
X = data[["color", "fabric", "length"]]
y = data["style"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["color", "fabric"]),
        ("num", "passthrough", ["length"])
    ]
)

model = RandomForestClassifier(random_state=42)
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", model)])
pipeline.fit(X, y)

st.title("👗 Dress Style Prediction AI App")

color = st.selectbox("Choose dress color", sorted(data["color"].unique()))
fabric = st.selectbox("Choose fabric type", sorted(data["fabric"].unique()))
length = st.slider("Select length in cm", 60, 200, 120)

if st.button("Predict Style"):
    new_dress = pd.DataFrame([{
        "color": color,
        "fabric": fabric,
        "length": length
    }])

    prediction = pipeline.predict(new_dress)[0]
    st.success(f"Predicted Dress Style: {prediction}")
