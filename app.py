import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Breast Cancer Prediction")

st.title("Breast Cancer Prediction Using ANN")
st.write("Enter the 30 feature values to get a model prediction.")

# Load the trained model and scaler
model = tf.keras.models.load_model("breast_cancer_ann.keras")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se",
    "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst",
    "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

with st.form("prediction_form"):
    values = []

    for name in feature_names:
        value = st.number_input(
            name,
            min_value=0.0,
            value=0.0,
            format="%.6f"
        )
        values.append(value)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array(values).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    probability = float(
        model.predict(input_scaled, verbose=0)[0][0]
    )

    if probability >= 0.5:
        st.error("Model prediction: Malignant")
    else:
        st.success("Model prediction: Benign")

    st.write(
        f"Estimated malignant probability: {probability:.2%}"
    )

    st.warning(
        "This is an educational ML project, not a medical diagnosis."
    )