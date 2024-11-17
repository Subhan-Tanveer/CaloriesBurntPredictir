import streamlit as st
import numpy as np
import pickle
import sklearn
import pandas as pd
from xgboost import XGBRegressor

# Load the model
loaded_model = pickle.load(open('trained_calories_model.pkl', 'rb'))

def prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    return loaded_model.predict(input_data_reshaped)[0]

# Streamlit app
def main():
    st.set_page_config(page_title="Calorie Burn Prediction", layout="centered", page_icon="🔥")

    # Styling for dark theme
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: #FFFFFF;
        }
        .main {
            border-radius: 15px;
            padding: 2rem;
            color: white;
        }
        .stButton>button {
            background-color: #FF5722;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 8px 16px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #E64A19;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown("<h1 style='text-align: center;'>🔥 Calorie Burn Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter your details below to get your calorie burn prediction! 🚴‍♂️</p>", unsafe_allow_html=True)

    # Form for user input
    with st.form("calorie_form"):
        st.subheader("Enter your details:")
        gender = st.radio("Select Gender 🧑‍🤝‍🧑", ["Male", "Female", "Other"])
        age = st.number_input("Age (years) 🎂", min_value=0, max_value=120, step=1)
        height = st.number_input("Height (cm) 📏", min_value=50.0, max_value=250.0, step=0.1)
        weight = st.number_input("Weight (kg) ⚖️", min_value=10.0, max_value=200.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm) ❤️", min_value=30.0, max_value=200.0, step=1.0)
        body_temp = st.number_input("Body Temperature (°C) 🌡️", min_value=35.0, max_value=42.0, step=0.1)
        duration = st.number_input("Duration (minutes) ⏱️", min_value=1, max_value=300, step=1)

        submitted = st.form_submit_button("🔥 Predict Calories Burnt")

    # Predict and display results
    if submitted:
        # Encode gender as numeric for model input
        gender_encoded = {"Male": 0, "Female": 1, "Other": 2}[gender]
        input_data = [gender_encoded, age, height, weight, heart_rate, body_temp, duration]

        try:
            result = prediction(input_data)
            st.success(f"🔥 Estimated Calories Burnt: **{result:.2f} kcal** 💪")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
