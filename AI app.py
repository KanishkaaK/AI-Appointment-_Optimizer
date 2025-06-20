import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load("miss_model.joblib")

# Load label encoders
label_encoder_doctor = joblib.load("label_encoder_doctor.joblib")
label_encoder_appointment_type = joblib.load("label_encoder_appointment_type.joblib")
label_encoder_contact_verified = joblib.load("label_encoder_contact_verified.joblib")

st.set_page_config(page_title="AI Appointment Optimizer", layout="centered")

# Title
st.title("AI Appointment Optimizer")

# Sidebar Inputs
st.sidebar.header("Input Appointment Details")

# Doctor selection
doctor = st.sidebar.selectbox("Select Doctor", label_encoder_doctor.classes_)

# Appointment time (6 AM to 8 AM only)
appointment_time = st.sidebar.selectbox(
    "Select Appointment Time (6 AM to 8 AM)",
    ["06:00 AM", "06:15 AM", "06:30 AM", "06:45 AM", "07:00 AM", "07:15 AM", "07:30 AM", "07:45 AM", "08:00 AM"]
)

# Day of week
day_of_week = st.sidebar.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

# Expected delay (mins)
delay_mins = st.sidebar.number_input("Expected Delay (mins)", min_value=0, max_value=60, value=0)

# Appointment type
appointment_type = st.sidebar.selectbox("Appointment Type", label_encoder_appointment_type.classes_)

# Patient Age
patient_age = st.sidebar.number_input("Patient Age", min_value=0, max_value=120, value=30)

# Gender
patient_gender = st.sidebar.radio("Patient Gender", ["Male", "Female"])

# Distance from Clinic (km)
distance_from_clinic_km = st.sidebar.number_input("Distance from Clinic (km)", min_value=0.0, max_value=100.0, value=5.0)

# Past No-Show Count
past_no_show_count = st.sidebar.number_input("Past No-Show Count", min_value=0, max_value=10, value=0)

# Contact number verified
contact_verified = st.sidebar.radio("Contact Number Verified", ["Yes", "No"])

# Predict Button
if st.button("Predict Appointment Status"):
    # Prepare input
    appointment_hour = int(datetime.strptime(appointment_time, "%I:%M %p").strftime("%H"))
    day_of_week_num = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].index(day_of_week)

    input_data = np.array([[
        label_encoder_doctor.transform([doctor])[0],
        appointment_hour,
        day_of_week_num,
        delay_mins,
        label_encoder_appointment_type.transform([appointment_type])[0],
        patient_age,
        0 if patient_gender == "Male" else 1,
        past_no_show_count,
        distance_from_clinic_km,
        label_encoder_contact_verified.transform([contact_verified])[0]
    ]])

    # Predict
    predicted_prob = model.predict_proba(input_data)[0][1]

    # Show result
    if predicted_prob < 0.5:
        st.success("Available - Appointment is likely to be attended.")
    else:
        st.error("Not Available - High chance of No-Show.")

    # Save log
    log_data = {
        "doctor": doctor,
        "appointment_time": appointment_time,
        "day_of_week": day_of_week,
        "delay_mins": delay_mins,
        "appointment_type": appointment_type,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "distance_from_clinic_km": distance_from_clinic_km,
        "past_no_show_count": past_no_show_count,
        "contact_verified": contact_verified,
        "predicted_prob": round(predicted_prob, 2),
        "prediction": "Available" if predicted_prob < 0.5 else "Not Available"
    }

    df_log = pd.DataFrame([log_data])
    df_log.to_csv("prediction_log.csv", mode='a', header=not pd.io.common.file_exists("prediction_log.csv"), index=False)

    st.info("Prediction logged ")
    with open("prediction_log.csv", "rb") as file:
        st.download_button(label="Download Prediction Log (CSV)", data=file, file_name="prediction_log.csv", mime="text/csv")
