import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Load the trained model
model = joblib.load("miss_model.joblib")

# Load label encoders
label_encoder_doctor = joblib.load("label_encoder_doctor.joblib")
label_encoder_appointment_type = joblib.load("label_encoder_appointment_type.joblib")
label_encoder_contact_verified = joblib.load("label_encoder_contact_verified.joblib")

st.set_page_config(page_title="AI Appointment Availability Predictor", layout="centered")

# Title
st.title("AI Appointment Availability Predictor")

# Sidebar Inputs
st.sidebar.header("Input Appointment Details")

# Doctor selection
doctor = st.sidebar.selectbox("Select Doctor", label_encoder_doctor.classes_)

# Appointment time (6 AM to 10 PM)
appointment_time = st.sidebar.selectbox(
    "Select Appointment Time (6 AM to 10 PM)",
    [
        "06:00 AM", "06:30 AM", "07:00 AM", "07:30 AM", "08:00 AM", "08:30 AM", "09:00 AM", "09:30 AM",
        "10:00 AM", "10:30 AM", "11:00 AM", "11:30 AM", "12:00 PM", "12:30 PM", "01:00 PM", "01:30 PM",
        "02:00 PM", "02:30 PM", "03:00 PM", "03:30 PM", "04:00 PM", "04:30 PM", "05:00 PM", "05:30 PM",
        "06:00 PM", "06:30 PM", "07:00 PM", "07:30 PM", "08:00 PM", "08:30 PM", "09:00 PM", "09:30 PM",
        "10:00 PM"
    ]
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
distance_from_clinic_km = st.sidebar.number_input("Distance from Clinic (km)", min_value=0, max_value=100, value=5)

# Past Miss Count
past_miss_count = st.sidebar.number_input("Past Miss Count", min_value=0, max_value=10, value=0)

# Contact Number (instead of verified)
contact_number = st.sidebar.text_input("Patient Contact Number", max_chars=15)

# Predict Button
if st.button("Predict Availability"):
    # Prepare input
    appointment_hour = int(datetime.strptime(appointment_time, "%I:%M %p").strftime("%H"))
    day_of_week_num = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].index(day_of_week)

    # Encode contact_verified as simple Yes/No based on number
    contact_verified_encoded = 1 if len(contact_number.strip()) > 0 else 0

    input_data = np.array([[
        label_encoder_doctor.transform([doctor])[0],
        appointment_hour,
        day_of_week_num,
        delay_mins,
        label_encoder_appointment_type.transform([appointment_type])[0],
        patient_age,
        0 if patient_gender == "Male" else 1,
        past_miss_count,
        distance_from_clinic_km,
        contact_verified_encoded
    ]])

    # Predict
    predicted_prob = model.predict_proba(input_data)[0][1]

    # Show result
    if predicted_prob < 0.5:
        st.success(" Available - Appointment is likely to be attended.")
        prediction = "Available"
    else:
        st.error(" Not Available - High risk of miss.")
        prediction = "Not Available"

    # Save log
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "doctor": doctor,
        "appointment_time": appointment_time,
        "day_of_week": day_of_week,
        "delay_mins": delay_mins,
        "appointment_type": appointment_type,
        "patient_age": patient_age,
        "patient_gender": patient_gender,
        "distance_from_clinic_km": distance_from_clinic_km,
        "past_miss_count": past_miss_count,
        "contact_number": contact_number,
        "prediction": prediction
    }

    df_log = pd.DataFrame([log_data])
    df_log.to_csv("Confirmation_report.csv", mode='a', header=not os.path.exists("Confirmation_report.csv"), index=False)

    st.info("Confirmed ")
    with open("Confirmation_report.csv", "rb") as file:
        st.download_button(label="Download Prediction Log (CSV)", data=file, file_name="Confirmation_report.csv", mime="text/csv")
