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


st.set_page_config(page_title="AI Appointment Availability Predictor", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>AI Appointment Availability Predictor</h1>", unsafe_allow_html=True)

# Display form on main page
with st.form("appointment_form"):
    st.subheader("Enter Appointment Details")

    col1, col2 = st.columns(2)

    with col1:
        doctor = st.selectbox("Select Doctor", label_encoder_doctor.classes_)

    with col2:
        appointment_time = st.selectbox(
            "Select Appointment Time (6 AM to 10 PM)",
            [
                "06:00 AM", "06:30 AM", "07:00 AM", "07:30 AM", "08:00 AM", "08:30 AM", "09:00 AM", "09:30 AM",
                "10:00 AM", "10:30 AM", "11:00 AM", "11:30 AM", "12:00 PM", "12:30 PM", "01:00 PM", "01:30 PM",
                "02:00 PM", "02:30 PM", "03:00 PM", "03:30 PM", "04:00 PM", "04:30 PM", "05:00 PM", "05:30 PM",
                "06:00 PM", "06:30 PM", "07:00 PM", "07:30 PM", "08:00 PM", "08:30 PM", "09:00 PM", "09:30 PM",
                "10:00 PM"
            ]
        )

    day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    delay_mins = st.number_input("Expected Delay (mins)", min_value=0, max_value=60, value=0)

    appointment_type = st.selectbox("Appointment Type", label_encoder_appointment_type.classes_)

    patient_age = st.slider("Patient Age", min_value=0, max_value=100, value=30)

    patient_gender = st.radio("Patient Gender", ["Male", "Female"], horizontal=True)

    distance_from_clinic_km = st.number_input("Distance from Clinic (km)", min_value=0, max_value=100, value=5)

    

    contact_number = st.text_input("Patient Contact Number", max_chars=10)

    # Submit button
    submitted = st.form_submit_button("Predict Availability")

# If button pressed
if submitted:
    # Prepare input
    appointment_hour = int(datetime.strptime(appointment_time, "%I:%M %p").strftime("%H"))
    day_of_week_num = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'].index(day_of_week)

    contact_verified_encoded = 1 if len(contact_number.strip()) > 0 else 0

    input_data = pd.DataFrame(
    [[doctor_encoded, appointment_hour_12, appointment_day_of_week, delay_mins, appointment_type_encoded,
      patient_age, gender_encoded, past_miss_count, distance_from_clinic_km, contact_verified_encoded]],
    columns=[
        'doctor_id_encoded',
        'appointment_hour',
        'day_of_week',
        'delay_mins',
        'appointment_type_encoded',
        'patient_age',
        'gender_encoded',
        'past_miss_count',
        'distance_from_clinic_km',
        'contact_verified_encoded'
    ]
)


    # Predict
    predicted_prob = model.predict_proba(input_data)[0][1]

    # Show result
    st.markdown("---")
    st.subheader("Prediction Result:")

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
        "contact_number": contact_number,
        "prediction": prediction
    }

    df_log = pd.DataFrame([log_data])
    df_log.to_csv("Confirmation_report.csv", mode='a', header=not os.path.exists("Confirmation_report.csv"), index=False)

    st.info("Confirmed ")

    with open("Confirmation_report.csv", "rb") as file:
        st.download_button(label="📥 Download Confirmation_report (CSV)", data=file, file_name="Confirmation_report.csv", mime="text/csv")
