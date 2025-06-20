import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Load model and encoders
model = joblib.load('miss_model.joblib')
label_encoder_doctor = joblib.load('label_encoder_doctor.joblib')
label_encoder_appointment_type = joblib.load('label_encoder_appointment_type.joblib')
label_encoder_contact_verified = joblib.load('label_encoder_contact_verified.joblib')

# App Title
st.set_page_config(page_title="AI Appointment Miss Predictor", layout="centered")
st.title("AI Appointment  Predictor ")

# Helper function: convert to 12hr format
def hour_12_format(hour):
    am_pm = "AM" if hour < 12 else "PM"
    hour_display = hour % 12
    hour_display = 12 if hour_display == 0 else hour_display
    return f"{hour_display}:00 {am_pm}"

# Inputs
selected_doctor = st.selectbox("Select Doctor", label_encoder_doctor.classes_)
doctor_encoded = label_encoder_doctor.transform([selected_doctor])[0]

appointment_hour_12 = st.selectbox(
    "Select Appointment Time (6 AM to 8 AM)",
    options=[6, 7, 8],
    format_func=hour_12_format
)

appointment_day_of_week = st.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x]
)

delay_mins = st.number_input("Expected Delay (mins)", min_value=0, max_value=60, value=0)

appointment_type = st.selectbox("Appointment Type", label_encoder_appointment_type.classes_)
appointment_type_encoded = label_encoder_appointment_type.transform([appointment_type])[0]

patient_age = st.slider("Patient Age", 1, 100, 30)
patient_gender = st.radio("Patient Gender", ["Male", "Female"])
gender_encoded = 0 if patient_gender == "Male" else 1

distance_from_clinic_km = st.number_input("Distance from Clinic (km)", min_value=0, max_value=100, value=5)



# Predict button
if st.button("Predict Miss Probability"):
    input_data = pd.DataFrame(
        [[doctor_encoded, appointment_hour_12, appointment_day_of_week, delay_mins, appointment_type_encoded,
          patient_age, gender_encoded, , distance_from_clinic_km, contact_verified_encoded]],
        columns=['doctor_id_encoded', 'appointment_hour', 'appointment_day_of_week', 'delay_mins', 'appointment_type_encoded',
                 'patient_age', 'patient_gender_encoded', , 'distance_from_clinic_km', ]
    )

    prob_miss = model.predict_proba(input_data)[0, 1]

    st.write(f"###  Predicted Miss Probability: **{prob_miss:.2f}**")
    st.progress(1 - prob_miss)

    if prob_miss < 0.3:
        st.success(" Available - Appointment is likely to be attended.")
        suggestion = "No action needed."
    elif 0.3 <= prob_miss < 0.6:
        st.warning(" Medium Risk - Recommend confirming with the client.")
        suggestion = "Recommend phone confirmation."
    else:
        st.error(" Not Available - High risk of miss.")
        suggestion = "Consider rescheduling or double-check with client."

    # Log
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'doctor_id': selected_doctor,
        'appointment_hour': hour_12_format(appointment_hour_12),
        'appointment_day_of_week': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][appointment_day_of_week],
        'delay_mins': delay_mins,
        'appointment_type': appointment_type,
        'patient_age': patient_age,
        'patient_gender': patient_gender,
        
        'distance_from_clinic_km': distance_from_clinic_km,
        
        'miss_probability': round(prob_miss, 2),
        'suggestion': suggestion
    }

    log_df = pd.DataFrame([log_entry])

    if os.path.exists("Confirmation_report.csv"):
        log_df.to_csv("Confirmation_report.csv", mode='a', header=False, index=False)
    else:
        log_df.to_csv("Confirmation_report.csv", index=False)

    st.info("Confirmation_report ")

# Download log
if os.path.exists("Confirmatiion_report.csv"):
    with open("Confirmatiion_report.csv", "rb") as f:
        st.download_button("📥 Download Confirmation_report(CSV)", f, file_name="Confirmation_report.csv")
