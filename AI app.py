import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model and encoders
model = joblib.load('miss_model.joblib')
le_appointment_type = joblib.load('label_encoder_appointment_type.joblib')
le_doctor = joblib.load('label_encoder_doctor.joblib')
le_gender = joblib.load('label_encoder_gender.joblib')

# Title
st.title("AI Appointment Availability Predictor (6 AM - 8 AM)")

# Helper: convert to 12hr string
def hour_12_format(hour):
    am_pm = "AM" if hour < 12 else "PM"
    hour_display = hour % 12
    hour_display = 12 if hour_display == 0 else hour_display
    return f"{hour_display}:00 {am_pm}"

# --- INPUTS ---

# Doctor select
selected_doctor = st.selectbox("Select Doctor", le_doctor.classes_)
doctor_encoded = le_doctor.transform([selected_doctor])[0]

# Appointment time (6AM to 8AM)
appointment_hour_12 = st.selectbox(
    "Select Appointment Time (12-Hour Format)",
    options=range(6, 9),  # 6AM - 8AM
    format_func=hour_12_format
)

# Day of week
appointment_day_of_week = st.selectbox(
    "Day of Week",
    options=[0,1,2,3,4,5,6],
    format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x]
)

# Delay
delay_mins = st.number_input("Expected Delay (mins)", min_value=0, value=0)

# Appointment Type
appointment_type = st.selectbox("Appointment Type", le_appointment_type.classes_)
appointment_type_encoded = le_appointment_type.transform([appointment_type])[0]

# Patient Age
patient_age = st.slider("Patient Age", 1, 100, 30)

# Patient Gender
patient_gender = st.selectbox("Patient Gender", le_gender.classes_)
gender_encoded = le_gender.transform([patient_gender])[0]

# Past No-show count
past_miss_count = st.number_input("Past Miss Count", min_value=0, value=0)

# Distance from clinic
distance_from_clinic_km = st.number_input("Distance from Clinic (km)", min_value=0.0, value=2.0)

# Contact Number
contact_number = st.text_input("Contact Number (optional)")
contact_verified_encoded = 1 if contact_number.strip() != '' else 0

# --- PREDICT BUTTON ---
if st.button("Predict Availability"):
    
    # Prepare input data
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
    predicted_prob = model.predict_proba(input_data)[0][1]  # probability of 'miss'
    
    # Show result (without number %)
    if predicted_prob < 0.3:
        st.success(" Appointment Available — likely to be attended.")
        suggestion = "No action needed."
    elif 0.3 <= predicted_prob < 0.6:
        st.warning(" Medium Risk — consider confirming with patient.")
        suggestion = "Phone confirmation suggested."
    else:
        st.error(" Not Available — high risk of no-show.")
        suggestion = "Consider follow-up or reschedule."
    
    # Log prediction
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'doctor_id': selected_doctor,
        'appointment_hour': appointment_hour_12,
        'day_of_week': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][appointment_day_of_week],
        'delay_mins': delay_mins,
        'appointment_type': appointment_type,
        'patient_age': patient_age,
        'patient_gender': patient_gender,
        'past_miss_count': past_miss_count,
        'distance_from_clinic_km': distance_from_clinic_km,
        'contact_number': contact_number,
        'availability': "Available" if predicted_prob < 0.3 else "Risk",
        'suggestion': suggestion
    }
    
    log_df = pd.DataFrame([log_entry])
    
    if os.path.exists("Confirmation_Report.csv"):
        log_df.to_csv("Confirmation_Report.csv", mode='a', header=False, index=False)
    else:
        log_df.to_csv("Confirmation_Report.csv", index=False)
    
    st.info("Confirmation_Report ")

# --- Download log ---
if os.path.exists("Confirmation_Report.csv"):
    with open("Confirmed_Report.csv", "rb") as f:
        st.download_button("📥 Download Confirmation_Report (CSV)", f, file_name="Confirmation_Report.csv")
