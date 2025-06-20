import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model and encoders
model = joblib.load('miss_model.joblib')
le_appointment_type = joblib.load('label_encoder.joblib')
le_doctor = joblib.load('label_encoder_doctor.joblib')
le_gender = joblib.load('label_encoder_gender.joblib')

# Title
st.title("AI Appointment miss Predictor (Advanced)")

# Helper function: convert to 12hr format
def hour_12_format(hour):
    am_pm = "AM" if hour < 12 else "PM"
    hour_display = hour % 12
    hour_display = 12 if hour_display == 0 else hour_display
    return f"{hour_display}:00 {am_pm}"

# Inputs
selected_doctor = st.selectbox("Select Doctor", le_doctor.classes_)
doctor_encoded = le_doctor.transform([selected_doctor])[0]

appointment_hour_12 = st.selectbox(
    "Select Appointment Time (12-Hour Format)",
    options=range(6, 9),
    format_func=hour_12_format
)

appointment_day_of_week = st.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x]
)

delay_mins = st.number_input("Expected Delay (mins)", min_value=0, value=0)

appointment_type = st.selectbox("Appointment Type", le_appointment_type.classes_)
appointment_type_encoded = le_appointment_type.transform([appointment_type])[0]

patient_age = st.slider("Patient Age", 1, 100, 30)
patient_gender = st.selectbox("Patient Gender", le_gender.classes_)
gender_encoded = le_gender.transform([patient_gender])[0]

past_no_show_count = st.number_input("Past miss Count", min_value=0, value=0)
distance_from_clinic_km = st.number_input("Distance from Clinic (km)", min_value=0.0, value=2.0)

contact_verified = st.selectbox("Contact Number Verified", ['Yes', 'No'])
contact_verified_encoded = 1 if contact_verified == 'Yes' else 0

# Predict button
if st.button("Predict miss Probability"):
    input_data = pd.DataFrame(
        [[doctor_encoded, appointment_hour_12, appointment_day_of_week, delay_mins, appointment_type_encoded,
          patient_age, gender_encoded, past_miss_count, distance_from_clinic_km, contact_verified_encoded]],
        columns=['doctor_id_encoded', 'appointment_hour', 'appointment_day_of_week', 'delay_mins', 'appointment_type_encoded',
                 'patient_age', 'patient_gender_encoded', 'past_miss_count', 'distance_from_clinic_km', 'contact_number_verified_encoded']
    )

    prob_miss = model.predict_proba(input_data)[0, 1]

    st.write(f"### 🎯 Predicted miss Probability: **{prob_miss:.2f}**")
    st.progress(1 - prob_miss)

    if prob_miss < 0.3:
        st.success(" Available - Appointment is likely to be attended.")
        suggestion = "No action needed."
    elif 0.3 <= prob_miss < 0.6:
        st.warning(" Medium Risk - Recommend confirming with the client.")
        suggestion = "Recommend phone confirmation."
    else:
        st.error(" Not Available - High risk of no-show.")
        suggestion = "Consider rescheduling or double-check with client."

    # Log
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'doctor_id': selected_doctor,
        'appointment_hour_12': hour_12_format(appointment_hour_12),
        'appointment_day_of_week': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][appointment_day_of_week],
        'delay_mins': delay_mins,
        'appointment_type': appointment_type,
        'patient_age': patient_age,
        'patient_gender': patient_gender,
        'past_miss_count': past_miss_count,
        'distance_from_clinic_km': distance_from_clinic_km,
        'contact_verified': contact_verified,
        'miss_probability': round(prob_miss, 2),
        'suggestion': suggestion
    }

    log_df = pd.DataFrame([log_entry])

    if os.path.exists("prediction_log.csv"):
        log_df.to_csv("prediction_log.csv", mode='a', header=False, index=False)
    else:
        log_df.to_csv("prediction_log.csv", index=False)

    st.info("Prediction logged ")

# Download log
if os.path.exists("prediction_log.csv"):
    with open("prediction_log.csv", "rb") as f:
        st.download_button("📥 Download Prediction Log (CSV)", f, file_name="prediction_log.csv")
