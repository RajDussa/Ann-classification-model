import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and preprocessing objects
model = load_model("trained_model.h5", compile=False)
scaler = pickle.load(open("standardscaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
onehot_encoder = pickle.load(open("one_hot_encoder.pkl", "rb"))

# Title
st.title("Customer Churn Prediction")

# User Input Form
st.header("Enter Customer Details")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Account Balance", value=50000.00)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ['Yes', 'No'])
is_active_member = st.selectbox("Is Active Member?", ['Yes', 'No'])
estimated_salary = st.number_input("Estimated Salary", value=100000.00)
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])

if st.button("Predict"):
    try:
        # Encode Gender (label encoding)
        gender_encoded = label_encoder.transform([gender])[0]

        # One-hot encode Geography
        geography_encoded = onehot_encoder.transform([[geography]]).toarray()  # shape (1, 3)

        # Combine all 12 features in correct order
        full_input = np.array([[credit_score, gender_encoded, age, tenure, balance,
                                num_of_products, int(has_cr_card == 'Yes'),
                                int(is_active_member == 'Yes'), estimated_salary]])
        
        # Add geography columns
        full_input = np.concatenate([full_input, geography_encoded], axis=1)  # shape (1, 12)

        # Scale all 12 features
        scaled_input = scaler.transform(full_input)

        # Predict
        prediction = model.predict(scaled_input)[0][0]
        churn = prediction > 0.5

        # Display result
        st.subheader("Prediction:")
        st.write(f"Churn Probability: {prediction:.4f}")
        st.success("Customer will **Churn**." if churn else "Customer will **Stay**.")
    
    except Exception as e:
        st.error(f"Error occurred: {e}")
