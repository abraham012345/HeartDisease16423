import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk

# Load the model and data
model = pk.load(open('Heart_disease_model.pkl', 'rb'))
data = pd.read_csv('heart_disease.csv')

# Set up the Streamlit app
st.title('Heart Disease Predictor')

# Sidebar for user inputs
st.sidebar.header('User Inputs')

# Gender selection
gender = st.sidebar.radio('Choose Gender', ('Male', 'Female','Transgender'))

gen = 1 if gender == 'Male' elseif
gen = 2 if gender == 'Transgender' else 0


# Patient details section
st.sidebar.subheader('Patient Details')

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25, step=1)

smoking_status = st.sidebar.radio("Current Smoker", ("Yes", "No"))

cigsPerDay = st.sidebar.slider("Cigarettes per Day", min_value=0, max_value=100, value=0, step=1,
                               key='cigsPerDay') if smoking_status == 'Yes' else 0

# Medical history section
st.sidebar.subheader('Medical History')

BPMeds = st.sidebar.radio("Blood Pressure Medications", ("Yes", "No"))

prevalentStroke = st.sidebar.radio("Had Stroke", ("Yes", "No"))

prevalentHyp = st.sidebar.radio("Prevalent Hypertension", ("Yes", "No"))

diabetes = st.sidebar.radio("Diabetes", ("Yes", "No"))

# Additional inputs...
totChol = st.sidebar.slider("Total Cholesterol", min_value=50, max_value=600, value=200, step=1)
sysBP = st.sidebar.slider("Systolic Blood Pressure", min_value=70, max_value=250, value=120, step=1)
# Add other inputs similarly

# Prediction button
if st.sidebar.button('Predict'):
    input_data = np.array([[age, 1 if smoking_status == 'Yes' else 0, cigsPerDay, 1 if BPMeds == 'Yes' else 0,
                            1 if prevalentStroke == 'Yes' else 0, 1 if prevalentHyp == 'Yes' else 0,
                            1 if diabetes == 'Yes' else 0, totChol, sysBP, 0, 0, 0, 0, gen]])
    
    output = model.predict(input_data)
    
    # Display prediction result
    st.header('Prediction Result')
    if output[0] == 0:
        st.write('Patient is Healthy, No Heart Disease')
    else:
        st.write('Patient May have Heart Disease')
