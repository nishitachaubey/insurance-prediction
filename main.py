import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Loading the trained model
model = joblib.load('medical_cost_model.pkl')

# main function
def main():
    st.title("Medical Insurance Cost Prediction")

    # features input 
    st.sidebar.header("Input Features")

    age = st.sidebar.slider('Age', 18, 100, 30)
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    bmi = st.sidebar.number_input('BMI', 15.0, 53.0, 25.0)
    children = st.sidebar.slider('Number of Children', 0, 5, 0)
    smoker = st.sidebar.selectbox('Smoker', ['yes', 'no'])
    region = st.sidebar.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Medical Insurance Cost: ${prediction[0]:.2f}")

if __name__ == '__main__':
    main()
