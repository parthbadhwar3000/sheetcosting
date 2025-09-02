import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

with open('model.pkl','rb') as file:
    model=pickle.load(file)

st.title("SS Sheet Cost Prediction(IIT Roorkee dataset)")

length=st.number_input("Length(in mm)")
width=st.number_input("Width (in mm)")

input_data=pd.DataFrame({
    'Length(in mm)': [length],
    'Width (in mm)': [width]
})


if st.button('Submit'):
    input_data_scaled=scaler.transform(input_data)
    prediction=model.predict(input_data_scaled)
    st.write(f"Predicted Cost ₹{prediction[0]:.2f}")
