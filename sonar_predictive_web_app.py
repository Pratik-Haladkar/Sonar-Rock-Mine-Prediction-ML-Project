# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:27:05 2026

@author: DELL
"""

import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("sonar_model.sav", "rb"))

# App title
st.title("🔍 Sonar Rock vs Mine Prediction")
st.write("Enter the sonar signal values to predict whether the object is a Rock or a Mine.")

# Input fields (Sonar dataset has 60 features)
inputs = []
for i in range(60):
    value = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=1.0, step=0.01)
    inputs.append(value)

# Predict button
if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.success("🟢 The object is a **Mine**")
    else:
        st.success("🔵 The object is a **Rock**")

    



