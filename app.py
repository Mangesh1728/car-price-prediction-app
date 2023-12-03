import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained XGBoost model
final_xgb_model = joblib.load('/content/final_xgboost_model.joblib')

def predict_price(horsepower, curbweight, enginesize):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'horsepower': [horsepower],
        'curbweight': [curbweight],
        'enginesize': [enginesize]
    })

    # Make a prediction using the loaded model
    prediction = final_xgb_model.predict(input_data)

    return prediction[0]

# Streamlit App
st.title('Car Price Prediction App')
st.markdown("<h2 style='text-align: center;'>Enter Car Features</h2>", unsafe_allow_html=True)

# Input form for user input (center-aligned)
col1, col2, col3 = st.columns(3)

with col1:
    horsepower = st.slider('Horsepower', min_value=50, max_value=300, value=150)

with col2:
    curbweight = st.slider('Curb Weight', min_value=1500, max_value=5000, value=3000)

with col3:
    enginesize = st.slider('Engine Size', min_value=50, max_value=500, value=200)

# Predict button (center-aligned)
if st.button('Predict Price'):
    prediction = predict_price(horsepower, curbweight, enginesize)
    st.markdown(f"<h3 style='text-align: center;'>Predicted Price: ${prediction:.2f}</h3>", unsafe_allow_html=True)
