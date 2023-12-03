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

# Sample car names for the dropdown list
car_names = [
    "Toyota Camry", "Honda Accord", "Ford Mustang", "Chevrolet Malibu", "Tesla Model S",
    "Nissan Altima", "BMW 3 Series", "Mercedes-Benz C-Class", "Audi A4", "Lexus ES",
    "Volkswagen Passat", "Hyundai Sonata", "Subaru Outback", "Mazda6", "Kia Optima",
    "Jeep Grand Cherokee", "Ford Explorer", "Toyota RAV4", "Honda CR-V", "Subaru Forester"
]

# Streamlit App
st.set_page_config(
    page_title="Car Price Prediction App",
    page_icon="🚗",
    layout="wide"
)

# Sidebar with user input
st.sidebar.title('Car Details')
car_name = st.sidebar.selectbox('Select Car Name', car_names)
horsepower = st.sidebar.slider('Horsepower', min_value=50, max_value=300, value=150)
curbweight = st.sidebar.slider('Curb Weight', min_value=1500, max_value=5000, value=3000)
enginesize = st.sidebar.slider('Engine Size', min_value=50, max_value=500, value=200)

# Predict button
if st.sidebar.button('Predict Price'):
    prediction = predict_price(horsepower, curbweight, enginesize)
    st.success(f"Predicted Price for {car_name}: ${prediction:.2f}")

# Main content area
st.title('Car Price Prediction App')

# Display car image or additional information if available
# You can customize this section based on your specific requirements

# Display additional information or insights about the car features
st.subheader('Car Features Overview')
st.markdown(f"**Car Name:** {car_name}")
st.markdown(f"**Horsepower:** {horsepower} HP")
st.markdown(f"**Curb Weight:** {curbweight} lbs")
st.markdown(f"**Engine Size:** {enginesize} cubic inches")

# You can add more sections or insights as needed

# Display a disclaimer or additional information at the bottom
st.sidebar.markdown(
    """
    **Disclaimer:** This app provides estimated car prices based on user input and a pre-trained machine learning model.
    The predictions are for informational purposes only and may not reflect the actual market prices.
    """
)