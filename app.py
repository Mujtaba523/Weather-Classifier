# Importing Libraries
import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import json
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import warnings
warnings.simplefilter('ignore')

# Utility Functions
def load_lottie(filepath):
    """
    Load and return the contents of a Lottie file.
    
    Args:
        filepath (str): Path to the Lottie file.
    
    Returns:
        dict: JSON contents of the Lottie file.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def load_models():
    """
    Load and return the trained models used for weather classification.
    
    Returns:
        dict: Dictionary containing the loaded models.
    """
    models = {
        "Decision Tree": joblib.load("Models/dt.pkl"),
        "Logistic Regression": joblib.load("Models/lr.pkl"),
        "K-Nearest Neighbors": joblib.load("Models/knn.pkl"),
        "Naive Bayes": joblib.load("Models/nb.pkl"),
        "Stochastic Gradient Descent": joblib.load("Models/sgd.pkl"),
        "Support Vector Machine": joblib.load("Models/svc.pkl"),
        "Gradient Boosting": joblib.load("Models/gb.pkl"),
        "Bagging Classifier": joblib.load("Models/bg.pkl"),
        "Feed-Forward Neural Network": tf.keras.models.load_model("Models/FFNN.h5"),
        "Recurrent Neural Network": tf.keras.models.load_model("Models/RNN.h5")
    }
    return models

# Initial configuration
st.set_page_config(
    page_title="⛅️SkyInsight: AI-Powered Weather Forecasting System",
    initial_sidebar_state="expanded",
    layout='wide',
)

models = load_models()    
# Display the logo in the sidebar
st.sidebar.image("Vectors/logo.png", use_column_width=True)
# Create the sidebar and select model
st.title("SkyInsight: AI-Powered Weather Forecasting System")
st.markdown("SkyInsight is a powerful system designed to accurately classify weather conditions as Overcast, Clear, or Foggy. This classifier employs advanced techniques from machine learning, deep learning, and transfer learning domains to achieve precise weather predictions.")
animation1 = load_lottie("Vectors/1.json")
st_lottie(animation1, loop=True, width=None, height=600)
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

# Classification
st.markdown("## Classification of Weather")
col1, col2 = st.columns(2)
with col1:
    # Weather attributes input form
    humidity = st.slider("Humidity", 0.0, 1.0, 0.5, step=0.1)
    time = st.slider("Time (hour)", 0, 23, 12)
    year = st.slider("Year", 2006, 2023, 2016)
    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day", 1, 31, 13)
with col2:
    # Weather attributes input form
    wind_speed = st.number_input("Wind Speed (km/h)", value=10, min_value=0, max_value=100, step=1)
    wind_bearing = st.number_input("Wind Bearing (degrees)", value=180, min_value=0, max_value=360, step=1)
    visibility = st.number_input("Visibility (km)", value=10.0, min_value=0.0, max_value=20.0, step=0.1)
    pressure = st.number_input("Pressure (millibars)", value=1000.0, min_value=800.0, max_value=1100.0, step=1.0)
    precip_type = st.radio("Precipitation Type", ["rain", "snow"])
    apparent_temp = st.number_input("Apparent Temperature (C)", value=25, min_value=-25, max_value=40, step=1)
    # Preparation of the input data
    input_data = pd.DataFrame({
        "Precip Type": [precip_type],
        "Apparent Temperature (C)": [apparent_temp],
        "Humidity": [humidity],
        "Wind Speed (km/h)": [wind_speed],
        "Wind Bearing (degrees)": [wind_bearing],
        "Visibility (km)": [visibility],
        "Pressure (millibars)": [pressure],
        "Time": [time],
        "Year": [year],
        "Month": [month],
        "Day": [day]
    })
    # Encoding
    mapping = {'rain': 0, 'snow': 1}
    input_data['Precip Type'] = input_data['Precip Type'].map(mapping)
    # Standardizing
    scaler = joblib.load("Models/scaler.pkl")
    scaled_input_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
    # Output Mappings
    classes = {0: "Clear", 1: "Foggy", 2: "Overcast"}
    # Perform weather classification
    if selected_model == "Recurrent Neural Network":
        # Reshape input data for RNN model
        scaled_input_data = np.expand_dims(scaled_input_data, axis=2)
        predictions = models[selected_model].predict(scaled_input_data)[0]
    elif selected_model == "Feed-Forward Neural Network":
        predictions = models[selected_model].predict(scaled_input_data)[0]
    else:
        predictions = models[selected_model].predict(scaled_input_data)[0]
        probabilities = models[selected_model].predict_proba(scaled_input_data)


# Display the predicted weather summary and probabilities
st.subheader("Result")
if selected_model in ["Recurrent Neural Network", "Feed-Forward Neural Network"]:
    predicted_class = np.argmax(predictions)
    predicted_prob = predictions[predicted_class]
    st.markdown(f"##### The classified weather summary is: {classes[predicted_class]}")
    st.markdown(f"##### The classified probability is: {predicted_prob*100:.2f}%")
else:
    max_prob_indices = np.argmax(probabilities, axis=1)
    for i, max_prob_index in enumerate(max_prob_indices):
        max_prob = probabilities[i, max_prob_index]
        # Print the highest probability and its corresponding class label
        st.write(f"##### The classified weather summary is: {classes[predictions]}")
        st.write(f"##### The classified probability is: {max_prob*100:.2f}%")

st.write('---')
st.subheader("Feature Importances")
if selected_model in ["Recurrent Neural Network", "Feed-Forward Neural Network"]:
    st.warning("Feature importances are not available for this model.")
else:
    if hasattr(models[selected_model], "feature_importances_"):
        importances = models[selected_model].feature_importances_
    elif hasattr(models[selected_model], "coef_"):
        importances = np.abs(models[selected_model].coef_[0])
    else:
        st.warning("Feature importances are not available for this model.")
        importances = None

    if importances is not None:
        df_importances = pd.DataFrame(
            {"Feature": scaled_input_data.columns, "Importance": importances}
        )
        df_importances = df_importances.sort_values("Importance", ascending=False)
        fig = go.Figure(
            go.Bar(
                x=df_importances["Feature"],
                y=df_importances["Importance"],
                marker_color="#ffc930"
            )
        )
        col_width = st.beta_columns([1])[0].width
        if col_width:
            fig.update_layout(width=col_width)
        fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Importance",
            font=dict(color="#ffc930", size=20),
            height=600,
        )
        fig.update_xaxes(showgrid=True, gridcolor="white")
        fig.update_yaxes(showgrid=True, gridcolor="white")
        
        # Display the chart using st.plotly_chart()
        st.plotly_chart(fig)
