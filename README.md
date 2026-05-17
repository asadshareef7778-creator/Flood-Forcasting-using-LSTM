# Flood-Forcasting-using-LSTM
A deep learning model built for three of the most flood effective districts of Punjab using Long Short Term Memory
Flood Guard AI – Flood Severity Prediction System

Flood Guard AI is a machine learning and deep learning based flood prediction project developed for district-level flood forecasting in Punjab, Pakistan. The goal of this project is to provide an early warning system that can help in disaster preparedness and flood risk management.

The system combines deep learning and machine learning models to forecast river discharge and classify flood severity levels for different districts.

Target Districts

• Sialkot
• Okara
• Multan

How the System Works

The project uses a two-stage prediction pipeline.

First, an LSTM model predicts future river discharge values using historical hydro-meteorological data.

Then, the predicted discharge values are passed to a classification model such as XGBoost or LightGBM to determine flood severity levels.

Flood Severity Levels

Level 0 — Normal Conditions  
Level 1 — Low Flood Risk  
Level 2 — Moderate Flood Risk  
Level 3 — Severe Flood Risk

Technologies Used

• Python  
• TensorFlow / Keras  
• Scikit-learn  
• XGBoost  
• LightGBM  
• Pandas  
• NumPy  
• Streamlit

Features

• District-wise flood prediction  
• Flood severity classification  
• Interactive Streamlit web application  
• Real-time predictions  
• Simple and user-friendly interface

Dataset Features

The models were trained using hydro-meteorological and environmental parameters including:

• Rainfall  
• River discharge  
• Temperature  
• Humidity  
• Wind speed  
• Pressure  
• Remote sensing features

Running the Project

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

Deployment

The application can be deployed using Streamlit Community Cloud for free public access.

Future Improvements

• Real-time weather API integration  
• Satellite imagery support  
• GIS-based flood visualization  
• Mobile application  
• Real-time flood alerts

Research Purpose


This project was developed as part of research focused on AI-driven flood early warning systems for Punjab, Pakistan, aiming to improve disaster preparedness through intelligent flood forecasting.

License

This project is intended for educational and research purposes.
An abstract has already been published on this Approach by myself in International Agri Roshan Watan Conference and has also been presented there.
