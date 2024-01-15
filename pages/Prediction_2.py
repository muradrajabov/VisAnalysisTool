import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = {}

# Load the data
df = pd.read_csv(r'C:\Users\Мурад\DataspellProjects\VisTool_v.1\Traffic_crashes_10000.csv')


# Drop rows with missing values in relevant columns
relevant_columns = ['FIRST_CRASH_TYPE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'CRASH_HOUR']
df = df.dropna(subset=relevant_columns)

# Convert categorical data to numerical using Label Encoding
label_encoders = {}
for column in relevant_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Features and target variable
X = df[['WEATHER_CONDITION', 'LIGHTING_CONDITION', 'CRASH_HOUR']]
y = df['FIRST_CRASH_TYPE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
# Predictions and evaluation
y_pred = model.predict(X_test)

# Classification report


# Accuracy Score


# Streamlit user interface for making predictions
st.title('Accident Type Prediction')

# User inputs for conditions
weather_condition = st.selectbox("Weather Condition", options=label_encoders['WEATHER_CONDITION'].classes_)
lighting_condition = st.selectbox("Lighting Condition", options=label_encoders['LIGHTING_CONDITION'].classes_)
crash_hour = st.slider("Hour of the Day", 0, 23)

if st.button("Predict Type of Crash"):
    # Prepare the input for prediction
    weather_encoded = label_encoders['WEATHER_CONDITION'].transform([weather_condition])[0]
    lighting_encoded = label_encoders['LIGHTING_CONDITION'].transform([lighting_condition])[0]
    hour_encoded = label_encoders['CRASH_HOUR'].transform([crash_hour])[0]
    input_data = np.array([[weather_encoded, lighting_encoded, hour_encoded]])

    # Make prediction
    prediction_encoded = model.predict(input_data)
    prediction = label_encoders['FIRST_CRASH_TYPE'].inverse_transform(prediction_encoded)
    st.write(f"The predicted type of crash is: {prediction[0]}")