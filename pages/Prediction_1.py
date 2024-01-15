from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pandas as pd
import streamlit as st
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = {}


def determine_severity(row):
    if row['INJURIES_FATAL'] > 0:
        return 3  # Fatal
    elif row['INJURIES_INCAPACITATING'] > 0:
        return 2  # Incapacitating
    elif row['INJURIES_NON_INCAPACITATING'] > 0 or row['INJURIES_REPORTED_NOT_EVIDENT'] > 0:
        return 1  # Non-incapacitating
    else:
        return 0  # No injury
st.title('Severity Type Prediction')
# Apply the function to each rows
def user_input_function():
    # Collect user inputs for each feature
    hour = st.slider("Hour of the Day", 0, 23, step=1)
    weather_condition = st.selectbox("Weather Condition", options=df['WEATHER_CONDITION'].unique())
    lighting_condition = st.selectbox("Lighting Condition", options=df['LIGHTING_CONDITION'].unique())
    st.session_state['input_data'] = {
        "CRASH_HOUR": hour,
        "WEATHER_CONDITION": weather_condition,
        "LIGHTING_CONDITION": lighting_condition,
    }
# Initialize your model and variables globally


df = pd.read_csv(r'C:\Users\Мурад\DataspellProjects\VisTool_v.1\Traffic_crashes_10000.csv')
# ... (other data preprocessing steps)
df['ACCIDENT_SEVERITY'] = df.apply(determine_severity, axis=1)
# Define your features and labels
X = df[['CRASH_HOUR', 'WEATHER_CONDITION', 'LIGHTING_CONDITION']]  # Add more features as needed
y = df['ACCIDENT_SEVERITY']  # This comes from your determine_severity function

# Convert categorical data to numerical if necessary
X = pd.get_dummies(X)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
# Store the columns from the training data to ensure the input data matches
model_columns = X.columns

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
user_input_function()
# Prediction section within the if statement for file upload
if st.button("Predict Severity of the accident"):
    input_data = st.session_state['input_data']
    if input_data:
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        # Ensure the input has the same columns as the training data
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]

        try:
            prediction = model.predict(input_df)
            st.write("Predicted Accident Severity:", prediction[0])
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.warning("Please provide all the input data.")

st.write("Model Cross-Validation Scores:", scores)
st.write("Training Classification Report")
st.text(classification_report(y_train, model.predict(X_train)))
st.write("Test Classification Report")
st.text(classification_report(y_test, model.predict(X_test)))