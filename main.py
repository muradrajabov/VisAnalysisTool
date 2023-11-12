import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pygwalker as pyg
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(page_title="VisTool", page_icon=":chart_with_upwards_trend:", layout="wide")
# Set a title for your app
st.title(" 	:chart_with_upwards_trend: Traffic Accident Data Visualization Tool")
st.markdown(
    """
    <style>
    div.block-container{padding-top:1rem;}
    </style>
    """,
    unsafe_allow_html=True
)
# Create a file uploader to allow users to upload their CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    def load_df():
        return df
    df = pd.read_csv(uploaded_file)
    # Create a separate page to display visualizations

    # Display a sample of the uploaded data

    # Basic manipulations of data
    # Creating separate columns for date and time for time series charts
    df['CRASH_DATE_SEP'] = pd.to_datetime(df['CRASH_DATE']).dt.date
    df['CRASH_TIME'] = pd.to_datetime(df['CRASH_DATE']).dt.time

    st.subheader("Choose Start Date and End Date of your dataset:")
    # Getting user an opportunity to select the minimum and maximum date in dataset
    col1, col2 = st.columns(2)
    df['CRASH_DATE_SEP'] = pd.to_datetime(df['CRASH_DATE_SEP'])
    startDate = df['CRASH_DATE_SEP'].min()
    endDate = df['CRASH_DATE_SEP'].max()
    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))

    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    df = df[(df['CRASH_DATE_SEP'] >= pd.Timestamp(date1)) & (df['CRASH_DATE_SEP'] <= pd.Timestamp(date2))]
    st.header("Sample Data")

    # Display a sample of the uploaded data
    st.write(df.head())

    # Representing basic statistical information
    st.subheader("Statistical Information")
    st.caption("Shape of the data(rows, columns):")
    st.write(df.shape)
    st.caption("Information about each column, and its data type")
    st.write(df.info())
    st.caption("Information about statistical measures:mean, count, standard deviation, and quantiles, etc.")
    st.write(df.describe())

    st.header("Visualizations")
    category_df_1 = df.groupby(by="WEATHER_CONDITION", as_index=False)["CRASH_RECORD_ID"].count()
    category_df_2 = df.groupby(by="LIGHTING_CONDITION", as_index=False)["CRASH_RECORD_ID"].count()
    st.subheader("Number of accidents by weather and lighting")

    fig1 = px.bar(category_df_1, x="WEATHER_CONDITION", y="CRASH_RECORD_ID", title="Weather")
    st.plotly_chart(fig1, use_container_width=True, height=200)

    fig2 = px.bar(category_df_2, x="LIGHTING_CONDITION", y="CRASH_RECORD_ID", title="Lighting")
    st.plotly_chart(fig2, use_container_width=True, height=200)

    # Visualizing geospatial data using PyGWalker
    st.subheader("Interactive map")
    pyg_html = pyg.walk(df, return_html=True)
    components.html(pyg_html, height=700, width=1200, scrolling=True)
else:
    st.warning("Please upload a CSV file to proceed.")

#%%
