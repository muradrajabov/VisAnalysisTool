import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pygwalker as pyg
import streamlit.components.v1 as components
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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

    def determine_severity(row):
        if row['INJURIES_FATAL'] > 0:
            return 3  # Fatal
        elif row['INJURIES_INCAPACITATING'] > 0:
            return 2  # Incapacitating
        elif row['INJURIES_NON_INCAPACITATING'] > 0 or row['INJURIES_REPORTED_NOT_EVIDENT'] > 0:
            return 1  # Non-incapacitating
        else:
            return 0  # No injury
    df['ACCIDENT_SEVERITY'] = df.apply(determine_severity, axis=1)
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
    category_df_1 = category_df_1.sort_values(by="CRASH_RECORD_ID", ascending=False)
    category_df_2 = df.groupby(by="LIGHTING_CONDITION", as_index=False)["CRASH_RECORD_ID"].count()
    category_df_2 = category_df_2.sort_values(by="CRASH_RECORD_ID", ascending=False)
    st.subheader("Number of accidents by weather and lighting")

    fig1 = px.bar(category_df_1, x="WEATHER_CONDITION", y="CRASH_RECORD_ID", title="Weather")
    st.plotly_chart(fig1, use_container_width=True, height=200)

    fig2 = px.bar(category_df_2, x="LIGHTING_CONDITION", y="CRASH_RECORD_ID", title="Lighting")
    st.plotly_chart(fig2, use_container_width=True, height=200)

    time_series_df = df.groupby('CRASH_DATE_SEP')['CRASH_RECORD_ID'].count().reset_index()
    fig_time = px.line(time_series_df, x='CRASH_DATE_SEP', y='CRASH_RECORD_ID', title='Trend of Accidents Over Time')
    st.plotly_chart(fig_time, use_container_width=True)

    severity_counts = df['ACCIDENT_SEVERITY'].value_counts().reset_index()
    fig_severity = px.pie(severity_counts, values='ACCIDENT_SEVERITY', names=severity_counts.index, title='Distribution of Accident Severity')
    st.plotly_chart(fig_severity, use_container_width=True)

    fig_cluster = px.scatter_mapbox(df, lat='LATITUDE', lon='LONGITUDE', color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10, title='Cluster Map of Accidents')
    fig_cluster.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_cluster, use_container_width=True)
    # Visualizing geospatial data using PyGWalker
    st.subheader("Interactive map")
    pyg_html = pyg.walk(df, return_html=True)
    components.html(pyg_html, height=700, width=1200, scrolling=True)

    # Add a heatmap to the visualizations
    st.subheader("Correlation Heatmap")
    st.caption("A heatmap showing correlations between numeric variables in the dataset.")
    # Selecting only numeric columns for the heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"size": 8})
    st.pyplot(plt)
else:
    st.warning("Please upload a CSV file to proceed.")


#%%
