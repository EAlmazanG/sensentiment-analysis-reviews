import streamlit as st
import pandas as pd

# Function to load data from uploaded file
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def extractPrefix(file_name):
    # Split the filename and extract the part before "_ml"
    return file_name.split('_ml')[0]

# File uploader for CSV selection
st.sidebar.header("Select CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    file_name = uploaded_file.name
    place = extractPrefix(file_name)
    st.write(place)

if data is not None:
    st.write("Here are the first 10 rows of your data:")
    st.dataframe(data.head(10))
else:
    st.write("Please upload a CSV file.")