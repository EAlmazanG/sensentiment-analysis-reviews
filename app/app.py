import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
from src import plots
from src import ml_processing

# Function to load data from uploaded file
@st.cache_data
def loadData(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def extractPrefix(file_name):
    # Split the filename and extract the part before "_ml"
    return file_name.split('_ml')[0]

def loadJson(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def reFormatEmbeddings(embedding_str):
    cleaned_str = re.sub(r'[\[\]\n]', '', embedding_str)
    embedding_list = list(map(float, cleaned_str.split()))
    return np.array(embedding_list, dtype=np.float32)
    return embedding_str

processed_path = '../data/processed/'
raw_path = '../data/raw/'

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Reviews Dashboard",
    page_icon="🍽️",
    layout="wide",
)

# File uploader for CSV selection and all the necesary data
st.sidebar.header("Select CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    ## Load all necessary data
    # Load reviews data and extract place from the file name
    reviews = loadData(uploaded_file)
    print(reviews[['embedding']])
    if 'embedding' in reviews.columns:
        # Convert embeddings from string to list of floats
        reviews['embedding'] = reviews['embedding'].apply(reFormatEmbeddings)

    file_name = uploaded_file.name
    place = extractPrefix(file_name)
    
    st.header(f"{place.upper()}")
    
    # Paths for the JSON and additional CSV files
    general_insights_file = os.path.join(processed_path, f"{place}_general_insights.json")
    worst_periods_file = os.path.join(processed_path, f"{place}_worst_periods_insights.json")
    sample_reviews_file = os.path.join(processed_path, f"{place}_sample_selected_reviews.csv")
    resume_file = os.path.join(raw_path, f"resumme_{place}.csv")
    
    # Load "place"_general_insights.json into a dictionary
    if os.path.exists(general_insights_file):
        general_insights = loadJson(general_insights_file)
        #st.write("General Insights:", general_insights)
    else:
        st.warning(f"{place}_general_insights.json not found in {processed_path}")

    # Load "place"_worst_periods_insights.json into a dictionary
    if os.path.exists(worst_periods_file):
        worst_periods_insights = loadJson(worst_periods_file)
        #st.write("Worst Periods Insights:", worst_periods_insights)
    else:
        st.warning(f"{place}_worst_periods_insights.json not found in {processed_path}")
    
    # Load "place"_sample_selected_reviews.csv into a DataFrame
    if os.path.exists(sample_reviews_file):
        sample_reviews = pd.read_csv(sample_reviews_file)
        #st.write("Sample Selected Reviews:")
        #st.dataframe(sample_reviews)
    else:
        st.warning(f"{place}_sample_selected_reviews.csv not found in {processed_path}")

    # Load resumme_"place".csv from ./data/raw into a DataFrame
    if os.path.exists(resume_file):
        resume = pd.read_csv(resume_file)
        #st.write(f"Resume data for {place}:")
        #st.dataframe(resume)
    else:
        st.warning(f"resumme_{place}.csv not found in {raw_path}")

else:
    st.write("Please upload a ML processed CSV file.")

sys.path.append(os.path.abspath(os.path.join('..')))
from src import plots

tab1, tab2, tab3 = st.tabs(["Main", "Detail", "Advanced"])

with tab1:
    st.markdown("<h2 style='text-align: center; color: #00000;'>Average Scores and Reviews Plot</h2>", unsafe_allow_html=True)
    fig = plots.plotAverageScoresAndReviews(reviews, resume, app=True)
    st.plotly_chart(fig, use_container_width=True)
    fig = plots.plotScoreTrends(reviews, app=True)
    st.plotly_chart(fig, use_container_width=True)

    st.header("Last reviews")
    col1, col2 = st.columns(2)
    with col1:
        # recent_best_reviews
        recent_best_reviews = sample_reviews[sample_reviews['sample_type'] == 'recent_best_reviews'][['date', 'rating_score','review', 'food_score', 'service_score', 'atmosphere_score', 'meal_type']]
        recent_best_reviews.rename(columns = {'review':'Review', 'rating_score':'Rating', 'meal_type':'Meal','food_score':'Food', 'service_score':'Service', 'atmosphere_score':'Ambient', 'date':'Date'}, inplace = True)
        st.markdown("<h3 style='text-align: left;'> 👍  Best!</h3>", unsafe_allow_html=True)
        st.dataframe(recent_best_reviews.head(5))
    with col2:
        # recent_worst_reviews
        recent_worst_reviews = sample_reviews[sample_reviews['sample_type'] == 'recent_worst_reviews'][['date', 'rating_score','review', 'food_score', 'service_score', 'atmosphere_score', 'meal_type']]
        recent_worst_reviews.rename(columns = {'review':'Review', 'rating_score':'Rating', 'meal_type':'Meal','food_score':'Food', 'service_score':'Service', 'atmosphere_score':'Ambient', 'date':'Date'}, inplace = True)
        st.markdown("<h3 style='text-align: left;'> 👎  Worst...</h3>", unsafe_allow_html=True)
        st.dataframe(recent_worst_reviews.head(5))
    
with tab2:
    pass

with tab3:
    st.markdown("<h2 style='text-align: center; color: #00000;'>Sentiment Plots</h2>", unsafe_allow_html=True)
    fig = plots.plotSentimentTrend(reviews, years_limit = 2, app = True)
    st.plotly_chart(fig, use_container_width=True)

    fig = plots.plotCommunities(reviews, app = True)
    st.plotly_chart(fig, use_container_width=True)

    embeddings_pca, fig = ml_processing.visualizeEmbeddingsPCA(reviews, plot = True, app = True)
    st.plotly_chart(fig, use_container_width=True)

    #embeddings_umap, fig = ml_processing.visualizeEmbeddingsUMAP(reviews, plot = True, app = True)
    #st.plotly_chart(fig, use_container_width=True)

    #fig = plots.plotKdistance(embeddings_umap, k= 10, method='PCA', app = True)
    #st.plotly_chart(fig, use_container_width=True)

    #fig = plots.plotKdistance(embeddings_pca, k= 10, method='UMAP', app = True)
    #st.plotly_chart(fig, use_container_width=True)

    #pca_clusters, fig = ml_processing.visualizeEmbeddingsPCA_with_DBSCAN(reviews, eps=0.5, min_samples=5, plot = True, app = True)
    #st.plotly_chart(fig, use_container_width=True)

    #umap_clusters, fig = ml_processing.visualizeEmbeddingsUMAP_with_DBSCAN(reviews, eps=0.5, min_samples=5, plot = True, app = True)
    #st.plotly_chart(fig, use_container_width=True)
