import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import sys
import importlib

import plotly.graph_objects as go
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('..')))
from src import plots
importlib.reload(plots)
from src import ml_processing
importlib.reload(ml_processing)

import tab_1
importlib.reload(tab_1)
import tab_3
importlib.reload(tab_3)
import header
importlib.reload(header)

# Function to load data from uploaded file
@st.cache_data
def loadData(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def loadAdditionalData(reviews, raw_path, processed_path):
    if 'embedding' in reviews.columns:
        # Convert embeddings from string to list of floats
        reviews['embedding'] = reviews['embedding'].apply(reFormatEmbeddings)

    file_name = uploaded_file.name
    place = extractPrefix(file_name)
    
    #st.markdown("<h3 style='text-align: center; color: #00000;'> 📊 ⭐ Sentiment Analysis Reviews ⭐ 📊</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #000000;'>🍴 {place.upper()} 🍴</h1>", unsafe_allow_html=True)

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
    
    return place, reviews, sample_reviews, resume, general_insights, worst_periods_insights 

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

# Filter data for the last periods based on filter_min and filter_max
def addFilters(reviews, filter_min, filter_max):
    if filter_min is not None:
        filter_min = pd.to_datetime(filter_min)
    if filter_max is not None:
        filter_max = pd.to_datetime(filter_max)

    reviews['date'] = pd.to_datetime(reviews['date'])
    # Set default values for start_date and end_date
    limit_date = reviews['date'].max()
    start_date = filter_min if filter_min is not None else limit_date - pd.DateOffset(years=1)
    end_date = filter_max if filter_max is not None else limit_date

    # Apply filtering
    selected_reviews = reviews[(reviews['date'] >= start_date) & (reviews['date'] <= end_date)]
    return selected_reviews

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
    
    place, reviews, sample_reviews, resume, general_insights, worst_periods_insights = loadAdditionalData(reviews, raw_path, processed_path)

    label_mapping = {
        'rating_score': 'Rating',
        'food_score': 'Food',
        'service_score': 'Service',
        'atmosphere_score': 'Ambient'
    }

    ## Header plots
    average_score = (resume['stars'] * resume['reviews']).sum() / resume['reviews'].sum()

    # Display average score with stars
    stars = "⭐️" * int(round(average_score))
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; padding: 0px;">
            <h1 style="font-size: 50px; color: #4CAF50; margin: 0;">{round(average_score, 2)} {stars}</h1>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(f"<h4 style='text-align: center; color: #000000;'></h4>", unsafe_allow_html=True)
    # Layout columns
    col1, col2 = st.columns([10, 12])

    with col1:
        fig_line = header.weekEvolution(reviews, label_mapping)
        st.markdown("<h4 style='text-align: left;'>📆 Last 4 weeks</h4>", unsafe_allow_html=True)
        st.plotly_chart(fig_line)

    # Donut chart for reviews distribution
    with col2:
        st.markdown("<h4 style='text-align: left;'>⭐ Distribution</h4>", unsafe_allow_html=True)
        col21, col22 = st.columns(2)
        with col21:
            color_scale = ['#4CAF50', '#8BC34A', '#FFEB3B', '#FFC107', '#F44336']  # Green to Red scale
            resume['stars_label'] = resume['stars'].apply(lambda x: '⭐' * x)  # Convert stars to labels
            fig_donut = go.Figure(
                go.Pie(
                    labels=resume['stars_label'],
                    values=resume['reviews'],
                    hole=0.5,
                    marker=dict(colors=color_scale),
                    textinfo='percent+label',
                    insidetextorientation='radial'
                )
            )
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(t=20, b=50, l=80, r=80),
                height=250,
                width=250
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Bar chart for reviews count by score
        with col22:
            st.markdown("<h4 style='text-align: center;'> </h4>", unsafe_allow_html=True)
            fig_bar = go.Figure(
                go.Bar(
                    x=resume['stars_label'],
                    y=resume['reviews'],
                    marker=dict(color=color_scale),
                    text=resume['reviews'],
                    textposition='auto'
                )
            )
            fig_bar.update_xaxes(showgrid=False)
            fig_bar.update_yaxes(showgrid=False, showticklabels=False)
            fig_bar.update_layout(
                margin=dict(t=10, b=10, l=10, r=20),
                height=200,
                width=300,
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([" 📋 Status ", " 📢 Customer Insigths ", " 🕵🏻‍♂️ Bad times Deep Dive ", " 🧪 ML Lab "])

    ## Tabs
    with tab1:
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'> 📋 Status 📋</h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        # Year Overview
        reviews['year'] = reviews['date'].dt.year
        recent_reviews = reviews[reviews['date'] >= reviews['date'].max() - pd.DateOffset(years=8)]
        yearly_avg_scores = recent_reviews.groupby('year')['rating_score'].mean()

        fig = go.Figure(
            go.Scatter(
                x=yearly_avg_scores.index.astype(str),
                y=yearly_avg_scores.values,
                mode='lines+markers+text',  # Añade texto a los puntos
                line=dict(color='#32CD32', width=4),
                text=[f"{val:.2f}" for val in yearly_avg_scores],  # Valores como etiquetas
                textposition="top center",  # Posición de las etiquetas
                hoverinfo="text"
            )
        )

        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(t=20, b=20),
            height=280
        )
        st.markdown("<h4 style='text-align: left; color: #00000;'>🗓️ Overview </h4>", unsafe_allow_html=True)
        st.write("Annual average rating trends over recent years show overall performance. Spot upward or downward shifts and compare average ratings in Food, Service, and Ambient to identify strengths and improvement areas.")
        col1, col2 = st.columns([6, 3])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
        # Categories
            columns = list(label_mapping.keys())[1:]  # Excluye 'rating_score' si no se necesita
            average_scores = [reviews[col].mean() for col in columns]
            colors = ['rgba(31, 119, 180, 0.8)', 'rgba(107, 174, 214, 0.8)', 'rgba(158, 202, 225, 0.8)']

            fig_categories = go.Figure(
                go.Bar(
                    x= np.round(average_scores, 2),
                    y=[label_mapping[col] for col in columns],  # Etiquetas usando label_mapping en orden
                    marker=dict(color=colors),
                    text=[f"{score:.2f}" for score in average_scores],
                    textposition='auto',
                    orientation='h',
                    name="Categories"
                )
            )
            fig_categories.update_layout(
                xaxis=dict(showgrid=False, range=[0, 5], tickvals=[0, 1, 2, 3, 4, 5]),
                yaxis=dict(showgrid=False),
                margin=dict(t=20, b=20),
                height=280
            )
            st.plotly_chart(fig_categories, use_container_width=True, key="fig_categories_tab1")

        # Trend Overview
        st.markdown("<h4 style='text-align: left ;'>📝 Monthly Overview</h4>", unsafe_allow_html=True)
        st.write("Provides a more detailed view of average scores for each category throughout the year. It highlights fluctuations over specific months, allowing stakeholders to observe seasonal variations or significant dips and peaks that may need further investigation.")
        recent_reviews = reviews[reviews['date'] >= reviews['date'].max() - pd.DateOffset(years=1)]
        fig_trend = tab_3.plotTrend(recent_reviews, label_mapping, app = True)
        st.plotly_chart(fig_trend, use_container_width=True, key="fig_month_trend_tab1")

        st.markdown("<h4 style='text-align: left; color: #00000;'>🚨 Last Reviews</h4>", unsafe_allow_html=True)
        st.write("Selection of recent reviews, separated into the best and worst ratings. Use this feedback to understand current strengths and pinpoint opportunities for improvement.")
        col1, col2 = st.columns(2)
        with col1:
            # recent_best_reviews
            recent_best_reviews = sample_reviews[sample_reviews['sample_type'] == 'recent_best_reviews'][['date', 'rating_score','review', 'food_score', 'service_score', 'atmosphere_score', 'meal_type']]
            recent_best_reviews.rename(columns = {'review':'Review', 'rating_score':'Rating', 'meal_type':'Meal','food_score':'Food', 'service_score':'Service', 'atmosphere_score':'Ambient', 'date':'Date'}, inplace = True)
            st.markdown("<h5 style='text-align: left;'> 👍  Best!</h5>", unsafe_allow_html=True)
            st.dataframe(recent_best_reviews, height= 600)
        with col2:
            # recent_worst_reviews
            recent_worst_reviews = sample_reviews[sample_reviews['sample_type'] == 'recent_worst_reviews'][['date', 'rating_score','review', 'food_score', 'service_score', 'atmosphere_score', 'meal_type']]
            recent_worst_reviews.rename(columns = {'review':'Review', 'rating_score':'Rating', 'meal_type':'Meal','food_score':'Food', 'service_score':'Service', 'atmosphere_score':'Ambient', 'date':'Date'}, inplace = True)
            st.markdown("<h5 style='text-align: left;'> 👎  Worst...</h5>", unsafe_allow_html=True)
            st.dataframe(recent_worst_reviews, height= 600)
        
    with tab2:
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'> 📢 Customer Insights 📢 </h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.write("These insights summarize key themes and feedback extracted from all user reviews, highlighting strengths, pain points, and areas for improvement.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='text-align: center;'>💪 Strengths!</h5>", unsafe_allow_html=True)
            for insight in general_insights['best']:
                st.success('👍 ' + insight)

        with col2:
            st.markdown("<h5 style='text-align: center;'>🤬 Pain Points...</h5>", unsafe_allow_html=True)
            for insight in general_insights['worst']:
                st.error('👎 ' + insight)

        _, col2, _ = st.columns([1, 3, 1])

        with col2:
            st.markdown("<h4 style='text-align: center;'>💡 Areas for Improvement</h4>", unsafe_allow_html=True)
            for insight in general_insights['improve']:
                st.warning('⚠️ ' + insight)

        st.markdown("<h3 style='text-align: center; color: #00000;'></h3>", unsafe_allow_html=True)
        ## Filters
        col1, col2, col3 = st.columns([6, 2, 2])
        with col1:
            pass
        with col2:
            filter_min_tab2 = st.date_input("Start Date", None, key="filter_min_tab2")
        with col3:
            filter_max_tab2 = st.date_input("End Date", None, key="filter_max_tab2")
        # Apply the filter function
        sample_reviews_filtered = addFilters(sample_reviews, filter_min_tab2, filter_max_tab2)
        reviews_filtered = addFilters(reviews, filter_min_tab2, filter_max_tab2)

        st.markdown("<h4 style='text-align: left; color: #00000;'> 💘 Sentiment</h4>", unsafe_allow_html=True)
        st.write("Monthly evolution of customer feelings over the past year, divided into positive, neutral, and negative categories. It highlights periods of higher satisfaction or concerns, helping to spot trends in customer sentiment.")
        fig = plots.plotSentimentTrend(reviews_filtered, years_limit = 2, app = True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<h4 style='text-align: left; color: #00000;'>🤓 Reviews Overview</h4>", unsafe_allow_html=True)
        st.write("Here are recent high and low reviews, summarizing both positive feedback and areas for improvement from individual customers. Quick glance at what customers value most and where improvements can be made.")
        col1, col2 = st.columns(2)
        with col1:
            # best_reviews
            best_reviews = sample_reviews_filtered[sample_reviews_filtered['sample_type'] == 'best_reviews_sample'][['date', 'rating_score','review', 'food_score', 'service_score', 'atmosphere_score', 'meal_type']]
            best_reviews.rename(columns = {'review':'Review', 'rating_score':'Rating', 'meal_type':'Meal','food_score':'Food', 'service_score':'Service', 'atmosphere_score':'Ambient', 'date':'Date'}, inplace = True)
            best_reviews.fillna('', inplace=True)
            st.markdown("<h5 style='text-align: left;'> 👍  Best!</h5>", unsafe_allow_html=True)
            st.dataframe(best_reviews, height=500)
        with col2:
            # worst_reviews
            worst_reviews = sample_reviews_filtered[sample_reviews_filtered['sample_type'] == 'worst_reviews_sample'][['date', 'rating_score','review', 'food_score', 'service_score', 'atmosphere_score', 'meal_type']]
            worst_reviews.rename(columns = {'review':'Review', 'rating_score':'Rating', 'meal_type':'Meal','food_score':'Food', 'service_score':'Service', 'atmosphere_score':'Ambient', 'date':'Date'}, inplace = True)
            worst_reviews.fillna('', inplace=True) 
            st.markdown("<h5 style='text-align: left;'> 👎  Worst...</h5>", unsafe_allow_html=True)
            st.dataframe(worst_reviews, height=500)

    with tab3:
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'> 🕵🏻‍♂️ Bad Times Deep Dive 🕵🏻‍♂️ </h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.write("Identify the lowest-rated periods based on customer reviews, highlighting specific issues and improvement opportunities during times of lower satisfaction.")
        
        ## Filters
        col1, col2, col3 = st.columns([6, 2, 2])
        with col2:
            filter_min_tab3 = st.date_input("Start Date", None, key="filter_min_tab3")
        with col3:
            filter_max_tab3 = st.date_input("End Date", None, key="filter_max_tab3")

        # Apply the filter function
        reviews_filtered = addFilters(reviews, filter_min_tab3, filter_max_tab3)
        
        ## Trend Overview
        st.markdown("<h4 style='text-align: left ;'>📝 Overview</h4>", unsafe_allow_html=True)
        st.write("Average monthly ratings across different categories. The chart helps pinpoint drops and peaks, providing context for periods with notable fluctuations in customer satisfaction.")

        fig = tab_3.plotTrend(reviews_filtered, label_mapping, app = True)
        st.plotly_chart(fig, use_container_width=True)

        ## Problems by period
        st.markdown("<h4 style='text-align: left ;'>🔍 Period details</h4>", unsafe_allow_html=True)
        st.write("Customer feedback for the selected period, including specific problems and actionable improvement suggestions based on customer reviews. It allows a closer look at the challenges during low-rated periods.")

        # Filter low_score_periods based on filter_min and filter_max
        from datetime import datetime
        dates = list(worst_periods_insights.keys())
        dates = [datetime.strptime(date, '%Y-%m') for date in dates]
        limit_date = max(dates)

        start_date = pd.to_datetime(filter_min_tab3 if filter_min_tab3 is not None else (limit_date - pd.DateOffset(years=1)))
        end_date = pd.to_datetime(filter_max_tab3 if filter_max_tab3 is not None else limit_date)
        worst_periods_insights_filtered = {date: data for date, data in worst_periods_insights.items() if start_date <= datetime.strptime(date, '%Y-%m') <= end_date}
        
        for i, (period, insights) in enumerate(sorted(worst_periods_insights_filtered.items(), key=lambda x: x[0], reverse=True)):
            expanded = True if i == 0 else False

            with st.expander(f"🗓️  {period}", expanded=expanded):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("<h5 style='text-align: center;'>🤬 Problems</h5>", unsafe_allow_html=True)
                    for problem in insights['problems']:
                        st.error('💔 ' + problem)

                with col2:
                    st.markdown("<h5 style='text-align: center;'>🔧 Areas for Improvement</h5>", unsafe_allow_html=True)
                    for improvement in insights['improve']:
                        st.warning('💡' + improvement)

                # Reviews for the specific period
                period_reviews = sample_reviews[(sample_reviews['month'] == period) & (sample_reviews['sample_type'] == 'low_score_reviews')][['date', 'rating_score', 'review', 'food_score', 'service_score', 'atmosphere_score', 'meal_type']]
                period_reviews.rename(columns={'review': 'Review', 'rating_score': 'Rating', 'meal_type': 'Meal', 'food_score': 'Food', 'service_score': 'Service', 'atmosphere_score': 'Ambient', 'date': 'Date'}, inplace=True)
                period_reviews.fillna('', inplace=True)
                if period_reviews.shape[0] > 0:
                    st.dataframe(period_reviews, height = 150)

    with tab4:
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'>🧪 ML Lab 🧪</h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #00000;'></h2>", unsafe_allow_html=True)
        st.write("...")
        
        ## Filters
        col1, col2, col3 = st.columns([6, 2, 2])
        with col2:
            filter_min_ml = st.date_input("Start Date", None, key="filter_min_ml")
        with col3:
            filter_max_ml = st.date_input("End Date", None, key="filter_max_ml")

        # Apply the filter function
        reviews_filtered = addFilters(reviews, filter_min_ml, filter_max_ml)

        fig = plots.plotCommunities(reviews, app = True)

        st.markdown("<h4 style='text-align: left ;'> Sentence Communities</h4>", unsafe_allow_html=True)
        st.write("Average monthly ratings across different categories. The chart helps pinpoint drops and peaks, providing context for periods with notable fluctuations in customer satisfaction.")
        st.plotly_chart(fig, use_container_width=True)

        st.write('Add topics')

        st.markdown("<h4 style='text-align: left ;'>🧩 Dimensional reduccion</h4>", unsafe_allow_html=True)
        st.write("Average monthly ratings across different categories. The chart helps pinpoint drops and peaks, providing context for periods with notable fluctuations in customer satisfaction.")
        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            embeddings_pca, fig = ml_processing.calculateAndVisualizeEmbeddingsPCA(reviews, plot = False, app = True)
            st.markdown("<h3 style='text-align: center ;'>PCA</h3>", unsafe_allow_html=True)
            st.write('Add topics')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            embeddings_umap, fig = ml_processing.calculateAndVisualizeEmbeddingsUMAP(reviews, plot = False, app = True)
            st.markdown("<h3 style='text-align: center ;'>UMAP</h3>", unsafe_allow_html=True)
            st.write('Add topics')
            st.plotly_chart(fig, use_container_width=True)

        st.write('lorem ipsum')
        col1, col2 = st.columns(2)
        with col1:
            fig = plots.plotKdistance(embeddings_umap, k= 10, method='PCA', app = True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plots.plotKdistance(embeddings_pca, k= 10, method='UMAP', app = True)
            st.plotly_chart(fig, use_container_width=True)

        st.write('lorem ipsum')
        col1, col2 = st.columns(2)
        with col1:
            pca_clusters, fig = ml_processing.calculateAndVisualizeEmbeddingsPCA_with_DBSCAN(reviews, eps=0.5, min_samples=5, plot = False, app = True)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            umap_clusters, fig = ml_processing.calculateAndVisualizeEmbeddingsUMAP_with_DBSCAN(reviews, eps=0.5, min_samples=5, plot = False, app = True)
            st.plotly_chart(fig, use_container_width=True)

        st.write('Add topics')

else:
    st.write("Please upload a ML processed CSV file to start.")