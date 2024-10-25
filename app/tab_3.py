import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import networkx as nx

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.abspath(os.path.join('..')))
from src import plots
from src import ml_processing


def plotTrend(reviews, label_mapping, app=False, filter_min=None, filter_max=None):
    # Convert date column to datetime format and create additional time columns
    reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')
    reviews['month'] = reviews['date'].dt.to_period('M')

    # Filter data for the last periods based on filter_min and filter_max
    limit_date = reviews['date'].max()
    if filter_min is None and filter_max is None:
        # If both filters are None, select data from the last year
        start_date = limit_date - pd.DateOffset(years=1)
        selected_reviews = reviews[(reviews['date'] >= start_date) & (reviews['date'] <= limit_date)]
    else:
        # Apply the filters if provided
        selected_reviews = reviews
        if filter_min is not None:
            selected_reviews = selected_reviews[selected_reviews['date'] >= filter_min]
        if filter_max is not None:
            selected_reviews = selected_reviews[selected_reviews['date'] <= filter_max]

    # Compute averages for the required periods using label_mapping keys
    columns_to_average = list(label_mapping.keys())
    monthly_avg_scores = selected_reviews.groupby('month')[columns_to_average].mean()
    
    # Create a figure to plot the trends
    fig = make_subplots(rows=1, cols=1)
    
    # Update the axis labels for each score to be more readable
    colors = ['#32CD32', 'rgba(31, 119, 180, 0.8)', 'rgba(107, 174, 214, 0.8)', 'rgba(158, 202, 225, 0.8)'] 
    for i, column in enumerate(monthly_avg_scores.columns):
        label = label_mapping[column]
        fig.add_trace(
            go.Scatter(x=monthly_avg_scores.index.astype(str), y=monthly_avg_scores[column],
                       mode='lines+markers', name=label, 
                       text=[f"{label} - {val:.2f}" for val in monthly_avg_scores[column]], 
                       hoverinfo="text", line=dict(color=colors[i], width=3 if i == 0 else 2)),
            row=1, col=1)


    # Analyze low scores and find high score
    _, low_score_periods = ml_processing.analyzeLowScores(reviews, 'rating_score', num_periods=3)
    high_score_period = monthly_avg_scores['rating_score'].idxmax()
    high_score_value = monthly_avg_scores['rating_score'].max()
    
    # Add annotations for low scores
    for i in range(len(low_score_periods)):
        if i > 0 and low_score_periods[i] - low_score_periods[i - 1] == 1:
            # If two periods are contiguous, combine them in one annotation
            fig.add_annotation(x=str(low_score_periods[i]), y=monthly_avg_scores.loc[low_score_periods[i], 'rating_score'] + 0.5,
                               text=f"Drop in {low_score_periods[i - 1].strftime('%B')} & {low_score_periods[i].strftime('%B')}",
                               showarrow=True, arrowhead=2, ax=0, ay=-40, row=1, col=1)
        elif i == 0 or low_score_periods[i] - low_score_periods[i - 1] != 1:
            fig.add_annotation(x=str(low_score_periods[i]), y=monthly_avg_scores.loc[low_score_periods[i], 'rating_score'] + 0.5,
                               text=f"Drop in {low_score_periods[i].strftime('%B')}",
                               showarrow=True, arrowhead=2, ax=0, ay=-40, row=1, col=1)
    
    # Add annotation for high score
    fig.add_annotation(x=str(high_score_period), y=high_score_value - 0.3,
                       text=f"High in {high_score_period.strftime('%B')}",
                       showarrow=True, arrowhead=2, ax=0, ay=40, row=1, col=1)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, title_text='Average Score', title_standoff=0)
    fig.update_layout(
                        showlegend=True, 
                        #title="Rating Trends",
                        #title_font=dict(size=20),
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="white",
                        height=350, width=1400,
                    )
    
    # Show or return the figure depending on the context
    if app:
        return fig
    else:
        fig.show()