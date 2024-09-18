# Sentiment Analysis and Review Classification Project

## Overview
This project aims to provide a simple, cost-effective solution for small online stores and startups to analyze customer reviews. The main goal is to extract insights from customer feedback that will help these businesses improve their products and services. By using Python tools and machine learning, this project helps to classify customer sentiment (positive, neutral, negative) and presents these insights in an easy-to-read dashboard using Power BI.

## Problem Statement
Small eCommerce businesses often lack the resources to conduct in-depth analysis of customer reviews. Understanding customer satisfaction through reviews can be crucial for product and service improvements. This project addresses this need by delivering a solution that is both accessible and scalable, allowing small businesses to efficiently analyze customer sentiment without the need for extensive technical knowledge or expensive tools.

## Technologies
- **Python Scrapy**: Used for scraping customer reviews from online stores.
- **CSV/Google Drive**: Data storage in CSV format, either locally or in Google Drive.
- **pandas**: Used for cleaning and processing the text data.
- **scikit-learn**: Implements a simple machine learning model for sentiment analysis.
- **Power BI**: Displays the results in an accessible and interactive dashboard.

## Project Phases
1. **Data Collection (Scraping)**: Reviews will be collected from online stores using Python's Scrapy framework and stored in CSV format.
2. **Data Storage**: The raw data will be saved in CSV files locally or in Google Drive.
3. **Data Cleaning and Processing**: The data will be processed using pandas to clean and prepare it for machine learning analysis.
4. **Sentiment Analysis (ML Model)**: A machine learning model, implemented in scikit-learn, will classify reviews as positive, neutral, or negative.
5. **Visualization**: The results of the analysis will be displayed in a Power BI dashboard for easy interpretation and insights.

## Folder Structure

```bash
sentiment-analysis-reviews/
│
├── data/                  
│   ├── raw/               # Raw data collected from Scrapy (CSV)
│   └── processed/         # Cleaned and processed data
│
├── notebooks/
│   └── scraping.ipynb       # Notebook for scrap the data
│   └── data_cleaning.ipynb  # Notebook for initial data exploration and cleaning
│   └── model_training.ipynb # Notebook for training and testing the ML model
│
├── src/                   
│   ├── scraping/          # Scrapy scripts for data ingestion
│   └── ml/                # ML scripts for data processing and model training
│       ├── preprocess.py  # Script for cleaning the data with pandas
│       └── train_model.py # Script for training the sentiment analysis model
│
├── reports/               
│   └── PowerBI/           # Exported Power BI reports and dashboards
│
├── requirements.txt       # Dependencies (pandas, scikit-learn, scrapy, etc.)
├── README.md              # Project documentation
└── .gitignore             # Ignored files for the repository
