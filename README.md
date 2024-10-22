# Sentiment Analysis and Review Classification Project

## Overview
This project aims to provide a simple, cost-effective solution for small online stores and startups to analyze customer reviews. The main goal is to extract insights from customer feedback that will help these businesses improve their products and services. By using Python tools and machine learning, this project helps to classify customer sentiment (positive, neutral, negative) and presents these insights in an easy-to-read dashboard using Power BI.

## Problem Statement
Small eCommerce businesses often lack the resources to conduct in-depth analysis of customer reviews. Understanding customer satisfaction through reviews can be crucial for product and service improvements. This project addresses this need by delivering a solution that is both accessible and scalable, allowing small businesses to efficiently analyze customer sentiment without the need for extensive technical knowledge or expensive tools.

## Technologies
- **Selenium**: Used for scraping customer reviews from online stores.
- **CSV/Google Drive**: Data storage in CSV format, either locally or in Google Drive.
- **pandas**: Used for cleaning and processing the text data.
- **scikit-learn**: Implements a simple machine learning model for sentiment analysis.
- **TBD**: Displays the results in an accessible and interactive dashboard.

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
│   ├── raw/                # Raw data collected from Scrapy (CSV)
│   └── processed/          # Cleaned and processed data
│
├── notebooks/
│   └── scraping.ipynb              # Notebook for scrap the data
│   └── data_cleaning.ipynb         # Notebook for initial data exploration and cleaning
│   └── sentiment_analysis.ipynb    # Notebook for training and testing the ML model
│
├── src/                   
│   ├── scraper.py          # Script for scrap the data with selenium
│   ├── cleaning.py         # Script for cleaning the data with pandas
│   ├── ml_processing.py    # Script with functions to do the sentiment analysis and word processing 
│   ├── ai_insights.py      # Script with functions to extract the insights using GPT
│   ├── plots.py            # Script with plot functions for sentiment analysis
│   └── sentiment.py        # Script for sentiment analysis and extraction of insights 
│
├── app/               
│   ├── app.py                 # Script with functions to extract the insights using GPT
│   ├── .py                 # Script with plot functions for sentiment analysis
│   └── .py                 # Script for sentiment analysis and extraction of insights 
│
├── requirements.txt        # Dependencies (pandas, scikit-learn, scrapy, etc.)
├── environment.yml         # Conda environment configuration
├── chromedriver            # chromedriver exec, download it from Chrome for Developers
├── openai_setup.yml        # Keys for co (dont sync it in your repo!)
├── README.md               # Project documentation
└── .gitignore              # Ignored files for the repository

```

## Phases of the project
### Data Collection

After setting up the project and understanding the folder structure, here's a step-by-step guide to using the `scraper.py` script to extract raw review data from Google Maps.

#### 1. **Run `scraper.py` to Extract Raw Data**
To start scraping data from a Google Maps page, you need to run the `scraper.py` script. The script takes two required arguments:
- **URL**: The Google Maps URL of the place you want to scrape.
- **CSV Name**: The name you want to give to the output CSV file.

```bash
python src/scraping/scraper.py "<Google_Maps_URL>" "<output_file_name>"
```

**Example**:
If you want to scrape reviews from a location such as "HD", run the following command:

```bash
python src/scraping/scraper.py "https://www.google.com/maps/place/Casa+Unai/@41.6414965,-0.8941244,15z/data=!4m8!3m7!1s0xd5914da17876fd3:0x567ce7a304ac2a65!8m2!3d41.643107!4d-0.8948281!9m1!1b1!16s%2Fg%2F11bx55_vgb?entry=ttu&g_ep=EgoyMDI0MDkxOC4xIKXMDSoASAFQAw%3D%3D" "hd"
```

This command will:
- Scrape reviews from the provided Google Maps URL.
- Store the raw review data in `collected_hd.csv` located in the `data/raw/` directory.
- Additionally, it will store the star rating summary in `resume_hd.csv` in the same folder.

#### 2. **View Output CSV Files**
Once the script finishes execution, two CSV files will be created in the `data/raw/` folder:
- `collected_reviews_<your_filename>.csv`: This file contains all the detailed reviews.
- `resume_<your_filename>.csv`: This file contains a summary of star ratings and the number of reviews for each rating level.

These CSV files can now be used for further analysis, processing, or visualization.

#### 3. **Adjust Chromedriver Path**
Ensure that the `chromedriver` executable is correctly referenced in the `scraper.py` script. If `chromedriver` is located elsewhere on your system, modify the following line in the `scraper.py` file to reflect the correct path:

```python
chromedriver_path = '../../chromedriver'  # Adjust this path based on your system
```

#### 4. **Customizing the Scraper**
The `scraper.py` file is modular and can be customized to fit different scraping needs. You can extend its functionality or adjust specific parameters like scroll behavior, pause times, or extraction details by modifying the appropriate sections of the code.

This completes the basic steps to scrape raw review data from Google Maps using the `scraper.py` script.

### Data Cleaning
Select the raw file using data_cleaning notebook and generate the processed file. Check the search_keyworkds variable to add the common expressions required to extract the desire info.

### Data Analysis
Once we have the cleaned data, we will extract the necessary information using ML, which we will use as input for the Streamlit.

```bash
python src/scraping/sentiment.py --name "<place_name>" --plot "<boolean_to_see_or_not_the_plots>"
```

The script will extract to the processed folder the insights extracted in json format, samples of some reviews in the periods of interest analysed, and the reviews with the necessary information already processed.

#### Techniques Used
- **Embeddings**: The notebook converts review texts into vector embeddings to capture semantic relationships between reviews, allowing for more nuanced analysis.
- **UMAP & PCA**: These dimensionality reduction techniques are applied to the embeddings to simplify their structure and facilitate visualization. UMAP focuses on preserving local relationships, while PCA captures the main variance in the data. After reducing dimensions, the notebook attempts clustering to group reviews with similar patterns, although results were inconclusive.
- **Low-Score Period Analysis**: The notebook specifically extracts reviews from periods with lower overall scores to analyze negative trends, helping to identify areas for improvement in customer experience.
- **Topic Extraction with LDA**: Latent Dirichlet Allocation (LDA) is used to uncover the main topics in the reviews, providing a clearer picture of what customers frequently discuss, both positively and negatively.
- **Community Analysis with TF-IDF**: Using TfidfVectorizer, the notebook groups reviews into communities based on similar keywords and phrases, enhancing the understanding of common customer concerns.

### Data Visualization
TBD