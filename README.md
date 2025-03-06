# predict-stock-values-and-stock-options-from-news
# Sentiment Analysis of Stock Tweets

## Overview
This project analyzes the relationship between **stock market trends** and **public sentiment** by integrating **financial data from Yahoo Finance** with **sentiment analysis of stock-related tweets**. The application is built using **Streamlit**, **Pandas**, and **Plotly** for data visualization.

## Features
- **Sentiment Analysis of Tweets**: Uses VADER sentiment analysis to classify stock-related tweets as **positive, neutral, or negative**.
- **Stock Price Analysis**: Displays stock price movements along with **30-day moving averages**.
- **Dual-Axis Visualization**: Overlays **sentiment scores** and **stock prices** on the same graph.
- **Interactive Dashboard**: Users can select **date ranges and specific stocks** to analyze trends dynamically.

## Installation
### Prerequisites
Ensure you have **Python 3.7+** installed and install the required dependencies:
```sh
pip install pandas streamlit plotly spacy vaderSentiment
```
Additionally, download the English language model for **spaCy**:
```sh
python -m spacy download en_core_web_sm
```

## Usage
Run the Streamlit application using the following command:
```sh
streamlit run app.py
```
The interactive dashboard will open in your web browser, allowing you to analyze sentiment trends and stock movements.

## Project Structure
```
.
├── Data/
│   ├── stock_yfinance_data.csv   # Stock market data
│   ├── stock_tweets.csv          # Tweet dataset
├── app.py                         # Main Streamlit app
├── README.md                      # Project documentation
└── requirements.txt                # List of dependencies
```

## Key Components
### **Sentiment Analysis**
- Uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)** to classify tweets into:
  - **1** → Positive
  - **0** → Neutral
  - **-1** → Negative
- Implements `classify_sentiment()` to return both sentiment class and score.

### **Data Preprocessing**
- The `preprocess_text()` function removes **URLs, mentions, special characters, and stopwords** using **spaCy**.
- Converts text into a cleaned format before sentiment classification.

### **Stock Price & Sentiment Visualization**
- **`plot_analysis()`** overlays:
  - **Stock price moving averages** (red line) on **left Y-axis**
  - **Sentiment score moving averages** (green dashed line) on **right Y-axis**
- **`plot_company_tweets()`** creates a bar chart of tweet counts categorized by sentiment.
- **`plot_company_stock()`** generates stock price plots with a 7-day moving average.

### **Interactive Dashboard**
- **`display_tweets()`** → Shows tweet sentiment trends.
- **`display_finance()`** → Displays stock market trends.
- **`display_sentiment()`** → Displays stock text or tweet sentiment score.
- **`display_analysis()`** → Combines sentiment and stock trends in one view.

## How It Works
1. **User selects a stock & date range** in the Streamlit sidebar.
2. **Application fetches & preprocesses data**:
   - Reads financial data from `stock_yfinance_data.csv`
   - Reads tweets from `stock_tweets.csv`
   - Computes **n-day moving averages** for **both stock prices & sentiment scores**
3. **Plotly visualizations** are updated dynamically.
4. **Users can analyze correlations** between stock price movements & social sentiment.

## Example Usage
- Select **Apple (AAPL)** and a **custom date range**.
- Observe if **positive sentiment spikes** align with **price increases**.
- Compare results across multiple stocks.

## Contributors
- **Iman Sherkat Bazazan** (imanshb1379@gmail.com) (iman.sherkatbazazan@alumni.esade.edu)

