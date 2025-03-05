
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import spacy
import re
from plotly.subplots import make_subplots
import math
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the English model
nlp = spacy.load("en_core_web_sm")

def classify_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']

    # Classify sentiment based on compound score
    if compound_score >= 0.05:
        return 1,compound_score  # Positive
    elif compound_score <= -0.05:
        return -1,compound_score  # Negative
    else:
        return 0,compound_score  # Neutral

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove @mentions (usernames)
    text = re.sub(r'@\w+', '', text)
    # Remove any non-alphabetical characters (optional step)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Process the text with spaCy
    doc = nlp(text.lower())  # Process the text
    # Remove stopwords, punctuation, and get clean tokens
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)  # Return the cleaned text as a single string

def plot_analysis(company, stock_data, tweet_data,n=30):
    """
    Returns traces for a 30-day moving average of stock prices and sentiment score trends,
    using dual Y-axes.
    
    :param company: Stock name (string)
    :param stock_data: DataFrame containing stock price data
    :param tweet_data: DataFrame containing sentiment scores from tweets
    :return: List of plotly traces
    """


    # Filter stock data for the given company
    comp_stock = stock_data[stock_data['Stock Name'] == company].copy()
    comp_stock['Date'] = pd.to_datetime(comp_stock['Date'])
    
    # Calculate 30-day moving average for stock prices
    comp_stock['Moving_Avg'] = comp_stock['Adj Close'].rolling(window=n).mean()
    comp_stock = comp_stock.dropna(subset=["Moving_Avg"])

    # Filter tweet sentiment data for the given company
    comp_tweets = tweet_data[tweet_data['Stock Name'] == company].copy()
    comp_tweets['Date'] = pd.to_datetime(comp_tweets['Date'])

    # Group by Stock Name and Date while calculating the average sentiment score
    comp_tweets = comp_tweets.groupby(["Stock Name", comp_tweets["Date"].dt.date]).agg({"Score": "mean"})

    #  Reset index to make "Date" a column again
    comp_tweets = comp_tweets.reset_index()

    #  Ensure "Date" is recognized as a datetime column
    comp_tweets['Date'] = pd.to_datetime(comp_tweets['Date'])

    comp_tweets['Moving_Avg'] = comp_tweets['Score'].rolling(window=n).mean()
 
    # Line plot for 30-day moving average (Primary Y-axis: Stock Price)
    trace_avg = go.Scatter(
        x=comp_stock["Date"],
        y=comp_stock["Moving_Avg"],
        mode="lines",
        line=dict(color="red"),
        name=f"{company} close price {n}-day Moving Avg",
        yaxis="y1"  # Assign to primary Y-axis
    )

    # Line plot for tweet sentiment score trends (Secondary Y-axis: Sentiment Score)
    trace_sentiment = go.Scatter(
        x=comp_tweets["Date"],
        y=comp_tweets["Moving_Avg"],
        mode="lines",
        line=dict(color="green", dash="dash"),
        name=f"{company} {n}-day Moving Avg Sentiment Score",
        yaxis="y2"  # Assign to secondary Y-axis
    )

    return trace_avg, trace_sentiment

def plot_company_tweets(company, tweet_counts):
    """Returns a bar plot for tweet counts of a given company, categorized by sentiment."""
    
    # Filter data for the selected company
    company_data = tweet_counts[tweet_counts['Stock Name'] == company]

    # Define colors for each sentiment
    sentiment_colors = {1: "green", 0: "gray", -1: "red"}

    # Create a list to store traces for each sentiment
    traces = []

    # Loop through each sentiment and create a bar trace
    for sentiment, color in sentiment_colors.items():
        sentiment_data = company_data[company_data["Sentiment"] == sentiment]
        traces.append(go.Bar(
            x=sentiment_data["Date"],
            y=sentiment_data["Tweet Count"],
            marker_color=color,
            name=f"{company} - Sentiment {sentiment}"
        ))

    return traces  # Return a list of traces for subplot usage

def plot_company_stock(company, stk):
    """Returns a scatter plot of stock prices with a moving average for a given company."""
    comp = stk[stk['Stock Name'] == company].copy()
    comp['Date'] = pd.to_datetime(comp['Date'])
    comp['Moving_Avg'] = comp['Adj Close'].rolling(window=7).mean()

    trace_stock = go.Scatter(
        x=comp["Date"],
        y=comp["Adj Close"],
        mode="markers",
        marker=dict(color="skyblue"),
        name=f"{company} Adjusted Close"
    )

    trace_avg = go.Scatter(
        x=comp["Date"],
        y=comp["Moving_Avg"],
        mode="lines",
        line=dict(color="red"),
        name=f"{company} 7-day Moving Avg"
    )
    
    return trace_stock, trace_avg

def create_subplots(companies, tweet_counts=None, stock_data=None, plot_type="both"):
    """
    Creates a subplot layout dynamically for tweet counts, stock prices, or both.
    
    :param companies: List of company names to visualize
    :param tweet_counts: DataFrame containing tweet counts (required if plot_type="tweets" or "both")
    :param stock_data: DataFrame containing stock price data (required if plot_type="stocks" or "both")
    :param plot_type: "tweets" for tweet plots, "stocks" for stock plots, "both" for both
    """
    
    num_charts = len(companies)
    rows = math.ceil(num_charts / int(num_charts**0.5))  # Two charts per row
    cols = int(num_charts**0.5)  # Two columns (tweets + stock) per company
   

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{comp} Tweets" for comp in companies] if plot_type == "tweets" else 
                       [f"{comp} Stock" for comp in companies] if plot_type == "stocks" else 
                       [f"{comp} Tweets" for comp in companies] + [f"{comp} Stock" for comp in companies],
        shared_xaxes=True
    )

    for idx, company in enumerate(companies):
        row = (idx // (cols // (2 if plot_type == "both" else 1))) + 1
        col = (idx % (cols // (2 if plot_type == "both" else 1))) + 1

        if plot_type in ["both", "tweets"] and tweet_counts is not None:
            tweet_traces = plot_company_tweets(company, tweet_counts)
            for trace in tweet_traces:  #  Fix: Add each trace separately
                fig.add_trace(trace, row=row, col=col)
        
        if plot_type in ["both", "stocks"] and stock_data is not None:
            stock_trace, avg_trace = plot_company_stock(company, stock_data)
            col = col + 1 if plot_type == "both" else col  # Shift to next column if "both"
            fig.add_trace(stock_trace, row=row, col=col)
            fig.add_trace(avg_trace, row=row, col=col)

    fig.update_layout(
        title=f"{plot_type.capitalize()} Analysis for Selected Companies",
        height=rows * 300,  # Adjusting height dynamically
        showlegend=False
    )

    return fig

def display_finance(df,date_range):
    """Displays financial stock data with interactive charts."""
    st.header("Financial Statistics")

    # Dropdown to select a company dynamically
    selected_stock = df["Stock Name"].unique()

    # Show plot for selected stock
    st.plotly_chart(create_subplots(selected_stock, tweet_counts=None, stock_data=df, plot_type="stocks"), use_container_width=True)


    st.write(f"Stock market data for used tickers from Yahoo Finance from {date_range[0]} till {date_range[1]}")
    st.dataframe(df)

def display_tweets(df,date_range):
    """Displays tweet statistics with interactive visualizations."""
    st.header("Tweet Statistics")

    # Process tweet counts
    df["Date"] = pd.to_datetime(df["Date"])
    tweet_counts = df.groupby(["Stock Name", df["Date"].dt.date,"Sentiment"]).size().reset_index(name="Tweet Count")

    # Dropdown to select a company dynamically
    selected_stock = tweet_counts["Stock Name"].unique()

    # Show plot for selected stock

    st.plotly_chart(create_subplots(selected_stock, tweet_counts=tweet_counts, stock_data=None, plot_type="tweets"), use_container_width=True)


    st.write(f"Tweets for corresponding stock tickers from {date_range[0]} till {date_range[1]}")
    st.dataframe(df)

def display_sentiment(df):
    st.title("Sentiment Analysis")
    st.write("Sentiment analysis of tweets")
    st.subheader("Enter your text below:")

    # Initialize session state for input text
    if "user_text" not in st.session_state:
        st.session_state.user_text = ""

    # Text area for input
    user_input = st.text_area("Type here...", value=st.session_state.user_text, height=150)

    # Layout for buttons (side by side)
    col1, col2 = st.columns([1, 1])
    empty_warning = "firs_place"
    with col1:
        if st.button("Analyze Sentiment"):
            if user_input.strip():
                sentiment_label, sentiment = classify_sentiment(preprocess_text(user_input))

                # Display result
                empty_warning = False
                st.write(f"**Sentiment:** {sentiment_label}")
                st.write(f"**Confidence Scores:** {sentiment}")
            else:
                empty_warning = True
    with col2:
        if st.button("Random"):
            if not df.empty:
                # Pick a random tweet from the DataFrame
                random_tweet = random.choice(df["Tweet"].dropna().tolist())
                st.session_state.user_text = random_tweet  # Store it in session state
                st.rerun()  # Rerun the app to update the text area
    
    if empty_warning == True:
        st.warning("Please enter some text before clicking the button!")
    elif empty_warning=='':
        st.success("Random tweet generated successfully!")
    elif empty_warning==False:
        st.success("Sentiment Analysis Complete!")

def display_analysis(df_tweet,df_finance):
    st.title("Related Analysis")
    st.write("Analysis of the relationship between tweet sentiment and stock prices")


    # Select a company from available stock names
    companies = df_finance["Stock Name"].unique()
    company = st.selectbox("Select a company:", companies)

    if company:
        st.write(f"Showing analysis for **{company}**")

        # Generate the plot
        fig = go.Figure()
        avg_number = st.number_input("Moving Average day",1, 365, 30)
        traces = plot_analysis(company, stock_data=df_finance, tweet_data=df_tweet,n=avg_number)

        for trace in traces:
            fig.add_trace(trace)

        fig.update_layout(
    title=f"{company} Stock Price, Moving Avg, and Sentiment Trend",
    xaxis=dict(title="Date"),
    yaxis=dict(
        title="Stock Price (30-day Avg)", 
        titlefont=dict(color="red"), 
        tickfont=dict(color="red"),
    ),
    yaxis2=dict(
        title="Sentiment Score", 
        titlefont=dict(color="green"), 
        tickfont=dict(color="green"),
        overlaying="y",  #  Allows it to overlay on the same graph
        side="right",  #  Puts it on the right side
    ),
    legend=dict(x=0, y=1.1),
)


        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
   
def get_sidebar_and_data():

    df_finance = pd.read_csv('Data/stock_yfinance_data.csv')
    df_finance['Date'] = pd.to_datetime(df_finance['Date'])
    df_tweets = pd.read_csv('Data/stock_tweets.csv')
    df_tweets['Date'] = pd.to_datetime(df_tweets['Date']).dt.tz_convert(None)
    
    st.sidebar.title("Time Slice Selector")

    # Date range picker
    date_range = st.sidebar.date_input(
    "Select Date Range:",
    [pd.Timestamp(2021, 9, 30), pd.Timestamp(2022, 9, 30)])
    start_date = date_range[0]
    end_date = date_range[1]

    if st.sidebar.button("Reset Date Range"):
        date_range = [pd.Timestamp(2021, 9, 30), pd.Timestamp(2022, 9, 30)]

    companies_list = list(df_tweets.sort_values(by='Stock Name')['Stock Name'].unique())
    selected = st.sidebar.multiselect("Select companies",
    companies_list)

    df_finance = df_finance[df_finance["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    df_tweets = df_tweets[df_tweets["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    
    if selected:
        df_finance = df_finance[df_finance['Stock Name'].isin(selected)]
        df_tweets = df_tweets[df_tweets['Stock Name'].isin(selected)]

    return df_tweets, df_finance,date_range

def main():
    st.title("Sentiment Analysis of Stock Tweets")

    df_tweets, df_finance,date_range = get_sidebar_and_data()
    tab_tweets, tab_finance,tab_sentiment,tab_analysis = st.tabs(["Tweets", "Finance","Sentiment Analysis","related Analysis"])

    with tab_tweets:
        display_tweets(df_tweets,date_range)
    with tab_finance:
        display_finance(df_finance,date_range)
    with tab_sentiment:
        display_sentiment(df_tweets)
    with tab_analysis:
        display_analysis(df_tweets,df_finance)

if __name__ == "__main__":
    main()
