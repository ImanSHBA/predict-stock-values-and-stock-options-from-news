import pandas as pd
import streamlit as st

df_tweets = pd.read_csv('Streamlit_prototype/Data/stock_tweets.csv')
st.title("Sentiment analysis for stock values and stock options from news")

st.header("Raw data")
st.write("Streamlit app to display Tweets about stocks")

st.dataframe(df_tweets)
st.write(f"Number of rows: {df_tweets.shape[0]}")
