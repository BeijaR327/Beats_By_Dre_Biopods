import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER
def _ensure_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

st.set_page_config(page_title="BioPods Sentiment", layout="wide")
st.title("Beats BioPods — Sentiment Explorer")

uploaded = st.file_uploader("Upload feedback CSV (schema: text, channel, created_at, rating, label optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No file uploaded. Falling back to sample data in repo's data/feedback_sample.csv")
    try:
        df = pd.read_csv("data/feedback_sample.csv")
    except Exception:
        df = pd.DataFrame({"text": []})

if df.empty:
    st.warning("No data available.")
    st.stop()

st.subheader("Raw Sample")
st.dataframe(df.head(20))

_ensure_vader()
sia = SentimentIntensityAnalyzer()
df['compound'] = df['text'].fillna("").astype(str).apply(lambda s: sia.polarity_scores(s)['compound'])

st.subheader("Sentiment Distribution")
fig = plt.figure()
df['compound'].hist(bins=20)
st.pyplot(fig)

st.subheader("Examples")
pos = df.sort_values('compound', ascending=False).head(5)[['text','compound']]
neg = df.sort_values('compound', ascending=True).head(5)[['text','compound']]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Most Positive**")
    st.table(pos)
with col2:
    st.markdown("**Most Negative**")
    st.table(neg)
