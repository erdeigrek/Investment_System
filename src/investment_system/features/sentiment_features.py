from transformers import pipeline
import nltk.data
from nltk.tokenize import PunktTokenizer
import feedparser
from bs4 import BeautifulSoup
import pandas as pd

nltk.download("punkt_tab")
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)


def split_sentences(text: str, language: str = "english") -> list[str]:

    tokenizer = PunktTokenizer(language)
    return tokenizer.tokenize(text)


def article_sentiment(sentences_results: list) -> float:
    total = 0.0
    for sentence in sentences_results:
        if sentence["label"] == "negative":
            total -= sentence["score"]
        elif sentence["label"] == "positive":
            total += sentence["score"]
        else:
            pass
    return total / len(sentences_results)


def analyze_article(text: str, language: str = "english") -> float:
    sentences = split_sentences(text, language)
    sentiment = sentiment_pipeline(sentences)
    return article_sentiment(sentiment)


def filter_articles(articles: list, company_names: list) -> list:
    filtered = [
        x
        for x in articles
        if any(form.lower() in x[0].lower() for form in company_names)
    ]
    return filtered


def get_sentiment_series(ticker: str) -> pd.Series:
    url = yahoo_rss_url(ticker)
    articles = download_article(url)
    scores = []
    dates = []
    for article in articles:
        score = analyze_article(article[1])
        scores.append(score)
        dates.append(article[2])
    data = {"date": dates, "sentiment_score": scores}
    df = pd.DataFrame(data)
    df["date"] = df["date"].dt.date
    df = df.groupby("date")["sentiment_score"].agg("mean")
    return df


def merge_sentiment_and_price(df: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return df.merge(df2, on="date")


from investment_system.ingestion.market_data import fetch_stooq_symbol
from datetime import date

df = fetch_stooq_symbol("AAPL", "us", date(2026, 5, 11), date(2026, 5, 12))
df2 = get_sentiment_series("AAPL")
print(merge_sentiment_and_price(df, df2))
