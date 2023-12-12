"""The purpose of this program is to perform sentiment analysis on a dataset of product reviews"""

import spacy
import pandas as pd
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """This function preprocesses the text to make it ready for sentiment analysis. It takes in standard text and
    returns text that has removed stop words and been lemmatised."""
    doc = nlp(text)

    # Lemmatise and remove stop words
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop]

    # Join the tokens back into a string
    processed_text = ' '.join(processed_tokens)

    return processed_text


def predict_sentiment(text):
    """This function performs sentiment analysis on text. It takes in text and returns a positive, negative or neutral
    label."""
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Perform sentiment analysis using TextBlob
    blob = TextBlob(preprocessed_text)
    sentiment_score = blob.sentiment.polarity

    # Label sentiment based on the score
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment


# Read in the product reviews file and select the column of interest
while True:
    try:
        dataframe = pd.read_csv("amazon_product_reviews.csv")
        reviews_data = dataframe['reviews.text']
        break
    except Exception:
        print("Your file can't be found. Please try again.")

# Test using a sample of Product Reviews:
for i in range(5):
    user_review = reviews_data[i]
    predicted_sentiment = predict_sentiment(user_review)
    print(user_review)
    print(f"Predicted Sentiment: {predicted_sentiment}")
