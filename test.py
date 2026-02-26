import joblib
import re

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(tweet):
    cleaned = clean_text(tweet)
    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]

while True:
    text = input("Enter tweet (or exit): ")
    if text.lower() == "exit":
        break
    print("Sentiment:", predict_sentiment(text))
