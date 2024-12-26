import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data['Text'], data['Emotion']

def preprocess_text(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer