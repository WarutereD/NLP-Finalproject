import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from textblob import TextBlob
import gradio as gr

# Load the dataset
df = pd.read_csv("/workspaces/NLP-Finalproject/swahili.csv")
df.head

# Text preprocessing
stop_words = stopwords.words("swahili")
df["maneno"] = df["maneno"].apply(lambda x: " ".join([word for word in re.sub('[^a-zA-Z0-9\s]', '', x).split() if word not in stop_words]))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["maneno"], df["lugha"], test_size=0.3, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train an SVM classifier
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_vec, y_train)

# Make predictions and print results
y_pred = svm.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print(f"Results for swahili.csv")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Polarity: {TextBlob(' '.join(df['maneno'])).sentiment.polarity:.2f}")

def predict_sentiment(text):
    # Preprocess the input text
    text = " ".join([word for word in re.sub('[^a-zA-Z0-9\s]', '', text).split() if word not in stop_words])
    # Vectorize the input text
    text_vec = vectorizer.transform([text])
    # Predict the sentiment of the input text
    sentiment = svm.predict(text_vec)[0]
    return sentiment

def predict_sentimentcsv(data):
    # Preprocess the new data
    data["maneno"] = data["maneno"].apply(lambda x: " ".join([word for word in re.sub('[^a-zA-Z0-9\s]', '', x).split() if word not in stop_words]))

    # Vectorize the new data
    new_data_vec = vectorizer.transform(data["maneno"])

    # Predict the sentiment of the new data
    sentiment = svm.predict(new_data_vec)

    # Calculate the polarity scores of the new data
    new_data_polarity = [TextBlob(text).sentiment.polarity for text in data["maneno"]]

    # Calculate the precision, recall, F1 score, and support for the new data
    precision_new, recall_new, f1_score_new, support_new = precision_recall_fscore_support(sentiment, data["lugha"], average='weighted')

    # Calculate the accuracy for the new data
    accuracy_new = accuracy_score(data["lugha"], sentiment)*100

    return sentiment, new_data_polarity, precision_new, recall_new, f1_score_new, support_new, accuracy_new
