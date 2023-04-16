import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

def process_text_from_url(url):
    # Scrape text data from a website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text_data = soup.get_text()

    # Create a DataFrame with the scraped text data
    data = pd.DataFrame({'text': [text_data]})

    # Remove URLs
    data['text'] = data['text'].apply(lambda x: re.sub(r'http\S+', '', x))

    # Remove special characters
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]+', ' ', x))

    # Convert text to lowercase
    data['text'] = data['text'].apply(lambda x: x.lower())

    # Tokenize the text data
    data['tokenized_text'] = data['text'].apply(lambda x: word_tokenize(x))

    # Remove stopwords
    stop_words = set(stopwords.words('swahili'))

    data['filtered_tokens'] = data['tokenized_text'].apply(lambda x: [token for token in x if token not in stop_words])

    stemmer = SnowballStemmer('english')

    data['stemmed_tokens'] = data['filtered_tokens'].apply(lambda x: [stemmer.stem(token) for token in x])

    # Display a sample of the stemmed data
    print(data['stemmed_tokens'].sample(min(10, len(data)), replace=True))

    lemmatizer = WordNetLemmatizer()

    data['lemmatized_tokens'] = data['filtered_tokens'].apply(lambda x: [lemmatizer.lemmatize(token) for token in x])

    # Display a sample of the lemmatized data
    print(data['lemmatized_tokens'].sample(min(10, len(data)), replace=True))

    all_words = ' '.join([word for tokens in data['lemmatized_tokens'] for word in tokens])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, stopwords=STOPWORDS).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    data.to_csv('Cleaned_Dataset_1.csv', index=False)
