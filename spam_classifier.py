# import libraries
import re
import joblib
from nltk.corpus import stopwords
import numpy as np

# Load model + vectorizer
model = joblib.load("spam_classifier_nb.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

def clean_email(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = [word for word in text.split() if word not in stopwords.words("english")]
    return " ".join(tokens)

def classify_email(text):
    clean_text = clean_email(text)
    X_text = vectorizer.transform([clean_text]).toarray()
    top_spam_words = ['call', 'free', 'u', 'txt', 'ur', 'mobile', 'stop', 'text', 'claim', 'reply', 'prize', 'get', 'p', 'new', 'urgent', 'send', 'nokia', 'cash', 'contact', 'please']
    
     # Manual features
    text_length = len(text)
    uppercase_ratio = sum(1 for word in text.split() if word.isupper() and len(word) > 1) / len(text) if len(text) > 0 else 0
    spam_words_freq = sum(1 for word in clean_text.split() if word in top_spam_words)

    X_manual = np.array([[text_length, uppercase_ratio, spam_words_freq]])

    # Combine
    X = np.hstack((X_text, X_manual))

    # Predict
    prediction = model.predict(X)
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == "__main__":
    sample = input("Enter an email to classify:\n")
    print(classify_email(sample))