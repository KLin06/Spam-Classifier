# 📧 Spam Email Classifier

This project uses Natural Language Processing (NLP) and machine learning to classify emails as **spam** or **ham** (not spam). It loads a dataset, processes the text, extracts features, trains multiple models, and evaluates their performance.

## 🔍 Features

- Cleans and preprocesses text (lowercasing, removing punctuation, stopwords)
- Extracts features:
  - TF-IDF vectorization
  - Email length
  - Uppercase word ratio
  - Frequency of common spam words
- Compares multiple models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Random Forest
  - Linear SVC
- Evaluates using accuracy, confusion matrix, and classification report
- Saves the final model and vectorizer using `joblib`

## 🧠 Dataset

Located at: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Original columns:
- `v1`: label (`ham` or `spam`)
- `v2`: email text

After processing:
- `class`: label
- `text`: original email
- `process_text`: cleaned email text
- Plus several engineered features

## 🛠️ Usage

1. Make sure you have Python 3.x installed
2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib
