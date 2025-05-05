#%%
# Set variables
DISPLAY = False
DEVELOPMENT = False

#%%
# import the libraries and data
from collections import Counter
import re

from IPython.display import display
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from scipy.stats import uniform

test_file = "home-data/spam.csv"
mail_data = pd.read_csv(test_file, encoding = "latin1")

#%% 
# display the data 
if DISPLAY:
    display(mail_data)
    display(mail_data.describe())

# %%
# process the columns 
mail_data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis = 1, inplace= True)
mail_data.rename(columns={"v1" : "class", "v2" : "text"}, inplace=True)
mail_data["type"] = mail_data["class"].map(lambda x: 1 if x == "spam" else 0)
# %%
# cleaning the data
mail_data.duplicated().sum()
mail_data.drop_duplicates(keep="first", inplace=True)

if DISPLAY:
    display(mail_data)
 
# %%
# dataset visualization

# length of each email type
mail_data["text_length"] = mail_data["text"].apply(len)
spam_email_length = mail_data[mail_data["class"] == "spam"]["text_length"].mean()
ham_email_length = mail_data[mail_data["class"] == "ham"]["text_length"].mean()
    
if DISPLAY:
    # amount of each type
    sns.countplot(data=mail_data, x="class")
    plt.title("Email Number by Type")
    plt.show()

    # length of each email type
    sns.barplot(x = ["ham length", "spam length"], y = [ham_email_length, spam_email_length])
    plt.title("Email Lengths")
    plt.ylabel("Text Length (Characters)")
    plt.show()

# %%
# Adding Features
'''
# searches for links
mail_data["contains_link"] = mail_data["text"].apply(lambda x: int(bool(re.search(r'(http[s]?://|www\.|\.[a-z]{2,3}/)', x))))
display(mail_data)
'''

# Ratio of uppercase to lowercase words
mail_data["uppercase_ratio"] = mail_data["text"].apply(lambda x: sum(1 for word in x.split() if word.isupper() and len(word) > 1)/len(x) if not len(x) == 0 else 0)

# Number of exclaimation words

# processing the data
mail_data["process_text"] = mail_data["text"].apply(str.lower)
mail_data["process_text"] = mail_data["process_text"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# removing the stopwords
stop_words = stopwords.words('english')
mail_data["process_text"] = mail_data["process_text"].apply(lambda x: [word for word in x.split() if word not in stop_words])

# recombining the words
mail_data["process_text"] = mail_data["process_text"].apply(lambda x: " ".join(x))

# length of each processed email type
mail_data["processed_text_length"] = mail_data["process_text"].apply(len)
spam_email_length = mail_data[mail_data["class"] == "spam"]["processed_text_length"].mean()
ham_email_length = mail_data[mail_data["class"] == "ham"]["processed_text_length"].mean()

# display the most common words in each type of chart
spam_mail = mail_data[mail_data["class"] == "spam"]["process_text"]
spam_words = " ".join(spam_mail).split()
spam_word_count = Counter(spam_words)
top_ten_spam_words = spam_word_count.most_common(10)

ham_mail = mail_data[mail_data["class"] == "ham"]["process_text"]
ham_words = " ".join(ham_mail).split()
ham_word_count = Counter(ham_words)
top_ten_ham_words = ham_word_count.most_common(10)
# %%
# display the data again after processing

if DISPLAY:
    sns.barplot(x = ["ham length", "spam length"], y = [ham_email_length, spam_email_length])
    plt.title("Email Lengths")
    plt.ylabel("Text Length (Characters)")
    plt.show()

    sns.barplot(x = [x for (x,y) in top_ten_spam_words], y = [y for (x,y) in top_ten_spam_words])
    plt.title("Most common words in Spam emails")
    plt.show()

    sns.barplot(x = [x for (x,y) in top_ten_ham_words], y = [y for (x,y) in top_ten_ham_words])
    plt.title("Most common words in Ham emails")
    plt.show()

#%%
# add the most common spam words as a feature
top_spam_words = [x for (x,y) in spam_word_count.most_common(20)]
top_ham_words = [x for (x,y) in ham_word_count.most_common(20)]
mail_data["spam_words_freq"] = mail_data["process_text"].apply(lambda x: sum(1 for word in x.split() if word in top_spam_words and word not in top_ham_words))

# %%
# prepare the dataset for the models
tfidf = TfidfVectorizer(max_features=3000)
X_text = tfidf.fit_transform(mail_data['process_text']).toarray()

# add and combine other features
X_manual = mail_data[["text_length", "uppercase_ratio", "spam_words_freq"]]
X = np.hstack((X_text, X_manual))

y = mail_data["type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 0)

#%% 
# tune the hyperparameters

if DEVELOPMENT:
    # bayes hyperparameter optimization
    bayes_param_distributions = {'alpha': uniform(0.001, 0.5)}
    bayes_random_search = RandomizedSearchCV(
        MultinomialNB(), 
        random_state=0,
        param_distributions=bayes_param_distributions,
        n_iter= 10,
        cv=3
    )
    bayes_random_search.fit(X_train, y_train)
    print(f"Best Hyperparameters for Bayes: {bayes_random_search.best_estimator_}")

    '''
    # logistic regression hyperparameter optimization
    log_reg_param_distributions = {
        'penalty': ['l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['lbfgs'],
        'max_iter': [100, 1000, 2500, 5000]
    }
    log_reg_random_search = RandomizedSearchCV(
        LogisticRegression(), 
        random_state=0,
        param_distributions=log_reg_param_distributions,
        n_iter= 10,
        cv=3
    )
    log_reg_random_search.fit(X_train, y_train)
    print(f"Best Hyperparameters for Logistic Regression: {log_reg_random_search.best_estimator_}")
    '''

    # random forest hyperparameter optimization
    random_forest_param_distributions = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    }
    random_forest_random_search = RandomizedSearchCV(
        RandomForestClassifier(), 
        random_state=0,
        param_distributions=random_forest_param_distributions,
        n_iter= 10,
        cv=3
    )
    random_forest_random_search.fit(X_train, y_train)
    print(f"Best Hyperparameters for Random Forest: {random_forest_random_search.best_estimator_}")

    # SVC hyperparameter optimization
    SVC_param_distributions = {'C': uniform(0.001, 10)}
    SVC_random_search = RandomizedSearchCV(
        LinearSVC(), 
        random_state=0,
        param_distributions=SVC_param_distributions,
        n_iter= 10,
        cv=3
    )
    SVC_random_search.fit(X_train, y_train)
    print(f"Best Hyperparameters for SVC: {SVC_random_search.best_estimator_}")

# %%
# train the bayes model
bayes_model = MultinomialNB(alpha=np.float64(0.19272075941288885))
bayes_model.fit(X_train, y_train)
bayes_predict = bayes_model.predict(X_test)
bayes_accuracy = accuracy_score(y_test, bayes_predict)
print(f"Bayes Accuracy: {bayes_accuracy}")
bayes_matrix = confusion_matrix(y_test, bayes_predict)
print(f"Bayes Confusion Matrix: \n {bayes_matrix}")
print(f"Bayes Classification Report: \n {classification_report(y_test, bayes_predict)}")

#%%
# train the logistic regression model
log_reg_model = LogisticRegression(random_state= 0, max_iter=2000)
log_reg_model.fit(X_train, y_train)
log_reg_predict = log_reg_model.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_predict)
print(f"Logistic Regression Accuracy: {log_reg_accuracy}")
log_reg_matrix = confusion_matrix(y_test, log_reg_predict)
print(f"Logistic Regression Confusion Matrix: \n {log_reg_matrix}")
print(f"Logistic Regression Classification Report: \n {classification_report(y_test, log_reg_predict)}")

# %%
# train the random forest model
forest_model = RandomForestClassifier(max_depth=100, min_samples_split=10, n_estimators=400, random_state= 0)
forest_model.fit(X_train, y_train)
forest_predict = forest_model.predict(X_test)
forest_accuracy = accuracy_score(y_test, forest_predict)
print(f"Forest Accuracy: {forest_accuracy}")
forest_matrix = confusion_matrix(y_test, forest_predict)
print(f"Forest Confusion Matrix: \n {forest_matrix}")
print(f"Forest Classification Report: \n {classification_report(y_test, forest_predict)}")

# %%
# train the SVC model
SVC_model = LinearSVC(random_state= 0, C=np.float64(4.376872112626925))
SVC_model.fit(X_train, y_train)
SVC_predict = SVC_model.predict(X_test)
SVC_accuracy = accuracy_score(y_test, SVC_predict)
print(f"SVC Accuracy: {SVC_accuracy}")
SVC_matrix = confusion_matrix(y_test, SVC_predict)
print(f"SVC Confusion Matrix: \n {SVC_matrix}")
print(f"SVC Classification Report: \n {classification_report(y_test, SVC_predict)}")

# %%
# I'm choosing to go with Bayes model
# Save the model to joblib
joblib.dump(bayes_model, "spam_classifier_nb.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")