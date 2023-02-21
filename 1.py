import nltk
import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data and label relevant information
data = pd.read_csv('legal_documents.csv')
# data['parties'] = np.where(
#     data['text'].str.contains('plaintiff|defendant'), 1, 0)
# data['timeline'] = np.where(data['text'].str.contains('on|before|after'), 1, 0)
# data['liabilities'] = np.where(
#     data['text'].str.contains('liable|liability'), 1, 0)

# Preprocess text using nltk and spacy
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
data['text'] = data['text'].apply(lambda x: ' '.join(nltk.word_tokenize(x)))
data['text'] = data['text'].apply(
    lambda x: ' '.join([token.lemma_ for token in nlp(x)]))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['result'], test_size=0.2, random_state=42)
print(X_train)
print("-------------------------")
print(X_test)
print("-------------------------")
print(y_train)
print("-------------------------")
print(y_test)
print("-------------------------")

# Extract features using TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a support vector machine (SVM) model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate model on testing set
y_pred = model.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
