# Import the necessary libraries:
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset into a pandas dataframe:

df = pd.read_csv('training_file.csv')

# Split the data into training and testing sets:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['feedback'], df['sentiment'], random_state=0)

# Vectorize the text data using the CountVectorizer:

vectorizer = CountVectorizer().fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)

# Train the Naive Bayes classifier:

nb = MultinomialNB()
nb.fit(X_train_vectorized, y_train)

# Evaluate the model performance:

X_test_vectorized = vectorizer.transform(X_test)
y_pred = nb.predict(X_test_vectorized)
print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Confusion matrix:', confusion_matrix(y_test, y_pred))

# Use the model to predict sentiment on new customer feedback:

new_feedback = ["your application is amazing great experience"]
new_feedback_vectorized = vectorizer.transform(new_feedback)
sentiment_pred = nb.predict(new_feedback_vectorized)
print('Predicted sentiment:', sentiment_pred[0])