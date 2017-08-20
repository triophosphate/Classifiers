import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import sys
import helpers


def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

# get the data
products = pd.read_csv("amazon_baby.csv")
# print list(products)

# clean the data
products = products.fillna({'review':''})
products['review_clean'] = products['review'].apply(remove_punctuation)
# print list(products)
# ignore all rating 3s since they tend to be neutral
products = products[products['rating'] != 3]

# extract sentiment positive: rating >=4, negative: rating <= 2
products['sentiment'] = products['rating'].apply(lambda rating: + 1 if rating > 3 else -1)
#print list(products)

# split into test data and training data randomly
# to get the same results at the test use json indexes
with open('module-2-assignment-test-idx.json') as data_file:
    test_idx = json.load(data_file)
test_data = products.iloc[test_idx]
# print len(test_data)

with open('module-2-assignment-train-idx.json') as data_file:
    train_idx = json.load(data_file)
train_data = products.iloc[train_idx]
# print len(train_data)

# compute word count in each review: bag of words
# use sparse matrix to store the collection of word count vectors
# because some words occur only in some reviews

# 1. Learn a vocabulary of all words in all reviews in the training data. Each word is a column
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b') #single word pattern
train_matrix = vectorizer.fit_transform(train_data['review_clean'])

# 2. Convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

# print train_matrix
# print test_matrix

# Train a sentiment classifier with logistic regression
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression() #call to get an instance of the linearRegression class
sentiment_model = logistic_regression.fit(train_matrix, train_data["sentiment"])

# There should be over 100,000 coefficients in this sentiment_model.
# Recall from the lecture that positive weights w_j correspond to weights that cause positive sentiment,
# while negative weights correspond to negative sentiment.
# Calculate the number of positive (>= 0, which is actually nonnegative) coefficients.
# Quiz question: How many weights are >= 0?


# X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
# C,S = np.cos(X), np.sin(X)


#plt.plot(X, C)
#plt.plot(X, S)

#plt.show()

