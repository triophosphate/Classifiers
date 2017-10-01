import numpy as np
import pandas as pd
import json
import sys
import helpers

# get the data
products = pd.read_csv("amazon_baby.csv")
print list(products)

# clean the data
products = products.fillna({'review':''})
products['review_clean'] = products['review'].apply(helpers.remove_punctuation)
print list(products)

# ignore all ratings of 3 since they tend to be neutral
products = products[products['rating'] != 3]

# extract sentiment positive: rating >=4, negative: rating <= 2
products['sentiment'] = products['rating'].apply(lambda rating: + 1 if rating > 3 else -1)
print list(products)

# split into test data and training data randomly
# to get the same results for the test do this using provided json indexes
test_data = helpers.get_data_from_json_indexes('module-2-assignment-test-idx.json', products)
print len(test_data)
train_data = helpers.get_data_from_json_indexes('module-2-assignment-train-idx.json', products)
print len(train_data)


# To train our classifier we will use the Countvectorizer  class to compute the
# word count in each review: 'bag of words'. Use sparse matrixes.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b') # single word pattern
print vectorizer

# 1. Learn a vocabulary of all words in all reviews in the training data. Each word is a column
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
print vectorizer.get_feature_names()
print train_matrix

# 2. Convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])
print test_matrix

# Train a sentiment classifier with logistic regression
# use an instance of the LinearRegression class
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
print "training the model..."
sentiment_model = logistic_regression.fit(train_matrix, train_data["sentiment"])
print "finished training the model!"

# There should be over 100,000 coefficients in this sentiment_model.
# Recall from the lecture that positive weights w_j correspond to weights that cause positive sentiment,
# while negative weights correspond to negative sentiment.
# Calculate the number of positive (>= 0, which is actually nonnegative) coefficients.
# Quiz question: How many weights are >= 0?

sentiment_model_nonnegative_weights = logistic_regression.coef_[logistic_regression.coef_ >= 0]
sentiment_model_negative_weights = logistic_regression.coef_[logistic_regression.coef_ < 0]

print "weights > = 0 ---> ", len(sentiment_model_nonnegative_weights)
print "weights < 0   ---> ", len(sentiment_model_negative_weights)

# digging deeper into test data, get a sample test data
sample_test_data = test_data[10:13]
print sample_test_data
print "maybe positive\n", sample_test_data['review'].iloc[0]
print "maybe negative\n", sample_test_data['review'].iloc[1]
print "also maybe negative\n", sample_test_data['review'].iloc[2]


# predictions for the sample test data
sample_test_matrix = vectorizer.transform(sample_test_data["review_clean"])
sample_scores = sentiment_model.decision_function(sample_test_matrix)
print sample_scores

sample_sentiments = sentiment_model.predict(sample_test_matrix)
print sample_sentiments

# Using the scores calculated previously, write code to calculate the probability
# that a sentiment is positive using the above formula.
# For each row, the probabilities should be a number in the range [0, 1].

print helpers.get_probability(sample_scores[0])
print helpers.get_probability(sample_scores[1])
print "%.6f" % helpers.get_probability(sample_scores[2])

# Now examine the whole test data set
# Using the sentiment_model, find the 20 reviews in the entire test_data with the highest probability
# of being classified as a positive review.
# We refer to these as the "most positive reviews."

scores = sentiment_model.decision_function(test_matrix)
probabilities = np.vectorize(helpers.get_probability)

# add a new column in the test_data with the calculated probabilities
test_data.loc[:,'probabilities'] = probabilities(scores)
print test_data['probabilities']

# sort test data according to probability
# Quiz Question: Which of the following products are represented in the 20 most positive reviews?

test_data_sorted = test_data.sort_values('probabilities', ascending=False)
print test_data_sorted[-20:]
# Quiz Question: Which of the following products are represented in the 20 most negative reviews?
print test_data_sorted[-20:]


# Evaluate the accuracy of the sentiment_model which is given by the formula:
# accuracy=# correctly classified examples/# total examples

#Quiz Question: What is the accuracy of the sentiment_model on the test_data? Round your answer to 2 decimal places (e.g. 0.76).
#Quiz Question: Does a higher accuracy value on the training_data always imply that the classifier is better?
print "accuracy: %.2f" % helpers.get_accuracy(test_data, 'sentiment', 'probabilities')

#Learn another classifier with fewer words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
      'work', 'product', 'money', 'would', 'return']

vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to significant words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

print "training the model..."
logistic_regression_simple = LogisticRegression() #call to get an instance of the linearRegression class
simple_model = logistic_regression_simple.fit(train_matrix_word_subset, train_data["sentiment"])

# Inspect the weights (coefficients) of the simple_model.
# First, build a table to store (word, coefficient) pairs.

simple_model_coef_table = pd.DataFrame({'coef': simple_model.coef_.flatten(), 'words':np.array(significant_words)})
print simple_model_coef_table
print len(simple_model_coef_table)
print len(significant_words)

# Sort the data frame by the coefficient value in descending order.

simple_model_coef_table_sorted = simple_model_coef_table.sort_values('coef', ascending=False)
print simple_model_coef_table_sorted

# Quiz Question: Consider the coefficients of simple_model.
# How many of the 20 coefficients (corresponding to the 20 significant_words) are positive for the simple_model?
print len(simple_model_coef_table_sorted[simple_model_coef_table_sorted['coef']>0])

#Quiz Question: Are the positive words in the simple_model also positive words in the sentiment_model?

sentiment_model_coef_table = pd.DataFrame({'coef':sentiment_model.coef_.flatten(), 'words':vectorizer.get_feature_names()})
for word in significant_words:
    print word, sentiment_model_coef_table[sentiment_model_coef_table['words'] == word]['coef']
