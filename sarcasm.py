#
# sarcasm.py
# Scacasm Detection
# Scikit-Learn/Naive Bayes Edition
#

import numpy as np
import csv
import random
import math

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from sarcasm_util import load_data, split_test_train

# Prepares text data for machine learning: 
# Performs feature extraction, followed by feature extraction into one hot ecoded
# sparse vectors using inverse document frequency.
# Returns converted sprse vectors
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
def preprocess_text(text_data):
    global vectorizer
    return vectorizer.transform(text_data)


if __name__ == "__main__":
    # Load and preprocess data
    labels, texts = load_data(2e+5)
    
    inputs = texts
    outputs = labels
    train_inputs, train_outputs, test_inputs, test_outputs = \
            split_test_train(inputs, outputs)
    
    vectorizer.fit(train_inputs)
    train_inputs = preprocess_text(train_inputs)

    # Train the classifier
    classifier = MultinomialNB()
    classifier.fit(train_inputs, train_outputs)
    
    # Test the classfier
    test_inputs = preprocess_text(test_inputs)
    test_predicts = classifier.predict(test_inputs)

    n_correct = 0
    for prediction, answer in zip(test_predicts, test_outputs):
        if prediction == answer:
            n_correct += 1
    
    print("Result:", n_correct/len(test_outputs) * 100.0)
    
    # Prompt to allow the studnets to try the program
    try:
        while True:
            statement = input("> ")
            inputs = preprocess_text([ statement ])
            predicts = classifier.predict(inputs)
            prediction = predicts[0]
            
            if prediction == 0:
                print("Not Sarcastic")
            else:
                print("Sarcastic")
    except EOFError:
        pass
