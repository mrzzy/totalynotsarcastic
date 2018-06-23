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

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)

if __name__ == "__main__":
    # Load and preprocess data
    labels, texts = load_data(2e+5)
    
    # TODO: split the data into test and train
    

    # Prepares text data for machine learning: 
    # Performs feature extraction, followed by feature extraction into one hot 
    # ecoded sparse vectors using inverse document frequency.
    # TODO: Transform the text/strings into vectors

    # Train the classifier
    classifier = MultinomialNB()
        
    # TODO: train the classifer using training data
    
    # Test the classfier
    # TODO: test the classifer by using the classifier to predict outputs for the 
    # test inputs. Then check answers and output the percent the the classifer 
    # got correct.

    
    # Prompt client to allow the students to try the program
    try:
        while True:
            statement = input("> ")
            inputs = vectorizer.transform([ statement ])
            predicts = classifier.predict(inputs)
            prediction = predicts[0]
            
            if prediction == 0:
                print("Not Sarcastic")
            else:
                print("Sarcastic")
    except EOFError:
        pass
