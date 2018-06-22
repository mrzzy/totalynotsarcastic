#
# sarcasm.py
# Scacasm Detection
#

import numpy as np
import csv
import random
import math
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer

# Loads data from CSV into a list of labels and a list of text data
def load_data(len_limit=None, filepath="./data/train-balanced-sarcasm.csv"):
    with open(filepath, "r") as f:
        labels = []
        texts = []
        reader = csv.DictReader(f)
        
        # Build numpy array dataframe
        for i, raw_row in enumerate(reader):
            labels.append(int(raw_row["label"]))
            texts.append(raw_row["comment"])
            # Only read len_limit rows of data if len_limit is defined
            if len_limit != None and i >= len_limit: break
    
        labels = np.asarray(labels)
        return (labels, texts)

# Prepares text data for machine learning: 
# Performs feature extraction, followed by feature extraction into one hot ecoded
# sparse vectors using inverse document frequency.
# Returns converted sprse vectors
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)
def preprocess_text(text_data):
    global vectorizer
    return vectorizer.transform(text_data)

# Split data into test and train sets to perform cross validation
# Splis data to test and train sets based on given factor, specifiy test size
# as a ratio of the entire datasetd
def split_test_train(inputs, outputs, test_ratio=0.3, random_state=0):
    data_len = len(inputs)
    
    # Shuffle the data to ensure relatively even distribution of data
    #random.seed(random_state)
    shuffle_order = list(range(data_len))
    random.shuffle(shuffle_order)
    
    shuffle_inputs = []
    shuffle_outputs = []
    for from_i in shuffle_order:
        shuffle_inputs.append(inputs[from_i])
        shuffle_outputs.append(outputs[from_i])
    
    # Split datasets into test and train
    divider = math.floor(data_len * test_ratio)
    
    test_inputs = shuffle_inputs[:divider]
    test_outputs = shuffle_outputs[:divider]
    
    train_inputs = shuffle_inputs[divider:]
    train_outputs = shuffle_outputs[divider:]

    return (train_inputs, train_outputs, test_inputs, test_outputs)


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
