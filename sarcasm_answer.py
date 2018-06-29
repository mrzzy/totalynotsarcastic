#
# sarcasm.py
# Scacasm Detection
# Scikit-Learn/Naive Bayes Edition
#

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from sarcasm_util import load_data, split_test_train

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None)

if __name__ == "__main__":
    # Load and preprocess data
    labels, texts = load_data()
    
    inputs = texts
    outputs = labels

    # Split the data into test and train sets
    train_inputs, train_outputs, test_inputs, test_outputs = \
            split_test_train(inputs, outputs)

    # Prepares text data for machine learning: 
    # Performs feature extraction, followed by feature extraction into
    # sparse vectors using inverse document frequency.  
    vectorizer.fit(train_inputs)
    train_inputs = vectorizer.transform(train_inputs)

    # Train the classifier
    classifier = MultinomialNB()
    classifier.fit(train_inputs, train_outputs)
    
    # Test the classfier
    test_inputs = vectorizer.transform(test_inputs)
    test_predicts = classifier.predict(test_inputs)

    # Check the answers
    n_correct = 0
    for i in range(len(test_predicts)):
        predicted = test_predicts[i]
        answer = test_outputs[i]
        
        if predicted == answer:
            n_correct = n_correct + 1
    
    print("Result: {:.2f}%".format(n_correct/len(test_outputs) * 100.0))
    
    # Prompt to allow the students to try the program
    try:
        print("Type 'quit' or 'exit' to exit")
        while True:
            statement = input("> ")
            if statement == "quit" or statement == "exit":
                break

            inputs = vectorizer.transform([ statement ])
            predicts = classifier.predict(inputs)
            prediction = predicts[0]
            
            if prediction == 0:
                print("Not Sarcastic")
            else:
                print("Sarcastic")
    except (EOFError, KeyboardInterrupt):
        pass
