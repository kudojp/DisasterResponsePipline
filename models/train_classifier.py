# import libraries

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle
from nltk.corpus import stopwords

nltk.download(['wordnet','stopwords'])


def load_data(database_filepath):
    """Method to load in data from the specified database_file

    Args:
        database_filepath : path to the database file to be loaded

    Return :
        X : matrix of messages and their genre
        Y : matrix containing 36 labels of messages
        category_names : series of 36 labels
    """

    # load data from database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("messages", con=engine)

    # split dataframe into x (messages and genres) / y (36 labels)
    x = df["message"]
    y = df.drop(axis=1, labels=["id", "message", "original", "genre"])

    return x.values, y.values, y.columns

def tokenize(text):
    """Method to make a list of tokens from sentences.

    Args:
        text : text to be tokenized

    Return:
        clean_tokens : list of clean tokens
    """

    # remove punctuations in the sentences
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # split sentences into tokens(words) in a list
    tokens = word_tokenize(text)

    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # create list where lemmatized tokens would be stored
    clean_tokens = []


    # lemmatize each token and store it in the list "clean_tokens"
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()

        """
        # remove stop words (This portion does not work...)
        if clean_tok not in set(stopwords.words("english")):
        """
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Method to return a GridSerach model object to be trained later.

    Return:
        cv : GridSearch model
    """

    pipeline = Pipeline([("vect",CountVectorizer(tokenizer=tokenize)),
                     ("tfidf",TfidfTransformer()),
                     ("clf", MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {
    "vect__ngram_range" : [(1,1), (1,2)],
    #"vect__max_df" : [0.5,1],
    "vect__max_features" : [None, 5000, 10000],
    #"tfidf__use_idf" : [True, False],
    "clf__estimator__n_estimators" : [50,100,200],
    "clf__estimator__min_samples_split" : [2,3,4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate and output the testing result of a model

    Args:
        model : a model used for testing
        X_test : x matrix of testing data
        Y_test : one-hot encoding y matrix of testing data
        category_names : ordered category names
    """

    Y_pred = model.predict(X_test)

    for i, category  in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    """Method to save a model in a specified pickle file

    Args:
        model : a model to be saved
        model_filepath : path to the file where a model would be stored
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/cleandata.db classifier.pkl')


if __name__ == '__main__':
    main()
