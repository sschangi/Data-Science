import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Returns two data frames, i.e., one containing the data, and the other containing the labels along with the label names.

    Parameters:
        database_filepath (string): The file path of the sql table holding the data.

    Returns:
        X (pandas data frame): A data frame containing the messages
        y (pandas data frame): A data frame contaiining the category labels
        y.columns (list of strings): A list of the category labels
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    
    X = df.iloc[:,2:3]
    y = df.iloc[:,-36:]
    return X, y, y.columns


def tokenize(text):
    '''
    Returns the list of tokens.

    Parameters:
        text (string): A sentence od phrase in english

    Returns:
        clean_tokens (list of strings): A list containing the tokens
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    Builds a sklearn ML pipeline using tokenizer, TfidfTransformer, and 
    RandomForest multioutput classifier.

    Returns:
        pipeline (Pipeline object): A sklearn Pipeline object
    '''
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tdfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [25, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose = 3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates a given ML model and prints the classification report, using 
    precision, recall, and f1-score.

    Parameters:
        model (sklearn model): A sklearn model object
        X_test (numpy array of strings): A numpy array holding the test data
        Y_test (numpy array of integers): A numpy array holding the label test data
        category_names (list of strings): A list of label categories
    '''
    
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print("category : ", category)
        print(classification_report(Y_test[:,i], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Saves the trained sklearn ML model as a pickle file.

    Parameters:
        model (sklearn ML model): A trained sklearn ML model object.
        model_filepath (string): The model file path.
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        X_train = X_train.values.reshape(X_train.values.shape[0])
        Y_train = Y_train.values
        
        idx = np.argwhere(~np.isnan(Y_train))
        indices = set()
        for index in idx:
            indices.add(index[0])
        
        X_train = X_train[list(indices)]
        Y_train = Y_train[list(indices)]
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        
        X_test = X_test.values.reshape(X_test.values.shape[0])
        Y_test = Y_test.values
        
        idx = np.argwhere(~np.isnan(Y_test))
        indices = set()
        for index in idx:
            indices.add(index[0])
            
        X_test = X_test[list(indices)]
        Y_test = Y_test[list(indices)]
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()