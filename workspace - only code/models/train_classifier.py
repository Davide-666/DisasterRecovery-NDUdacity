# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

import warnings

warnings.simplefilter('ignore')

def load_data(database_filepath):
    """
    Loads data from SQL Database
    Args:
    database_filepath: SQL database file
    Returns:
    X pandas_dataframe: Features dataframe
    Y pandas_dataframe: Target dataframe
    category_names list: Target labels 
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(Y.columns.values)
    return X, Y, category_names

def tokenize(text):
    """
    Tokenizes text data
    Args:
    text str: Messages as text data
    Returns:
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize words
    tokens = word_tokenize(text)
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return stemmed

def performance_metric(y_true, y_pred):
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_list.append(f1)
    score = np.median(f1_list)
    return score

def build_model():
    """
    Build model with GridSearchCV
    Returns:
    Trained model after performing grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'vect__min_df': [5], #[1, 5],
                  'tfidf__use_idf': [True], #[True, False],
                  'clf__estimator__n_estimators': [25], #[10, 25], 
                  'clf__estimator__min_samples_split': [10] #[10, 20]
                 }
    scorer = make_scorer(performance_metric)
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 10)
    return cv
        
    
def get_eval_metrics(actual, predicted, col_names):
    metrics = []
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        metrics.append([accuracy, precision, recall, f1])
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
    return metrics_df
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Shows model's performance on test data
    Args:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """
    Y_pred = model.predict(X_test)
    eval_metrics = get_eval_metrics(np.array(Y_test), Y_pred, category_names)
    print(eval_metrics)


def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file    
    Args:
    model: Trained model
    model_filepath: Filepath to save the model
    """    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()