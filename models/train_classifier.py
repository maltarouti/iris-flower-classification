from inspect import Parameter
import sys
import pickle 
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine

import warnings 
warnings.filterwarnings("ignore")


def load_data(database_path):
    """
    Loads data from SQLite database and returns a pandas dataframe.
    :param database_path: Path to SQLite database.
    :return: Pandas dataframe.
    """
    engine = create_engine('sqlite:///{}'.format(database_path))
    dataset = pd.read_sql_table('dataset', engine)
    return dataset

def split_dataset(dataset):
    """
    Splits the dataset into training and test sets.
    :param dataset: Pandas dataframe.
    :return: x_train, x_test, y_train, y_test.
    """
    x = dataset.iloc[:, :4]
    y = dataset.iloc[:, 4]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test

def build_mode():
    """
    Builds a model using the training dataset and returns the model.
    :return: Model.
    """

    model = LogisticRegression()

    parameters = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [100, 500],
    }

    model = GridSearchCV(model, parameters, cv=5, scoring='accuracy',)


    return model

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the model using the test dataset.
    :param model: Model.
    :param x_test: Test dataset.
    :param y_test: Test labels.
    :return: None.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {}'.format(accuracy))
    
def save_model(model, filename):
    """
    Saves the model to a pickle file.
    :param model: Model.
    :param path: Path to pickle file.
    :return: None.
    """

    filename += '.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Main function.
    :return: None.
    """
    if len(sys.argv) == 3:
        database_path, model_path = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_path))
        dataset = load_data(database_path)
        print('Splitting data...')
        x_train, x_test, y_train, y_test = split_dataset(dataset)
        print('Building model...')
        model = build_mode()
        print('Training model...')
        model.fit(x_train, y_train)
        print('Evaluating model...')
        evaluate_model(model, x_test, y_test)
        print('Saving model...\n    MODEL: {}'.format(model_path))
        save_model(model, model_path)
        print('Trained model saved!')
    else:
        print('Please provide the following argument:')
        print('    - Path to the SQLite database.')
        print('    - Path to the output pickle file.')

if __name__ == "__main__":
    main()