"""
__author__ = 'Fadhil Salih'
__email__ = 'fadhilsalih94@gmail.com'
__date__ = '2021-05-01'
__dataset__ = 'https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho'
__connect__ = 'https://www.linkedin.com/in/fadhilsalih/'
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import datetime
import pickle


def number_of_years(df):
    """Function to create a column with the number of years since the car was bought and delete the column with the year of purchase
    :param df: input dataframe
    :return: df
    """
    current_date = datetime.datetime.now()
    date = current_date.date()
    year = int(date.strftime("%Y"))

    df['number_of_years'] = year - df['Year']
    df.drop(['Year'], axis=1, inplace=True)
    return df


def one_hot_encode(df):
    """Function to one-hot encode categorical variables
    :param df: input dataframe
    :return: df
    """
    df = pd.get_dummies(df, drop_first=True)
    return df


def correlation_check(df):
    """Function to view correlation between features using heat map
    :param df: input dataframe
    :return: None
    """
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(10, 10))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()


def missing_values_check(df):
    """Function to check if the data has missing values
    :param df: input dataframe
    :return: None
    """
    count = len(df.columns[df.isnull().any()])
    check = True if count > 0 else False
    if check:
        print("Warning! Dataset has missing values")
    else:
        print("Dataset has no missing values")
    return


def target_variable_split(df):
    """Function to split dataset into predictor variables and target variable
    :param df: input dataframe
    :return: X, y
    """
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return (X, y)


def important_features(df):
    """Function to plot graph of important features for better visualization
    :param df: input dataframe
    :return: None
    """
    X, y = target_variable_split(df)
    model = ExtraTreesRegressor()
    model.fit(X, y)
    important = pd.Series(model.feature_importances_, index=X.columns)
    important.nlargest(5).plot(kind='barh')
    plt.show()


def test_train_split(df):
    """Function to split the train and test dataset
    :param df: input dataframe
    :return: None
    """
    X, y = target_variable_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    return(X_train, X_test, y_train, y_test)


def hyperparameter_selection_rf():
    """Function to select hyperparameters for random forest regressor model
    :param: None
    :return: random_grid
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 15, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    return(random_grid)


def results(rf_random, X_test, y_test):
    """Function to produce results of the trained model
    :param: rf_random, X_test, y_test
    :return: None
    """
    print(rf_random.best_score_)
    predictions = rf_random.predict(X_test)
    sns.displot(y_test-predictions, kde=True)
    plt.show()
    plt.scatter(y_test, predictions)
    plt.title("Scatterplot")
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(y_test, predictions)))


def final_model(df):
    """Function to train the model and call the save_pickle function
    :param: rf_random, X_test, y_test
    :return: None
    """

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=hyperparameter_selection_rf(),
                                   scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
    X_train, X_test, y_train, y_test = test_train_split(df)
    rf_random.fit(X_train, y_train)
    results(rf_random, X_test, y_test)
    save_pickle(rf_random)


def save_pickle(rf_random):
    """Function to pickle the model
    :param: rf_random
    :return: None
    """
    # open a file, where you ant to store the data
    file = open('random_forest_regression_model.pkl', 'wb')

    # dump information to that file
    pickle.dump(rf_random, file)


def main():
    car = pd.read_csv('car data.csv')

    car.drop(['Car_Name'], axis=1, inplace=True)

    car = number_of_years(car)

    car = one_hot_encode(car)

    print(car.head())

    correlation_check(car)

    missing_values_check(car)

    important_features(car)

    final_model(car)


if __name__ == '__main__':
    main()
