import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import sklearn as sk
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import dump
from joblib import load
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_pipeline(classifier, isScaled, selectFeatures):
    """
    This function takes a scikit-learn classifier and builds a ML
    pipeline that applies a series of transformations.

    Parameters:
    - classifier: The machine learning classifer of interest ex. KNN
    - isScaled(bool): Should we scale data? True or False
    - selectFeatures(bool): Should we perform RFE? True or False

    Returns:
    - pipe(Pipeline): A ML pipeline capable of transforming data

    """
    try:
        if selectFeatures:
            if isScaled:
                pipe = Pipeline(steps=[("imputer", SimpleImputer()),
                                       ("scale", StandardScaler()),
                                       ("rfe", RFE(estimator=classifier)),
                                       ("clf", classifier)])
            else:
                pipe = Pipeline(steps=[("imputer", SimpleImputer()),
                                       ("rfe", RFE(estimator=classifier)),
                                       ("clf", classifier)])
        else:
            if isScaled:
                pipe = Pipeline(steps=[("imputer", SimpleImputer()),
                                       ("scale", StandardScaler()),
                                       ("clf", classifier)])
            else:
                pipe = Pipeline(steps=[("imputer", SimpleImputer()),
                                       ("clf", classifier)])
        return pipe

    except Exception as inst:
        print("Run-Time Error:", type(inst))
        sys.exit(1)


def run_gridsearch(pipe, param_grid, num_folds, metric,
                   X_train, y_train, output_model):
    """
    This function runs a grid search over a combination of
    hyperparameters using k-fold cross validation to select the best
    estimator

    Parameters:
    - pipe(Pipeline): A ML pipeline capable of transforming data
    - param_grid(dict): The desired hyperparameters and ranges to tune
    - num_folds(int): The number of folds for cross validation
    - metric(str): The evaluation metric  ex. "f1"
    - X_train(ndarray): A numpy array containing 1-test_size fraction of
                        the original dataset
    - y_train(ndarray): A numpy array containing class labels for X_train
    - output_model: The name of the output file containing the best model

    Returns:
    - best_params_(dict): The best performing hyperparameters
    - best_score_(float): The average validation score over k-folds
    - best_estimator_: The actual best performing model

    """

    try:
        search = GridSearchCV(pipe, param_grid, cv=num_folds,
                              scoring=metric, n_jobs=-1,
                              return_train_score=True)
        search.fit(X_train, y_train)
        dump(search.best_estimator_, output_model+".pkl")
        return search.best_params_, search.best_score_, search.best_estimator_, search.cv_results_  # noqa: E501

    except Exception as inst:
        print("Run-Time Error:", type(inst))
        sys.exit(1)


def import_data(csv_file):
    """
    This function imports data from a .csv file as a pandas dataframe

    Parameters:
    - csv_file(str): A .csv file path

    Returns:
    - df(DataFrame): A pandas DataFrame containing features and class labels

    """
    try:
        df = pd.read_csv(csv_file)
        return df

    except FileNotFoundError as inst:
        print("Run-Time Error:", type(inst))
        sys.exit(1)


def make_correlation_heatmap(df, title, outfile):
    """
    This function creates a correlation heatmap for a dataframe
    of features and saves it to a user-defined file.

    Parameters:
    - df(DataFrame): A dataframe of X features and y label
    - title: The title of the correlation heat map
    - outfile(str): The file where the heat map will be saved

    Returns:
    - None, but a .png file is created

    """

    try:
        df_corr = df.corr()
        plt.figure(figsize=(10, 10))
        plt.title(title)
        mask = np.zeros_like(df_corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            ax = sns.heatmap(df_corr, fmt=".2f", mask=mask, annot=True)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.savefig(outfile)

    except Exception as inst:
        print("Run-Time Error:", type(inst))
        sys.exit(1)


def one_hot_encode(df, col_label):
    """
    This function one-hot encodes a single categorical feature in a DataFrame
    and drops the original feature column from the DataFrame

    Parameters:
    - df(DataFrame): A pandas DataFrame containing features and class labels
    - col_label(str): The column label of categorical feature

    Returns:
    - df(DataFrame): A pandas DataFrame after one-hot encoding

    """
    try:
        df_enc = pd.get_dummies(df[col_label])
        df = pd.concat([df_enc, df], axis=1)
        df.drop([col_label], axis=1, inplace=True)

    except KeyError as inst:
        print("Run-Time Error:", type(inst))
        sys.exit(1)

    return df


def split_data(df, test_size, label_col):
    """
    This function splits a pandas DataFrame into X_train, X_test, y_train, and
    y_test sets according to user-defined percentage.

    Parameters:
    - df(DataFrame): A pandas DataFrame containing features and class labels
    - test_size(float): The fraction of original DataFrame held out for testing
                        Ex. 0.2
    - label_col(str): The column of the DataFrame that contains the class label

    Returns:
    - X_train(ndarray): A numpy array containing 1-test_size fraction of
                        the original dataset
    - y_train(ndarray): A numpy array containing the class labels for X_train
    - X_test(ndarray): A numpy array containing test_size fraction of
                       the original dataset
    - y_test(ndarray): A numpy array containing the class labels for X_test

    """
    try:
        train, test = train_test_split(df, test_size=test_size)
        X_train = train.loc[:, train.columns != label_col].values
        y_train = train[label_col].values
        X_test = test.loc[:, train.columns != label_col].values
        y_test = test[label_col].values

        return X_train, y_train, X_test, y_test

    except Exception as inst:
        print("Run-Time Error:", type(inst))
        sys.exit(1)


def make_confusion_matrix(y_true, y_pred, output_file):
    """
    This function creates a confusion matrix in the form of a heat map

    Parameters:
    - y_true(list): The true class labels for X_test
    - y_pred(list): The predicted class labels for X_test
    - output_file(str): The file path for saving the confusion matrix

    Returns:
    - None; a .png file of the confusion matrix is created

    """
    try:
        data = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(data, columns=np.unique(y_true),
                             index=np.unique(y_true))
        df_cm.index.name = "Actual"
        df_cm.columns.name = "Predicted"
        plt.figure(figsize=(10, 10))
        plt.title("Confusion matrix")
        cm = sns.heatmap(df_cm, annot=True)
        plt.savefig(output_file)

    except Exception as inst:
        print("Run-Time Error:", type(inst))
        sys.exit(1)


def evaluate_test_set(y_test, y_pred, output_file):
    """
    This function creates a confusion matrix of the actual
    vs. predicted class labels and returns the f1_score

    Parameters:
    - y_test(ndarray): A numpy array containing the class labels for X_test
    - y_test(ndarray): A numpy array containing the predicted
                       class labels for X_test
    - output_file(str): The file path for saving the confusion matrix

    Returns:
    - score(float): The f1_score for the model's predictions on the test set

    """
    try:
        make_confusion_matrix(y_test, y_pred, output_file)
        return f1_score(y_test, y_pred)

    except Exception as inst:
        print("Run-Time Error:", type(inst))
        sys.exit(1)
