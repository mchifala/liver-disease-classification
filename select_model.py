import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from liver_functions import import_data
from liver_functions import one_hot_encode
from liver_functions import split_data
from liver_functions import build_pipeline
from liver_functions import run_gridsearch
from liver_functions import evaluate_test_set
from joblib import dump
from joblib import load
import json
from time import perf_counter


def select_model(config_file):
    """
    This function uploads a configuration file and uses its contents
    to load a dataset, split the dataset, build a ML pipe, and run
    a grid search using k-fold cross validation

    Parameters:
    - config_file(str): The path to a json file containing configurations

    Returns:
    None; the best estimator is output to a .pkl model for use by rest server

    """

    with open(config_file, 'r') as file:
        config = json.load(file)

    df = import_data(config["file"])
    df = one_hot_encode(df, config["one_hot_encode"])
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    X_train, y_train, X_test, y_test = split_data(df, 0.2,
                                                  config["class_label"])
    if config["model_type"] == "tree":
        pipe = build_pipeline(DecisionTreeClassifier(),
                              config["isScaled"], config["selectFeatures"])

    elif config["model_type"] == "svm":
        pipe = build_pipeline(SVC(kernel=config["kernel"]),
                              config["isScaled"], config["selectFeatures"])

    elif config["model_type"] == "knn":
        pipe = build_pipeline(KNeighborsClassifier(),
                              config["isScaled"], config["selectFeatures"])

    param_grid = config["param_grid"]

    t0 = perf_counter()
    params, score, model, results = run_gridsearch(pipe, param_grid,
                                                   config["num_folds"],
                                                   config["metric"], X_train,
                                                   y_train,
                                                   config["output_model"])
    t1 = perf_counter()

    print("Grid search runtime:", t1-t0)
    print("# of hyperparameter combinations:",
          len(results["mean_test_score"]))
    print("Avg. time per combination:",
          (t1-t0)/(len(results["mean_test_score"])))
    print("Best parameters:", params)
    print("Best validation score:", score)
    print("Best model:", model)
    if config["selectFeatures"]:
        print("# of features selected:",
              model.named_steps["rfe"].n_features_)
        print("Features:", np.array(df.columns[:-1])[model.named_steps["rfe"].support_])  # noqa: E501
        print("Feature ranking:",
              np.array(model.named_steps["rfe"].ranking_)[model.named_steps["rfe"].support_])  # noqa: E501
    pd.DataFrame(results).to_csv(config["cv_file"])

    y_pred = model.predict(X_test)
    evaluate_test_set(y_test, y_pred, "confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration file")
    parser.add_argument("config_file",
                        type=str,
                        help="Path of the config file")
    args = parser.parse_args()
    select_model(args.config_file)
