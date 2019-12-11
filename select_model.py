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

from joblib import dump, load
import json

def select_model(config_file):
    """
    
    """
    with open(config_file, 'r') as file:
        config = json.load(file)

    df = import_data(config["file"])
    df = one_hot_encode(df, config["one_hot_encode"])
    df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    X_train, y_train, X_test, y_test = split_data(df, 0.2, config["class_label"])
    pipe = build_pipeline(DecisionTreeClassifier(), 
                          config["isScaled"], config["selectFeatures"])
    param_grid = config["param_grid"]
    
    params, score, model = run_gridsearch(pipe, param_grid, config["num_folds"], 
                                          config["metric"], X_train, y_train, 
                                          config["output_model"])

    print("Best parameters:", params)
    print("Best validation score:", score)
    print("Best model:", model)
    if config["selectFeatures"]:
        print("Number of features selected:", model.named_steps["rfe"].n_features_)
        print("Feature mask:", model.named_steps["rfe"].support_)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration file")

    parser.add_argument("config_file",
                        type=str,
                        help="Path of the config file")
    
    args = parser.parse_args()
    select_model(args.config_file)
    
    
    
    
    
    
    
    