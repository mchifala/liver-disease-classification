'''
This function enables visualizations of different
machine learning classifier algorithms while varying
different parameters. Dataset used is availible at:
https://www.kaggle.com/jeevannagaraj/indian-liver-patient-dataset
main:
    Parameters
    ----------
    * --classifier_type : type of classifier to be examined
        + 'KNN' --> K-nearest neighbors classifier
        + 'SVM' --> Support vector machines
        + 'DT' --> Decison tree
        + 'VC' --> Voting classifier using combination of the above methods
    * --output_folder : name of folder containing visualizations

    Returns
    -------
    * Folder containing visualizations
'''

import pandas as pd
import sklearn as sk
import seaborn as sns
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


from liver_functions import import_data
from liver_functions import one_hot_encode
from liver_functions import split_data
from liver_functions import build_pipeline
from liver_functions import run_gridsearch

import argparse
import os

def learning_viz(classifier, output_folder):
    # Create output folder
    if os.path.exists(str(output_folder)) is False:
        os.mkdir(output_folder)

    if classifier == 'KNN':
        X_train, y_train, X_test, y_test = setup()

        # Build pipeline
        pipe = build_pipeline(KNeighborsClassifier(), True)

        # Run KNN classifier, vary neighbors from (1, 100) and store f1 scores for testing and training
        param_grid = dict(clf__n_neighbors=list(range(1, 100)))
        params, score, model, cv_results_ = run_gridsearch(pipe, param_grid, 10, "f1", X_train, y_train)
        grid_results = pd.DataFrame.from_dict(cv_results_)

        # Process output
        param_clf__n_neighbors = np.array(grid_results[["param_clf__n_neighbors"]].values, dtype='float64')
        mean_test_score = np.array(grid_results[["mean_test_score"]].values, dtype='float64')
        mean_train_score = np.array(grid_results[["mean_train_score"]].values, dtype='float64')

        # Create plots
        plt.plot(param_clf__n_neighbors, mean_train_score, color='skyblue', linewidth=4)
        plt.plot(param_clf__n_neighbors, mean_test_score, color='red', linewidth=4)
        plt.legend(["train score", "test score"])
        plt.xlabel("Nearest neighbors used")
        plt.ylabel("f1 score")
        plt.savefig(str(output_folder) + '/KNN.png')

    if classifier == 'SVM':
        X_train, y_train, X_test, y_test = setup()

        # Build pipeline
        svm_pipe = build_pipeline(SVC(), True)

        # Run SVM classifier, vary classifiers store f1 score
        # Kernel: rbf
        # Degree: 1 to 10
        # Gamma: 0 to 0.5
        svm_param_grid = {'clf__C': list(np.linspace(1, 10, num=10)), 'clf__kernel': ['rbf'], 'clf__gamma': list(np.logspace(0, 0.5, num=10))}
        svm_params, svm_score, svm_model, svm_cv_results_ = run_gridsearch(svm_pipe, svm_param_grid, 10, "f1", X_train, y_train)
        svm_grid_results = pd.DataFrame.from_dict(svm_cv_results_)

        # Process output
        svm_params_clf__C = np.array(svm_grid_results[["param_clf__C"]].values, dtype='float64')
        svm_params_clf__gamma = np.array(svm_grid_results[["param_clf__gamma"]].values,  dtype='float64')
        svm_params_mean_train_score = np.array(svm_grid_results[["mean_train_score"]].values,  dtype='float64')
        svm_params_mean_test_score = np.array(svm_grid_results[["mean_test_score"]].values, dtype='float64')
        # Create interpolated surface
        f_train = sp.interpolate.interp2d(svm_params_clf__C, svm_params_clf__gamma, svm_params_mean_train_score, kind='linear', )
        f_test = sp.interpolate.interp2d(svm_params_clf__C, svm_params_clf__gamma, svm_params_mean_test_score, kind='linear', )

        # More formatting
        x = np.log10(svm_params_clf__C.flatten())
        y = np.log10(svm_params_clf__gamma.flatten())
        z_train = f_train(x, y)
        z_test = f_test(x, y)

        # Create surface plot
        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, z_train, cmap=cm.PuBu_r)
        plt.title('Training Metrics')
        plt.xlabel("Regularization parameter")
        plt.ylabel("Gamma")
        cbar = fig.colorbar(cs)
        cbar.set_label('f1 score')
        plt.savefig(str(output_folder) + '/SVM_rbf_train.png')

        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, z_test, cmap=cm.PuBu_r)
        plt.title('Testing Metrics')
        plt.xlabel("Regularization parameter")
        plt.ylabel("Gamma")
        cbar = fig.colorbar(cs)
        cbar.set_label('f1 score')
        plt.savefig(str(output_folder) + '/SVM_rbf_test.png')

        # Run SVM classifier, vary classifiers store f1 score
        # Kernel: linear
        # Degree: 1 to 10
        # Gamma: 0 to 0.5
        svm_param_grid = {'clf__C': list(np.linspace(1, 10, 25)), 'clf__kernel': ['rbf'],
                          'clf__gamma': list(np.linspace(0, 0.5, 25))}
        svm_params, svm_score, svm_model, svm_cv_results_ = run_gridsearch(svm_pipe, svm_param_grid, 10, "f1", X_train,
                                                                           y_train)
        svm_grid_results = pd.DataFrame.from_dict(svm_cv_results_)

        # Process output
        svm_params_clf__C = np.array(svm_grid_results[["param_clf__C"]].values, dtype='float64')
        svm_params_clf__gamma = np.array(svm_grid_results[["param_clf__gamma"]].values, dtype='float64')
        svm_params_mean_train_score = np.array(svm_grid_results[["mean_train_score"]].values, dtype='float64')
        svm_params_mean_test_score = np.array(svm_grid_results[["mean_test_score"]].values, dtype='float64')
        # Create interpolated surface
        f_train = sp.interpolate.interp2d(svm_params_clf__C, svm_params_clf__gamma, svm_params_mean_train_score,
                                          kind='linear', )
        f_test = sp.interpolate.interp2d(svm_params_clf__C, svm_params_clf__gamma, svm_params_mean_test_score,
                                         kind='linear', )

        # More formatting
        x = svm_params_clf__C.flatten()
        y = svm_params_clf__gamma.flatten()
        z_train = f_train(x, y)
        z_test = f_test(x, y)

        # Create surface plot
        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, z_train, cmap=cm.PuBu_r)
        plt.title('Training Metrics')
        plt.xlabel("Regularization parameter")
        plt.ylabel("Gamma")
        cbar = fig.colorbar(cs)
        cbar.set_label('f1 score')
        plt.savefig(str(output_folder) + '/SVM_lin_train.png')

        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, z_test, cmap=cm.PuBu_r)
        plt.title('Testing Metrics')
        plt.xlabel("Regularization parameter")
        plt.ylabel("Gamma")
        cbar = fig.colorbar(cs)
        cbar.set_label('f1 score')
        plt.savefig(str(output_folder) + '/SVM_lin_test.png')

        # Run SVM classifier, vary classifiers store f1 score
        # Kernel: sigmoid
        # Degree: 1 to 10
        # Gamma: 0 to 0.5
        svm_param_grid = {'clf__C': list(np.linspace(1, 10, 25)), 'clf__kernel': ['sigmoid'],
                          'clf__gamma': list(np.linspace(0, 0.5, 25))}
        svm_params, svm_score, svm_model, svm_cv_results_ = run_gridsearch(svm_pipe, svm_param_grid, 10, "f1", X_train,
                                                                           y_train)
        svm_grid_results = pd.DataFrame.from_dict(svm_cv_results_)

        # Process output
        svm_params_clf__C = np.array(svm_grid_results[["param_clf__C"]].values, dtype='float64')
        svm_params_clf__gamma = np.array(svm_grid_results[["param_clf__gamma"]].values, dtype='float64')
        svm_params_mean_train_score = np.array(svm_grid_results[["mean_train_score"]].values, dtype='float64')
        svm_params_mean_test_score = np.array(svm_grid_results[["mean_test_score"]].values, dtype='float64')
        # Create interpolated surface
        f_train = sp.interpolate.interp2d(svm_params_clf__C, svm_params_clf__gamma, svm_params_mean_train_score,
                                          kind='linear', )
        f_test = sp.interpolate.interp2d(svm_params_clf__C, svm_params_clf__gamma, svm_params_mean_test_score,
                                         kind='linear', )

        # More formatting
        x = svm_params_clf__C.flatten()
        y = svm_params_clf__gamma.flatten()
        z_train = f_train(x, y)
        z_test = f_test(x, y)

        # Create surface plot
        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, z_train, cmap=cm.PuBu_r)
        plt.title('Training Metrics')
        plt.xlabel("Regularization parameter")
        plt.ylabel("Gamma")
        cbar = fig.colorbar(cs)
        cbar.set_label('f1 score')
        plt.savefig(str(output_folder) + '/SVM_sig_train.png')

        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, z_test, cmap=cm.PuBu_r)
        plt.title('Testing Metrics')
        plt.xlabel("Regularization parameter")
        plt.ylabel("Gamma")
        cbar = fig.colorbar(cs)
        cbar.set_label('f1 score')
        plt.savefig(str(output_folder) + '/SVM_sig_test.png')

        # Run SVM classifier, vary classifiers store f1 score
        # Kernel: poly
        # Degree: 1 to 10
        # Gamma: 0 to 0.5

        svm_param_grid = {'clf__degree': list(range(1,10)), 'clf__kernel': ['poly'], 'clf__gamma': list(np.linspace(0, 0.5, 25))}
        svm_params, svm_score, svm_model, svm_cv_results_ = run_gridsearch(svm_pipe, svm_param_grid, 10, "f1", X_train, y_train)
        svm_grid_results = pd.DataFrame.from_dict(svm_cv_results_)

        # Process output
        svm_params_clf__deg = np.array(svm_grid_results[["param_clf__degree"]].values, dtype='float64')
        svm_params_clf__gamma = np.array(svm_grid_results[["param_clf__gamma"]].values,  dtype='float64')
        svm_params_mean_train_score = np.array(svm_grid_results[["mean_train_score"]].values,  dtype='float64')
        svm_params_mean_test_score = np.array(svm_grid_results[["mean_test_score"]].values, dtype='float64')
        # Create interpolated surface
        f_train = sp.interpolate.interp2d(svm_params_clf__deg, svm_params_clf__gamma, svm_params_mean_train_score, kind='linear', )
        f_test = sp.interpolate.interp2d(svm_params_clf__deg, svm_params_clf__gamma, svm_params_mean_test_score, kind='linear', )

        # More formatting
        x = svm_params_clf__deg.flatten()
        y = svm_params_clf__gamma.flatten()
        z_train = f_train(x, y)
        z_test = f_test(x, y)

        # Create surface plots
        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, z_train, cmap=cm.PuBu_r)
        plt.title('Training Metrics')
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Gamma")
        cbar = fig.colorbar(cs)
        cbar.set_label('f1 score')
        plt.savefig(str(output_folder) + '/SVM_poly_train.png')

        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, z_test, cmap=cm.PuBu_r)
        plt.title('Testing Metrics')
        plt.xlabel("Polynomial Degree")
        plt.ylabel("Gamma")
        cbar = fig.colorbar(cs)
        cbar.set_label('f1 score')
        plt.savefig(str(output_folder) + '/3D_SVM_poly_test.png')

        # Create linear plots
        fig = plt.subplots()
        plt.plot("param_clf__degree", 'mean_train_score', data=svm_grid_results, color='skyblue', linewidth=4)
        plt.plot("param_clf__degree", 'mean_test_score', data=svm_grid_results, color='red', linewidth=4)
        plt.legend(["train score", "test score"])
        plt.xlabel("Degree")
        plt.ylabel("f1 score")
        plt.savefig(str(output_folder) + '/2D_SVM_poly_test_deg.png')

        fig = plt.subplots()
        plt.plot("param_clf__gamma", 'mean_train_score', data=svm_grid_results, color='skyblue', linewidth=4)
        plt.plot("param_clf__gamma", 'mean_test_score', data=svm_grid_results, color='red', linewidth=4)
        plt.legend(["train score", "test score"])
        plt.xlabel("Gamma")
        plt.ylabel("f1 score")
        plt.savefig(str(output_folder) + '/2D_SVM_poly_test_gam.png')

    if classifier == 'DT':
        X_train, y_train, X_test, y_test = setup()

        # Build Pipe
        tree_pipe = build_pipeline(DecisionTreeClassifier(), True)

        # Run DecisionTreeClassifier classifier,
        # vary classifiers store f1 score
        # max_depth: poly
        # min_samples_split: 1 to 10
        # max_features: 0 to 0.5

        tree_param_grid = {'clf__max_depth': list(np.linspace(1, 50, 20, dtype=int)), 'clf__min_samples_split': list(np.linspace(2, 100, 20, dtype=int)),
                           'clf__max_features': list(range(1, 10))}
        tree_params, tree_score, tree_model, tree_cv_results_ = run_gridsearch(tree_pipe, tree_param_grid, 10, "f1",
                                                                               X_train, y_train)
        tree_grid_results = pd.DataFrame.from_dict(tree_cv_results_)

        # Process output
        tree_params_clf__max_depth = np.array(tree_grid_results[["param_clf__max_depth"]].values, dtype='float64')
        tree_params_clf__min_samples_split = np.array(tree_grid_results[["param_clf__min_samples_split"]].values,
                                                      dtype='float64')
        tree_params_clf__max_features = np.array(tree_grid_results[["param_clf__max_features"]].values, dtype='float64')
        tree_params_mean_train_score = np.array(tree_grid_results[["mean_train_score"]].values, dtype='float64')
        tree_params_mean_test_score = np.array(tree_grid_results[["mean_test_score"]].values, dtype='float64')

        # Create 4D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(tree_params_clf__max_depth, tree_params_clf__min_samples_split, tree_params_clf__max_features,
                         c=np.reshape(tree_params_mean_train_score, len(tree_params_mean_train_score)), cmap=plt.hot())
        fig.colorbar(img)
        plt.title('Tree Training Performance')
        ax.set_xlabel('Maximum tree depth')
        ax.set_ylabel('Minimum sample split')
        ax.set_zlabel('Maximum features')
        plt.savefig(str(output_folder) + '/4D_tree_train.png')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        img = ax.scatter(tree_params_clf__max_depth, tree_params_clf__min_samples_split, tree_params_clf__max_features,
                         c=np.reshape(tree_params_mean_test_score, len(tree_params_mean_train_score)), cmap=plt.hot())
        fig.colorbar(img)
        plt.title('Tree Testing Performance')
        ax.set_xlabel('Maximum tree depth')
        ax.set_ylabel('Minimum sample split')
        ax.set_zlabel('Maximum features')
        plt.savefig(str(output_folder) + '/4D_tree_test.png')

    if classifier == 'VC':
        pass

def setup():
    # import the data
    df = import_data("Indian Liver Patient Dataset (ILPD).csv")

    # One hot encode gender classifier
    df = one_hot_encode(df, "gender")

    # Split data with test, train, split function
    X_train, y_train, X_test, y_test = split_data(df, 0.2, "is_patient")

    return X_train, y_train, X_test, y_test

def main():
    parser = argparse.ArgumentParser(description='Create machine learning visualizations')

    parser.add_argument('--classifier_type', type=str,
                        help='name of desired classifier', required=True)
    parser.add_argument('--output_folder', type=str,
                        help='name of the output folder for visualizations', required=True)

    args = parser.parse_args()

    # Run function
    learning_viz(args.classifier_type, args.output_folder)




if __name__ == '__main__':
    main()

