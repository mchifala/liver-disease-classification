import pandas as pd
import sklearn as sk
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def import_data(csv_file):
    """
    This function imports data from a .csv file as a pandas dataframe
    
    Parameters:
    - csv_file(str): A .csv file path
    
    Returns:
    - df(DataFrame): A pandas DataFrame containing features and class labels
    
    """
    return pd.read_csv(csv_file)

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
    df_enc = pd.get_dummies(df[col_label])
    df = pd.concat([df_enc, df], axis = 1)
    df.drop([col_label], axis = 1, inplace = True)
    
    return df

def split_data(df, test_size, label_col):
    """
    This function splits a pandas DataFrame into X_train, X_test, y_train, and
    y_test sets according to user-defined percentage.
    
    Parameters:
    - df(DataFrame): A pandas DataFrame containing features and class labels
    - test_size(float): The fraction of original DataFrame held out for testing. Ex. 0.2
    - label_col(str): The column of the DataFrame that contains the class labels
    
    Returns:
    - X_train(ndarray): A numpy array containing 1-test_size fraction of 
                        the original dataset
    - y_train(ndarray): A numpy array containing the class labels for X_train
    - X_test(ndarray): A numpy array containing test_size fraction of 
                       the original dataset
    - y_test(ndarray): A numpy array containing the class labels for X_test
    
    """
    train, test = train_test_split(df, test_size = test_size)
    X_train = train.loc[:, train.columns != label_col].values
    y_train = train[label_col].values
    X_test = test.loc[:, train.columns != label_col].values
    y_test = test[label_col].values
    
    return X_train, y_train, X_test, y_test