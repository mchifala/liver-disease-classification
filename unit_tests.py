import unittest
import numpy as np
import pandas as pd
import string
import random
import os

from liver_functions import import_data
from liver_functions import one_hot_encode
from liver_functions import split_data
from liver_functions import make_correlation_heatmap
from liver_functions import make_confusion_matrix
from liver_functions import evaluate_test_set


class TestFunctions(unittest.TestCase):

    def make_char_file(self):
        """
        This helper function makes a .csv file of random characters
        called test_data.csv

        Returns:
        - data(ndarray): An n x n array of random characters

        """
        n = 5
        data = np.random.choice(list(string.ascii_lowercase),
                                size=(n, n), replace=False)
        np.savetxt("test_data.csv", data, fmt="%c", delimiter=",")
        return data

    def make_num_df(self):
        """
        This helper function makes a n x n DataFrame of random integers
        between 0 to 100 with no repeats

        Returns:
        - df(DataFrame): A  n x n DataFrame of random integers

        """
        n = 5
        data = np.random.choice(100, size=(n, n), replace=False)
        return pd.DataFrame(data)

    def make_gender_df(self):
        """
        This helper function makes a n x 1 DataFrame of "male" and
        "female" entries

        Returns:
        - df(DataFrame): a n x 1 DataFrame of "male" and "female" entries

        """
        n = 50
        gender = [random.choice(["Male", "Female"]) for i in range(n)]
        df = pd.DataFrame(gender, columns=["gender"])
        return df

    def test_heatmap(self):
        """
        This test function ensures that the make_correlation_heatmap
        function creates a .png file containing the heat map

        """
        if os.path.exists("heat.png"):
            os.remove("heat.png")
        df = self.make_num_df()
        make_correlation_heatmap(df, "Test", "heat.png")
        self.assertEqual(True, os.path.exists("heat.png"))

    def test_one_hot(self):
        """
        This test function ensures that the one_hot_encode
        functions properly encodes the chosen feature

        """
        df = self.make_gender_df()
        df_enc = df.copy()
        df_enc["Male"] = df_enc["gender"].apply(lambda x: 1 if x == "Male" else 0)  # noqa: E501
        df_enc["Female"] = df_enc["gender"].apply(lambda x: 1 if x == "Female" else 0)  # noqa: E501
        df_enc.drop(columns=["gender"],  inplace=True)
        pd.testing.assert_frame_equal(df_enc, one_hot_encode(df, "gender"),
                                      check_like=True,
                                      check_dtype=False)

    def test_data_import(self):
        """
        This test_function ensures that the import_data function
        properly imports data into a DataFrame

        """
        data = self.make_char_file()
        df = pd.DataFrame(data, columns=[x for x in data[0, :]])
        df = df.reindex(df.index.drop(0)).reset_index(drop=True)
        df_import = import_data("test_data.csv")
        pd.testing.assert_frame_equal(df, df_import, check_less_precise=0)

    def test_split(self):
        """
        This test function ensures that the split_data function
        properly splits the original data set into X_train, y_train,
        X_test, and y_test subsets.

        """
        frac = 0.2
        data = self.make_char_file()
        n = len(data)
        df = pd.DataFrame(data, columns=[str(x) for x in range(n)])
        X_train, y_train, X_test, y_test = split_data(df, frac, str(n-1))

        self.assertEqual(np.shape(X_train), (round((1-frac)*n), n-1))
        self.assertEqual(np.shape(X_test), (round((frac)*n), n-1))
        self.assertEqual(np.shape(y_train), (round((1-frac)*n), ))
        self.assertEqual(np.shape(y_test), (round((frac)*n), ))

    def test_confusion_matrix(self):
        """
        This test function ensures that the make_confusion_matrix
        function creates a .png file containing the confusion matrix

        """
        if os.path.exists("cm.png"):
            os.remove("cm.png")
        y_pred = np.random.choice(2, 100)
        y_true = np.random.choice(2, 100)
        make_confusion_matrix(y_true, y_pred, "cm.png")
        self.assertEqual(True, os.path.exists("cm.png"))

    def test_f1_score_one(self):
        """
        This test function ensures that the evaluate_test_set
        function properly calls test_confusion_matrix and
        returns an f1_score of 1

        """
        y_pred = [1, 1, 0, 0]
        y_true = [1, 1, 0, 0]
        f1 = evaluate_test_set(y_true, y_pred, "cm.png")
        self.assertEqual(1, f1)

    def test_f1_score_zero(self):
        """
        This test function ensures that the evaluate_test_set
        function properly calls test_confusion_matrix and
        returns an f1_score of 0

        """
        y_pred = [0, 0, 1, 1]
        y_true = [1, 1, 0, 0]
        f1 = evaluate_test_set(y_true, y_pred, "cm.png")
        self.assertEqual(0, f1)


if __name__ == '__main__':
    unittest.main()
