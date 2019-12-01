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

class TestFunctions(unittest.TestCase):
    
    def make_char_file(self):
        n = 5
        data = np.random.choice(list(string.ascii_lowercase),
                                     size=(n,n), replace = False)
        np.savetxt("test_data.csv", data, fmt= "%c", delimiter=",")
        return data
    
    def make_num_df(self):
        n = 5
        data = np.random.choice(100, size=(n,n), replace = False)
        return pd.DataFrame(data)
    
    def make_gender_df(self):
        n = 50
        gender = [random.choice(["Male", "Female"]) for i in range(n)]
        df = pd.DataFrame(gender, columns=["gender"])
        return df
    
    def test_heatmap(self):
        df = self.make_num_df()
        make_correlation_heatmap(df, "Test", "test.png")
        self.assertEqual(True, os.path.exists("test.png"))
         
    def test_one_hot(self):
        df = self.make_gender_df()
        df_enc = df.copy()
        df_enc["Male"] = df_enc["gender"].apply(lambda x: 1 if x == "Male" else 0)
        df_enc["Female"] = df_enc["gender"].apply(lambda x: 1 if x == "Female" else 0)
        df_enc.drop(columns = ["gender"],  inplace = True)
        pd.testing.assert_frame_equal(df_enc, one_hot_encode(df, "gender"),
                                      check_like = True, 
                                      check_dtype = False)

    def test_data_import(self):
        data = self.make_char_file()
        df = pd.DataFrame(data, columns= [x for x in data[0,:]])
        df = df.reindex(df.index.drop(0)).reset_index(drop=True)
        df_import = import_data("test_data.csv")
        pd.testing.assert_frame_equal(df, df_import, check_less_precise = 0) 

    
    def test_split(self):
        frac = 0.2
        data = self.make_char_file()
        n = len(data)
        df = pd.DataFrame(data, columns = [str(x) for x in range(n)])
        X_train, y_train, X_test, y_test = split_data(df, frac, str(n-1))
        
        self.assertEqual(np.shape(X_train), (round((1-frac)*n), n-1))
        self.assertEqual(np.shape(X_test), (round((frac)*n), n-1))
        self.assertEqual(np.shape(y_train), (round((1-frac)*n), ))
        self.assertEqual(np.shape(y_test), (round((frac)*n), ))
                                   
if __name__ == '__main__':        
    unittest.main()