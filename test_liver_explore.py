"""Unittesting framework for liver_explore.py
Parameters
----------
None
Returns
-------
None
"""

import unittest
import liver_explore
import os
import shutil


# Testing no file creation
class TestFileNonExistance(unittest.TestCase):

    def test_file_does_not_exist(self):
        shutil.rmtree('output_loc')
        self.assertFalse(os.path.exists('output_loc' + '/KNN.png'))
        self.assertFalse(os.path.exists(
            'output_loc' + '/2D_SVM_poly_test_deg.png'))
        self.assertFalse(os.path.exists(
            'output_loc' + '/2D_SVM_poly_test_gam.png'))
        self.assertFalse(os.path.exists(
            'output_loc' + '/3D_SVM_poly_test.png'))
        self.assertFalse(os.path.exists(
            'output_loc' + '/SVM_lin_test.png'))
        self.assertFalse(os.path.exists('output_loc' + '/SVM_lin_train.png'))
        self.assertFalse(os.path.exists('output_loc' + '/SVM_poly_test.png'))
        self.assertFalse(os.path.exists('output_loc' + '/SVM_poly_train.png'))
        self.assertFalse(os.path.exists('output_loc' + '/SVM_rbf_test.png'))
        self.assertFalse(os.path.exists('output_loc' + '/SVM_rbf_train.png'))
        self.assertFalse(os.path.exists('output_loc' + '/SVM_sig_test.png'))
        self.assertFalse(os.path.exists('output_loc' + '/SVM_sig_train.png'))
        self.assertFalse(os.path.exists('output_loc' + '/4D_tree_test.png'))
        self.assertFalse(os.path.exists('output_loc' + '/4D_tree_train.png'))


# Testing file creation
class TestFileExistance(unittest.TestCase):

    def test_file_exists_KNN(self):
        liver_explore.learning_viz('KNN', 'output_loc')
        self.assertTrue(os.path.exists('output_loc' + '/KNN.png'))

    def test_file_exists_SVM(self):
        liver_explore.learning_viz('SVM', 'output_loc')
        self.assertTrue(os.path.exists(
            'output_loc' + '/2D_SVM_poly_test_deg.png'))
        self.assertTrue(os.path.exists(
            'output_loc' + '/2D_SVM_poly_test_gam.png'))
        self.assertTrue(os.path.exists('output_loc' + '/3D_SVM_poly_test.png'))
        self.assertTrue(os.path.exists('output_loc' + '/SVM_lin_test.png'))
        self.assertTrue(os.path.exists('output_loc' + '/SVM_poly_train.png'))
        self.assertTrue(os.path.exists('output_loc' + '/SVM_sig_test.png'))
        self.assertTrue(os.path.exists('output_loc' + '/SVM_sig_test.png'))
        self.assertTrue(os.path.exists('output_loc' + '/SVM_rbf_train.png'))
        self.assertTrue(os.path.exists('output_loc' + '/SVM_rbf_test.png'))

    def test_file_exists_DT(self):
        liver_explore.learning_viz('DT', 'output_loc')
        self.assertTrue(os.path.exists('output_loc' + '/4D_tree_test.png'))
        self.assertTrue(os.path.exists('output_loc' + '/4D_tree_train.png'))


if __name__ == '__main__':
    unittest.main()
