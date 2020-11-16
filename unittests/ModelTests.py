#!/usr/bin/env python
"""
model tests
"""


import unittest
## import model specific functions and variables
from model import *

root_path = r"C:\Users\DineshNaik\IBM-AI-Learning\Aavail-Capstone-Proj"
class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """
        data_dir = os.path.join(root_path, "data", "cs-train")
        ## train the model
        model_train(data_dir,test=True)
        self.assertTrue(os.path.exists(SAVED_MODEL))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## load the model
        model = model_load(training=True)
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

    def test_03_predict(self):
        """
        test the predict functionality
        """

        ## load model first
        model = model_load(training=True)
        
        ## example predict
        country = 'United Kingdom'
        year = '2018'
        month = '01'
        day = '05'

        result = model_predict(country,year,month,day,model)
        y_pred = result['y_pred']
        self.assertTrue(y_pred in dir(model))

        
### Run the tests
if __name__ == '__main__':
    unittest.main()
