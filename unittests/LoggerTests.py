#!/usr/bin/env python
"""
model tests
"""

import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
from datetime import date
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from logger import update_train_log, update_predict_log



class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs", "train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## YOUR CODE HERE
        ## Call the update_train_log() function from logger.py with arbitrary input values and test if the log file 
        ## exists in you file system using the assertTrue() base method from unittest.
        ## update the log
        data_shape = (100, 10)
        eval_test = {'rmse': 0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"

        update_train_log(self,data_shape, eval_test, runtime,
                         model_version, model_version_note, test=True)

        self.assertTrue(os.path.exists(log_file))

        
    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """
        
        log_file = os.path.join("logs", "train-test.log")
        
        ## YOUR CODE HERE
        ## Log arbitrary values calling update_train_log from logger.py. Then load the data
        ## from this log file and assert that the loaded data is the same as the data you logged.
        ## update the log
        data_shape = (100, 10)
        eval_test = {'rmse': 0.5}
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"

        update_train_log(self,data_shape, eval_test, runtime,
                         model_version, model_version_note, test=True)

        df = pd.read_csv(log_file)
        logged_eval_test = [literal_eval(i) for i in df['eval_test'].copy()][-1]
        self.assertEqual(eval_test, logged_eval_test)


    def test_03_predict(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs", "predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## YOUR CODE HERE
        ## Call the update_predict_log() function from logger.py with arbitrary input values and test if the log file 
        ## exists in you file system using the assertTrue() base method from unittest.
        ## update the log
        y_pred = [0]
        y_proba = [0.6, 0.4]
        runtime = "00:00:02"
        model_version = 0.1
        query = ['United Kingdom', '2018', '01', '05']

        update_predict_log(self,y_pred, y_proba, query, runtime,
                           model_version, test=True)

        self.assertTrue(os.path.exists(log_file))

    
    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """ 

        log_file = os.path.join("logs", "predict-test.log")

        ## YOUR CODE HERE
        ## Log arbitrary values calling update_predict_log from logger.py. Then load the data
        ## from this log file and assert that the loaded data is the same as the data you logged.
        y_pred = [0]
        y_proba = [0.6, 0.4]
        runtime = "00:00:02"
        model_version = 0.1
        query = ['United Kingdom', '2018', '01', '05']

        update_predict_log(self,y_pred, y_proba, query, runtime,
                           model_version, test=True)

        df = pd.read_csv(log_file)
        logged_y_pred = [literal_eval(i) for i in df['y_pred'].copy()][-1]
        self.assertEqual(y_pred,logged_y_pred)

### Run the tests
if __name__ == '__main__':
    unittest.main()
      
