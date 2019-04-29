### Train Model . py
from model.data_processing.xgboost_data_processing import DataProcesser as XGBoostDataProcesser
from model.data_processing.logistic_regression_data_processing import DataProcesser as LRDataProcesser
from model.model import Model, ModelTrainingData
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier

import os
import pandas as pd
import pickle
import time


# Helper function to save out pickle object
def save_obj(obj, directory_path, file_name):
    pickle.dump(obj, open('{}/{}.p'.format(directory_path, file_name), 'wb'))


def train_model(model_type):
    df = pd.read_csv('data/DR_Demo_Lending_Club.csv')
    model_version = '0.1.%d' % int(time.time()) # creates new patch version when ran.
    save_directory = 'saved_models/{}/{}'.format(model_type, model_version)

    # Make new directory if one does not exist.
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if model_type == 'xgboost_model':
        d = XGBoostDataProcesser()
        model_definition = XGBClassifier(n_estimators=100, scale_pos_weight=6.77)
        model_fit_parameters = {
            'eval_metric': 'logloss'
        }
    elif model_type == 'logistic_regression_model':
        d = LRDataProcesser()
        model_definition = SGDClassifier(loss='log', penalty="l2", n_iter=1000)
        model_fit_parameters = {}

    # Run raw data through data transformers
    X, y = d.fit_transform(df)

    # Split data into Train, Validate, and K Folds, store for later.
    mtd = ModelTrainingData(X, y)

    # Save out ModelTrainingData and DataProcesser
    save_obj(obj=d, directory_path=save_directory, file_name='data_processer')
    save_obj(obj=mtd, directory_path=save_directory, file_name='training_data')

    # Train on all data model
    model_name = 'model_trained_on_all_data'
    m = Model(model_definition)
    m.fit(mtd.X_all_data, mtd.y_all_data, model_fit_parameters)
    save_obj(obj=m, directory_path=save_directory, file_name=model_name)

    # Train verification model
    model_name = 'model_trained_on_validation_data'
    m = Model(model_definition=model_definition)
    m.fit(mtd.X_train, mtd.y_train, model_fit_parameters)
    save_obj(obj=m, directory_path=save_directory, file_name=model_name)

    # Train k-fold models
    for fold in mtd.folds:
        X = mtd.folds[fold]['X_train']
        y = mtd.folds[fold]['y_train']
        model_name = 'model_trained_on_k_{}_data'.format(fold)

        m = Model(model_definition=model_definition)
        m.fit(X, y, model_fit_parameters)
        save_obj(obj=m, directory_path=save_directory, file_name=model_name)


if __name__ == "__main__":
    model_types = ['logistic_regression_model', 'xgboost_model']
    for model_type in model_types:
        train_model(model_type)
