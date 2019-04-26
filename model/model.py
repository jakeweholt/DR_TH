from data_transformers import RawDataTransformer
from model_training import ModelTrainingData
import numpy as np

class Model:

    def __init__(self, model_data, model_definiton):
        self.target_variable_column = 'is_bad'
        self.raw_data_transformer = RawDataTransformer()
        self.model_definiton = model_definiton
        X_transformed, y = self.raw_data_transformer.fit_transform(model_data)
        self.model_training_data = ModelTrainingData(X_transformed, y)

    def fit_all_data(self, model_fit_parameters):
        X = self.model_training_data.X_raw
        y = self.model_training_data.y_raw
        self.model_definiton.fit(X=X, y=np.asarray(y).reshape(-1,), **model_fit_parameters)

    def predict(X, self):
        pass
