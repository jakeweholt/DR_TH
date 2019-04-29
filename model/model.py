import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


class Model:

    def __init__(self, model_definition):
        self.model_definition = model_definition

    def fit(self, X, y, model_fit_parameters):
        self.X = X
        self.y = y
        self.model_definition.fit(X=X, y=np.asarray(y).reshape(-1,), **model_fit_parameters)

    def predict(self, X_new):
        return(self.model_definition.predict(X_new))

    def predict_proba(self, X_new):
        return(self.model_definition.predict_proba(X_new))


class ModelTrainingData:
    """
    ModelTrainingData is a convenience class for storing model training data.
    It takes cleaned and transformed model features and a target variable, and
    creates a training/validation data split (based on validation_size), then it 
    creates k_folds number of stratified folds for crossvalidation.

    :param X: Cleaned and transformed model features.
    :param y: Target variable.
    :param validation_size: The size of valition set (0-1).
    :param k_folds: Number of stratified folds to create.
    :param random_state: random state for train_test_split, and stratifiedKFolds.

    :field X_raw: Unsplit features.
    :field y_raw: Unsplit target variable.
    :field X_train: Training features.
    :field X_validate: Validation features.
    :field y_train: Training target variable.
    :field y_validate: Validation target variable.
    :field folds: Dictionary of k folds. Keys = ['fold_1', ..., 'fold_k']
    """

    def __init__(self, X, y, validation_size=0.15, k_folds=5, random_state=1234):
        self.validation_size = validation_size
        self.k_folds = k_folds
        self.random_state = random_state
        self.X_all_data = X
        self.y_all_data = y
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(
            X,
            y,
            test_size=validation_size,
            stratify=y,
            random_state=random_state)
        self._create_k_folds()

    def _create_k_folds(self):
        skf = StratifiedKFold(n_splits=self.k_folds, random_state=self.random_state, shuffle=True)
        i = 0
        self.folds = {}
        for train, test in skf.split(self.X_train, self.y_train):
            self.folds['fold_{}'.format(i)] = {
                'X_train': self.X_train.iloc[train],
                'y_train': self.y_train.iloc[train],
                'X_test': self.X_train.iloc[test],
                'y_test': self.y_train.iloc[test],
            }
            i += 1
