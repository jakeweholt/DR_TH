import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


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
        self.X_raw = X
        self.y_raw = y
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(
            self.X_transformed,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state)
        self._create_k_folds(k=5)

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


drop_na_rows_in_cols = ['delinq_2yrs',
                        'days_since_earliest_cr_line',
                        'inq_last_6mths',
                        'open_acc',
                        'pub_rec',
                        'total_acc',
                        'revol_util',
                        'purpose',
                        'annual_inc',
                        'emp_length']

drop_columns = ['collections_12_mths_ex_med',
                'pymnt_plan',
                'initial_list_status',
                'mths_since_last_record',
                'mths_since_last_delinq',
                'zip_code',
                'addr_state']


def clean_and_transform_training_data(df):
    """
    Takes a raw dataframe from the DR_Demo_Lending_Club.csv dataset, cleans and transforms data for modeling.
    Rational and methodology outlined in /analysis/01_data_cleaning.ipynb

    :param df: Raw data from the DR_Demo_Lending_Club.csv dataset.
    :return: A clean version of df, ready for modeling.
    """

    # Create new variable from earliest_cr_line(datetime) =
    # current_date - earliest_cr_line = days since earliest credit line
    current_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['days_since_earliest_cr_line'] = (current_date - df['earliest_cr_line']).dt.days
    df = df.drop('earliest_cr_line', axis=1)

    # Convert emp_length to float64 from str.
    df.loc[df['emp_length'] == 'na', 'emp_length'] = np.nan
    df['emp_length'] = df['emp_length'].astype('float64')

    # Drop NA Values from the following columns
    df = df.dropna(axis=0, subset=drop_na_rows_in_cols)
    assert len(df) > 0, "NA value found in row, df is empty."

    # Drop columns
    df = df.drop(drop_columns, axis=1)

    # Dropping very specific outliers/non-sensical rows
    df = df[df['revol_util'] <= 100]

    # Creating mapping to consolidate 'VERIFIED - income' and
    # 'VERIFIED - income source' as simply ''VERIFIED - income'
    value_map = {
        'VERIFIED - income': 'VERIFIED - income',
        'VERIFIED - income source': 'VERIFIED - income',
        'not verified': 'not verified'
    }
    df['verification_status'] = [value_map[x] for x in df['verification_status']]

    # Mapping lower represented groups to an 'other' bucket
    purpose_cat_count = df.groupby(['purpose_cat']).count()['Id']
    valid_values = list(purpose_cat_count[purpose_cat_count > 100].index)
    df['purpose_cat'] = [p if p in valid_values else 'other' for p in df['purpose_cat']]

    # Dropping following text columns for V1
    df = df.drop(['emp_title', 'Notes', 'purpose'], axis=1)

    # Splitting out categorical data to dummies:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    categorical_columns = df.select_dtypes(exclude=numerics)

    for col in categorical_columns:
        dummies = pd.get_dummies(categorical_columns[col])
        df = df.drop(col, axis=1)
        df = pd.concat([df, dummies], axis=1)

    # Finally drop Id columns
    df = df.drop('Id', axis=1)

    return df
