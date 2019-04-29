import numpy as np
import sklearn
from sklearn_pandas import DataFrameMapper, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from .data_processing import preprocess_data


stateful_transforms = DataFrameMapper([
    (['emp_length'], StandardScaler()),
    (['annual_inc'], [FunctionTransformer(np.log10), StandardScaler()]),
    (['debt_to_income'], StandardScaler()),
    (['inq_last_6mths'], StandardScaler()),
    (['open_acc'], [FunctionTransformer(np.log10), StandardScaler()]),
    (['pub_rec'], StandardScaler()),
    (['revol_bal'], StandardScaler()),
    (['revol_util'], StandardScaler()),
    (['total_acc'], [FunctionTransformer(np.log10), StandardScaler()]),
    (['days_since_earliest_cr_line'], StandardScaler()),
    ('purpose_cat', sklearn.preprocessing.LabelBinarizer()),
    ('home_ownership', sklearn.preprocessing.LabelBinarizer()),
    ('verification_status', sklearn.preprocessing.LabelBinarizer()),
    ('policy_code', sklearn.preprocessing.LabelBinarizer())],
    input_df=True,
    df_out=True)


class DataProcesser:

    def __init__(self):
        self.stateful_transforms = stateful_transforms

    def fit_transform(self, df):
        X, y = preprocess_data(df)
        return self.stateful_transforms.fit_transform(X), y

    def transform(self, X):
        assert target_variable_column not in X.columns, "remove target_variable_column from X before transforming"
        X = preprocess_data(X)
        return self.stateful_transforms.transform(X)
