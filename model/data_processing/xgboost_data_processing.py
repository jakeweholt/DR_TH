import sklearn
from sklearn_pandas import DataFrameMapper
from .data_processing import preprocess_data

stateful_transforms = DataFrameMapper([
    ('emp_length', None),
    ('annual_inc', None),
    ('debt_to_income', None),
    ('delinq_2yrs', None),
    ('inq_last_6mths', None),
    ('open_acc', None),
    ('pub_rec', None),
    ('revol_bal', None),
    ('revol_util', None),
    ('total_acc', None),
    ('mths_since_last_major_derog', None),
    ('days_since_earliest_cr_line', None),
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
