import datetime
import numpy as np
import pandas as pd

target_variable_column = 'is_bad'

na_rows_to_drop = ['delinq_2yrs',
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
                'addr_state',
                'emp_title',
                'Notes',
                'purpose',
                'Id']

valid_purpose_cats_values = ['debt consolidation',
                             'credit card',
                             'other',
                             'home improvement',
                             'major purchase',
                             'small business',
                             'car',
                             'wedding',
                             'medical',
                             'moving',
                             'educational',
                             'debt consolidation small business']

verification_status_map = {
    'VERIFIED - income': 'VERIFIED - income',
    'VERIFIED - income source': 'VERIFIED - income',
    'not verified': 'not verified'
}

def preprocess_data(df):
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
    df = df.dropna(axis=0, subset=na_rows_to_drop)
    assert len(df) > 0, "All rows filtered out due to NA drop row, df is empty. Check values in %s" % na_rows_to_drop

    # Drop columns
    df = df.drop(drop_columns, axis=1)

    # Dropping very specific outliers/non-sensical rows
    df = df[df['revol_util'] <= 100]
    assert len(df) > 0, "revol_util out of bounds, valid input = [0,100]"

    # Creating mapping to consolidate 'VERIFIED - income' and
    # 'VERIFIED - income source' as simply ''VERIFIED - income'
    try:
        df['verification_status'] = [verification_status_map[x] for x in df['verification_status']]
    except KeyError:
        print('Invalid verification_status')

    # Mapping lower represented groups to an 'other' bucket
    df['purpose_cat'] = [p if p in valid_purpose_cats_values else 'other' for p in df['purpose_cat']]

    assert len(df) > 0, "Final df is empty"

    if target_variable_column in df.columns:
        X = df.drop(target_variable_column, axis=1)
        y = df.filter([target_variable_column])
        return X, y
    else:
        X = df
        return X