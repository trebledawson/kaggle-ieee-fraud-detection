# ############################################################################ #
# preprocessing.py                                                             #
# Author: Glenn Dawson (2019)                                                  #
# --------                                                                     #
# Utilities file for IEEE Fraud Detection Challenge (Kaggle).                  #
# ############################################################################ #

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    data = pd.read_csv('train_transaction.csv')
    test = pd.read_csv('test_transaction.csv')

    # Subtract common element from TransactionID
    data['TransactionID'] -= 2987000
    test['TransactionID'] -= 2987000

    # Subtract common element from TransactionDT
    data['TransactionDT'] -= 86400
    test['TransactionDT'] -= 86400

    # Leave TransactionAmt as float

    # Categorical encoding
    categorical = ['ProductCD'] + \
                  ['card' + str(i) for i in range(1, 7)] + \
                  ['addr' + str(i) for i in range(1, 3)] + \
                  ['P_emaildomain', 'R_emaildomain'] + \
                  ['M' + str(i) for i in range(1, 10)]
    for feature in categorical:
        categorical_encode(feature, data, test)

    # Fill missing values in distx columns with -1
    for col in ['dist' + str(i) for i in range(1, 3)]:
        data[col].fillna(-1, inplace=True)
        test[col].fillna(-1, inplace=True)

    # Fill missing values in Cx columns with -1
    for col in ['C' + str(i) for i in range(1, 15)]:
        data[col].fillna(-1, inplace=True)
        test[col].fillna(-1, inplace=True)

    # Replace missing and negative values in Dx columns with -1
    for col in ['D' + str(i) for i in range(1, 16)]:
        remove_negative(col, data)
        data[col].fillna(-1, inplace=True)
        test[col].fillna(-1, inplace=True)

    # Fill missing values in Vx columns with -10
    for col in ['V' + str(i) for i in range(1, 340)]:
        data[col].fillna(-10, inplace=True)
        test[col].fillna(-10, inplace=True)

    # Check class balance
    # 20663 instances of fraud, or 3.5% of the training dataset

    # Export new data files
    data.to_csv('train_transaction_clean.csv')
    test.to_csv('test_transaction_clean.csv')


def col_check(col, data, test):
    """
    Used for gathering column statistics.
    """
    is_numeric = pd.api.types.is_numeric_dtype(data[col])
    show_unique_elements = False
    show_set_differences = False
    show_indices_of_nan = False
    print('---')
    print('Statistics for column "' + col + '"')
    print('---')

    print('Train')
    print('Total length:', len(data[col]))
    data_set = set(data[col].unique())
    print('Number of unique elements:', len(data_set))
    print('Number of missing elements:', np.sum(pd.isnull(data[col])))
    if is_numeric:
        print('Number of negative elements:', sum(data[col] < 0))
    if show_unique_elements:
        print(data[col].unique())
    if show_indices_of_nan:
        print('Indices of NaN elements:',
              data.index[data[col].isnull() == True].tolist())
    print('---')

    print('Test')
    print('Total length:', len(test[col]))
    test_set = set(test[col].unique())
    print('Number of unique elements:', len(test_set))
    print('Number of missing elements:', np.sum(pd.isnull(test[col])))
    if is_numeric:
        print('Number of negative elements:', sum(test[col] < 0))
    if show_unique_elements:
        print(test[col].unique())
    print('---')

    print('Number of shared unique elements:',
          len(data_set & test_set))

    if show_set_differences:
        print('Elements in test but not in data:',
              test_set - data_set)
        print('Elements in data but not in test:',
              data_set - test_set)

    print('---')


def col_check_v(col, data, test, df):
    """
    Used for gathering columns statistics for Vx columns.
    """
    indices = ['train_total', 'train_unique', 'train_missing',
               'train_negative', 'test_total', 'test_unique', 'test_missing',
               'train_negative', 'shared_unique']
    df = pd.DataFrame(index=indices)
    for col in ['V' + str(i) for i in range(1, 340)]:
        print(col)
        check = []
        data_set = set(data[col].unique())
        test_set = set(test[col].unique())

        # Train
        check.append(len(data[col]))
        check.append(len(data_set))
        check.append(np.sum(pd.isnull(data[col])))
        check.append(sum(data[col] < 0))

        # Test
        check.append(len(test[col]))
        check.append(len(test_set))
        check.append(np.sum(pd.isnull(test[col])))
        check.append(sum(test[col] < 0))

        # Shared unique
        check.append(len(data_set & test_set))

        # Add to df
        df[col] = check
    df.to_csv('col_check_v.csv')


def categorical_encode(col, data, test):
    """
    Fill missing values and encode categorical data. Values seen in the
    testing data but not the training data are encoded as "unseen" regardless
    of their true value.
    """
    is_numeric = pd.api.types.is_numeric_dtype(data[col])
    print_library = False

    # Fill NaN with -1 (if numeric) or 'missing' (if string)
    if is_numeric:
        data[col].fillna(-1, inplace=True)
        test[col].fillna(-1, inplace=True)
    else:
        data[col].fillna('missing', inplace=True)
        test[col].fillna('missing', inplace=True)

    # Make library of seen values in training set
    if is_numeric:
        library = [int(i) for i in sorted(list(set(data[col])))]
    else:
        library = list(set(data[col]))

    # Replace unseen values in test set with -2 (if numeric) or 'unseen' (if
    # string).

    if is_numeric:
        library.append(-2)
        test.loc[~test[col].isin(library), col] = -2
    else:
        library.append('unseen')
        test.loc[~test[col].isin(library), col] = 'unseen'

    if print_library:
        print(library)
        print('---')

    # Encode labels
    le = LabelEncoder()
    le.fit(library)
    data[col] = le.transform(data[col])
    test[col] = le.transform(test[col])


def remove_negative(col, data):
    """
    Remove negative values from column. Only used in Dx columns in the
    training data.
    """
    is_numeric = pd.api.types.is_numeric_dtype(data[col])

    if is_numeric:
        data.loc[data[col] < 0, col] = -1

if __name__ == '__main__':
    main()