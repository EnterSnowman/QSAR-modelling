import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif, GenericUnivariateSelect, mutual_info_classif


def get_data(train_dataset, test_dataset, number_of_important_features=5):
    # read data and drop na
    train = pd.read_csv(train_dataset)
    train.dropna(inplace=True)
    test = pd.read_csv(test_dataset)
    test.dropna(inplace=True)

    # form train
    X_train = train.drop(columns=[train.columns[0], 'SMILES', 'log(IGC50-1) {measured, converted}'])
    Y_train = train[['log(IGC50-1) {measured, converted}']]

    # get k most important features
    # selector = GenericUnivariateSelect(mutual_info_classif, mode='k_best', param=number_of_important_features)
    selector = SelectKBest(f_classif, k=number_of_important_features)
    print(X_train.values.shape)
    print(Y_train.values.shape)
    selector.fit(X_train, Y_train)
    X_new = selector.transform(X_train.values)
    print(X_new.shape)
    print(selector.get_support(indices=True))
    indicies = selector.get_support(indices=True)

    X_train = X_train.iloc[:, indicies]

    X_test = test.drop(columns=[test.columns[0], 'SMILES', 'log(IGC50-1) {measured, converted}'])
    Y_test = test[['log(IGC50-1) {measured, converted}']]

    X_test = X_test.iloc[:, indicies]
    X_train, X_test, Y_train, Y_test = train_test_split(np.vstack((X_train.values, X_test.values)),
                                                        np.vstack((Y_train.values, Y_test.values)),
                                                        test_size=0.1)
    return X_train, Y_train.ravel(), X_test, Y_test.ravel()
