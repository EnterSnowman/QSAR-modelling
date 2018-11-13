import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from model.classification.data_loader import get_data


class Train:
    def __init__(self):
        super().__init__()

    def train_multiple(train_dataset, test_dataset, number_of_features=30):
        features, X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset, number_of_features)
        grids = []
        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # create cross-validation split
        cv = 5
        n_jobs = 1
        # logistic regression
        clf = LogisticRegression()
        params = {'penalty': ['l1', 'l2']}
        gr = get_grid(clf, params, cv, n_jobs)
        grids.append(gr)

        # # svm
        # C = [0.01, 0.1, 1, 10, 100, 1000]
        # gamma = [0.1, 1, 10]
        # kernel = ['rbf', 'linear', 'sigmoid']
        # clf = SVC()
        # params = {'C': C, 'gamma': gamma, 'kernel': kernel}
        # gr = GridSearchCV(clf, params, cv=5, n_jobs=2, verbose=10)
        # grids.append(gr)

        # nearest neighbours
        params = {'n_neighbors': [2, 3, 5, 7], 'weights': ['uniform', 'distance']}
        clf = KNeighborsClassifier()
        gr = get_grid(clf, params, cv, n_jobs)
        grids.append(gr)

        # decision tree
        clf = DecisionTreeClassifier()
        params = {'max_depth': np.arange(3, 11, 2), 'min_samples_leaf': np.arange(3, 15, 3)}
        gr = get_grid(clf, params, cv, n_jobs)
        grids.append(gr)

        # ada boost
        DTC = DecisionTreeClassifier(random_state=11, max_features="auto", class_weight="balanced", max_depth=None)
        clf = AdaBoostClassifier(DTC)
        params = {'n_estimators': [50, 100, 150], 'learning_rate': [1.0, 0.1, 0.01]}
        gr = get_grid(clf, params, cv, n_jobs)
        grids.append(gr)

        # # GradientBoostingClassifier
        # clf = GradientBoostingClassifier()
        # params = {'n_estimators': [50, 100, 150], 'learning_rate': [1.0, 0.1, 0.01], 'max_depth': [1, 5, 10]}
        # gr = get_grid(clf, params, cv, n_jobs)
        # grids.append(gr)
        #
        # # random forest
        # clf = RandomForestClassifier()
        # params = {'n_estimators': [50, 100, 150]}
        # gr = get_grid(clf, params, cv, n_jobs)
        # grids.append(gr)
        for grid in grids:
            grid.fit(X_train, Y_train)
            print(grid.best_estimator_, " -> ", grid.best_score_)


    def get_grid(clf, params, cv, n_jobs=1, verbose=0):
        return GridSearchCV(clf, params, cv=cv, n_jobs=n_jobs, verbose=verbose)


# if __name__ == '__main__':
#     features = [300, 400, 500]
#     for f in features:
#         train_multiple('../../data/classification/ames_challenge_train_all_descriptors_v3.csv',
#                    '../../data/classification/ames_challenge_test_all_descriptors_v3.csv', f)
