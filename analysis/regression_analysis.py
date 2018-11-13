import itertools

import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import r2_score

from model.regression.data_loader import get_data
from utils.saver_loader import load_sklearn_model
from scipy import stats


class Analysis:
    def __init__(self):
        super().__init__()

    def plot_regression_lines(elf, train_dataset, test_dataset, number_of_features):
        scores, indicies, features, X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset,
                                                                                number_of_features)
        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)
        features_to_plot = scores.argsort()[-110:-100]
        print(features_to_plot.shape)
        clf = load_sklearn_model('toxicity_regressor')
        print(clf)
        # clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        ff = ['GETAWAY_3', 'MolLogP', 'MORSE_81', 'MORSE_124']
        meas = {'GETAWAY_3': ', µm', 'MolLogP': '', 'MORSE_81': ', log(µm)', 'MORSE_124': ', log(µm)'}
        print(r2_score(Y_test, y_pred))
        i = 0
        for f in ff:
            plt.figure(i, figsize=(8, 6))
            i += 1
            plt.xlabel(f + meas[f])
            plt.ylabel('log(IGC50)^(-1), -log(mmol/L)')
            slope, intercept, r_value, p_value, std_err = stats.linregress(X_test[:, features.index(f)], Y_test)
            line = slope * X_test[:, features.index(f)] + intercept

            plt.plot(X_test[:, features.index(f)], Y_test, 'b.', X_test[:, features.index(f)], line, markersize=10,
                     label=u'Observations')
            lists = sorted(zip(*[X_test[:, features.index(f)], y_pred]))
            new_x, new_y = list(zip(*lists))

            # plt.plot(new_x, new_y, 'r', label=u'Prediction')
        plt.show()

#
# if __name__ == '__main__':
#     plot_regression_lines('../data/regression/t_pyriformis_train_all_descriptors.csv',
#                           '../data/regression/t_pyriformis_test_all_descriptors.csv', 350)
