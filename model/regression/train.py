from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from model.regression.data_loader import get_data
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from utils.logger import save_log_lines, save_important_features_in_list
from utils.saver_loader import save_sklearn_model


def train_multiple(train_dataset, test_dataset, logs_folder, number_of_features=30):
    features, X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset, number_of_features)
    grids = []
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    log_lines = ["number of features: " + str(number_of_features)]
    cv = 5
    # simple linear models

    clf = LinearRegression()
    params = {'normalize': [True, False]}
    gr = get_grid(clf, params, cv)
    grids.append(gr)

    clf = Lasso()
    params = {'alpha': [1.0, 0.1, 0.01]}
    gr = get_grid(clf, params, cv)
    grids.append(gr)

    clf = Ridge()
    params = {'alpha': [1.0, 0.1, 0.01]}
    gr = get_grid(clf, params, cv)
    grids.append(gr)

    clf = ElasticNet()
    params = {'alpha': [1.0, 0.1, 0.01], 'l1_ratio': [0.2, 0.5, 0.8]}
    gr = get_grid(clf, params, cv)
    grids.append(gr)

    clf = BayesianRidge()
    params = {'alpha_1': [1e-06, 1e-05, 1e-04, 1e-03], 'alpha_2': [1e-06, 1e-05, 1e-04, 1e-03],
              'lambda_1': [1e-06, 1e-05, 1e-04, 1e-03], 'lambda_2': [1e-06, 1e-05, 1e-04, 1e-03], }
    gr = get_grid(clf, params, cv)
    grids.append(gr)

    # ensembles

    clf = RandomForestRegressor()
    params = {'n_estimators': [100, 300, 500, 1000]}
    gr = get_grid(clf, params, cv)
    grids.append(gr)

    clf = GradientBoostingRegressor()
    params = {'n_estimators': [100, 300, 500, 1000], 'learning_rate': [1.0, 0.1, 0.01]}
    gr = get_grid(clf, params, cv)
    grids.append(gr)
    best_est = ""
    best_score = -11111111.
    for grid in grids:
        grid.fit(X_train, Y_train)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_est = str(grid.best_estimator_)
        log_lines.append(str(grid.best_estimator_))
        log_lines.append("Score: " + str(grid.best_score_))
        log_lines.append("\n")
    log_lines.append("Best model: " + best_est)
    log_lines.append("Best score: " + str(best_score))
    save_log_lines(logs_folder, log_lines)


def train_and_save_final_model(number_of_features, train_dataset, test_dataset, model_folder):
    scores, indicies, features, X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset, number_of_features)
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    clf = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1)
    clf.fit(X_train, Y_train)
    print(r2_score(Y_test, clf.predict(X_test)))
    save_sklearn_model(clf, model_folder)
    save_important_features_in_list(features, model_folder, "regressor")


def get_grid(clf, params, cv, n_jobs=1, verbose=0):
    return GridSearchCV(clf, params, cv=cv, n_jobs=n_jobs, verbose=verbose)


if __name__ == '__main__':
    # num_features = [10]
    # for f in num_features:
    #     train_multiple('../../data/regression/t_pyriformis_train_all_descriptors.csv',
    #                    '../../data/regression/t_pyriformis_test_all_descriptors.csv',
    #                    '../../results/logs/regression/',
    #                    f)
    train_and_save_final_model(350, '../../data/regression/t_pyriformis_train_all_descriptors.csv',
                               '../../data/regression/t_pyriformis_test_all_descriptors.csv',
                               '../../ui/saved_predictors/toxicity_regressor')
