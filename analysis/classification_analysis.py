import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from model.classification.data_loader import get_data
from utils.saver_loader import load_keras_model


class Analysis:
    def __init__(self):
        super().__init__()

    def show_correlations(self, file):
        data = pd.read_csv(file)
        data = data.drop(data.columns[0], axis=1)
        data['AMES {measured}'].replace(to_replace={'active': 1, 'inactive': 0}, inplace=True)
        corrmat = data.corr()

        k = 11
        nlargest = corrmat.abs().nlargest(k, ['AMES {measured}'])
        important_corr = nlargest[nlargest.index]
        print(important_corr)
        self.plot_correlations(important_corr)

    def show_pair_relations(self, data, features_to_show, label, multiwindow=False):
        # data = pd.read_csv(file)
        # data = data.drop(data.columns[0], axis=1)
        # data['AMES {measured}'].replace(to_replace={'active': 1, 'inactive': 0}, inplace=True)
        #
        #
        # fig = plt.figure(figsize=(8, 8))
        colors = ['red', 'green']
        f_len = features_to_show.shape[0]

        if multiwindow:
            for i in range(0, f_len):
                for j in range(0, f_len):
                    plt.figure(i * 10 + j)
                    plt.title(features_to_show[i] + " : " + features_to_show[j])
                    plt.scatter(data[features_to_show[i]], data[features_to_show[j]], c=data['AMES {measured}'],
                                cmap=matplotlib.colors.ListedColormap(colors))
                    plt.xlabel(features_to_show[i])
                    plt.ylabel(features_to_show[j])
            plt.show()
        else:
            f, axes = plt.subplots(f_len, f_len)
            f.suptitle('Relations')
            for i in range(0, f_len):
                for j in range(0, f_len):
                    axes[i, j].scatter(data[features_to_show[i]], data[features_to_show[j]], c=data['AMES {measured}'],
                                       cmap=matplotlib.colors.ListedColormap(colors))
                    axes[i, j].set_xlabel(features_to_show[i])
                    axes[i, j].set_ylabel(features_to_show[j])
            # plt.scatter(data['SlogP_VSA8'], data['fr_benzene'], c=data['AMES {measured}'],
            #             cmap=matplotlib.colors.ListedColormap(colors))
            plt.show()

    def plot_correlations(self, df):
        sns.heatmap(df, annot=True)
        plt.show()

    # def show_important_features(file, number_of_output_features=6, multiwindow=False):
    #     data = pd.read_csv(file)
    #     data.dropna(inplace=True)
    #     data = data.drop([data.columns[0], 'SMILES'], axis=1)
    #     data['AMES {measured}'].replace(to_replace={'active': 1, 'inactive': -1}, inplace=True)
    #     X = data.drop(columns=[data.columns[0], 'AMES {measured}'])
    #     Y = data[['AMES {measured}']]
    #     selector = SelectKBest(f_classif, k=number_of_output_features)
    #     selector.fit(X, Y)
    #     X_new = selector.transform(X)
    #     features_to_show = X.columns[selector.get_support(indices=True)].base
    #     # show_pair_relations(data, features_to_show, 'AMES {measured}', multiwindow)
    #     # print(X_new)
    #     # print(selector.get_support(indices=True))
    #     # print(X.columns[selector.get_support(indices=True)])
    #     # print(X_new[:4, :2])
    #     # print(X.iloc[:4, selector.get_support(indices=True)[:2]])

    # show_correlations('../data/classification/ames_challenge_train_all_descriptors_v2.csv')
    # show_pair_relations('../data/classification/ames_challenge_train_all_descriptors_v2.csv')
    # show_important_features('../data/classification/ames_challenge_train_all_descriptors_v3.csv', 3)

    def plot_classification_hyperplanes(self, train_dataset, test_dataset, number_of_features):
        scores, indicies, features, X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset,
                                                                                number_of_features)
        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)
        features_to_plot = scores.argsort()[-10:-6]
        print(features_to_plot.shape)
        clf = load_keras_model('ames_classifier')
        y_pred = clf.predict(X_test)
        colors1 = ['red', 'green']
        colors2 = ['yellow', 'blue']
        i = 0
        for f1 in features_to_plot:
            for f2 in features_to_plot:
                if f1 != f2:
                    plt.figure(i, figsize=(10, 7))
                    i += 1

                    plt.subplot(2, 1, 1)
                    plt.title("Distribution of real and predicted values")
                    plt.scatter(X_test[:, f1], X_test[:, f2], c=Y_test,
                                cmap=matplotlib.colors.ListedColormap(colors1))
                    plt.xlabel(features[f1])
                    plt.ylabel(features[f2])

                    plt.subplot(2, 1, 2)
                    plt.scatter(X_test[:, f1], X_test[:, f2], c=y_pred.ravel(),
                                cmap=matplotlib.colors.ListedColormap(colors2))
                    plt.xlabel(features[f1])
                    plt.ylabel(features[f2])

        plt.show()


# if __name__ == '__main__':
#     plot_classification_hyperplanes('../data/classification/ames_challenge_train_all_descriptors_v3.csv',
#                                     '../data/classification/ames_challenge_test_all_descriptors_v3.csv', 200)
