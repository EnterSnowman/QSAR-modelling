from tensorflow.python.keras import Sequential, Input, Model
from tensorflow.python.keras.layers import Dense, Dropout, Average
from keras.callbacks import EarlyStopping
from model.classification.data_loader import get_data
from utils.logger import save_experiment_results, save_log_lines, save_important_features_in_list
from utils.saver_loader import save_keras_model

from sklearn.preprocessing import MinMaxScaler


class NNTrain:
    def __init__(self):
        super().__init__()
    def train_neural_network(train_dataset, test_dataset, logs_folder, number_of_features=30, batch_size=64, num_epochs=50,
                             patience=4):
        features, X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset, number_of_features)
        log_lines = []
        log_lines.append("number of features: " + str(number_of_features))
        log_lines.append("batch size: " + str(batch_size))
        log_lines.append("epochs: " + str(num_epochs))
        log_lines.append("early stopping patience: " + str(patience))
        log_lines.append("")
        best_params = []
        best_acc = 0
        archs = get_learning_params()
        for arch in archs:
            # for num_epochs in list_of_epochs:
            #     log_lines.append("epochs: " + str(num_epochs))
            #     for batch_size in list_of_batches:

            log_lines.append("Layers:" + str(arch))
            model = Sequential()
            for i, l in enumerate(arch):
                activation = 'relu'
                if i == len(arch) - 1:
                    activation = 'sigmoid'
                if i == 0:
                    model.add(Dense(units=l, input_dim=number_of_features, activation=activation))
                else:
                    model.add(Dense(units=l, activation=activation))
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            early_stopping_monitor = EarlyStopping(patience=patience)
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=num_epochs, batch_size=batch_size,
                      callbacks=[early_stopping_monitor])
            acc = model.evaluate(X_test, Y_test)[1]
            log_lines.append("validation accuracy: " + str(acc))
            if acc > best_acc:
                best_acc = acc
                best_params = [batch_size, num_epochs, arch]
            log_lines.append("")
        log_lines.append("Best val_acc: " + str(best_acc))
        log_lines.append("Best params:" + str(best_params))
        save_log_lines(logs_folder, log_lines)
        # last_validation_accuracy = model.evaluate(X_test, Y_test)[1]
        # # print(last_validation_accuracy)
        # save_experiment_results('NN', summary, last_validation_accuracy, logs_folder,
        #                         {'sizes of layers': sizes_of_layers, 'epochs': num_epochs, 'batch size': batch_size,
        #
        #
        #
        #                     'number of features': number_of_features})


    def train_and_save_ensemble(number_of_features, train_dataset, test_dataset, model_folder, batch_size=64,
                                num_epochs=50,
                                patience=10):
        features, X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset, number_of_features,
                                                              test_size=0.1)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # ensemble
        nets = []
        nets.append([800, 400, 400, 1])
        nets.append([50, 100, 50, 1])
        nets.append([600, 600, 1])
        nets.append([300, 450, 350, 1])
        nets.append([50, 50, 1])
        nets.append([400, 20, 50, 1])

        input_shape = X_train[0].shape
        model_input = Input(shape=input_shape)
        models = []
        for net in nets:
            for i, l in enumerate(net):
                activation = 'relu'
                if i == len(net) - 1:
                    activation = 'sigmoid'
                if i == 0:
                    x = Dense(units=l, activation=activation)(model_input)
                else:
                    x = Dense(units=l, activation=activation)(x)
            model = Model(model_input, x)
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            model.summary()
            early_stopping_monitor = EarlyStopping(patience=patience)
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=num_epochs, batch_size=batch_size,
                      callbacks=[early_stopping_monitor])
            models.append(model)
        outputs = [model.outputs[0] for model in models]
        y = Average()(outputs)
        ensemble = Model(model_input, y, name='ensemble')
        ensemble.summary()
        ensemble.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
        acc = ensemble.evaluate(X_test, Y_test)[1]
        print("val accuracy: ", acc)
        save_keras_model(ensemble, model_folder)
        save_important_features_in_list(features, model_folder, "classifier")


    def train_and_save_final_model(number_of_features, train_dataset, test_dataset, model_folder, batch_size=64,
                                   num_epochs=50,
                                   patience=10):
        scores, indicies, features, X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset, number_of_features,
                                                              test_size=0.1)
        arch = [800, 400, 400, 1]
        for i, l in enumerate(arch):
            model = Sequential()
            activation = 'tanh'
            if i == len(arch) - 1:
                activation = 'sigmoid'
            if i == 0:
                model.add(Dense(units=l, input_dim=number_of_features, activation=activation))
            else:
                # model.add(Dropout(0.5))
                model.add(Dense(units=l, activation=activation))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        early_stopping_monitor = EarlyStopping(patience=patience)
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=num_epochs, batch_size=batch_size,
                  callbacks=[early_stopping_monitor])
        acc = model.evaluate(X_test, Y_test)[1]
        print("val accuracy: ", acc)
        save_keras_model(model, model_folder)
        save_important_features_in_list(features, model_folder, "classifier")


    def get_learning_params():
        archs_file = open("list_of_architectures.txt", "r")
        archs = [[int(i) for i in a.split(" ")] for a in archs_file.read().splitlines()]
        archs_file.close()
        return archs


# if __name__ == '__main__':
#     # f = [10, 20, 30, 50, 100, 200, 300, 350, 400]
#     # for fe in f:
#     #     train_neural_network('../../data/classification/ames_challenge_train_all_descriptors_v3.csv',
#     #                          '../../data/classification/ames_challenge_test_all_descriptors_v3.csv',
#     #                          '../../results/logs/classification/', fe, patience=3)
#
#     train_and_save_final_model(200, '../../data/classification/ames_challenge_train_all_descriptors_v3.csv',
#                                '../../data/classification/ames_challenge_test_all_descriptors_v3.csv',
#                                '../../ui/saved_predictors/ames_classifier', patience=5)
