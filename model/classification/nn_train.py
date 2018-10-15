from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from keras.callbacks import EarlyStopping
from model.classification.data_loader import get_data
from utils.logger import save_experiment_results, save_log_lines


def train_neural_network(train_dataset, test_dataset, logs_folder, number_of_features=30, batch_size=64, num_epochs=50,
                         patience=4):
    X_train, Y_train, X_test, Y_test = get_data(train_dataset, test_dataset, number_of_features)
    log_lines = []
    log_lines.append("number of features: " + str(number_of_features))
    log_lines.append("batch size: " + str(batch_size))
    log_lines.append("epochs: " + str(num_epochs))
    log_lines.append("early stopping patience: " + str(patience))
    log_lines.append("")
    best_params = []
    best_acc = 0
    archs, _, _ = get_learning_params()
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
    #                          'number of features': number_of_features})


def get_learning_params():
    archs_file = open("list_of_architectures.txt", "r")
    archs = [[int(i) for i in a.split(" ")] for a in archs_file.read().splitlines()]
    archs_file.close()
    params_file = open("learning_params.txt", "r")
    l = params_file.read().splitlines()
    list_of_batches = [int(i) for i in l[0].split(" ")]
    list_of_epochs = [int(i) for i in l[1].split(" ")]
    return archs, list_of_batches, list_of_epochs


if __name__ == '__main__':
    train_neural_network('../../data/classification/ames_challenge_train_all_descriptors_v3.csv',
                         '../../data/classification/ames_challenge_test_all_descriptors_v3.csv', '../../results/logs/classification/', 200, patience=3)
    # get_archs_from_file()
