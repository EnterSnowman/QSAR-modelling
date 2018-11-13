from datetime import datetime


def save_experiment_results(model_type, model_summary, validation_accuracy, logs_folder, another_params=None):
    t = datetime.now()

    f = open(str(logs_folder + model_type + " " + t.strftime("%d_%m_%Y %H_%M_%S")) + ".txt", "x")
    f.write(str(model_summary) + '\n')
    f.write("validation accuracy : " + str(validation_accuracy) + "\n")
    for k in another_params.keys():
        f.write(str(k) + ' : ' + str(another_params[k]) + "\n")
    f.close()


def save_log_lines(logs_folder, log_lines):
    t = datetime.now()
    f = open(str(logs_folder + "grid_search " + t.strftime("%d_%m_%Y %H_%M_%S")) + ".txt", "x")
    f.write('\n'.join(log_lines))
    f.close()


def save_important_features_in_list(features, folder, t):
    f = open(str(folder + "/" + t + "_" + str(len(features))) + ".txt", "w")
    f.write('\n'.join(features))
    f.close()


def get_important_features_from_list(name):
    features_file = open(name, "r")
    features = [a for a in features_file.read().splitlines()]
    features_file.close()
    return features
