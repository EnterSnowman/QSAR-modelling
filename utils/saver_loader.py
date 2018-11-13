from sklearn.externals import joblib
from keras.models import load_model


def save_sklearn_model(clf, name):
    joblib.dump(clf, name + '.joblib')


def load_sklearn_model(name):
    return joblib.load(name + '.joblib')


def save_keras_model(clf, name):
    clf.save(name + ".h5")


def load_keras_model(name):
    return load_model(name+".h5")
