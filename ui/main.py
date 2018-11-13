import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets

import design
import numpy as np
import subprocess
import time
from utils.calculation import compute_features
from utils.logger import get_important_features_from_list
from utils.saver_loader import load_keras_model, load_sklearn_model

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


class MolWatcher:
    DIRECTORY_TO_WATCH = "mol"

    def __init__(self, app):
        self.observer = Observer()
        self.app = app

    def run(self):
        handler = MolHandler(self.app)
        self.observer.schedule(handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        # try:
        #     while True:
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     self.observer.stop()
        # self.observer.join()


class MolHandler(PatternMatchingEventHandler):
    patterns = ["*.smi"]

    def __init__(self, app):
        super().__init__()
        self.app = app

    def process(self, event):
        """
        event.event_type
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        # the file will be processed there
        print(event.src_path, event.event_type)
        mol_file = open(event.src_path, "r")
        mol = [a for a in mol_file.read().splitlines()]
        print("Mol", mol)
        if len(mol) > 0:
            self.app.put_smiles_to_input(mol[0])
        mol_file.close()
        # print now only for degug

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)


class QSARApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.compute_button.clicked.connect(self.compute_and_show_properties)
        self.editor_button.clicked.connect(self.show_molecule_editor)
        self.ames_classifier = load_keras_model("saved_predictors/ames_classifier")
        self.toxicity_regressor = load_sklearn_model("saved_predictors/toxicity_regressor")
        self.ames_features = get_important_features_from_list("saved_predictors/ames_classifier/classifier_200.txt")
        self.tox_features = get_important_features_from_list("saved_predictors/toxicity_regressor/regressor_350.txt")
        print(self.ames_classifier)
        print(self.toxicity_regressor)

        self.watcher = MolWatcher(self)
        self.watcher.run()
        # ames_inputs = compute_features("OCC1=CC=CO1", self.ames_features)
        # print(ames_inputs.shape)
        # print(self.ames_classifier.predict(np.array([ames_inputs])))

    def compute_and_show_properties(self):
        print(self.smiles_input.text())
        print(self.ames_classifier)
        print(self.toxicity_regressor)
        print("calc tox")
        tox_inputs = compute_features(self.smiles_input.text(), get_important_features_from_list(
            "saved_predictors/toxicity_regressor/regressor_350.txt"))
        print("calc ames")
        ames_inputs = compute_features(self.smiles_input.text(), self.ames_features)
        mut = self.ames_classifier.predict(np.array([ames_inputs]))
        tox = self.toxicity_regressor.predict(np.array([tox_inputs]))
        self.toxicity_result.setText(str(tox[0])[:6] + " -log(mmol/L)")
        if mut[0, 0] >= 0.5:
            self.mutagenity_result.setText("Active")
        else:
            self.mutagenity_result.setText("Inactive")

    def show_molecule_editor(self):
        process = subprocess.Popen(['java', '-jar', 'jchempaint-3.3-1210.jar'], stdout=subprocess.PIPE)

    def put_smiles_to_input(self, smiles):
        self.smiles_input.setText(smiles)
        print(smiles)


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = QSARApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
