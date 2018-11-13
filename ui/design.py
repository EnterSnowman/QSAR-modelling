# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(694, 283)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 120, 671, 77))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.toxicity_result = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.toxicity_result.setFont(font)
        self.toxicity_result.setText("")
        self.toxicity_result.setObjectName("toxicity_result")
        self.verticalLayout.addWidget(self.toxicity_result)
        self.mutagenity_result = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.mutagenity_result.setFont(font)
        self.mutagenity_result.setObjectName("mutagenity_result")
        self.verticalLayout.addWidget(self.mutagenity_result)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(10, 10, 671, 102))
        self.widget1.setObjectName("widget1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(self.widget1)
        self.label.setEnabled(True)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.smiles_input = QtWidgets.QLineEdit(self.widget1)
        self.smiles_input.setObjectName("smiles_input")
        self.horizontalLayout_2.addWidget(self.smiles_input)
        self.editor_button = QtWidgets.QPushButton(self.widget1)
        self.editor_button.setObjectName("editor_button")
        self.horizontalLayout_2.addWidget(self.editor_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.compute_button = QtWidgets.QPushButton(self.widget1)
        self.compute_button.setObjectName("compute_button")
        self.verticalLayout_3.addWidget(self.compute_button)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Toxicity and mutagenicity predictor"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">Toxicity against T.Pyriformis:</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">Ames mutagenicity:</span></p></body></html>"))
        self.mutagenity_result.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\"/></p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">Input SMILES notation of compound:</span></p></body></html>"))
        self.editor_button.setText(_translate("MainWindow", "Open molecule editor"))
        self.compute_button.setText(_translate("MainWindow", "Compute properties"))

