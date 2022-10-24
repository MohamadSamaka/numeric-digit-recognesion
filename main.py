from app.app import *
from os import sys
from neuralNetwork.neuralNetwork import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(NeuralNetwork())
    app.exec()
