from os import sys
from defaults import Defualts
from PyQt6.QtGui import QPainter, QPen, QKeySequence, QShortcut, QPixmap
from PyQt6.QtCore import QSize, Qt, QPoint
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox, QHBoxLayout, QLabel, QWidget, QVBoxLayout


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow, Defualts):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        self.ShortcutsInit()
        self.UiInit()
        self.show()

    def UiInit(self):
        self.configureMainWindow()
        self.createMainLayout()
        self.createBoard()
        self.createPredictionArea()
        self.mainView.addWidget(self.imageContainer)
        self.mainView.addWidget(self.predictionLabel)

    
    def configureMainWindow(self):
        self.windowSize = QSize(self.HEIGHT, self.WIDTH)
        self.resize(self.windowSize)

    def createMainLayout(self):
        self.mainLayoutContaier = QWidget(self)
        self.setCentralWidget(self.mainLayoutContaier)
        self.mainLayoutContaier.resize(self.windowSize)
        self.mainView = QVBoxLayout()
        self.mainLayoutContaier.setLayout(self.mainView)

    def createBoard(self):
        self.imageContainer = QLabel()
        self.imageContainer.resize(self.windowSize)
        self.pixmap = QPixmap(self.imageContainer.size())
        self.pixmap.fill(Qt.GlobalColor.black)
        self.imageContainer.setPixmap(self.pixmap)
        self.mainView.addChildWidget(self.imageContainer)

    def createPredictionArea(self):
        self.predictionLabel = QLabel("none")
        self.predictionLabel.setStyleSheet("background:green;")
        self.predictionLabel .setAlignment(Qt.AlignmentFlag.AlignCenter)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            canvasPainter = QPainter(self.pixmap)
            canvasPainter.setPen(QPen(self.brushColor, self.brushSize))
            canvasPainter.drawLine(self.lastPoint, event.position())
            self.imageContainer.setPixmap(self.pixmap)
            self.lastLineDraw.append([self.lastPoint, event.position()])
            self.lastPoint = event.position()
            canvasPainter.end()

       
    def ShortcutsInit(self):
        self.msgSc = QShortcut(QKeySequence('Ctrl+Z'), self)
        self.msgSc.activated.connect(self.undo)
        self.quitSc = QShortcut(QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(QApplication.instance().quit)


    def undo(self):
        if self.drawEvents:
            self.drawEvents.pop()
            self.ReDraw()


    def ReDraw(self):
        self.pixmap.fill(Qt.GlobalColor.black)
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(self.brushColor, self.brushSize, Qt.PenStyle.SolidLine)) 
        for line in self.drawEvents:
            for point in line:
                painter.drawLine(*point)


        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.lastPoint = event.position()

    
    def mouseReleaseEvent(self, event):
        self.drawEvents.append(self.lastLineDraw)
        self.lastLineDraw = []


    def resizeEvent(self, event):
        # self.windowSize = self.mainLayoutContaier.size()
        self.pixmap = self.pixmap.scaled(self.windowSize)
        self.ReDraw()
    

app = QApplication(sys.argv)

window = MainWindow()

app.exec()