from . import defaults
from PyQt6.QtGui import QPainter, QPen, QKeySequence, QShortcut, QPixmap, QImage
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QLabel, QWidget, QVBoxLayout
from numpy import array


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow, defaults.Defualts):
    PICTURE_CHANGED = False
    def __init__(self, NN):
        self.predicionColors = ["#FF0000", "#E31C00", "#C63900", "#AA5500", "#8E7100",
                                "#718E00", "#55AA00", "#39C600", "#1CE300", "#00FF00", "#0d6ba5"]
        super().__init__()
        self.UiInit()
        self.show()
        self.NN = NN


    def UiInit(self):
        self.ShortcutsInit()
        self.configureMainWindow()
        self.createMainLayout()
        self.createBoard()
        self.createPredictionArea()
        self.mainView.addWidget(self.imageContainer)
        self.mainView.addWidget(self.predictionsLayoutContainer)

    
    def configureMainWindow(self):
        self.setWindowTitle("My App")
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
        self.predictionsLayoutContainer = QWidget(self)
        self.predictionsLayout = QHBoxLayout()
        self.predictionLabels = []
        for i in range(10):
            label = QLabel(f"{i}\n")
            label.setStyleSheet(f"background: {self.predicionColors[-1]}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.predictionLabels.append(label)
            self.predictionsLayout.addWidget(self.predictionLabels[i])
        self.predictionsLayoutContainer.setLayout(self.predictionsLayout)


    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            canvasPainter = QPainter(self.pixmap)
            canvasPainter.setPen(QPen(self.brushColor, self.brushSize))
            canvasPainter.drawLine(self.lastStartPoint, event.position())
            self.imageContainer.setPixmap(self.pixmap)
            self.lastLineDraw.append([self.lastStartPoint, event.position()])
            self.lastStartPoint = event.position()
            canvasPainter.end()
            

    def convertDrawingToMatrix(self):
        img = self.pixmap.scaled(28,28).toImage().convertToFormat(QImage.Format.Format_Grayscale8)
        imgarray = array(img.constBits().asarray(28*28)).astype('float32')/255.0
        predictions = self.NN.runTest(imgarray, verbose = False)
        predictionsSorted = predictions.copy()
        predictionsSorted.sort()
        mappedColors = {key:val for key,val in zip(predictionsSorted, self.predicionColors)}
        for i, (val, label) in enumerate(zip(predictions,self.predictionLabels)):
            label.setText(f"{i}\n{val}")
            label.setStyleSheet(f"background: {mappedColors[val]}")


    def ShortcutsInit(self):
        self.msgSc = QShortcut(QKeySequence('Ctrl+Z'), self)
        self.msgSc.activated.connect(self.undo)
        self.quitSc = QShortcut(QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(QApplication.instance().quit)


    def undo(self):
        if self.drawEvents:
            self.drawEvents.pop()
            self.ReDraw()
            if not self.drawEvents:
                for i in range(10):
                    self.predictionLabels[i].setText(f"{i}\n")
                    self.predictionLabels[i].setStyleSheet(f"background: {self.predicionColors[-1]}")


    def ReDraw(self):
        self.pixmap.fill(Qt.GlobalColor.black)
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(self.brushColor, self.brushSize, Qt.PenStyle.SolidLine)) 
        for line in self.drawEvents:
            for point in line:
                painter.drawLine(*point)
        self.imageContainer.setPixmap(self.pixmap)
        if self.drawEvents:
            self.convertDrawingToMatrix()
            self.update()

        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.lastStartPoint = event.position()

    
    def mouseReleaseEvent(self, event):
        self.drawEvents.append(self.lastLineDraw)
        self.lastLineDraw = []
        if self.drawEvents:
            self.convertDrawingToMatrix()
            self.update()


    def resizeEvent(self, event):
        # self.windowSize = self.mainLayoutContaier.size()
        self.pixmap = self.pixmap.scaled(self.windowSize)
        self.ReDraw()
