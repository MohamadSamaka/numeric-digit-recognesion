from ast import keyword
from ctypes import pointer
from curses import keyname
import sys
# from PyQt6 import QtGui
from PyQt6 import QtGui
from PyQt6.QtGui import QImage, QPainter, QPen, QKeySequence, QShortcut
from PyQt6.QtCore import QSize, Qt, QPoint, QLine
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        # button = QPushButton("Press Me!")
        # button.setMaximumSize(button.sizeHint())
        # self.setGeometry(500,500, 400, 400)
        # button.move(200 - int(button.width()/2),0)
        # self.layout().addWidget(button)

        self.setGeometry(500,500, 400, 400)
        self.ShortcutsInit()
        self.drawEvents = []
        self.lastLineDraw = []


        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.black)
        self.drawing = False
        self.brushSize = 2
        self.brushColor = Qt.GlobalColor.white
        self.lastPoint = QPoint()

       
    def ShortcutsInit(self):
        self.msgSc = QShortcut(QKeySequence('Ctrl+Z'), self)
        self.msgSc.activated.connect(self.undo)

        self.quitSc = QShortcut(QKeySequence('Ctrl+Q'), self)
        self.quitSc.activated.connect(QApplication.instance().quit)


    def undo(self):
        if self.drawEvents:
            self.drawEvents.pop()
            self.image.fill(Qt.GlobalColor.black)
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.PenStyle.SolidLine)) 
            for line in self.drawEvents:
                for point in line:
                    painter.drawLine(*point)
            self.update()

        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.lastPoint = event.position()


    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.PenStyle.SolidLine)) 
            painter.drawLine(self.lastPoint, event.position())
            self.lastLineDraw.append([self.lastPoint, event.position()])
            self.lastPoint = event.position()
            self.update()
    
    def mouseReleaseEvent(self, event):
        self.drawEvents.append(self.lastLineDraw)
        self.lastLineDraw = []


    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        canvasPainter.setPen(QPen(self.brushColor, 20))
        canvasPainter.end()


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()