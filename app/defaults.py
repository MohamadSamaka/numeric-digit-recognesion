from PyQt6.QtCore import Qt, QPoint

class Defualts:
    def __init__(self):
        self.HEIGHT = 800
        self.WIDTH = 800
        self.drawing = False
        self.brushSize = 40
        self.brushColor = Qt.GlobalColor.white
        self.lastStartPoint = QPoint()
        self.drawEvents = []
        self.lastLineDraw = []
        