from PyQt6.QtCore import Qt, QPoint
class Defaluts:
    def __init__(self):
        self.HEIGHT = 800
        self.WIDTH = 800
        self.drawing = False
        self.brushSize = 2
        self.brushColor = Qt.GlobalColor.red
        self.lastPoint = QPoint()
        self.drawEvents = []
        self.lastLineDraw = []