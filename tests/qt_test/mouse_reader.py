#!/usr/bin/env python3

import sys
from PyQt6 import QtWidgets

class GraphicsScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

    def mouseMoveEvent(self, event):
        self.posX = event.scenePos().x()
        self.parent().parent().setPosition(event.scenePos().x()) # <-- crawl up the ancestry


class GraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.setMouseTracking(True)
        scene = GraphicsScene(self)
        self.setScene(scene)


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.Layout = QtWidgets.QVBoxLayout()
        self.gw = GraphicsView(self) # <-- pass self here
        self.Layout.addWidget(self.gw)
        self.label = QtWidgets.QLabel("Coordinate: x") # wanna show coorinates here correspond to the mouse movement.
        self.Layout.addWidget(self.label)
        self.setLayout(self.Layout)

    def setPosition(self, posX): # <-- this is a setter, not a getter
        self.label.setText("Coordinate: x" + str(posX)) # <-- use argument posX
        self.repaint()
        print(posX)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())