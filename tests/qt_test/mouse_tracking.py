import sys
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)

        self.setMouseTracking(True)

        self.label = QLabel("Mouse coordinates", parent)
        # Without this mouseMoveEvent of QLabel will be called
        # instead of mouseMoveEvent of MainWindow!
        self.label.setMouseTracking(True)

        self.setCentralWidget(self.label)

    def mouseMoveEvent(self, event):
        self.setStyleSheet("background-color: red;")
        #self.label.setText('Mouse coordinates: ( %d : %d )' % (event.x(), event.y()))
        print(event.pos())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())