import sys
from PyQt6 import QtWidgets, QtCore, QtGui


app = QtWidgets.QApplication(sys.argv)
screen_geometry = app.primaryScreen().geometry()

print(screen_geometry)

print(screen_geometry.height())

print(screen_geometry.width())
