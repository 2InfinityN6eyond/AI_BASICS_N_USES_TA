import os
import time
import cv2
import sys 
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from image_plotter import ImagePlotter
from mediapipe_visualizer import ThreeDimensionVisualizer

class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self
    ):
        super(MainWindow, self).__init__()

        self.initUI()


    def initUI(self) :

        self.three_dimention_visualizer = ThreeDimensionVisualizer()

        self.record_curr_frame_button = QtWidgets.QPushButton(
            "record_curr_frame", self
        )
        self.record_curr_frame_button.setCheckable(True)
        self.record_curr_frame_button.setChecked(False)
        self.record_curr_frame_button.setShortcut("Ctrl+S")

        self.webcam_image_plotter = ImagePlotter(500, 500)
        self.face_plotter = ImagePlotter(250, 250)
        self.face_vis_plotter = ImagePlotter(250, 250)
        self.left_eye_plotter = ImagePlotter(250, 180)
        self.left_eye_vis_plotter = ImagePlotter(250, 180)
        self.right_eye_plotter = ImagePlotter(250, 180)
        self.right_eye_vis_plotter = ImagePlotter(250, 180)

        eye_plot_layout = QtWidgets.QGridLayout()
        eye_plot_layout.addWidget(self.left_eye_plotter, 1, 1)
        eye_plot_layout.addWidget(self.left_eye_vis_plotter, 2, 1)
        eye_plot_layout.addWidget(self.right_eye_plotter, 1, 2)
        eye_plot_layout.addWidget(self.right_eye_vis_plotter, 2, 2)

        face_plot_layout = QtWidgets.QHBoxLayout()
        face_plot_layout.addWidget(self.face_plotter)
        face_plot_layout.addWidget(self.face_vis_plotter)

        image_plot_layout = QtWidgets.QVBoxLayout()
        image_plot_layout.addWidget(self.webcam_image_plotter)
        image_plot_layout.addLayout(face_plot_layout)
        image_plot_layout.addLayout(eye_plot_layout)

        main_layout = QtWidgets.QHBoxLayout()

        '''
        self.image_plotter_3.update(
            np.zeros((405, 720, 3), dtype=np.uint8) + 255
        )
        self.image_plotter_4.update(
            np.zeros((405, 720, 3), dtype=np.uint8) + 255
        )
        '''

        main_layout.addLayout(image_plot_layout )
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.show()
        #self.showMaximized()

    def saveData(self) :
        if self.record_curr_frame_button.isChecked() :
            if (
                    self.configs_n_vals["pressure_sensor_data"] and \
                    self.configs_n_vals["image_data"] # and \
                    #self.configs_n_vals["homography"]
            ) :
                self.to_data_writer.put({
                    "pressure_sensor" : self.configs_n_vals["pressure_sensor_data"],
                    "images" : self.configs_n_vals["image_data"],
                    "homography" : self.configs_n_vals["homography"]
                })

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Space :
            self.record_curr_frame_button.setChecked(True)

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Space and not a0.isAutoRepeat() :
            self.record_curr_frame_button.setChecked(False)

        
if __name__ == "__main__" :
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())