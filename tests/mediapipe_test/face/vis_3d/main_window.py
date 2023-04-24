import os
import time
import cv2
import sys 
import numpy as np
import multiprocessing
from multiprocessing import shared_memory

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

from image_plotter import ImagePlotter
from mediapipe_visualizer import ThreeDimensionVisualizer

class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        frame_width     :int,
        frame_height    :int,
        shm_name        :str,
        img_queue_size  :int,
    ):
        super(MainWindow, self).__init__()
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.shm_name       = shm_name
        self.img_queue_size = img_queue_size

        shm = shared_memory.SharedMemory(name=self.shm_name)
        self.image_queue = np.ndarray(
            (self.img_queue_size, self.frame_height, self.frame_width, 3),
            dtype=np.uint8, buffer = shm.buf
        )

        self.initUI()

    def initUI(self) :
        self.three_dimention_visualizer = ThreeDimensionVisualizer()
        self.three_dimention_visualizer.setMinimumSize(500, 500)
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

        '''
        self.webcam_image_plotter.update(
            np.zeros((500, 500, 3), dtype=np.uint8) + 150
        )
        self.face_plotter.update(
            np.zeros((250, 250, 3), dtype=np.uint8) + 150
        )
        self.face_vis_plotter.update(
            np.zeros((250, 250, 3), dtype=np.uint8) + 150
        )
        self.left_eye_plotter.update(
            np.zeros((250, 180, 3), dtype=np.uint8) + 150
        )
        self.left_eye_vis_plotter.update(
            np.zeros((250, 180, 3), dtype=np.uint8) + 150
        )
        self.right_eye_plotter.update(
            np.zeros((250, 180, 3), dtype=np.uint8) + 150
        )
        self.right_eye_vis_plotter.update(
            np.zeros((250, 200, 3), dtype=np.uint8) + 150
        )
        '''

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

        main_layout.addWidget(self.three_dimention_visualizer)

        main_layout.addLayout(image_plot_layout)


        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.show()
        #self.showMaximized()

    def updateWhole(self, face_landmark_dict) :
        """
        update three dimension plot, image plotters.
        """
        self.three_dimention_visualizer.updateWhole(face_landmark_dict)
        image_queue_idx = face_landmark_dict["image_idx"]
        face_landmarks_array  = face_landmark_dict["face"]

        print(image_queue_idx)
        image = self.image_queue[image_queue_idx]
        print(image[0][0])

        return

        image = self.image_queue[image_queue_idx]
        image = image.copy()


        if face_landmarks_array is None : 
            return

        full_lt_rb = np.array([
            face_landmarks_array.min(axis=0),
            face_landmarks_array.max(axis=0),
        ])[:, :2] * image.shape[-2:-4:-1]
        full_lt_rb = full_lt_rb.flatten().astype(int)

        left_eye_lt_rb = np.array([
            face_landmarks_array[
                np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
            ].min(axis=0),
            face_landmarks_array[
                np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)).flatten()
            ].max(axis=0),
        ])[:, :2] * image.shape[-2:-4:-1]
        left_eye_lt_rb = left_eye_lt_rb * 2 - left_eye_lt_rb.mean(axis=0)
        left_eye_lt_rb = left_eye_lt_rb.flatten().astype(int)

        right_eye_lt_rb = np.array([
            face_landmarks_array[
                np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()
            ].min(axis=0),
            face_landmarks_array[
                np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)).flatten()
            ].max(axis=0),
        ])[:, :2] * image.shape[-2:-4:-1]
        right_eye_lt_rb = right_eye_lt_rb * 2 - right_eye_lt_rb.mean(axis=0)
        right_eye_lt_rb = right_eye_lt_rb.flatten().astype(int)


        #print(image.shape)
        #self.webcam_image_plotter.update(image)
        '''
        self.face_plotter.update(image[
            full_lt_rb[1]:full_lt_rb[3], full_lt_rb[0]:full_lt_rb[2], :
        ])
        self.face_vis_plotter.update(vis_image[
            full_lt_rb[1]:full_lt_rb[3], full_lt_rb[0]:full_lt_rb[2], :
        ])

        self.left_eye_plotter.update(image[
            left_eye_lt_rb[1]:left_eye_lt_rb[3],
            left_eye_lt_rb[0]:left_eye_lt_rb[2],
            :
        ])

        self.right_eye_plotter.update(image[
            right_eye_lt_rb[1]:right_eye_lt_rb[3],
            right_eye_lt_rb[0]:right_eye_lt_rb[2],
            :
        ])
        '''

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