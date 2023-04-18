import sys
import sys
import numpy as np
import threading
import multiprocessing

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from face_reader import FaceReader
from visualize_3d_pose import MediaPipeVisualizer

class DataBridge(QtCore.QThread) :
    landmark_acquired = QtCore.pyqtSignal(dict)

    def __init__(self, stop_flag, from_holistic_reader) :
        super(DataBridge, self).__init__()
        self.stop_flag = stop_flag
        self.from_holistic_reader = from_holistic_reader

    def run(self) :
        while not self.stop_flag.is_set() :
            self.landmark_acquired.emit(
                self.from_holistic_reader.get()
            )


if __name__ == "__main__" :
    app = QtWidgets.QApplication(sys.argv)

    landmark_queue = multiprocessing.Queue()
    holistic_reader = FaceReader(landmark_queue)

    thread_stop_flag = threading.Event()
    data_bridge = DataBridge(thread_stop_flag, landmark_queue)

    plot_widget = MediaPipeVisualizer()
    plot_widget.show()

    data_bridge.landmark_acquired.connect(
        lambda landmark_dict : plot_widget.updateWhole(landmark_dict)
    )

    holistic_reader.start()
    data_bridge.start()

    app.exec()

    thread_stop_flag.set()
    sys.exit()