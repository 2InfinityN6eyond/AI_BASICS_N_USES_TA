import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import multiprocessing
from multiprocessing import shared_memory
import cv2

from image_plotter import ImagePlotter

class DataBridge(QtCore.QThread) :
    frame_captured = QtCore.pyqtSignal(int)

    def __init__(
        self,
        shm_name,
        image_queue_size,
        frame_width,
        frame_height,
        stop_flag,
    ) :
        super(DataBridge, self).__init__()
        self.shm_name = shm_name
        self.image_queue_size = image_queue_size
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.stop_flag = stop_flag

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.shm = shared_memory.SharedMemory(
            name = self.shm_name,
            size = frame_height * frame_width * 3
        )
        self.image_queue = np.ndarray(
            (self.image_queue_size, self.frame_height, self.frame_width, 3),
            dtype = np.uint8, buffer = self.shm.buf
        )

    def run(self) :
        image_idx_iterator = iter(self.idx_iterator())
        while not self.stop_flag.is_set() :
            ret, frame = self.cap.read()
            if not ret :
                print("no frame!!")
                continue
            idx = next(image_idx_iterator)
            self.image_queue[idx, :, :, :] = frame[:, :, :]
            self.frame_captured.emit(idx)

    def idx_iterator(self) :
            idx = 0
            while True :
                yield idx
                idx += 1
                idx %= self.image_queue_size

class MainWindow(QtWidgets.QMainWindow) :
    def __init__(
        self,
        shm_name,
        image_queue_size,
        frame_width,
        frame_height
    ) :
        super(MainWindow, self).__init__()

        self.shm_name = shm_name
        self.image_queue_size = image_queue_size
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.init_ui()

        self.shm = shared_memory.SharedMemory(
            name = self.shm_name,
            size = (
                self.image_queue_size * frame_height * frame_width * 3
            )
        )
        self.image_queue = np.ndarray(
            (self.image_queue_size, self.frame_height, self.frame_width, 3),
            dtype = np.uint8, buffer = self.shm.buf
        )

    def init_ui(self) :
        self.image_plotter = ImagePlotter(self.frame_width, self.frame_height)
        self.setCentralWidget(self.image_plotter)

    def update(self, idx) :
        frame = self.image_queue[idx, :, :, :]
        self.image_plotter.update(frame)

if __name__ == "__main__" :

    image_queue_size = 4
    frame_width = 1920
    frame_height = 1080

    shm = shared_memory.SharedMemory(
        create = True,
        size   = image_queue_size * frame_width * frame_height * 3
    )

    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow(
        shm.name,
        image_queue_size,
        frame_width,
        frame_height
    )
    main_window.show()
    
    stop_flag = multiprocessing.Event()
    data_bridge = DataBridge(
        shm_name = shm.name,
        image_queue_size = image_queue_size,
        frame_width = frame_width, frame_height = frame_height,
        stop_flag = stop_flag
    )
    data_bridge.frame_captured.connect(main_window.update)
    data_bridge.start()

    app.exec()

    stop_flag.set()
    data_bridge.join()