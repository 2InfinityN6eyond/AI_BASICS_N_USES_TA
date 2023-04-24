import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import multiprocessing
from multiprocessing import shared_memory
import cv2

from image_plotter import ImagePlotter
from camera_reader import CameraReader
from face_landmark_processor import FaceLandmarkProcessor

class DataBridge(QtCore.QThread) :
    frame_captured = QtCore.pyqtSignal(dict)

    def __init__(
        self,
        shm_name,
        image_queue_size,
        frame_width,
        frame_height,
        stop_flag,
        data_queue,
    ) :
        super(DataBridge, self).__init__()
        self.shm_name           = shm_name
        self.image_queue_size   = image_queue_size
        self.frame_width        = frame_width
        self.frame_height       = frame_height
        self.stop_flag          = stop_flag
        self.data_queue         = data_queue

        self.shm = shared_memory.SharedMemory(
            name = self.shm_name,
            size = frame_height * frame_width * 3
        )
        self.image_queue = np.ndarray(
            (self.image_queue_size, self.frame_height, self.frame_width, 3),
            dtype = np.uint8, buffer = self.shm.buf
        )

    def run(self) :
        while not self.stop_flag.is_set() :
            data = self.data_queue.get()
            self.frame_captured.emit(data)


class MainWindow(QtWidgets.QMainWindow) :
    def __init__(
        self,
        shm_name,
        image_queue_size,
        frame_width,
        frame_height,
    ) :
        super(MainWindow, self).__init__()

        self.shm_name           = shm_name
        self.image_queue_size   = image_queue_size
        self.frame_width        = frame_width
        self.frame_height       = frame_height
        
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

    def update(self, data) :
        idx = data["image_idx"]
        frame = self.image_queue[idx, :, :, :]
        self.image_plotter.update(frame)

        print(data)


if __name__ == "__main__" :

    image_queue_size = 4
    frame_width = 1920
    frame_height = 1080

    shm = shared_memory.SharedMemory(
        create = True,
        size   = image_queue_size * frame_width * frame_height * 3
    )
    img_idx_queue = multiprocessing.Queue()
    data_queue = multiprocessing.Queue()
    stop_flag = multiprocessing.Event()

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow(
        shm.name,
        image_queue_size,
        frame_width,
        frame_height
    )
    main_window.show()

    data_bridge = DataBridge(
        shm_name = shm.name,
        image_queue_size = image_queue_size,
        frame_width = frame_width, frame_height = frame_height,
        data_queue = data_queue,
        stop_flag = stop_flag
    )
    data_bridge.frame_captured.connect(main_window.update)
    data_bridge.start()
    
    face_processor = FaceLandmarkProcessor(
        frame_width     = frame_width,
        frame_height    = frame_height,
        shm_name        = shm.name,
        img_queue_size  = image_queue_size,
        img_idx_queue   = img_idx_queue,
        data_queue      = data_queue,
        stop_flag       = stop_flag,
    )
    face_processor.start()

    camera_reader = CameraReader(
        frame_width = frame_width,
        frame_height = frame_height,
        shm_name = shm.name,
        img_queue_size = image_queue_size,
        img_idx_queue = img_idx_queue,
        stop_flag = stop_flag
    )
    camera_reader.start()

    app.exec()

    stop_flag.set()
    face_processor.join()