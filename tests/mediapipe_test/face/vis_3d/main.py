import sys
import platform
import numpy as np
import cv2
import multiprocessing
from multiprocessing import shared_memory
import argparse

from pynput import mouse

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from camera_reader import CameraReader
from face_landmark_processor import FaceLandmarkProcessor
from mediapipe_visualizer import ThreeDimensionVisualizer
from main_window import MainWindow

class DataBridge(QtCore.QThread) :
    landmark_acquired = QtCore.pyqtSignal(dict)
    mouse_changed = QtCore.pyqtSignal(list)
    
    def __init__(self, stop_flag, data_queue) :
        super(DataBridge, self).__init__()
        self.stop_flag = stop_flag
        self.data_queue = data_queue

    def run(self) :
        listener = mouse.Listener(
            on_move = lambda x, y : self.mouse_changed.emit(
                [int(x), int(y), None, None, None, None]
            ),
            on_click = lambda x, y, button, pressed : self.mouse_changed.emit(
                [int(x), int(y), None, None, button, pressed]
            ),
            on_scroll = lambda x, y, dx, dy : self.mouse_changed.emit(
                [int(x), int(y), int(dx), int(dy), None, None]
            )
        )
        listener.start()

        while not self.stop_flag.is_set() :
            self.landmark_acquired.emit(
                self.data_queue.get()
            )
        
        listener.stop()
        listener.join()



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_idx",       default=0, type=int)
    parser.add_argument("--image_queue_size", default=10, type=int)
    parser.add_argument("--image_width",      default=1552, type=int)
    parser.add_argument("--image_height",     default=1552, type=int)
    args = parser.parse_args()
    camera_idx = args.camera_idx

    SYSTEM_NAME = platform.system()
    if SYSTEM_NAME == "Windows" :
        VID_CAP_FLAG = cv2.CAP_DSHOW
    if SYSTEM_NAME == "Darwin" :
        VID_CAP_FLAG = None

    print(camera_idx)

    # open webcam first and get image size.
    # image shape should be square
    cap = cv2.VideoCapture(camera_idx, VID_CAP_FLAG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.image_height)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print("webcam image size set to {}X{}".format(
        frame_width, frame_height
    ))

    # initialize shared memory for storing camera frames
    shm = shared_memory.SharedMemory(
        create = True,
        size   = args.image_queue_size * frame_width * frame_height * 3 
    )
    shm_name = shm.name

    # initialize pyqt app
    app = QtWidgets.QApplication(sys.argv)
    screen_geometry = app.primaryScreen().geometry()
    main_window = MainWindow(
        screen_width    = screen_geometry.width(),
        screen_height   = screen_geometry.height(),
        frame_width     = frame_width,
        frame_height    = frame_height,
        shm_name        = shm_name,
        img_queue_size  = args.image_queue_size,
    )
    main_window.show()

    # flag for stop program. chile processes stop if stop_flag set.
    stop_flag = multiprocessing.Event()
    # queue accross processes for sending index where image is stored in shm
    img_idx_queue = multiprocessing.Queue()
    # queue accross processes for sending face mesh data
    data_queue = multiprocessing.Queue()

    camera_reader = CameraReader(
        camera_idx      = camera_idx,
        frame_width     = frame_width,
        frame_height    = frame_height,
        shm_name        = shm_name,
        img_queue_size  = args.image_queue_size,
        img_idx_queue   = img_idx_queue,
        stop_flag       = stop_flag,
    )
    face_landmark_processor = FaceLandmarkProcessor(
        frame_width     = frame_width,
        frame_height    = frame_height,
        shm_name        = shm_name,
        img_queue_size  = args.image_queue_size,
        img_idx_queue   = img_idx_queue,
        data_queue      = data_queue,
        stop_flag       = stop_flag,
    )
    data_bridge = DataBridge(
        stop_flag = stop_flag,
        data_queue = data_queue
    )
    
    data_bridge.landmark_acquired.connect(
        lambda ladmark_dict : main_window.updateFaceData(ladmark_dict)
    )
    data_bridge.mouse_changed.connect(
        lambda mouse_data : main_window.updateMouseData(mouse_data)
    )

    data_bridge.start()
    face_landmark_processor.start()
    camera_reader.start()

    app.exec()

    stop_flag.set()
    camera_reader.join()
    face_landmark_processor.join()

    shm.close()
    shm.unlink()
    sys.exit()