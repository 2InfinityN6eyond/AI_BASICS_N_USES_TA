import cv2
import platform
import numpy as np
import multiprocessing
from multiprocessing import shared_memory

class CameraReader(multiprocessing.Process) :
    def __init__(
        self,
        frame_width     :int,
        frame_height    :int,
        shm_name        :str,
        img_queue_size  :int,
        img_idx_queue   :multiprocessing.Queue, 
        stop_flag       :multiprocessing.Event
    ) :
        super(CameraReader, self).__init__()
        self.frame_width     = frame_width
        self.frame_height    = frame_height
        self.shm_name        = shm_name
        self.img_queue_size  = img_queue_size
        self.img_idx_queue   = img_idx_queue
        self.stop_flag       = stop_flag

    def run(self) :
        image_idx_iterator = iter(self.idx_iterator())

        shm = shared_memory.SharedMemory(name=self.shm_name)
        image_queue = np.ndarray(
            (self.img_queue_size, self.frame_height, self.frame_width, 3),
            dtype=np.uint8, buffer = shm.buf
        )

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert (
            frame_width == self.frame_width and frame_height == self.frame_height
        ), "opend webcam has frame size {}X{}, which is defferent from configured image size {}X{}\n".format(
            frame_width, frame_height, self.frame_width, self.frame_height,
        )
        while not self.stop_flag.is_set() and cap.isOpened() :
            image_idx = next(image_idx_iterator)
            success_flag, frame = cap.read()
            if not success_flag :
                print("webcam read failed")

            #print(frame.shape, frame_width, frame_height)

            image_queue[image_idx][:,:,:] = frame[:,:,:]
            #image_queue[image_idx][:,:,:] = frame[:,::-1,:]
            self.img_idx_queue.put(image_idx)   
        cap.release()
        shm.close()

    def idx_iterator(self) :
            idx = 0
            while True :
                yield idx
                idx += 1
                idx %= self.img_queue_size
