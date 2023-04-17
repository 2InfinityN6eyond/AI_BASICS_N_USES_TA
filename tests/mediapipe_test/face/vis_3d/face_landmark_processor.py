import numpy as np
import multiprocessing
from multiprocessing import shared_memory

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceLandmarkProcessor(multiprocessing.Process) :
    def __init__(
            self,
            frame_width     :int,
            frame_height    :int,
            shm_name        :str,
            img_queue_size  :int,
            img_idx_queue   :multiprocessing.Queue,
            data_queue      :multiprocessing.Queue,
            stop_flag       :multiprocessing.Event,
            max_num_faces   :int  = 1,
        ) :
        super(FaceLandmarkProcessor, self).__init__()
        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.shm_name       = shm_name
        self.img_queue_size = img_queue_size
        self.img_idx_queue  = img_idx_queue
        self.data_queue     = data_queue
        self.stop_flag      = stop_flag
        self.max_num_faces  = max_num_faces

    def run(self) :
        shm = shared_memory.SharedMemory(name=self.shm_name)
        image_queue = np.ndarray(
            (self.img_queue_size, self.frame_height, self.frame_width, 3),
            dtype=np.uint8, buffer = shm.buf
        )
        with mp_face_mesh.FaceMesh(
            max_num_faces            = self.max_num_faces,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5
        ) as face_mesh :
            while not self.stop_flag.is_set() :
                image_idx = self.img_idx_queue.get()
                image = image_queue[image_idx]
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                data_dict = {"image_idx": image_idx, "face": None}
                if results.multi_face_landmarks :
                    face_landmark_array = np.array(list(map(
                        lambda kp : [kp.x, kp.y, kp.z],
                        results.multi_face_landmarks[0].landmark
                    )))
                    data_dict["face"] = face_landmark_array
                self.data_queue.put(data_dict)
            shm.close()