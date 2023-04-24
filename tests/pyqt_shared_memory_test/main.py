import cv2
import numpy as np
import multiprocessing
from multiprocessing import shared_memory


if __name__ == "__main__" :
    shm = shared_memory.SharedMemory(
        create = True,
        size = 10
    )
    