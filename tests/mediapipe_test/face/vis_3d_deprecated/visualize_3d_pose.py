import sys
import numpy as np
import mediapipe as mp

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


def edge_list_2_path(edge_list) :
    tesel = edge_list
    change_occured = True
    while change_occured :
        change_occured = False
        for idx in range(len(tesel)) :
            target_edge = tesel[idx]

            inner_changed = False
            for e in tesel :
                if e != target_edge and len(e) < 3 :
                    edge = e

                    if target_edge[-1] == edge[0] :
                        target_edge.append(edge[-1])
                        tesel.remove(edge)
                        change_occured = True
                        inner_changed = True
                        break

                    if target_edge[0] == edge[-1] :
                        target_edge.insert(0, edge[0])
                        tesel.remove(edge)
                        change_occured = True
                        inner_changed  = True
                        break
            
            if inner_changed :
                break

    change_occured = True
    while change_occured :
        change_occured = False
        for idx in range(len(tesel)) :
            source_path = tesel[idx]

            inner_changed = False
            for target_path in tesel :

                if target_path == source_path :
                    continue

                if source_path[-1] == target_path[-1] :
                    target_path.reverse()
                    source_path += target_path[1:]
                    tesel.remove(target_path)
                    change_occured = True
                    inner_changed = True
                    break

                if source_path[0] == target_path[0] :
                    source_path.reverse()
                    target_path += source_path[1:]
                    tesel.remove(source_path)
                    change_occured = True
                    inner_changed = True
                    break

            if inner_changed :
                break

    return tesel

edge_list = np.array(
    list(mp_face_mesh.FACEMESH_TESSELATION) +
    list(mp_face_mesh.FACEMESH_FACE_OVAL) +
    list(mp_face_mesh.FACEMESH_LIPS) +
    list(mp_face_mesh.FACEMESH_IRISES) +
    list(mp_face_mesh.FACEMESH_LEFT_EYE) +
    list(mp_face_mesh.FACEMESH_RIGHT_EYE) +
    list(mp_face_mesh.FACEMESH_LEFT_EYEBROW) +
    list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
).tolist()

FACE_PATH_LIST = edge_list_2_path(edge_list)
LEFT_PATH = edge_list_2_path( np.array(list(mp_face_mesh.FACEMESH_LEFT_IRIS)).tolist() )[0]
RIGHT_PATH = edge_list_2_path( np.array(list(mp_face_mesh.FACEMESH_RIGHT_IRIS)).tolist() )[0]

class MediaPipeVisualizer(gl.GLViewWidget) :
    def __init__(self) -> None :
        super().__init__()

        gx = gl.GLGridItem(color=pg.mkColor((100, 50, 50)))
        gx.rotate(90, 0, 1, 0)
        #gx.translate(-1.5, 0, 0)
        self.addItem(gx)
        gy = gl.GLGridItem(color=pg.mkColor((50, 100, 50)))
        gy.rotate(90, 1, 0, 0)
        #gy.translate(0, -1.5, 0)
        self.addItem(gy)
        gz = gl.GLGridItem(color=pg.mkColor((50, 50, 100)))
        #gz.translate(0, 0, -1.5)
        self.addItem(gz)

        self.setCameraParams(elevation = -90, azimuth = -90)

        self.left_hand_line_list = []
        self.right_hand_line_list = []
        self.face_line_list = []
        self.pose_line_list = []

    def updateFace(
        self,
        landmark_list,
        image_shape,
        set_origin_as_origin = True,
        always_place_center = False,
    ) :
        for line_item in self.face_line_list :
            self.removeItem(line_item)
        landmark_array = np.array(landmark_list)
        if always_place_center :
            landmark_array -= landmark_array.mean(axis=0)
        elif set_origin_as_origin :
            landmark_array -= np.array([0.5, 0.5, 0])
        denormalizer = np.array([image_shape[0], image_shape[1], image_shape[0]]) / image_shape[0]
        self.face_line_list = list(map(
            lambda face_path : gl.GLLinePlotItem(
                pos = landmark_array[face_path] * denormalizer,
                color = pg.mkColor((255, 0, 0)),
                width = 2,
                antialias = True
            ),            
            FACE_PATH_LIST
        ))
        

        for item in self.face_line_list :
            if item :
                self.addItem(item)

    def updateWhole(self, landmark_dict) :
        if "face" in landmark_dict :
            self.updateFace(
                landmark_dict["face"],
                landmark_dict["image_shape"]
            )