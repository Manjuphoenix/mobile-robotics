import numpy as np
import open3d as o3d
import cv2



pcd = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460517.915522575.pcd")
points = np.asarray(pcd.points)

camera_intrinsics = np.array([[6.442133178710937500e+02, 0.00000e+00, 6.52677368164062500e+02],
                                [0.0000e+00, 6.434055786132812500e+02, 3.712568664550781250e+02],
                                [0.0000e+00, 0.0000e+00, 1.00000e+00]])

rotation_matrix = np.array([],[],[])

translation = np.array([])

