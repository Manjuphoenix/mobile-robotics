import numpy as np
import open3d as o3d
import cv2
from transformations import quaternion_matrix, translation_matrix, concatenate_matrices



p = np.array([
        0.07093786809565478,
        0.012665553133702897,
        -0.08437335095076476,
        0.4051169630628588,
        0.4006810126971365,
        0.5797585743574961,
        0.5824216408768541,
    ])


############################### Library to compute the transformations ##############################

tvec = np.array([p[0], p[1], p[2]])
quat = np.array([p[6], p[3], p[4], p[5]])

T_trans = translation_matrix(tvec)
T_rot = quaternion_matrix(quat)

########### Final Transformation matrix ######################
T_fin = concatenate_matrices(T_trans, T_rot)


############# Mount flip correction #########################
# tmp_D = np.diag([1, -1, -1, 1])
# T_fin = T_fin @ tmp_D
tmp_D = np.diag([1, -1, -1]).astype(float)

############ Read point cloud data to account for this flip ##########

# pcd = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460386.414835930.pcd")
pcd = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460674.916070223.pcd")
points = np.asarray(pcd.points)

# import ipdb; ipdb.set_trace()
############### Account for the mount flip on raw point clouds ###########
points = points @ tmp_D




rotation_matrix_fin = T_fin[:-1, :-1]     # Mistake - did not convert the frames lidar to camera


# import ipdb; ipdb.set_trace()
# rotation_matrix_fin = rotation_matrix_fin @ np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) 

############ This is to compute the inverse of the square matrix ################
rotation_matrix_fin = np.linalg.inv(rotation_matrix_fin)    
"""
If the above is not done, the points get accumulated to the left of the image only...
Simply cuz the lidar points are not rotated (emperically - after rotation there are 
more points seen on the image as horizontal dim is greater than vertical dim (W > H))
"""

translation_vector_fin = T_fin[:3, 3:]


#####################################################################################################


# camera intrinsics
K_intrinsics = np.array([[6.442133178710937500e+02, 0.000000000000000000e+00, 6.526773681640625000e+02],
[0.000000000000000000e+00, 6.434055786132812500e+02, 3.712568664550781250e+02],
[0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])


# pcd = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460517.915522575.pcd")
# points = np.asarray(pcd.points)


points_in_camera_frame = (rotation_matrix_fin @ points.T + translation_vector_fin)


new_points = np.ascontiguousarray(points_in_camera_frame.T, dtype=np.float64)

"""
This is a single channel or one dim mask 
(bool values - yes or no based on the condition) 
Later this mask is applied to all 3 dimensions independently 
to filter across all 3 dim (x, y, z)

Analogy - consider the binary mask for images one channel mask is overlayed
for all 3 channels and we get rgb image with colors only to the mask segments
and black color pixels (Zero valued) in rest all places.

So 3 channels here is 3 dimensions and nothing much..
"""
mask = new_points[:, 2] > 0     


filtered_new_points = new_points[mask]
"""
Only positive Z dim data is considered and rest all are thrown away..
"""



#################### Projecting points on to the image plane ########################
# import ipdb; ipdb.set_trace()
# 
uv = (K_intrinsics @ filtered_new_points.T).T
uv = uv[:, :2] / uv[:, 2:3]

###################################################################################
# File 3: 1776460674.916070223

# image = cv2.imread("/home/jarvis/a1/assets/q1_data/rgb/1776460386.414835930.png")
image = cv2.imread("/home/jarvis/a1/assets/q1_data/rgb/1776460674.916070223.png")

h, w = image.shape[:2]
# import ipdb; ipdb.set_trace()

in_bounds = (uv[:, 0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:, 1] < h)


for (u, v), z in zip(uv[in_bounds], filtered_new_points[in_bounds][:, 2]):
    color = int(255 * min(z / 50, 1))
    # cv2.circle(image, (int(u), int(v)), 1, (0, 255 - color, color), 2)
    cv2.drawMarker(image, (int(u), int(v)), (0, 255 - color, color), 
    markerType = cv2.MARKER_CROSS, markerSize=3, thickness=2)


# cv2.imwrite("projected_1776460386.414835930.png", image)
cv2.imwrite("projected_1776460674.916070223.png", image)
