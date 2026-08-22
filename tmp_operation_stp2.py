import numpy as np
import open3d as o3d
import cv2



p = np.array([
        0.07093786809565478,
        0.012665553133702897,
        -0.08437335095076476,
        0.4051169630628588,
        0.4006810126971365,
        0.5797585743574961,
        0.5824216408768541,
    ])


x, y, z = p[0], p[1], p[2]

translation_vector = np.array([[x, y, z]]).T

qx, qy, qz, qw = p[3], p[4], p[5], p[6]


# Compute the matrix elements
r00 = 1 - 2 * (qy**2 + qz**2)
r01 = 2 * (qx*qy - qw*qz)
r02 = 2 * (qx*qz + qw*qy)

r10 = 2 * (qx*qy + qw*qz)
r11 = 1 - 2 * (qx**2 + qz**2)
r12 = 2 * (qy*qz - qw*qx)

r20 = 2 * (qx*qz - qw*qy)
r21 = 2 * (qy*qz + qw*qx)
r22 = 1 - 2 * (qx**2 + qy**2)


rotation_matrix = np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])


rotation_matrix = np.linalg.inv(rotation_matrix)


transition_matrix = np.zeros([4,4])
transition_matrix[:-1, :-1] += rotation_matrix
transition_matrix[:3, 3:] += translation_vector
transition_matrix[-1, -1] += 1

tmp_D = np.diag([1, -1, -1, 1])

transition_matrix_flipped = tmp_D @ transition_matrix

# import ipdb; ipdb.set_trace()

rotation_matrix_fin = transition_matrix_flipped[:-1, :-1]
translation_vector_fin = transition_matrix[:3, 3:]


pcd = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460517.915522575.pcd")
points = np.asarray(pcd.points)

print(type(pcd.points)) #<class 'open3d.cpu.pybind.utility.Vector3dVector'>


# camera intrinsics
K_intrinsics = np.array([[6.442133178710937500e+02, 0.000000000000000000e+00, 6.526773681640625000e+02],
[0.000000000000000000e+00, 6.434055786132812500e+02, 3.712568664550781250e+02],
[0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

# o3d.utility.Vector3dVector


points_in_camera_frame = (rotation_matrix_fin @ points.T + translation_vector_fin)


new_points = np.ascontiguousarray(points_in_camera_frame.T, dtype=np.float64)
mask = new_points[:, 2] > 0
filtered_new_points = new_points[mask]


# import ipdb; ipdb.set_trace()
uv = (K_intrinsics @ filtered_new_points.T).T
uv = uv[:, :2] / uv[:, 2:3]


image = cv2.imread("/home/jarvis/a1/assets/q1_data/rgb/1776460517.915522575.png")

h, w = image.shape[:2]
# import ipdb; ipdb.set_trace()
# in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:1] >= 0) & (uv[:, 0] < h)

in_bounds = (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)

for (u, v), z in zip(uv[in_bounds], filtered_new_points[in_bounds][:, 2]):
    color = int(255 * min(z / 50, 1))
    cv2.circle(image, (int(u), int(v)), 1, (0, 255 - color, color), -1)


cv2.imwrite("projected_img.png", image)


import ipdb; ipdb.set_trace()

pcd_new = o3d.geometry.PointCloud()
pcd_new.points = o3d.utility.Vector3dVector(new_points)

o3d.visualization.draw_geometries([pcd_new]) # This works

print(new_points)