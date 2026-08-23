import numpy as np
import open3d as o3d



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


print(rotation_matrix)


pcd = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460517.915522575.pcd")
points = np.asarray(pcd.points)

print(type(pcd.points)) #<class 'open3d.cpu.pybind.utility.Vector3dVector'>


# camera intrinsics
K = np.array([[6.442133178710937500e+02, 0.000000000000000000e+00, 6.526773681640625000e+02],
[0.000000000000000000e+00, 6.434055786132812500e+02, 3.712568664550781250e+02],
[0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])

# o3d.utility.Vector3dVector


points_in_camera_frame = (rotation_matrix @ points.T + translation_vector)

new_points = np.ascontiguousarray(points_in_camera_frame.T, dtype=np.float64)

# import ipdb; ipdb.set_trace()

pcd_new = o3d.geometry.PointCloud()
pcd_new.points = o3d.utility.Vector3dVector(new_points)

# retrived_back_point_cloud = o3d.utility.Vector3dVector(points_in_camera_frame)

# import ipdb; ipdb.set_trace()
# o3d.visualization.draw_geometries([pcd_new.points])       # Does not work
o3d.visualization.draw_geometries([pcd_new]) # This works

print(new_points)