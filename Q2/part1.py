import hashlib
from dataclasses import dataclass

import numpy
import numpy as np
import math

@dataclass(frozen=True)
class Q2Instance:
    primary_euler_angles: numpy.ndarray
    secondary_euler_angles: numpy.ndarray
    gimbal_alpha: float
    gimbal_gamma: float
    gimbal_offset: float


def seed_from_text(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_signed_angle(
    random_generator: numpy.random.Generator,
    minimum_magnitude: float,
    maximum_magnitude: float,
) -> float:
    magnitude = float(random_generator.uniform(minimum_magnitude, maximum_magnitude))
    sign = -1.0 if int(random_generator.integers(0, 2)) == 0 else 1.0
    return sign * magnitude


def generate_q2_instance(roll_number: str) -> Q2Instance:
    normalized_roll_number = roll_number.strip()
    if not normalized_roll_number or normalized_roll_number.upper() == "TODO":
        raise ValueError("replace ROLL_NUMBER with your roll number")

    random_generator = numpy.random.default_rng(
        seed_from_text(f"{normalized_roll_number}|public")
    )
    primary_angles = numpy.array(
        [
            sample_signed_angle(random_generator, 0.45, 1.20),
            sample_signed_angle(random_generator, 0.30, 0.90),
            sample_signed_angle(random_generator, 0.50, 1.40),
        ]
    )
    secondary_angles = numpy.array(
        [
            sample_signed_angle(random_generator, 0.25, 1.00),
            sample_signed_angle(random_generator, 0.20, 0.80),
            sample_signed_angle(random_generator, 0.35, 1.25),
        ]
    )
    gimbal_alpha = sample_signed_angle(random_generator, 0.40, 1.30)
    gimbal_gamma = sample_signed_angle(random_generator, 0.40, 1.30)
    gimbal_offset = float(random_generator.uniform(0.35, 0.95))
    return Q2Instance(
        primary_euler_angles=primary_angles,
        secondary_euler_angles=secondary_angles,
        gimbal_alpha=gimbal_alpha,
        gimbal_gamma=gimbal_gamma,
        gimbal_offset=gimbal_offset,
    )


ROLL_NUMBER = "2026702017"
q2_instance = generate_q2_instance(ROLL_NUMBER)
q2_instance




def euler_xyz_to_matrix(
    alpha: float,
    beta: float,
    gamma: float,
) -> numpy.ndarray:
    """
    alpha is the rotation about x axis,
    beta is the rotation about y axis,
    gamma is the rotation about the z axis.
    output: numpy array with dim 3x3 which is nothing but the rotation matrix for the given euler angles.
    """

    """
    First compute the rotation matrix along the z axis and similarly for y and x axis respectively.
    Once the rotation matrices around the different axis are obtained then just
    multiply them in the specific order.
    """

    ############ Rotation about the x axis ###############
    R_x = np.array([[1.000, 0.000, 0.000],
                    [0.000, math.cos(alpha), -math.sin(alpha)],
                    [0.000, math.sin(alpha), math.cos(alpha)]])

    ############ Rotation about the y axis ###############
    R_y = np.array([[math.cos(beta), 0.000, math.sin(beta)],
                    [0.000, 1.000, 0.000],
                    [-math.sin(beta), 0.000, math.cos(beta)]])

    ############ Rotation about the z axis ###############
    R_z = np.array([[math.cos(gamma), -math.sin(gamma), 0.000],
                    [math.sin(gamma), math.cos(gamma), 0.0000],
                    [0.000, 0.000, 1.000]])

    R_tmp = R_y @ R_x
    fin_rotation_matrix = R_z @ R_tmp
    # raise NotImplementedError("Implement the fixed-axis X-Y-Z conversion")

    return fin_rotation_matrix


def axis_angle_to_matrix(axis: numpy.ndarray, angle: float) -> numpy.ndarray:
    """
    axis: numpy array [x, y, z] is the axis vector that has to be rotated by the given angle
    """
    K_x, K_y, K_z = axis[0], axis[1], axis[2]

    v_t = (1-math.cos(angle))
    s_t = math.sin(angle)
    c_t = math.cos(angle)

    angle_axis_rotation = np.array([[(K_x*K_x*v_t + c_t), ((K_x*K_y*v_t) - (K_z*s_t)), ((K_x*K_z*v_t)+K_y*s_t)],
                                    [(K_x*K_y*v_t + K_z*s_t), (K_y*K_y*v_t + c_t), (K_y*K_z*v_t - K_x*s_t)],
                                    [(K_x*K_z*v_t - K_y*s_t), (K_y*K_z*v_t+K_x*s_t), (K_z*K_z*v_t + c_t)]])


    # raise NotImplementedError("Implement the angle-axis conversion")
    return angle_axis_rotation


def matrix_to_axis_angle(
    rotation_matrix: numpy.ndarray,
) -> tuple[numpy.ndarray, float]:
    """
    rotation_matrix is the input which has to be converted back to the axis vector k and angle
    """
    angle = math.acos((rotation_matrix[1][1] + rotation_matrix[2][2] + rotation_matrix[3][3] -1)/2)
    
    k_vector = (1/(2*math.sin(angle))*(np.array([[rotation_matrix[3][2] - rotation_matrix[2][3],
                                                [rotation_matrix[1][3] - rotation_matrix[3][1],
                                                [rotation_matrix[2][1] - rotation_matrix[1][2]]]]])))

    # raise NotImplementedError("Implement the inverse angle-axis conversion")
    return (k_vector, angle)


def quaternion_to_matrix(quaternion: numpy.ndarray) -> numpy.ndarray:
    """
    quaternion: a numpy array with 4 elements, [x, y, z, w] -> x, y, z are the unit coordinate axes,
    and w is the real number that represents the analge of rotation...
    """

    q = quaternion

    quat_rotation_matrix = np.array([[(q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])],
                                    [2*(q[0]*q[3] + q[1]*q[2]), (q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
                                    [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[0]*q[1] + q[2]*q[3]), (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)]])
    # raise NotImplementedError("Implement the quaternion conversion")

    return quat_rotation_matrix


def matrix_to_quaternion(rotation_matrix: numpy.ndarray) -> numpy.ndarray:
    """
    roatation_matrix: given rotation matrix we have to compute the quaternion back..
    """ 

    r = rotation_matrix
    q_0 = (0.5)*(math.sqrt(1+ r[1][1] + r[2][2] + r[3][3]))

    q_1, q_2, q_3 = (1/4*q_0)*np.array([[r[3][2]-r[2][3],
                                                    r[1][3] - r[3][1],
                                                    r[2][1] - r[1][2]]])
    # raise NotImplementedError("Implement the inverse quaternion conversion")
    return np.array(q_0, q_1, q_2, q_3)


ip = q2_instance.primary_euler_angles



########### Orthogonality Error #################
def orthogonality_error(R):
    return np.linalg.norm(R @ R.T - np.eye(3))



########################## Testing for table generation #############################
######################### Method1 #################################
euler_rotation_matrix = euler_xyz_to_matrix(ip[0], ip[1], ip[2])

import ipdb; ipdb.set_trace()

"""
Points to verify:
P1 = [1, 0, 0]
P2 = [0, 1, 0]
P3 = [0, 0, 1]
P4 = [1, 2, 3]


FOR determinant check
Roll Number: det = 1
P1 = , P2 = , P3 = , P4 = ,

FOR Orthogonality error:
Rol Number: 
P1 = 2.2291027918243403e-16, P2 = , P3 = , P4 = ,

FOR Matrix Reconstruction Error:
Rol Number: 
P1 = , P2 = , P3 = , P4 = ,

FOR minimum transformed point error:
Rol Number: 
P1 = , P2 = , P3 = , P4 = ,

"""


######################### Method1 #################################



######################### Method2 #################################




######################### Method3 #################################




######################### Method4 #################################



######################### Method5 #################################


###################### FOR Visualization purposes ########################

import open3d as o3d
import numpy as np

# Single point
point = np.array([[1.0, 0.0, 0.0]])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point)
pcd.paint_uniform_color([1.0, 0.0, 0.0])  # make it visibly red

new_point = point @ euler_rotation_matrix

new_pcd = o3d.geometry.PointCloud()
new_pcd.points = o3d.utility.Vector3dVector(new_point)


# --- Line connecting origin -> new_point ---
origin = np.array([[0.0, 0.0, 0.0]])
line_points = np.vstack([origin, new_point])  # row0=origin, row1=new_point
lines = [[0, 1]]
colors = [[0.0, 0.0, 1.0]]  # blue line

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(line_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)


# Coordinate frame (modern API)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# Use a Visualizer so we can control point size
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(axes)
vis.add_geometry(new_pcd)
vis.add_geometry(line_set)

opt = vis.get_render_option()
opt.point_size = 15.0  # default is often 5.0 or smaller-looking

ctr = vis.get_view_control()
ctr.set_zoom(1.8)
ctr.set_front([0.7, -0.3, 0.6])
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, 0, 1])

vis.run()

vis.capture_screen_image("output.png", do_render=True)

vis.destroy_window()