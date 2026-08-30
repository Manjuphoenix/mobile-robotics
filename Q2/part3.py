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

    # import ipdb; ipdb.set_trace()
    angle = math.acos((rotation_matrix[0][0] + rotation_matrix[1][1] + rotation_matrix[2][2] -1)/2)
    
    # import ipdb; ipdb.set_trace()
    k_vector = 1/(2*math.sin(angle))*(np.array([rotation_matrix[2][1] - rotation_matrix[1][2],
                                                rotation_matrix[0][2] - rotation_matrix[2][0],
                                                rotation_matrix[1][0] - rotation_matrix[0][1]]))

    # raise NotImplementedError("Implement the inverse angle-axis conversion")
    return (k_vector, angle)


def quaternion_to_matrix(quaternion: numpy.ndarray) -> numpy.ndarray:
    """
    quaternion: a numpy array with 4 elements, [x, y, z, w] -> x, y, z are the unit coordinate axes,
    and w is the real number that represents the analge of rotation...
    """
    
    x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    quaternion[0] = w
    quaternion[1] = x
    quaternion[2] = y
    quaternion[3] = z

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
    q_0 = (0.5)*(math.sqrt(1+ r[0][0] + r[1][1] + r[2][2]))

    q_vector = (1/(4*q_0))*np.array([r[2][1]-r[1][2],
                                                    r[0][2] - r[2][0],
                                                    r[1][0] - r[0][1]])
    # raise NotImplementedError("Implement the inverse quaternion conversion")
    return np.array([q_vector[0], q_vector[1], q_vector[2], q_0])


ip = q2_instance.primary_euler_angles


#######################################PART 3################################################################

def quaternion_multiply(
    left_quaternion: numpy.ndarray,
    right_quaternion: numpy.ndarray,
) -> numpy.ndarray:
    # raise NotImplementedError("Implement the Hamilton product")
    x1, y1, z1, w1 = left_quaternion[0], left_quaternion[1], left_quaternion[2], left_quaternion[3]
    x2, y2, z2, w2 = right_quaternion[0], right_quaternion[1], right_quaternion[2], right_quaternion[3]

    return np.array([w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2,
                     w1*w2 - x1*x2 - y1*y2 - z1*z2,])



primary_angles = q2_instance.primary_euler_angles
secondary_angles = q2_instance.secondary_euler_angles


R_primary = euler_xyz_to_matrix(*primary_angles)
R_secondary = euler_xyz_to_matrix(*secondary_angles)


primary_quat = matrix_to_quaternion(R_primary)   # Pass the primary rotation matrix here...
second_quat = matrix_to_quaternion(R_secondary)   # Pass the secondary rotation matrix here...

hamilton_quat = quaternion_multiply(primary_quat, second_quat)
hamilaton_rotation_matrix = quaternion_to_matrix(hamilton_quat)

primary_rotation_matrix = quaternion_to_matrix(primary_quat)
secondary_rotation_matrix = quaternion_to_matrix(second_quat)

abc = secondary_rotation_matrix @ primary_rotation_matrix

error = np.linalg.norm(hamilaton_rotation_matrix - abc, "fro")


import ipdb; ipdb.set_trace()
if np.allclose(hamilaton_rotation_matrix, secondary_rotation_matrix @ primary_rotation_matrix):
    print("HEY its working and both are the same....")