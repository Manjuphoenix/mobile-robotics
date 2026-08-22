from transformations import quaternion_matrix, translation_matrix, concatenate_matrices
import numpy as np

p = np.array([
        0.07093786809565478,
        0.012665553133702897,
        -0.08437335095076476,
        0.4051169630628588,
        0.4006810126971365,
        0.5797585743574961,
        0.5824216408768541,
    ])


# x, y, z = p[0], p[1], p[2]
# qx, qy, qz, qw = p[3], p[4], p[5], p[6]


tvec = np.array([p[0], p[1], p[2]])
quat = np.array([p[6], p[3], p[4], p[5]])

T_trans = translation_matrix(tvec)
T_rot = quaternion_matrix(quat)

T_fin = concatenate_matrices(T_trans, T_rot)

import ipdb; ipdb.set_trace()