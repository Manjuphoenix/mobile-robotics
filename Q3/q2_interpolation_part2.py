import numpy as np
import numpy
import open3d as o3d
import os
import imageio
import math


def random_unit_quaternion(rng):
    """
    Generate a uniformly random unit quaternion using Shoemake's method.
    Returns quaternion as (w, x, y, z).
    """
    u1, u2, u3 = rng.uniform(0, 1, 3)
    
    q_w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q_x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q_y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q_z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    
    return np.array([q_w, q_x, q_y, q_z])


def quaternion_to_rotation_matrix(q):
    """
    Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    
    return R


def random_transformation_matrix(translation_range=(-1.0, 1.0), seed=None):
    """
    Generate a random 4x4 homogeneous rigid transformation matrix
    using only NumPy (no SciPy).
    
    Parameters:
        translation_range : tuple (min, max) for random translation components
        seed              : optional int for reproducibility
    
    Returns:
        T : 4x4 numpy array
    """
    rng = np.random.default_rng(seed)
    
    # Uniformly random rotation via random unit quaternion
    q = random_unit_quaternion(rng)
    R = quaternion_to_rotation_matrix(q)
    
    # Random translation vector
    t = rng.uniform(translation_range[0], translation_range[1], size=3)
    
    # Assemble 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def interpolate_transforms(T1, T2, step):
    R1, R2 = T1[:3, :3], T2[:3, :3]
    Tl1, Tl2 = T1[:3, 3:], T2[:3, 3:]

    T = np.eye(4)

    T = (1-step)*T1 + step*T2

    return T


def rotation_matrix_to_euler_angles(rotation_matrix):
    R = rotation_matrix
    pitch = np.arctan2(-R[2][0], math.sqrt(R[0][0]**2 + R[1][0]**2))
    yaw = np.arctan2(R[1][0]/math.cos(pitch), R[0][0]/math.cos(pitch))
    roll = np.arctan2(R[2][1]/math.cos(pitch), R[2][2]/math.cos(pitch))

    return np.array([roll, pitch, yaw])



def euler_xyz_to_rotation_matrix(
    euler_angle
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

    alpha = euler_angle[0]
    beta = euler_angle[1]
    gamma = euler_angle[2]

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



def check(path, start, end, name="path"):
    """
    Sanity check for one interpolated path. Provided for you - do not modify.

    path  : sequence of 4x4 matrices, path[0] at s = 0 and path[-1] at s = 1
    start : the 4x4 pose the path is supposed to begin at
    end   : the 4x4 pose the path is supposed to end at

    Prints the things that are cheap to get wrong: whether the path actually
    lands on the two poses you asked for, and whether the rotation block is
    still a rotation at every step along the way.
    """
    path = np.asarray(path, dtype=float)
    assert path.ndim == 3 and path.shape[1:] == (4, 4), (
        f"expected a sequence of 4x4 matrices, got shape {path.shape}"
    )

    rotations = path[:, :3, :3]
    start_gap = np.abs(path[0] - np.asarray(start, dtype=float)).max()
    end_gap = np.abs(path[-1] - np.asarray(end, dtype=float)).max()
    orthogonality = np.linalg.norm(
        np.transpose(rotations, (0, 2, 1)) @ rotations - np.eye(3), axis=(1, 2)
    )
    determinants = np.linalg.det(rotations)

    print(f"{name}: {len(path)} poses")
    print(f"  endpoints           : |T_0 - start| = {start_gap:.2e},"
          f"   |T_N - end| = {end_gap:.2e}")
    print(f"  max ||R^T R - I||_F : {orthogonality.max():.3e}")
    print(f"  det(R) range        : [{determinants.min():.4f},"
          f" {determinants.max():.4f}]")
    if max(start_gap, end_gap) > 1e-8:
        print("  ^^ this path does not land on the poses you asked for.")
    return {"orthogonality": orthogonality, "determinant": determinants}



def geodesic_angle(T_prev, T_curr):
    R_prev = T_prev[:3, :3]
    R_curr = T_curr[:3, :3]

    # print((np.trace(np.clip(R_curr.T @ R_prev, -1, 1))-1)/2)
    return math.degrees(math.acos((np.trace(np.clip(R_curr.T @ R_prev, -1, 1))-1)/2))



if __name__ == "__main__":
    # T1 = random_transformation_matrix(translation_range=(-1.0, 1.0), seed=1)
    T1 = np.array([[-0.02364325,  0.93781144,  0.34633901, 17.94597789],
       [ 0.55996768, -0.27456738,  0.7816962 , -7.52674192],
       [ 0.82817703,  0.21242049, -0.51865243, -3.06694204],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    print("T1 =\n", T1)
    # print("\nT2 =\n", T2)
    
    # Sanity checks: rotation part should be orthonormal with det = 1
    # for name, T in [("T1", T1), ("T2", T2)]:
    #     R = T[:3, :3]
    #     print(f"\n{name} det(R) = {np.linalg.det(R):.4f}")
    #     print(f"{name} R @ R.T ≈ I:\n{np.round(R @ R.T, 4)}")

    # Read the toothless point cloud data here..
    # pcd = o3d.io.read_point_cloud("/Users/manjunath/mobile-robotics/assets/toothless.ply")
    pcd = o3d.io.read_point_cloud("../assets/toothless.ply")
    points = np.asarray(pcd.points)
    P0 = np.asarray(pcd.points)

    # o3d.visualization.draw_geometries([pcd])

    R1 = T1[:3, :3]
    Tl1 = T1[:3, 3:]


    # T2 = random_transformation_matrix(translation_range=(-1.0, 1.0), seed=10)
    T2 = np.array([[ -0.91200342,  -0.28221969,  -0.29766056, -21.04307262],
       [  0.09231014,   0.56584878,  -0.81932533,   0.76827699],
       [  0.39966061,  -0.77470459,  -0.49000428, -21.84482376],
       [  0.        ,   0.        ,   0.        ,   1.        ]])
    R2 = T2[:3, :3]
    Tl2 = T2[:3, 3:]

    points_in_frame1 = (R1 @ points.T + Tl1)
    new_points_f1 = np.ascontiguousarray(points_in_frame1.T, dtype=np.float64)

    pcd_f1 = o3d.geometry.PointCloud()
    pcd_f1.points =  o3d.utility.Vector3dVector(new_points_f1)


    points_in_frame2 = (R1 @ points.T + Tl1)
    new_points_f2 = np.ascontiguousarray(points_in_frame2.T, dtype=np.float64)

    pcd_f2 = o3d.geometry.PointCloud()
    pcd_f2.points =  o3d.utility.Vector3dVector(new_points_f2)

    # pcd = o3d.io.read_point_cloud("/Users/manjunath/mobile-robotics/assets/q1_data/pcd/1776460674.916070223.pcd")
    # points = np.asarray(pcd.points)

    # import ipdb; ipdb.set_trace()
    
    # output_dir = "linear_interpolation"
    output_dir = "linear_interpolation_part2"
    n_steps = 70

    os.makedirs(output_dir, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=800, height=600)

    all_frames = []
    frame_paths = []

    tmp_paths = []
    geod_error = []

    for i, s in enumerate(np.linspace(0, 1, n_steps)):
        # T_s = interpolate_transforms(T1, T2, s)
        # # import ipdb; ipdb.set_trace()
        e1, e2 = rotation_matrix_to_euler_angles(R1), rotation_matrix_to_euler_angles(R2)

        diff = e2 - e1
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        e_s = e1 + s * diff

        Rs = euler_xyz_to_rotation_matrix(e_s)
        t_s = (1-s)*Tl1 + s*Tl2

        T_s = np.eye(4)
        T_s[:3, :3] = Rs
        T_s[:3, 3:] = t_s
        tmp_paths.append(T_s)

        # import ipdb; ipdb.set_trace()

        # P_s = ((np.asarray(pcd_f1.points) @ T_s[:3, :3]).T + T_s[:3, 3:]).T
        P_s = ((P0 @ T_s[:3, :3]).T + T_s[:3, 3:]).T
        all_frames.append(P_s)

        
        if i > 0:
            geodesic_error = geodesic_angle(tmp_paths[-2], T_s)
            geod_error.append(geodesic_error)



        # pcd_frame_for_gif = o3d.geometry.PointCloud()
        # pcd_frame_for_gif.points = o3d.utility.Vector3dVector(P_s)

        # if pcd.has_colors():
        #     pcd_frame_for_gif.colors = pcd.colors

        # vis.clear_geometries()
        # vis.add_geometry(pcd_frame_for_gif)
        # vis.poll_events()
        # vis.update_renderer()

        # frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        # vis.capture_screen_image(frame_path)
        # frame_paths.append(frame_path)


    # all_points_stacked = np.vstack(all_frames)
    # global_min = all_points_stacked.min(axis=0)
    # global_max = all_points_stacked.max(axis=0)
    # global_center = (global_min + global_max) / 2.0
    # global_extent = np.linalg.norm(global_max - global_min)  # diagonal size of the swept volume

    # # Add one geometry object that we will just mutate every frame (not re-add)
    # pcd_frame_for_gif = o3d.geometry.PointCloud()
    # pcd_frame_for_gif.points = o3d.utility.Vector3dVector(
    #     np.ascontiguousarray(all_frames[0], dtype=np.float64)
    # )

    # vis.add_geometry(pcd_frame_for_gif)

    # ctr = vis.get_view_control()
    # ctr.set_lookat(global_center)
    # ctr.set_front([0.5, -0.5, 0.5])
    # ctr.set_up([0.0, 0.0, 1.0])
    # zoom_value = np.clip(0.0001 / (global_extent + 1e-6), 2.0, 20.0)
    # ctr.set_zoom(zoom_value)


    # for i, P_s in enumerate(all_frames):
    #     pcd_frame_for_gif.points = o3d.utility.Vector3dVector(
    #         np.ascontiguousarray(P_s, dtype=np.float64)
    #     )
    #     vis.update_geometry(pcd_frame_for_gif)   # tell it the points changed
    #     vis.poll_events()
    #     vis.update_renderer()

    #     frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
    #     vis.capture_screen_image(frame_path)
    #     frame_paths.append(frame_path)

    # vis.destroy_window()
    import ipdb; ipdb.set_trace()

    # images = [imageio.imread(p) for p in frame_paths]
    # imageio.mimsave("linear_interpolation_part2.gif", images, fps=15)
    # print(f"Images saved succesfully at{output_dir}")

