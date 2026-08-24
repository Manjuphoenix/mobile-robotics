import numpy as np
import open3d as o3d


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


if __name__ == "__main__":
    T1 = random_transformation_matrix(translation_range=(-0.1, 0.4), seed=1)
    
    print("T1 =\n", T1)
    # print("\nT2 =\n", T2)
    
    # Sanity checks: rotation part should be orthonormal with det = 1
    # for name, T in [("T1", T1), ("T2", T2)]:
    #     R = T[:3, :3]
    #     print(f"\n{name} det(R) = {np.linalg.det(R):.4f}")
    #     print(f"{name} R @ R.T ≈ I:\n{np.round(R @ R.T, 4)}")

    # Read the toothless point cloud data here..
    pcd = o3d.io.read_point_cloud("/home/jarvis/a1/assets/toothless.ply")
    points = np.asarray(pcd.points)
    # o3d.visualization.draw_geometries([pcd])

    R1 = T1[:3, :3]
    Tl1 = T1[:3, 3:]


    T2 = random_transformation_matrix(translation_range=(-0.5, 0.9), seed=10)
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

    pcd = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460674.916070223.pcd")
    points = np.asarray(pcd.points)

    import ipdb; ipdb.set_trace()

