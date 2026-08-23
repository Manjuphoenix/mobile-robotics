"""
Gimbal Lock Illustration in Open3D
-----------------------------------
Renders an animated GIF showing three gimbal rings (yaw=Z/blue, pitch=Y/green,
roll=X/red) plus a payload coordinate frame. As the pitch angle sweeps toward
90 degrees, the roll ring's rotation axis swings into alignment with the yaw
ring's axis -- illustrating gimbal lock (loss of one rotational DOF).

Requirements:
    pip install open3d imageio numpy

Run:
    python gimbal_lock_open3d.py

Output:
    gimbal_lock.gif  (in the same directory)
"""

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import imageio.v2 as imageio
import os

OUT_DIR = "frames_gimbal_lock"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- rotation helpers ----------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_aligning_z_to(n):
    """Rotation matrix mapping the +Z axis onto unit vector n (Rodrigues)."""
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    c = np.dot(z, n)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else rot_x(np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# ---------- build base ring meshes (in local XY plane, normal = Z) ----------
def make_ring(radius, tube_radius, color):
    torus = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=radius, tube_radius=tube_radius,
        radial_resolution=40, tubular_resolution=20)
    torus.paint_uniform_color(color)
    torus.compute_vertex_normals()
    return torus


yaw_ring = make_ring(1.0, 0.025, [0.20, 0.45, 0.95])   # blue,  Z axis
pitch_ring = make_ring(0.8, 0.025, [0.25, 0.80, 0.30])  # green, Y axis
roll_ring = make_ring(0.6, 0.025, [0.90, 0.25, 0.20])   # red,   X axis

yaw_base_verts = np.asarray(yaw_ring.vertices).copy()
pitch_base_verts = np.asarray(pitch_ring.vertices).copy()
roll_base_verts = np.asarray(roll_ring.vertices).copy()

payload = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.45)
payload_base_verts = np.asarray(payload.vertices).copy()

material = rendering.MaterialRecord()
material.shader = "defaultLit"

# ---------- offscreen renderer ----------
W, H = 640, 640
renderer = rendering.OffscreenRenderer(W, H)
renderer.scene.set_background([1, 1, 1, 1])
renderer.scene.scene.set_sun_light([-0.3, -0.3, -0.9], [1, 1, 1], 60000)
renderer.scene.scene.enable_sun_light(True)

renderer.setup_camera(60.0, [0, 0, 0], [2.2, 1.6, 1.6], [0, 0, 1])

names = ["yaw", "pitch", "roll", "payload"]


def update_scene(yaw, pitch, roll):
    for n in names:
        if renderer.scene.has_geometry(n):
            renderer.scene.remove_geometry(n)

    R_yaw = rot_z(yaw)
    R_yaw_pitch = R_yaw @ rot_y(pitch)
    R_full = R_yaw_pitch @ rot_x(roll)

    n_yaw = np.array([0, 0, 1.0])              # world Z, fixed
    n_pitch = R_yaw @ np.array([0, 1.0, 0])     # tilts with yaw
    n_roll = R_yaw_pitch @ np.array([1.0, 0, 0])  # tilts with yaw+pitch

    yaw_ring.vertices = o3d.utility.Vector3dVector(
        yaw_base_verts @ rotation_aligning_z_to(n_yaw).T)
    pitch_ring.vertices = o3d.utility.Vector3dVector(
        pitch_base_verts @ rotation_aligning_z_to(n_pitch).T)
    roll_ring.vertices = o3d.utility.Vector3dVector(
        roll_base_verts @ rotation_aligning_z_to(n_roll).T)
    payload.vertices = o3d.utility.Vector3dVector(
        payload_base_verts @ R_full.T)

    for mesh in (yaw_ring, pitch_ring, roll_ring):
        mesh.compute_vertex_normals()

    renderer.scene.add_geometry("yaw", yaw_ring, material)
    renderer.scene.add_geometry("pitch", pitch_ring, material)
    renderer.scene.add_geometry("roll", roll_ring, material)
    renderer.scene.add_geometry(
        "payload", payload,
        rendering.MaterialRecord())  # coord frame uses vertex colors


# ---------- animation schedule ----------
# Pitch sweeps 0 -> 95 -> 0 degrees; roll spins continuously so you can see
# it stop producing independent motion once pitch nears 90 (gimbal lock).
frames = []
n_frames = 90
pitch_deg = np.concatenate([
    np.linspace(0, 95, n_frames // 2),
    np.linspace(95, 0, n_frames // 2),
])
roll_deg = np.linspace(0, 360 * 2, n_frames)  # spins twice over the loop
yaw_deg = np.zeros(n_frames)  # keep yaw fixed so the lock is easy to see

for i in range(n_frames):
    update_scene(np.radians(yaw_deg[i]),
                 np.radians(pitch_deg[i]),
                 np.radians(roll_deg[i]))
    img = renderer.render_to_image()
    path = os.path.join(OUT_DIR, f"frame_{i:03d}.png")
    o3d.io.write_image(path, img)
    frames.append(path)

# ---------- assemble GIF ----------
images = [imageio.imread(f) for f in frames]
imageio.mimsave("gimbal_lock.gif", images, duration=0.05, loop=0)
print("Saved gimbal_lock.gif")