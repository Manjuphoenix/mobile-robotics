import open3d as o3d
import numpy as np
import os
import sys

sys.path.append("..")
#import open3d_tutorial as o3dtut

#o3dtut.interactive = not "CI"  in os.environ

pcd1 = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460517.915522575.pcd")
pcd2 = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460517.915522575.pcd")
pcd3 = o3d.io.read_point_cloud("/home/jarvis/a1/assets/q1_data/pcd/1776460517.915522575.pcd")
#print(np.asarray(pcd.points))
#o3d.visualization.draw_geometries([pcd1])
#o3d.visualization.draw_geometries([pcd2])
o3d.visualization.draw_geometries([pcd3])

