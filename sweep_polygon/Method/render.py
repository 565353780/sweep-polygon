import numpy as np
import open3d as o3d


def renderPoints(points: np.ndarray) -> bool:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd])
    return True
