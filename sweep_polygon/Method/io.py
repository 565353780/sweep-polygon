import os
import numpy as np
import open3d as o3d
from tqdm import trange

from sweep_polygon.Method.path import createFileFolder, removeFile


def saveAsWNNCNormalPcdFile(
    pcd: o3d.geometry.PointCloud,
    save_pcd_file_path: str,
    overwrite: bool = False,
) -> bool:
    if os.path.exists(save_pcd_file_path):
        if not overwrite:
            return True

        removeFile(save_pcd_file_path)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    createFileFolder(save_pcd_file_path)
    print("[INFO][io::saveAsWNNCNormalPcdFile]")
    print("\t start saving pure pcd data...")
    with open(save_pcd_file_path, "w") as f:
        for i in trange(points.shape[0]):
            x, y, z = points[i]
            nx, ny, nz = normals[i]
            f.write(f"{x} {y} {z} {nx} {ny} {nz}\n")
    return True
