import numpy as np
import open3d as o3d
from typing import Union


def getPcd(
    points: np.ndarray, normals: Union[np.ndarray, None] = None
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def renderPoints(points: np.ndarray, normals: Union[np.ndarray, None] = None) -> bool:
    pcd = getPcd(points, normals)
    o3d.visualization.draw_geometries([pcd])
    return True


def createNormalLineSet(points: np.ndarray, normals: np.ndarray, length=0.1):
    """
    points: (N,3)
    normals: (N,3) 单位法线
    length: 法线箭头长度
    """
    lines = []
    line_points = []
    for i, (pt, normal) in enumerate(zip(points, normals)):
        line_points.append(pt)
        line_points.append(pt + normal * length)
        lines.append([2 * i, 2 * i + 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))

    # 线颜色（可选）
    colors = [[1, 0, 0] for _ in lines]  # 红色
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set
