import os
import numpy as np
import open3d as o3d

from sweep_polygon.Data.sweep_polygon import SweepPolygon
from sweep_polygon.Method.io import saveAsWNNCNormalPcdFile
from sweep_polygon.Method.render import cerateNormalLineSet, getPcd


def generate_ellipse_polygon(a, b, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.stack((x, y), axis=1)


def naca4_coordinates(m=0.02, p=0.4, t=0.12, c=1.0, n_points=50):
    """
    生成NACA 4位数翼型截面坐标
    m: 最大弯度，0.02代表2%
    p: 弯度位置，0.4代表40%
    t: 厚度，0.12代表12%
    c: 弦长，默认1.0
    n_points: 顶点数(上下面总和)

    返回：shape (n_points, 2) 顶点坐标数组[x,y]
    """

    # x分布，余弦分布更均匀，前缘点密集
    beta = np.linspace(0, np.pi, n_points // 2)
    x_upper = c * (0.5 * (1 - np.cos(beta)))

    # 厚度分布方程(标准NACA公式)
    yt = (
        5
        * t
        * c
        * (
            0.2969 * np.sqrt(x_upper / c)
            - 0.1260 * (x_upper / c)
            - 0.3516 * (x_upper / c) ** 2
            + 0.2843 * (x_upper / c) ** 3
            - 0.1015 * (x_upper / c) ** 4
        )
    )

    # 弯度线
    yc = np.zeros_like(x_upper)
    dyc_dx = np.zeros_like(x_upper)

    for i, x in enumerate(x_upper):
        if x < p * c:
            yc[i] = m / (p**2) * (2 * p * (x / c) - (x / c) ** 2) * c
            dyc_dx[i] = 2 * m / (p**2) * (p - x / c)
        else:
            yc[i] = (
                m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * (x / c) - (x / c) ** 2) * c
            )
            dyc_dx[i] = 2 * m / ((1 - p) ** 2) * (p - x / c)

    theta = np.arctan(dyc_dx)

    # 上表面坐标
    xu = x_upper - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    # 下表面坐标
    xl = x_upper + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # 合并，注意上下表面点序，使轮廓闭合（从后缘上表面 -> 前缘 -> 后缘下表面）
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])

    vertices = np.vstack([x_coords, y_coords]).T
    return vertices


def test():
    sample_resolution = 20000

    # 构建多边形
    vertices = naca4_coordinates(
        m=0.02, p=0.4, t=0.12, c=1.0, n_points=sample_resolution
    )

    sweep_polygon = SweepPolygon()

    sweep_polygon.addPolygon(vertices, [0, 0, 0])
    sweep_polygon.addPolygon(vertices * 1.2, [1, 0, 4])
    sweep_polygon.addPolygon(vertices * 1.5, [1.2, 0, 8])

    # 均匀采样 t 并查询坐标
    ts = np.linspace(0, 1, 2000, endpoint=False)
    hs = np.linspace(0, 8, 400, endpoint=False)

    t_h_array = np.array(np.meshgrid(ts, hs)).T.reshape(-1, 2)

    sampled_points = sweep_polygon.queryPoints(t_h_array)

    sampled_normals = -1.0 * sweep_polygon.queryNormals(t_h_array)

    pcd = getPcd(sampled_points, sampled_normals)

    normal_lineset = cerateNormalLineSet(sampled_points, sampled_normals, length=1.0)

    saveAsWNNCNormalPcdFile(pcd, "./output/sweep_polygon.ply", overwrite=True)

    o3d.visualization.draw_geometries([pcd, normal_lineset])
