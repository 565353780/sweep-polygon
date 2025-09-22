import numpy as np
import open3d as o3d

from sweep_polygon.Data.sweep_polygon import SweepPolygon
from sweep_polygon.Method.io import saveAsWNNCNormalPcdFile
from sweep_polygon.Method.render import createNormalLineSet, getPcd


def generate_ellipse_polygon(a, b, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.stack((x, y), axis=1)


def naca4_airfoil_polygon_sharp_trailing_edge(code: str, n_points=100):
    """
    生成NACA翼型闭合多边形顶点（尾缘厚度强制为0，尖锐尾缘）

    code: 4位字符串，例如 '2412'
    n_points: 上下表面各自采样点数（含端点）

    返回：
    points: (2*n_points-1, 2) ndarray，闭合多边形顶点
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must be a 4-digit string.")

    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100

    if p == 0:
        p = 0.0001

    x = np.linspace(0, 1, n_points)

    # 厚度分布
    yt = (t / 0.2) * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )

    # 弦线中线
    yc = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= p:
            yc[i] = m / p**2 * (2 * p * xi - xi**2)
        else:
            yc[i] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xi - xi**2)

    dyc_dx = np.gradient(yc, x)
    theta = np.arctan(dyc_dx)

    # 上表面点，x从尾缘到前缘逆序排列
    xu = x[::-1]
    yu = yc[::-1] + yt[::-1] * np.cos(theta[::-1])
    xu = xu - yt[::-1] * np.sin(theta[::-1])

    # 下表面点，x从前缘到尾缘
    xl = x
    yl = yc - yt * np.cos(theta)
    xl = xl + yt * np.sin(theta)

    # 拼接闭合多边形顶点（避免重复尾缘点）
    lower = np.vstack((xl[1:-1], yl[1:-1])).T
    upper = np.vstack((xu, yu)).T

    points = np.vstack((upper, lower))

    return points


def test():
    sample_resolution = 400

    # 构建多边形
    vertices = naca4_airfoil_polygon_sharp_trailing_edge("2412", sample_resolution)

    sweep_polygon = SweepPolygon()

    sweep_polygon.addPolygon(vertices, [0, 0, 0])
    sweep_polygon.polygons[0].renderVertices()
    exit()
    sweep_polygon.addPolygon(vertices * 1.2, [1, 0, 4])
    sweep_polygon.addPolygon(vertices * 1.5, [1.2, 0, 8])

    # 均匀采样 t 并查询坐标
    ts = np.linspace(0, 1, sample_resolution, endpoint=False)
    hs = np.linspace(0, 8, 1, endpoint=False)

    t_h_array = np.array(np.meshgrid(ts, hs)).T.reshape(-1, 2)

    sampled_points = sweep_polygon.queryPoints(t_h_array)

    sampled_normals = -1.0 * sweep_polygon.queryNormals(t_h_array)

    pcd = getPcd(sampled_points, sampled_normals)

    normal_lineset = createNormalLineSet(sampled_points, sampled_normals, length=1.0)

    saveAsWNNCNormalPcdFile(pcd, "./output/sweep_polygon.ply", overwrite=True)

    o3d.visualization.draw_geometries([pcd, normal_lineset])
