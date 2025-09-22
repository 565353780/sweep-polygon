import numpy as np
import open3d as o3d

from sweep_polygon.Data.polygon2d import Polygon2D
from sweep_polygon.Method.render import createNormalLineSet, getPcd


def generate_ellipse_polygon(a, b, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.stack((x, y), axis=1)


def test():
    # 椭圆参数
    a, b = 2.0, 1.0
    num_polygon_vertices = 50

    # 构建多边形
    ellipse_vertices = generate_ellipse_polygon(a, b, num_polygon_vertices)

    polygon1 = Polygon2D(ellipse_vertices, [0, 0, 0])
    polygon2 = Polygon2D(ellipse_vertices, [1, 0, 4])
    polygon3 = Polygon2D(ellipse_vertices, [1.2, 0, 8])

    # 均匀采样 t 并查询坐标
    ts = np.linspace(0, 1, 100, endpoint=False)

    vn1 = polygon1.query(ts)
    vn2 = polygon2.query(ts)
    vn3 = polygon3.query(ts)

    merge_vn = np.vstack([vn1, vn2, vn3])

    merge_v = merge_vn[:, :3]
    merge_n = merge_vn[:, 3:]

    merge_pcd = getPcd(merge_v, merge_n)

    merge_normals = createNormalLineSet(merge_v, merge_n, length=0.1)

    o3d.visualization.draw_geometries([merge_pcd, merge_normals])
