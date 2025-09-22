import numpy as np

from sweep_polygon.Data.polygon2d import Polygon2D
from sweep_polygon.Method.render import renderPoints


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

    sampled_points1 = polygon1.queryPoints(ts)
    sampled_points2 = polygon2.queryPoints(ts)
    sampled_points3 = polygon3.queryPoints(ts)

    merge_pts = np.vstack([sampled_points1, sampled_points2, sampled_points3])

    renderPoints(merge_pts)
