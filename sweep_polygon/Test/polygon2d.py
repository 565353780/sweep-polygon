import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from sweep_polygon.Data.polygon2d import Polygon2D


def generate_ellipse_polygon(a, b, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.stack((x, y), axis=1)


def visualize_polygon_and_samples(polygon: Polygon2D, samples: np.ndarray):
    # 创建LineSet用于显示polygon边界
    verts = polygon.vertices
    lines = [[i, (i + 1) % len(verts)] for i in range(len(verts))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(
            np.hstack([verts, np.zeros((len(verts), 1))])
        ),  # (N,3)
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(lines))

    # 创建点云用于显示采样点
    samples_3d = np.hstack([samples, np.zeros((len(samples), 1))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(samples_3d)

    # 给采样点上色（彩虹色）
    colors = plt.get_cmap("hsv")(np.linspace(0, 1, len(samples)))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([line_set, pcd])


def test():
    # 椭圆参数
    a, b = 2.0, 1.0
    num_polygon_vertices = 50

    # 构建多边形
    ellipse_vertices = generate_ellipse_polygon(a, b, num_polygon_vertices)
    polygon = Polygon2D(ellipse_vertices)

    # 均匀采样 t 并查询坐标
    ts = np.linspace(0, 1, 200, endpoint=False)
    sampled_points = polygon.queryPoints(ts)

    # 可视化
    visualize_polygon_and_samples(polygon, sampled_points)
