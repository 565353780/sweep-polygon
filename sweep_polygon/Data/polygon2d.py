import numpy as np
import open3d as o3d
from typing import Union
from scipy.interpolate import interp1d

from sweep_polygon.Method.render import createNormalLineSet, getPcd


class Polygon2D(object):
    def __init__(
        self, vertices: np.ndarray, position: Union[np.ndarray, list, None] = None
    ) -> None:
        self.vertices = vertices

        if position is None:
            self.position = np.zeros(3)
        else:
            position = np.asarray(position)
            if position.shape != (3,):
                raise ValueError("Position must be a 3-element array")
            self.position = position

        self.normals = self.compute_vertex_normals()

        self.vn = np.hstack([self.vertices, self.normals])

        self.vt = self.compute_vertex_length_t()

        self.createLinearQueryFunc()
        return

    @property
    def vertices3d(self) -> np.ndarray:
        vertices = np.zeros([self.vertices.shape[0], 3])
        vertices[:, :2] = self.vertices
        vertices += self.position
        return vertices

    @property
    def normals3d(self) -> np.ndarray:
        normals = np.zeros([self.normals.shape[0], 3])
        normals[:, :2] = self.normals
        return normals

    def compute_vertex_normals(self):
        verts = self.vertices
        N = verts.shape[0]
        normals = np.zeros_like(verts)

        for i in range(N):
            v_prev = verts[(i - 1) % N]
            v_curr = verts[i]
            v_next = verts[(i + 1) % N]

            e1 = v_curr - v_prev
            e2 = v_next - v_curr

            # 单位化
            e1_norm = e1 / np.linalg.norm(e1)
            e2_norm = e2 / np.linalg.norm(e2)

            bisector = e1_norm + e2_norm
            if np.allclose(bisector, 0):
                # 两边共线，法线直接取垂直e1或e2
                normal = np.array([-e1_norm[1], e1_norm[0]])
            else:
                # 法线为bisector的法线
                normal = np.array([-bisector[1], bisector[0]])
                normal /= np.linalg.norm(normal)

            normals[i] = normal

        # 可选：判断多边形顶点顺序，调整法线方向使其外指
        if self.is_clockwise():
            normals = -normals

        return normals

    def is_clockwise(self):
        """判断多边形顶点顺序是否顺时针，常用面积法"""
        verts = self.vertices
        x = verts[:, 0]
        y = verts[:, 1]
        return np.sum((x[:-1] * y[1:] - x[1:] * y[:-1])) > 0

    def compute_vertex_length_t(self) -> np.ndarray:
        # 计算每条边长度
        verts_ext = np.vstack([self.vertices, self.vertices[0]])  # 尾部加头部闭合
        edges = verts_ext[1:] - verts_ext[:-1]
        edge_lengths = np.linalg.norm(edges, axis=1)

        # 累积长度
        cum_length = np.cumsum(edge_lengths)
        total_length = cum_length[-1]

        # 顶点对应的归一化参数t（首顶点t=0）
        vt = np.hstack(([0], cum_length[:-1])) / total_length
        return vt

    def createLinearQueryFunc(self) -> bool:
        expand_vn = np.vstack([self.vn, self.vn[0, :]])
        expand_vt = np.hstack([self.vt, [1.0]])
        self.query_func = interp1d(
            expand_vt, expand_vn, kind="linear", axis=0, fill_value="extrapolate"
        )
        return True

    def query(self, t: np.ndarray) -> np.ndarray:
        valid_t_idxs = np.where((t >= 0.0) & (t <= 1.0))

        valid_t = t[valid_t_idxs]

        query_vn = self.query_func(valid_t)

        expand_vn = np.zeros([t.shape[0], 6])
        expand_vn[valid_t_idxs, :2] = query_vn[:, :2]
        expand_vn[valid_t_idxs, :3] += self.position
        expand_vn[valid_t_idxs, 3:5] = query_vn[:, 2:]
        return expand_vn

    def renderVertices(self, normal_length: float = 0.01) -> bool:
        vertices = self.vertices3d
        normals = self.normals3d

        pcd = getPcd(vertices, normals)
        normal_line_set = createNormalLineSet(vertices, normals, normal_length)

        o3d.visualization.draw_geometries([pcd, normal_line_set])
        return True
