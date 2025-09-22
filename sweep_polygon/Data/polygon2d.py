import numpy as np
import open3d as o3d
from typing import Union

from sweep_polygon.Method.render import createNormalLineSet, getPcd


class Polygon2D(object):
    def __init__(
        self, vertices: np.ndarray, position: Union[np.ndarray, list, None] = None
    ) -> None:
        self.vertices = vertices

        if position is None:
            self.position = np.zeros(3)  # [x,y,z]
        else:
            position = np.asarray(position)
            if position.shape != (3,):
                raise ValueError("Position must be a 3-element array")
            self.position = position

        self.normals = self.compute_vertex_normals()

        self.vt = np.empty(0)  # 顶点对应的参数化t
        self.update()

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

    def center(self) -> np.ndarray:
        # 用顶点均值作为中心，再加上位置偏移（x,y,z）
        local_center = np.mean(self.vertices, axis=0)  # (x,y)
        global_center = np.array([local_center[0], local_center[1], 0]) + self.position
        return global_center

    def update(self) -> bool:
        # 计算每条边长度
        verts_ext = np.vstack([self.vertices, self.vertices[0]])  # 尾部加头部闭合
        edges = verts_ext[1:] - verts_ext[:-1]
        edge_lengths = np.linalg.norm(edges, axis=1)

        # 累积长度
        cum_length = np.cumsum(edge_lengths)
        total_length = cum_length[-1]

        # 顶点对应的归一化参数t（首顶点t=0）
        self.vt = np.hstack(([0], cum_length[:-1])) / total_length

        return True

    def queryPoints(self, t: np.ndarray) -> np.ndarray:
        # 参数t归一化到[0,1)
        t = np.mod(t, 1.0)

        # 为方便插值，把vt和vertices闭合一圈
        vt_ext = np.hstack((self.vt, 1.0))
        verts_ext = np.vstack([self.vertices, self.vertices[0]])

        result_points = []
        for val in t:
            # 找到参数val对应的边索引
            idx = np.searchsorted(vt_ext, val) - 1
            idx = np.clip(idx, 0, len(self.vt) - 1)

            # 计算插值比例
            segment_length = vt_ext[idx + 1] - vt_ext[idx]
            if segment_length == 0:
                # 处理特殊情况，边长为0，直接取顶点
                pt_2d = verts_ext[idx]
            else:
                local_t = (val - vt_ext[idx]) / segment_length
                pt_2d = (1 - local_t) * verts_ext[idx] + local_t * verts_ext[idx + 1]

            # 转成三维点，z分量来自position的第三个分量
            pt_3d = np.array([pt_2d[0], pt_2d[1], 0]) + self.position
            result_points.append(pt_3d)

        return np.array(result_points)

    def queryNormals(self, t: np.ndarray) -> np.ndarray:
        t = np.mod(t, 1.0)

        vt_ext = np.hstack((self.vt, 1.0))
        verts_ext = np.vstack([self.vertices, self.vertices[0]])

        normals = []

        for val in t:
            # 找到对应的边索引
            idx = np.searchsorted(vt_ext, val) - 1
            idx = np.clip(idx, 0, len(self.vt) - 1)

            # 判断是否与某个顶点重合（用一个容差，比如1e-8）
            close_to_vertex = np.isclose(val, self.vt, atol=1e-8)
            if np.any(close_to_vertex):
                vertex_idx = np.argmax(close_to_vertex)
                n = len(self.vertices)

                # 计算前一条边方向向量
                prev_idx = (vertex_idx - 1) % n
                edge_prev = verts_ext[vertex_idx] - verts_ext[prev_idx]
                edge_prev /= np.linalg.norm(edge_prev)

                # 计算后一条边方向向量
                next_idx = vertex_idx
                edge_next = verts_ext[next_idx + 1] - verts_ext[next_idx]
                edge_next /= np.linalg.norm(edge_next)

                # 平均方向
                direction = edge_prev + edge_next
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    # 极端情况，前后方向完全相反，直接用前方向
                    direction = edge_prev
                else:
                    direction /= norm
            else:
                # 在边上，方向是该边向量
                edge = verts_ext[idx + 1] - verts_ext[idx]
                norm = np.linalg.norm(edge)
                if norm == 0:
                    # 边长为0，直接取零向量（或者其他处理）
                    direction = np.array([0.0, 0.0])
                else:
                    direction = edge / norm

            # 计算法线向量 [-y, x]
            normal_2d = np.array([-direction[1], direction[0]])
            normal_2d /= np.linalg.norm(normal_2d)  # 保证单位向量

            # 转换成三维，z=0，考虑位置偏移
            normal_3d = np.array([normal_2d[0], normal_2d[1], 0.0])

            normals.append(normal_3d)

        return np.array(normals)

    def renderVertices(self, normal_length: float = 0.01) -> bool:
        vertices = self.vertices3d
        normals = self.normals3d

        pcd = getPcd(vertices, normals)
        normal_line_set = createNormalLineSet(vertices, normals, normal_length)

        o3d.visualization.draw_geometries([pcd, normal_line_set])
        return True
