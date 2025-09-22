import numpy as np


class Polygon2D(object):
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = vertices
        self.vt = np.empty(0)  # 顶点对应的参数化t
        self.update()
        return

    def center(self) -> np.ndarray:
        # 用顶点均值作为中心
        return np.mean(self.vertices, axis=0)

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
                pt = verts_ext[idx]
            else:
                local_t = (val - vt_ext[idx]) / segment_length
                pt = (1 - local_t) * verts_ext[idx] + local_t * verts_ext[idx + 1]

            result_points.append(pt)

        return np.array(result_points)
