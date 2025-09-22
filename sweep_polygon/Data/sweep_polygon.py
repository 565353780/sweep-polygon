import numpy as np
from typing import Union
from copy import deepcopy
from scipy.interpolate import interp1d

from sweep_polygon.Data.polygon2d import Polygon2D


class SweepPolygon(object):
    def __init__(self):
        self.polygons = []
        return

    def addPolygon(
        self, vertices: np.ndarray, position: Union[np.ndarray, list, None] = None
    ) -> bool:
        polygon = Polygon2D(vertices, position)
        self.polygons.append(polygon)

        self.polygons.sort(key=lambda p: p.position[2])
        return True

    @property
    def heights(self) -> np.ndarray:
        heights = [p.position[2] for p in self.polygons]
        return np.array(heights)

    @property
    def size(self) -> int:
        return len(self.polygons)

    def createLinearQueryFunc(self) -> bool:
        polygon_num = self.size

        weights = np.eye(polygon_num)

        self.query_func = interp1d(
            self.heights, weights, kind="linear", axis=0, fill_value="extrapolate"
        )
        return True

    def query(self, t_h_array: np.ndarray) -> np.ndarray:
        """
        输入:
            t_h_array: np.ndarray, shape (N, 2)
                每行是 [t, h]，t ∈ [0,1) 表示多边形边界参数，
                h 表示高度值
        返回:
            points: np.ndarray, shape (N, 6)
                对应每个 (t,h) 点的三维坐标
        """

        query_vn = np.zeros([t_h_array.shape[0], 6])

        query_t = t_h_array[:, 0]
        query_h = t_h_array[:, 1]

        weights = self.query_func(query_h)
        for i in range(self.size):
            curr_weights = weights[:, i]

            valid_weight_idxs = np.where(curr_weights > 0)[0]

            curr_query_t = deepcopy(query_t)
            curr_query_t[curr_weights == 0] = -1.0

            curr_vn = self.polygons[i].query(curr_query_t)

            query_vn[valid_weight_idxs] += (
                curr_weights[valid_weight_idxs].reshape(-1, 1)
                * curr_vn[valid_weight_idxs]
            )

        return query_vn
