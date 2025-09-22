import numpy as np
from typing import Union

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

    def queryPoints(self, t_h_array: np.ndarray) -> np.ndarray:
        """
        输入:
            t_h_array: np.ndarray, shape (N, 2)
                每行是 [t, h]，t ∈ [0,1) 表示多边形边界参数，
                h 表示高度值
        返回:
            points: np.ndarray, shape (N, 3)
                对应每个 (t,h) 点的三维坐标
        """
        if not self.polygons:
            raise RuntimeError("No polygons in SweepPolygon")

        t_array = t_h_array[:, 0]
        h_array = t_h_array[:, 1]

        layer_heights = self.heights
        n_layers = len(self.polygons)

        # 限制h范围，但仍保持原h方便判断越界
        h_min, h_max = layer_heights[0], layer_heights[-1]

        N = t_h_array.shape[0]
        points = np.empty((N, 3))

        for i in range(N):
            t = t_array[i]
            h = h_array[i]

            if h <= h_min:
                # 低于最底层，直接取最底层点
                points[i] = self.polygons[0].queryPoints(np.array([t]))[0]
            elif h >= h_max:
                # 高于最高层，直接取最高层点
                points[i] = self.polygons[-1].queryPoints(np.array([t]))[0]
            else:
                # 找上下层插值
                idx = np.searchsorted(layer_heights, h, side="right") - 1
                low_idx = max(0, idx)
                high_idx = min(n_layers - 1, low_idx + 1)

                h_low = layer_heights[low_idx]
                h_high = layer_heights[high_idx]
                alpha = (h - h_low) / (h_high - h_low)

                p_low = self.polygons[low_idx].queryPoints(np.array([t]))[0]
                p_high = self.polygons[high_idx].queryPoints(np.array([t]))[0]

                points[i] = (1 - alpha) * p_low + alpha * p_high

        return points

    def queryNormals(self, t_h_array: np.ndarray) -> np.ndarray:
        """
        输入:
            t_h_array: np.ndarray, shape (N, 2)
                每行是 [t, h]，t ∈ [0,1) 表示多边形边界参数，
                h 表示高度值
        返回:
            points: np.ndarray, shape (N, 3)
                对应每个 (t,h) 点的三维坐标
        """
        if not self.polygons:
            raise RuntimeError("No polygons in SweepPolygon")

        t_array = t_h_array[:, 0]
        h_array = t_h_array[:, 1]

        layer_heights = self.heights
        n_layers = len(self.polygons)

        # 限制h范围，但仍保持原h方便判断越界
        h_min, h_max = layer_heights[0], layer_heights[-1]

        N = t_h_array.shape[0]
        points = np.empty((N, 3))

        for i in range(N):
            t = t_array[i]
            h = h_array[i]

            if h <= h_min:
                # 低于最底层，直接取最底层点
                points[i] = self.polygons[0].queryNormals(np.array([t]))[0]
            elif h >= h_max:
                # 高于最高层，直接取最高层点
                points[i] = self.polygons[-1].queryNormals(np.array([t]))[0]
            else:
                # 找上下层插值
                idx = np.searchsorted(layer_heights, h, side="right") - 1
                low_idx = max(0, idx)
                high_idx = min(n_layers - 1, low_idx + 1)

                h_low = layer_heights[low_idx]
                h_high = layer_heights[high_idx]
                alpha = (h - h_low) / (h_high - h_low)

                p_low = self.polygons[low_idx].queryNormals(np.array([t]))[0]
                p_high = self.polygons[high_idx].queryNormals(np.array([t]))[0]

                points[i] = (1 - alpha) * p_low + alpha * p_high

        return points
