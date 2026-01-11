# core/planning/astar_planner.py

"""
A*路径规划模块
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
from loguru import logger


class OccupancyGrid3D:
    """3D占用栅格地图"""

    def __init__(self, size: Tuple[int, int, int] = (100, 100, 50),
                 resolution: float = 1.0,
                 origin: np.ndarray = None):
        """
        初始化3D栅格地图

        Args:
            size: 栅格尺寸 (nx, ny, nz)
            resolution: 栅格分辨率（米）
            origin: 地图原点
        """
        self.size = size
        self.resolution = resolution
        self.origin = origin if origin is not None else np.zeros(3)

        # 占用栅格 (0=空闲, 1=占用)
        self.grid = np.zeros(size, dtype=np.uint8)

        logger.debug(f"创建3D栅格地图: {size}, 分辨率: {resolution}m")

    def world_to_grid(self, position: np.ndarray) -> Tuple[int, int, int]:
        """世界坐标转栅格索引"""
        grid_pos = (position - self.origin) / self.resolution
        return tuple(np.clip(grid_pos.astype(int), 0, np.array(self.size) - 1))

    def grid_to_world(self, indices: Tuple[int, int, int]) -> np.ndarray:
        """栅格索引转世界坐标"""
        return np.array(indices) * self.resolution + self.origin + self.resolution / 2

    def is_occupied(self, position: np.ndarray) -> bool:
        """检查位置是否被占用"""
        indices = self.world_to_grid(position)
        return bool(self.grid[indices])

    def is_valid(self, indices: Tuple[int, int, int]) -> bool:
        """检查索引是否有效"""
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= self.size[i]:
                return False
        return True

    def set_occupied(self, position: np.ndarray, occupied: bool = True):
        """设置位置占用状态"""
        indices = self.world_to_grid(position)
        if self.is_valid(indices):
            self.grid[indices] = 1 if occupied else 0

    def add_box_obstacle(self, min_corner: np.ndarray, max_corner: np.ndarray):
        """添加长方体障碍物"""
        min_idx = self.world_to_grid(min_corner)
        max_idx = self.world_to_grid(max_corner)

        for x in range(min_idx[0], max_idx[0] + 1):
            for y in range(min_idx[1], max_idx[1] + 1):
                for z in range(min_idx[2], max_idx[2] + 1):
                    if self.is_valid((x, y, z)):
                        self.grid[x, y, z] = 1

    def add_sphere_obstacle(self, center: np.ndarray, radius: float):
        """添加球形障碍物"""
        center_idx = self.world_to_grid(center)
        radius_idx = int(np.ceil(radius / self.resolution))

        for dx in range(-radius_idx, radius_idx + 1):
            for dy in range(-radius_idx, radius_idx + 1):
                for dz in range(-radius_idx, radius_idx + 1):
                    idx = (center_idx[0] + dx, center_idx[1] + dy, center_idx[2] + dz)
                    if self.is_valid(idx):
                        world_pos = self.grid_to_world(idx)
                        if np.linalg.norm(world_pos - center) <= radius:
                            self.grid[idx] = 1


@dataclass(order=True)
class AStarNode:
    """A*节点"""
    f_score: float
    position: Tuple[int, int, int] = field(compare=False)
    g_score: float = field(compare=False)
    parent: Optional['AStarNode'] = field(default=None, compare=False)


class AStarPlanner:
    """A*路径规划器"""

    # 26邻域偏移
    NEIGHBORS_26 = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx != 0 or dy != 0 or dz != 0:
                    NEIGHBORS_26.append((dx, dy, dz))

    def __init__(self, grid: OccupancyGrid3D,
                 diagonal_cost: float = 1.414,
                 heuristic_weight: float = 1.0):
        """
        初始化A*规划器

        Args:
            grid: 占用栅格地图
            diagonal_cost: 对角线移动代价
            heuristic_weight: 启发式权重
        """
        self.grid = grid
        self.diagonal_cost = diagonal_cost
        self.heuristic_weight = heuristic_weight

    def plan(self, start: np.ndarray, goal: np.ndarray,
             max_iterations: int = 100000) -> Optional[List[np.ndarray]]:
        """
        规划路径

        Args:
            start: 起点（世界坐标）
            goal: 终点（世界坐标）
            max_iterations: 最大迭代次数

        Returns:
            路径点列表，如果找不到路径返回None
        """
        start_idx = self.grid.world_to_grid(start)
        goal_idx = self.grid.world_to_grid(goal)

        # 检查起点和终点
        if not self.grid.is_valid(start_idx):
            logger.warning("起点无效")
            return None
        if not self.grid.is_valid(goal_idx):
            logger.warning("终点无效")
            return None
        if self.grid.grid[start_idx]:
            logger.warning("起点被占用")
            return None
        if self.grid.grid[goal_idx]:
            logger.warning("终点被占用")
            return None

        # 初始化
        open_set = []
        closed_set: Set[Tuple[int, int, int]] = set()

        start_node = AStarNode(
            f_score=self._heuristic(start_idx, goal_idx),
            position=start_idx,
            g_score=0.0,
            parent=None
        )

        heapq.heappush(open_set, start_node)
        g_scores = {start_idx: 0.0}

        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1

            # 取出f值最小的节点
            current = heapq.heappop(open_set)

            # 到达目标
            if current.position == goal_idx:
                path = self._reconstruct_path(current)
                logger.info(f"A*找到路径，{len(path)}个点，迭代{iterations}次")
                return path

            if current.position in closed_set:
                continue

            closed_set.add(current.position)

            # 扩展邻居
            for offset in self.NEIGHBORS_26:
                neighbor_pos = (
                    current.position[0] + offset[0],
                    current.position[1] + offset[1],
                    current.position[2] + offset[2]
                )

                # 检查有效性
                if not self.grid.is_valid(neighbor_pos):
                    continue
                if neighbor_pos in closed_set:
                    continue
                if self.grid.grid[neighbor_pos]:
                    continue

                # 计算代价
                move_cost = self._move_cost(offset)
                tentative_g = current.g_score + move_cost

                if neighbor_pos in g_scores and tentative_g >= g_scores[neighbor_pos]:
                    continue

                g_scores[neighbor_pos] = tentative_g
                f_score = tentative_g + self.heuristic_weight * self._heuristic(neighbor_pos, goal_idx)

                neighbor_node = AStarNode(
                    f_score=f_score,
                    position=neighbor_pos,
                    g_score=tentative_g,
                    parent=current
                )

                heapq.heappush(open_set, neighbor_node)

        logger.warning(f"A*未找到路径，迭代{iterations}次")
        return None

    def _heuristic(self, pos: Tuple[int, int, int], goal: Tuple[int, int, int]) -> float:
        """计算启发式值（欧几里得距离）"""
        return np.sqrt(
            (pos[0] - goal[0]) ** 2 +
            (pos[1] - goal[1]) ** 2 +
            (pos[2] - goal[2]) ** 2
        ) * self.grid.resolution

    def _move_cost(self, offset: Tuple[int, int, int]) -> float:
        """计算移动代价"""
        non_zero = sum(1 for o in offset if o != 0)
        if non_zero == 1:
            return self.grid.resolution
        elif non_zero == 2:
            return self.grid.resolution * self.diagonal_cost
        else:
            return self.grid.resolution * 1.732  # sqrt(3)

    def _reconstruct_path(self, node: AStarNode) -> List[np.ndarray]:
        """重建路径"""
        path = []
        current = node

        while current is not None:
            world_pos = self.grid.grid_to_world(current.position)
            path.append(world_pos)
            current = current.parent

        path.reverse()
        return path


class PathSmoother:
    """路径平滑器"""

    @staticmethod
    def smooth_path(path: List[np.ndarray],
                    weight_data: float = 0.5,
                    weight_smooth: float = 0.1,
                    tolerance: float = 0.001,
                    max_iterations: int = 1000) -> List[np.ndarray]:
        """
        使用梯度下降平滑路径

        Args:
            path: 原始路径
            weight_data: 数据权重
            weight_smooth: 平滑权重
            tolerance: 收敛容差
            max_iterations: 最大迭代次数

        Returns:
            平滑后的路径
        """
        if len(path) <= 2:
            return path.copy()

        # 转换为数组
        path_array = np.array(path)
        smoothed = path_array.copy()

        for _ in range(max_iterations):
            change = 0.0

            for i in range(1, len(smoothed) - 1):
                for j in range(3):
                    old_val = smoothed[i, j]

                    # 数据项
                    data_term = weight_data * (path_array[i, j] - smoothed[i, j])

                    # 平滑项
                    smooth_term = weight_smooth * (
                            smoothed[i - 1, j] + smoothed[i + 1, j] - 2 * smoothed[i, j]
                    )

                    smoothed[i, j] += data_term + smooth_term
                    change += abs(old_val - smoothed[i, j])

            if change < tolerance:
                break

        return [smoothed[i] for i in range(len(smoothed))]

    @staticmethod
    def interpolate_path(path: List[np.ndarray],
                         max_segment_length: float = 1.0) -> List[np.ndarray]:
        """
        路径插值

        Args:
            path: 原始路径
            max_segment_length: 最大段长度

        Returns:
            插值后的路径
        """
        if len(path) < 2:
            return path.copy()

        interpolated = [path[0]]

        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            segment = end - start
            length = np.linalg.norm(segment)

            if length > max_segment_length:
                num_segments = int(np.ceil(length / max_segment_length))
                for j in range(1, num_segments):
                    t = j / num_segments
                    point = start + t * segment
                    interpolated.append(point)

            interpolated.append(end)

        return interpolated

    @staticmethod
    def simplify_path(path: List[np.ndarray],
                      epsilon: float = 0.5) -> List[np.ndarray]:
        """
        使用Douglas-Peucker算法简化路径

        Args:
            path: 原始路径
            epsilon: 简化容差

        Returns:
            简化后的路径
        """
        if len(path) <= 2:
            return path.copy()

        # 找到距离最远的点
        start = path[0]
        end = path[-1]

        max_dist = 0.0
        max_idx = 0

        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len > 0:
            line_unit = line_vec / line_len

            for i in range(1, len(path) - 1):
                point_vec = path[i] - start
                proj_len = np.dot(point_vec, line_unit)
                proj_len = np.clip(proj_len, 0, line_len)
                proj_point = start + proj_len * line_unit
                dist = np.linalg.norm(path[i] - proj_point)

                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

        # 递归简化
        if max_dist > epsilon:
            left = PathSmoother.simplify_path(path[:max_idx + 1], epsilon)
            right = PathSmoother.simplify_path(path[max_idx:], epsilon)
            return left[:-1] + right
        else:
            return [start, end]
