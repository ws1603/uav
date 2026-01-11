# # ui/widgets/simple_3d_view.py
#
# """
# 简化3D视图组件 - 使用pyqtgraph
# """
#
# import numpy as np
# from PyQt5.QtWidgets import QWidget, QVBoxLayout
# from PyQt5.QtCore import pyqtSignal
# from loguru import logger
#
# try:
#     import pyqtgraph as pg
#     import pyqtgraph.opengl as gl
#
#     HAS_OPENGL = True
# except ImportError:
#     HAS_OPENGL = False
#     logger.warning("pyqtgraph.opengl不可用，3D视图将使用2D替代")
#
#
# class Simple3DView(QWidget):
#     """简化3D视图"""
#
#     def __init__(self, parent=None):
#         super().__init__(parent)
#
#         self.layout = QVBoxLayout(self)
#         self.layout.setContentsMargins(0, 0, 0, 0)
#
#         # 轨迹历史
#         self.trajectory = []
#         self.max_trajectory_points = 1000
#
#         # 目标位置
#         self.target_position = np.array([0, 0, -10])
#
#         if HAS_OPENGL:
#             self._setup_3d_view()
#         else:
#             self._setup_2d_fallback()
#
#         logger.info("3D视图初始化完成")
#
#     def _setup_3d_view(self):
#         """设置3D视图"""
#         self.view_3d = gl.GLViewWidget()
#         self.layout.addWidget(self.view_3d)
#
#         # 设置背景色
#         self.view_3d.setBackgroundColor(0.1, 0.1, 0.15, 1)
#
#         # 设置相机
#         self.view_3d.setCameraPosition(distance=50, elevation=30, azimuth=45)
#
#         # 添加网格
#         grid = gl.GLGridItem()
#         grid.setSize(100, 100)
#         grid.setSpacing(5, 5)
#         grid.translate(0, 0, 0)
#         self.view_3d.addItem(grid)
#
#         # 添加坐标轴
#         axis = gl.GLAxisItem()
#         axis.setSize(10, 10, 10)
#         self.view_3d.addItem(axis)
#
#         # 无人机模型（简化为点+线框）
#         self.drone_scatter = gl.GLScatterPlotItem(
#             pos=np.array([[0, 0, 0]]),
#             size=15,
#             color=(0, 1, 0, 1),  # 绿色
#             pxMode=True
#         )
#         self.view_3d.addItem(self.drone_scatter)
#
#         # 无人机机臂
#         self._create_drone_arms()
#
#         # 目标点
#         self.target_scatter = gl.GLScatterPlotItem(
#             pos=np.array([[0, 0, 0]]),
#             size=20,
#             color=(1, 0, 0, 1),  # 红色
#             pxMode=True
#         )
#         self.view_3d.addItem(self.target_scatter)
#
#         # 轨迹线
#         self.trajectory_line = gl.GLLinePlotItem(
#             pos=np.array([[0, 0, 0], [0, 0, 0]]),
#             color=(0, 0.7, 1, 0.8),
#             width=2,
#             antialias=True
#         )
#         self.view_3d.addItem(self.trajectory_line)
#
#         # 到目标的连线
#         self.target_line = gl.GLLinePlotItem(
#             pos=np.array([[0, 0, 0], [0, 0, 0]]),
#             color=(1, 1, 0, 0.5),
#             width=1,
#             antialias=True
#         )
#         self.view_3d.addItem(self.target_line)
#
#     def _create_drone_arms(self):
#         """创建无人机机臂"""
#         self.drone_arms = []
#         arm_length = 2.0
#
#         # 四个机臂（X型布局）
#         arm_positions = [
#             [arm_length, arm_length, 0],  # 前右
#             [-arm_length, arm_length, 0],  # 前左
#             [-arm_length, -arm_length, 0],  # 后左
#             [arm_length, -arm_length, 0],  # 后右
#         ]
#
#         for i, end_pos in enumerate(arm_positions):
#             arm = gl.GLLinePlotItem(
#                 pos=np.array([[0, 0, 0], end_pos]),
#                 color=(0.8, 0.8, 0.8, 1),
#                 width=3
#             )
#             self.view_3d.addItem(arm)
#             self.drone_arms.append(arm)
#
#         # 连接对角机臂
#         self.body_lines = []
#         for i in range(4):
#             j = (i + 2) % 4
#             line = gl.GLLinePlotItem(
#                 pos=np.array([arm_positions[i], arm_positions[j]]),
#                 color=(0.5, 0.5, 0.5, 0.5),
#                 width=1
#             )
#             self.view_3d.addItem(line)
#             self.body_lines.append(line)
#
#     def _setup_2d_fallback(self):
#         """设置2D回退视图"""
#         from PyQt5.QtWidgets import QLabel
#         self.label = QLabel("3D视图不可用\n请安装 PyOpenGL:\npip install PyOpenGL PyOpenGL_accelerate")
#         self.label.setStyleSheet("""
#             background-color: #1a1a2e;
#             color: #888;
#             font-size: 14px;
#             padding: 20px;
#         """)
#         from PyQt5.QtCore import Qt
#         self.label.setAlignment(Qt.AlignCenter)
#         self.layout.addWidget(self.label)
#
#     def update_drone_state(self, position: np.ndarray, euler_angles: np.ndarray):
#         """更新无人机状态"""
#         if not HAS_OPENGL:
#             return
#
#         # 转换坐标（NED到可视化坐标：X->X, Y->Y, Z->-Z）
#         vis_pos = np.array([position[0], position[1], -position[2]])
#
#         # 更新无人机位置
#         self.drone_scatter.setData(pos=np.array([vis_pos]))
#
#         # 更新机臂（根据姿态旋转）
#         arm_length = 2.0
#         roll, pitch, yaw = euler_angles
#
#         # 旋转矩阵（简化版）
#         cy, sy = np.cos(yaw), np.sin(yaw)
#         cp, sp = np.cos(pitch), np.sin(pitch)
#         cr, sr = np.cos(roll), np.sin(roll)
#
#         R = np.array([
#             [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
#             [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
#             [-sp, cp * sr, cp * cr]
#         ])
#
#         # 更新机臂位置
#         arm_ends_body = [
#             np.array([arm_length, arm_length, 0]),
#             np.array([-arm_length, arm_length, 0]),
#             np.array([-arm_length, -arm_length, 0]),
#             np.array([arm_length, -arm_length, 0]),
#         ]
#
#         for i, arm in enumerate(self.drone_arms):
#             end_world = R @ arm_ends_body[i] + vis_pos
#             arm.setData(pos=np.array([vis_pos, end_world]))
#
#         # 更新到目标的连线
#         target_vis = np.array([
#             self.target_position[0],
#             self.target_position[1],
#             -self.target_position[2]
#         ])
#         self.target_line.setData(pos=np.array([vis_pos, target_vis]))
#
#         # 添加到轨迹
#         self.trajectory.append(vis_pos.copy())
#         if len(self.trajectory) > self.max_trajectory_points:
#             self.trajectory.pop(0)
#
#         # 更新轨迹线
#         if len(self.trajectory) > 1:
#             self.trajectory_line.setData(pos=np.array(self.trajectory))
#
#     def set_target_position(self, position: np.ndarray):
#         """设置目标位置"""
#         self.target_position = position.copy()
#
#         if HAS_OPENGL:
#             vis_pos = np.array([position[0], position[1], -position[2]])
#             self.target_scatter.setData(pos=np.array([vis_pos]))
#
#     def clear_trajectory(self):
#         """清除轨迹"""
#         self.trajectory = []
#         if HAS_OPENGL:
#             self.trajectory_line.setData(pos=np.array([[0, 0, 0], [0, 0, 0]]))
#
#     def reset_view(self):
#         """重置视图"""
#         if HAS_OPENGL:
#             self.view_3d.setCameraPosition(distance=50, elevation=30, azimuth=45)
#         self.clear_trajectory()


# ui/widgets/simple_3d_view.py（修复坐标转换）

"""
简化3D视图组件 - 修复版
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from loguru import logger

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    logger.warning("pyqtgraph.opengl不可用")


class Simple3DView(QWidget):
    """简化3D视图"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 轨迹历史
        self.trajectory = []
        self.max_trajectory_points = 2000

        # 目标位置 (NED)
        self.target_position = np.array([0, 0, -10])

        if HAS_OPENGL:
            self._setup_3d_view()
        else:
            self._setup_fallback()

        logger.info("3D视图初始化完成")

    def _ned_to_vis(self, ned_pos: np.ndarray) -> np.ndarray:
        """
        NED坐标转可视化坐标
        NED: X北, Y东, Z下
        可视化: X右(东), Y前(北), Z上

        转换: vis_x = ned_y, vis_y = ned_x, vis_z = -ned_z
        """
        return np.array([ned_pos[1], ned_pos[0], -ned_pos[2]])

    def _setup_3d_view(self):
        """设置3D视图"""
        self.view = gl.GLViewWidget()
        self.layout.addWidget(self.view)

        # 背景色
        self.view.setBackgroundColor(0.08, 0.08, 0.12, 1)

        # 相机位置
        self.view.setCameraPosition(distance=60, elevation=25, azimuth=45)

        # 地面网格
        grid = gl.GLGridItem()
        grid.setSize(100, 100)
        grid.setSpacing(5, 5)
        grid.setColor((0.3, 0.3, 0.3, 0.5))
        self.view.addItem(grid)

        # 坐标轴 (可视化坐标系)
        # X轴 - 红色 (东)
        axis_x = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [15, 0, 0]]),
            color=(1, 0.3, 0.3, 1), width=2
        )
        self.view.addItem(axis_x)

        # Y轴 - 绿色 (北)
        axis_y = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 15, 0]]),
            color=(0.3, 1, 0.3, 1), width=2
        )
        self.view.addItem(axis_y)

        # Z轴 - 蓝色 (上)
        axis_z = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 15]]),
            color=(0.3, 0.3, 1, 1), width=2
        )
        self.view.addItem(axis_z)

        # 无人机位置标记
        self.drone_point = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]]),
            size=20,
            color=(0, 1, 0.5, 1),
            pxMode=True
        )
        self.view.addItem(self.drone_point)

        # 无人机机体 (简化为十字)
        self.drone_body = []
        for i in range(2):
            line = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=(0.9, 0.9, 0.9, 1),
                width=3
            )
            self.view.addItem(line)
            self.drone_body.append(line)

        # 目标点
        self.target_point = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 10]]),
            size=25,
            color=(1, 0.2, 0.2, 1),
            pxMode=True
        )
        self.view.addItem(self.target_point)

        # 目标点标记圈
        theta = np.linspace(0, 2 * np.pi, 32)
        circle = np.column_stack([np.cos(theta) * 2, np.sin(theta) * 2, np.zeros(32)])
        self.target_circle = gl.GLLinePlotItem(
            pos=circle,
            color=(1, 0.3, 0.3, 0.8),
            width=2
        )
        self.view.addItem(self.target_circle)

        # 轨迹线
        self.trajectory_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0.1]]),
            color=(0.2, 0.8, 1, 0.9),
            width=2.5,
            antialias=True
        )
        self.view.addItem(self.trajectory_line)

        # 到目标的引导线
        self.guide_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            color=(1, 1, 0, 0.4),
            width=1
        )
        self.view.addItem(self.guide_line)

        # 高度指示线（从无人机到地面）
        self.altitude_line = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, 0]]),
            color=(0.5, 0.5, 1, 0.5),
            width=1
        )
        self.view.addItem(self.altitude_line)

    def _setup_fallback(self):
        """2D回退"""
        label = QLabel("3D视图不可用\n\n请安装:\npip install PyOpenGL PyOpenGL_accelerate")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            background-color: #1a1a2e;
            color: #888;
            font-size: 14px;
        """)
        self.layout.addWidget(label)

    def update_drone_state(self, position: np.ndarray, euler_angles: np.ndarray):
        """更新无人机状态"""
        if not HAS_OPENGL:
            return

        # NED转可视化坐标
        vis_pos = self._ned_to_vis(position)

        # 更新无人机位置
        self.drone_point.setData(pos=np.array([vis_pos]))

        # 更新机体（根据姿态旋转）
        roll, pitch, yaw = euler_angles
        arm = 2.0  # 机臂长度（可视化用）

        # 计算旋转后的机臂端点
        # 注意：这里yaw需要调整因为坐标系转换
        cy, sy = np.cos(-yaw), np.sin(-yaw)  # 负号因为坐标转换
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        # 简化旋转（主要考虑yaw）
        arm1_dir = np.array([cy - sy, cy + sy, 0]) * arm / 1.414
        arm2_dir = np.array([cy + sy, -cy + sy, 0]) * arm / 1.414

        self.drone_body[0].setData(pos=np.array([
            vis_pos - arm1_dir,
            vis_pos + arm1_dir
        ]))
        self.drone_body[1].setData(pos=np.array([
            vis_pos - arm2_dir,
            vis_pos + arm2_dir
        ]))

        # 更新引导线
        target_vis = self._ned_to_vis(self.target_position)
        self.guide_line.setData(pos=np.array([vis_pos, target_vis]))

        # 更新高度指示线
        ground_pos = vis_pos.copy()
        ground_pos[2] = 0
        self.altitude_line.setData(pos=np.array([vis_pos, ground_pos]))

        # 添加到轨迹
        self.trajectory.append(vis_pos.copy())
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)

        # 更新轨迹线
        if len(self.trajectory) >= 2:
            self.trajectory_line.setData(pos=np.array(self.trajectory))

    def set_target_position(self, position: np.ndarray):
        """设置目标位置"""
        self.target_position = position.copy()

        if HAS_OPENGL:
            vis_pos = self._ned_to_vis(position)
            self.target_point.setData(pos=np.array([vis_pos]))

            # 更新目标圈位置
            theta = np.linspace(0, 2 * np.pi, 32)
            r = 2
            circle = np.column_stack([
                vis_pos[0] + np.cos(theta) * r,
                vis_pos[1] + np.sin(theta) * r,
                np.ones(32) * vis_pos[2]
            ])
            self.target_circle.setData(pos=circle)

    def clear_trajectory(self):
        """清除轨迹"""
        self.trajectory = []
        if HAS_OPENGL:
            self.trajectory_line.setData(pos=np.array([[0, 0, 0], [0, 0, 0.1]]))

    def reset_view(self):
        """重置视图"""
        if HAS_OPENGL:
            self.view.setCameraPosition(distance=60, elevation=25, azimuth=45)
        self.clear_trajectory()
