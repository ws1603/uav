# ui/panels/chart_panel.py

import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QCheckBox, QPushButton, QComboBox, QLabel
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from core.physics.quadrotor_dynamics import DroneState


class ChartPanel(QWidget):
    """实时图表面板"""

    def __init__(self, max_points: int = 1000, parent=None):
        super().__init__(parent)
        self.max_points = max_points

        # 数据缓冲区
        self.time_data = deque(maxlen=max_points)
        self.position_data = {
            'x': deque(maxlen=max_points),
            'y': deque(maxlen=max_points),
            'z': deque(maxlen=max_points)
        }
        self.velocity_data = {
            'vx': deque(maxlen=max_points),
            'vy': deque(maxlen=max_points),
            'vz': deque(maxlen=max_points)
        }
        self.attitude_data = {
            'roll': deque(maxlen=max_points),
            'pitch': deque(maxlen=max_points),
            'yaw': deque(maxlen=max_points)
        }
        self.motor_data = {
            f'motor_{i}': deque(maxlen=max_points) for i in range(4)
        }

        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 工具栏
        toolbar = QHBoxLayout()

        self.combo_time_range = QComboBox()
        self.combo_time_range.addItems(["10秒", "30秒", "60秒", "全部"])
        self.combo_time_range.currentIndexChanged.connect(self._on_time_range_changed)
        toolbar.addWidget(QLabel("时间范围:"))
        toolbar.addWidget(self.combo_time_range)

        toolbar.addStretch()

        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self.clear)
        toolbar.addWidget(btn_clear)

        btn_export = QPushButton("导出")
        btn_export.clicked.connect(self._export_data)
        toolbar.addWidget(btn_export)

        layout.addLayout(toolbar)

        # 图表标签页
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # 位置图表
        self.position_plot = self._create_plot_widget(
            "位置", ["X (m)", "Y (m)", "Z (m)"],
            [('r', 'X'), ('g', 'Y'), ('b', 'Z')]
        )
        self.tab_widget.addTab(self.position_plot['widget'], "位置")

        # 速度图表
        self.velocity_plot = self._create_plot_widget(
            "速度", ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"],
            [('r', 'Vx'), ('g', 'Vy'), ('b', 'Vz')]
        )
        self.tab_widget.addTab(self.velocity_plot['widget'], "速度")

        # 姿态图表
        self.attitude_plot = self._create_plot_widget(
            "姿态", ["Roll (°)", "Pitch (°)", "Yaw (°)"],
            [('r', 'Roll'), ('g', 'Pitch'), ('b', 'Yaw')]
        )
        self.tab_widget.addTab(self.attitude_plot['widget'], "姿态")

        # 电机图表
        self.motor_plot = self._create_plot_widget(
            "电机转速", ["RPM"] * 4,
            [('r', 'M1'), ('g', 'M2'), ('b', 'M3'), ('y', 'M4')]
        )
        self.tab_widget.addTab(self.motor_plot['widget'], "电机")

        # 3D轨迹图
        self.trajectory_widget = self._create_trajectory_widget()
        self.tab_widget.addTab(self.trajectory_widget, "3D轨迹")

    def _create_plot_widget(self, title: str, y_labels: list, curves: list) -> dict:
        """创建绘图部件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建绘图区
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('w')
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.setLabel('bottom', '时间', units='s')
        plot_widget.addLegend()

        # 创建曲线
        curve_items = []
        for color, name in curves:
            pen = pg.mkPen(color=color, width=2)
            curve = plot_widget.plot([], [], pen=pen, name=name)
            curve_items.append(curve)

        layout.addWidget(plot_widget)

        return {
            'widget': widget,
            'plot': plot_widget,
            'curves': curve_items
        }

    def _create_trajectory_widget(self) -> QWidget:
        """创建3D轨迹部件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 使用pyqtgraph的3D功能
        try:
            from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLScatterPlotItem, GLGridItem

            self.gl_widget = GLViewWidget()
            self.gl_widget.setCameraPosition(distance=50)

            # 添加网格
            grid = GLGridItem()
            grid.setSize(100, 100)
            grid.setSpacing(10, 10)
            self.gl_widget.addItem(grid)

            # 轨迹线
            self.trajectory_line = GLLinePlotItem(
                pos=np.zeros((1, 3)),
                color=(0.2, 0.6, 1.0, 1.0),
                width=2,
                antialias=True
            )
            self.gl_widget.addItem(self.trajectory_line)

            # 当前位置点
            self.current_pos_scatter = GLScatterPlotItem(
                pos=np.zeros((1, 3)),
                color=(1.0, 0.0, 0.0, 1.0),
                size=10
            )
            self.gl_widget.addItem(self.current_pos_scatter)

            layout.addWidget(self.gl_widget)
            self.has_3d = True

        except ImportError:
            # 如果没有OpenGL支持，显示2D俯视图
            self.trajectory_plot_2d = pg.PlotWidget()
            self.trajectory_plot_2d.setBackground('w')
            self.trajectory_plot_2d.setLabel('bottom', 'X (m)')
            self.trajectory_plot_2d.setLabel('left', 'Y (m)')
            self.trajectory_plot_2d.setAspectLocked(True)

            self.trajectory_curve_2d = self.trajectory_plot_2d.plot(
                [], [], pen=pg.mkPen('b', width=2)
            )
            self.current_pos_2d = self.trajectory_plot_2d.plot(
                [], [], pen=None, symbol='o', symbolBrush='r', symbolSize=10
            )

            layout.addWidget(self.trajectory_plot_2d)
            self.has_3d = False

        return widget

    def add_data_point(self, time: float, state: DroneState):
        """添加数据点"""
        self.time_data.append(time)

        # 位置
        self.position_data['x'].append(state.position[0])
        self.position_data['y'].append(state.position[1])
        self.position_data['z'].append(-state.position[2])  # 转换为高度

        # 速度
        self.velocity_data['vx'].append(state.velocity[0])
        self.velocity_data['vy'].append(state.velocity[1])
        self.velocity_data['vz'].append(state.velocity[2])

        # 姿态（转换为度）
        euler = np.degrees(state.euler_angles)
        self.attitude_data['roll'].append(euler[0])
        self.attitude_data['pitch'].append(euler[1])
        self.attitude_data['yaw'].append(euler[2])

        # 电机
        for i in range(4):
            self.motor_data[f'motor_{i}'].append(state.motor_speeds[i])

        # 更新图表
        self._update_plots()

    def _update_plots(self):
        """更新图表显示"""
        if len(self.time_data) < 2:
            return

        time_array = np.array(self.time_data)

        # 位置图表
        self.position_plot['curves'][0].setData(
            time_array, np.array(self.position_data['x'])
        )
        self.position_plot['curves'][1].setData(
            time_array, np.array(self.position_data['y'])
        )
        self.position_plot['curves'][2].setData(
            time_array, np.array(self.position_data['z'])
        )

        # 速度图表
        self.velocity_plot['curves'][0].setData(
            time_array, np.array(self.velocity_data['vx'])
        )
        self.velocity_plot['curves'][1].setData(
            time_array, np.array(self.velocity_data['vy'])
        )
        self.velocity_plot['curves'][2].setData(
            time_array, np.array(self.velocity_data['vz'])
        )

        # 姿态图表
        self.attitude_plot['curves'][0].setData(
            time_array, np.array(self.attitude_data['roll'])
        )
        self.attitude_plot['curves'][1].setData(
            time_array, np.array(self.attitude_data['pitch'])
        )
        self.attitude_plot['curves'][2].setData(
            time_array, np.array(self.attitude_data['yaw'])
        )

        # 电机图表
        for i in range(4):
            self.motor_plot['curves'][i].setData(
                time_array, np.array(self.motor_data[f'motor_{i}'])
            )

        # 更新3D轨迹
        self._update_trajectory()

    def _update_trajectory(self):
        """更新3D轨迹"""
        if len(self.position_data['x']) < 2:
            return

        x = np.array(self.position_data['x'])
        y = np.array(self.position_data['y'])
        z = np.array(self.position_data['z'])

        if self.has_3d:
            # 3D轨迹
            pos = np.column_stack([x, y, z])
            self.trajectory_line.setData(pos=pos)
            self.current_pos_scatter.setData(pos=pos[-1:])
        else:
            # 2D俯视图
            self.trajectory_curve_2d.setData(x, y)
            self.current_pos_2d.setData([x[-1]], [y[-1]])

    def clear(self):
        """清空数据"""
        self.time_data.clear()
        for data in [self.position_data, self.velocity_data,
                     self.attitude_data, self.motor_data]:
            for key in data:
                data[key].clear()

        # 清空图表
        for plot_dict in [self.position_plot, self.velocity_plot,
                          self.attitude_plot, self.motor_plot]:
            for curve in plot_dict['curves']:
                curve.setData([], [])

    def _on_time_range_changed(self, index):
        """时间范围改变"""
        time_ranges = [10, 30, 60, None]  # None表示全部
        selected_range = time_ranges[index]

        if selected_range and len(self.time_data) > 0:
            current_time = self.time_data[-1]
            min_time = current_time - selected_range

            for plot_dict in [self.position_plot, self.velocity_plot,
                              self.attitude_plot, self.motor_plot]:
                plot_dict['plot'].setXRange(min_time, current_time)

    def _export_data(self):
        """导出数据"""
        from PyQt5.QtWidgets import QFileDialog
        import csv

        filename, _ = QFileDialog.getSaveFileName(
            self, "导出图表数据", "", "CSV文件 (*.csv)"
        )

        if filename:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 写入表头
                headers = ['时间', 'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz',
                           'Roll', 'Pitch', 'Yaw', 'M1', 'M2', 'M3', 'M4']
                writer.writerow(headers)

                # 写入数据
                for i in range(len(self.time_data)):
                    row = [
                        self.time_data[i],
                        self.position_data['x'][i],
                        self.position_data['y'][i],
                        self.position_data['z'][i],
                        self.velocity_data['vx'][i],
                        self.velocity_data['vy'][i],
                        self.velocity_data['vz'][i],
                        self.attitude_data['roll'][i],
                        self.attitude_data['pitch'][i],
                        self.attitude_data['yaw'][i],
                        self.motor_data['motor_0'][i],
                        self.motor_data['motor_1'][i],
                        self.motor_data['motor_2'][i],
                        self.motor_data['motor_3'][i]
                    ]
                    writer.writerow(row)

            from loguru import logger
            logger.info(f"图表数据已导出到: {filename}")