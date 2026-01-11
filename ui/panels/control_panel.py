# ui/panels/control_panel.py

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QSlider, QPushButton, QComboBox, QFormLayout,
    QTabWidget, QCheckBox, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from loguru import logger

from simulation.engine.simulation_core import SimulationEngine


class ControlPanel(QWidget):
    """控制面板"""

    target_changed = pyqtSignal(np.ndarray, float)  # position, yaw

    def __init__(self, engine: SimulationEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 使用标签页组织
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # 目标位置标签页
        target_tab = self._create_target_tab()
        tab_widget.addTab(target_tab, "目标位置")

        # 控制模式标签页
        mode_tab = self._create_mode_tab()
        tab_widget.addTab(mode_tab, "控制模式")

        # 快捷操作
        quick_group = QGroupBox("快捷操作")
        quick_layout = QVBoxLayout(quick_group)

        # 预设位置按钮
        preset_layout = QHBoxLayout()

        btn_takeoff = QPushButton("起飞")
        btn_takeoff.clicked.connect(lambda: self._set_preset_target(0, 0, -10))
        preset_layout.addWidget(btn_takeoff)

        btn_land = QPushButton("降落")
        btn_land.clicked.connect(lambda: self._set_preset_target(0, 0, -0.5))
        preset_layout.addWidget(btn_land)

        btn_hover = QPushButton("悬停")
        btn_hover.clicked.connect(self._hover_at_current)
        preset_layout.addWidget(btn_hover)

        quick_layout.addLayout(preset_layout)

        # 航点按钮
        waypoint_layout = QHBoxLayout()

        btn_wp1 = QPushButton("航点1")
        btn_wp1.clicked.connect(lambda: self._set_preset_target(20, 0, -15))
        waypoint_layout.addWidget(btn_wp1)

        btn_wp2 = QPushButton("航点2")
        btn_wp2.clicked.connect(lambda: self._set_preset_target(20, 20, -15))
        waypoint_layout.addWidget(btn_wp2)

        btn_wp3 = QPushButton("航点3")
        btn_wp3.clicked.connect(lambda: self._set_preset_target(0, 20, -15))
        waypoint_layout.addWidget(btn_wp3)

        btn_home = QPushButton("返航")
        btn_home.clicked.connect(lambda: self._set_preset_target(0, 0, -10))
        waypoint_layout.addWidget(btn_home)

        quick_layout.addLayout(waypoint_layout)

        layout.addWidget(quick_group)

        # 添加弹簧
        layout.addStretch()

    def _create_target_tab(self) -> QWidget:
        """创建目标位置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 位置设置
        pos_group = QGroupBox("目标位置 (NED坐标系)")
        pos_layout = QFormLayout(pos_group)

        # X (北)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-500, 500)
        self.spin_x.setSingleStep(1)
        self.spin_x.setValue(0)
        self.spin_x.setSuffix(" m")
        pos_layout.addRow("X (北):", self.spin_x)

        # Y (东)
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(-500, 500)
        self.spin_y.setSingleStep(1)
        self.spin_y.setValue(0)
        self.spin_y.setSuffix(" m")
        pos_layout.addRow("Y (东):", self.spin_y)

        # Z (高度)
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(-200, 0)
        self.spin_z.setSingleStep(1)
        self.spin_z.setValue(-10)
        self.spin_z.setSuffix(" m")
        pos_layout.addRow("高度:", self.spin_z)

        # 偏航角
        self.spin_yaw = QDoubleSpinBox()
        self.spin_yaw.setRange(-180, 180)
        self.spin_yaw.setSingleStep(5)
        self.spin_yaw.setValue(0)
        self.spin_yaw.setSuffix(" °")
        pos_layout.addRow("偏航角:", self.spin_yaw)

        layout.addWidget(pos_group)

        # 高度滑块
        alt_group = QGroupBox("高度快速调节")
        alt_layout = QVBoxLayout(alt_group)

        self.slider_altitude = QSlider(Qt.Horizontal)
        self.slider_altitude.setRange(1, 100)
        self.slider_altitude.setValue(10)
        self.slider_altitude.valueChanged.connect(self._on_altitude_slider_changed)
        alt_layout.addWidget(self.slider_altitude)

        self.label_altitude = QLabel("10 m")
        self.label_altitude.setAlignment(Qt.AlignCenter)
        alt_layout.addWidget(self.label_altitude)

        layout.addWidget(alt_group)

        # 应用按钮
        btn_apply = QPushButton("应用目标位置")
        btn_apply.clicked.connect(self._apply_target)
        layout.addWidget(btn_apply)

        layout.addStretch()

        return widget

    def _create_mode_tab(self) -> QWidget:
        """创建控制模式标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 控制模式选择
        mode_group = QGroupBox("控制模式")
        mode_layout = QFormLayout(mode_group)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "位置控制",
            "速度控制",
            "姿态控制",
            "手动控制"
        ])
        mode_layout.addRow("模式:", self.combo_mode)

        layout.addWidget(mode_group)

        # 自动驾驶选项
        auto_group = QGroupBox("自动驾驶")
        auto_layout = QVBoxLayout(auto_group)

        self.check_auto_takeoff = QCheckBox("自动起飞")
        auto_layout.addWidget(self.check_auto_takeoff)

        self.check_auto_land = QCheckBox("低电量自动降落")
        auto_layout.addWidget(self.check_auto_land)

        self.check_geofence = QCheckBox("启用地理围栏")
        self.check_geofence.setChecked(True)
        auto_layout.addWidget(self.check_geofence)

        self.check_collision_avoid = QCheckBox("启用避障")
        auto_layout.addWidget(self.check_collision_avoid)

        layout.addWidget(auto_group)

        layout.addStretch()

        return widget

    def _on_altitude_slider_changed(self, value):
        """高度滑块变化"""
        self.label_altitude.setText(f"{value} m")
        self.spin_z.setValue(-value)

    def _apply_target(self):
        """应用目标位置"""
        position = np.array([
            self.spin_x.value(),
            self.spin_y.value(),
            self.spin_z.value()
        ])
        yaw = np.radians(self.spin_yaw.value())

        self.target_changed.emit(position, yaw)
        logger.info(f"设置目标: 位置={position}, 偏航={np.degrees(yaw):.1f}°")

    def _set_preset_target(self, x: float, y: float, z: float):
        """设置预设目标"""
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)
        self.spin_z.setValue(z)
        self._apply_target()

    def _hover_at_current(self):
        """在当前位置悬停"""
        state = self.engine.drone_state
        self.spin_x.setValue(state.position[0])
        self.spin_y.setValue(state.position[1])
        self.spin_z.setValue(state.position[2])
        self._apply_target()
