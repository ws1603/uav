# ui/panels/info_panel.py

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel,
    QProgressBar, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from core.physics.quadrotor_dynamics import DroneState


class InfoPanel(QWidget):
    """飞行信息面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 位置信息
        pos_group = QGroupBox("位置信息")
        pos_layout = QFormLayout(pos_group)

        self.label_x = QLabel("0.00 m")
        self.label_y = QLabel("0.00 m")
        self.label_z = QLabel("0.00 m")
        self.label_altitude = QLabel("0.00 m")

        pos_layout.addRow("X (北):", self.label_x)
        pos_layout.addRow("Y (东):", self.label_y)
        pos_layout.addRow("Z (下):", self.label_z)
        pos_layout.addRow("高度:", self.label_altitude)

        layout.addWidget(pos_group)

        # 速度信息
        vel_group = QGroupBox("速度信息")
        vel_layout = QFormLayout(vel_group)

        self.label_vx = QLabel("0.00 m/s")
        self.label_vy = QLabel("0.00 m/s")
        self.label_vz = QLabel("0.00 m/s")
        self.label_speed = QLabel("0.00 m/s")

        vel_layout.addRow("Vx:", self.label_vx)
        vel_layout.addRow("Vy:", self.label_vy)
        vel_layout.addRow("Vz:", self.label_vz)
        vel_layout.addRow("总速度:", self.label_speed)

        layout.addWidget(vel_group)

        # 姿态信息
        att_group = QGroupBox("姿态信息")
        att_layout = QFormLayout(att_group)

        self.label_roll = QLabel("0.00°")
        self.label_pitch = QLabel("0.00°")
        self.label_yaw = QLabel("0.00°")

        att_layout.addRow("横滚 (Roll):", self.label_roll)
        att_layout.addRow("俯仰 (Pitch):", self.label_pitch)
        att_layout.addRow("偏航 (Yaw):", self.label_yaw)

        layout.addWidget(att_group)

        # 电机信息
        motor_group = QGroupBox("电机状态")
        motor_layout = QGridLayout(motor_group)

        self.motor_bars = []
        motor_labels = ["前右", "后左", "前左", "后右"]

        for i in range(4):
            label = QLabel(motor_labels[i])
            label.setAlignment(Qt.AlignCenter)
            motor_layout.addWidget(label, 0, i)

            bar = QProgressBar()
            bar.setOrientation(Qt.Vertical)
            bar.setRange(0, 12000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(60)
            motor_layout.addWidget(bar, 1, i)
            self.motor_bars.append(bar)

            rpm_label = QLabel("0")
            rpm_label.setAlignment(Qt.AlignCenter)
            motor_layout.addWidget(rpm_label, 2, i)

        self.motor_rpm_labels = [
            motor_layout.itemAtPosition(2, i).widget() for i in range(4)
        ]

        layout.addWidget(motor_group)

        # 系统状态
        sys_group = QGroupBox("系统状态")
        sys_layout = QFormLayout(sys_group)

        self.label_battery = QLabel("100%")
        self.battery_bar = QProgressBar()
        self.battery_bar.setRange(0, 100)
        self.battery_bar.setValue(100)

        sys_layout.addRow("电池:", self.battery_bar)

        self.label_signal = QLabel("良好")
        sys_layout.addRow("信号:", self.label_signal)

        self.label_gps = QLabel("已定位 (12颗卫星)")
        sys_layout.addRow("GPS:", self.label_gps)

        layout.addWidget(sys_group)

        # 添加弹簧
        layout.addStretch()

    def update_state(self, state: DroneState):
        """更新显示状态"""
        # 位置
        self.label_x.setText(f"{state.position[0]:.2f} m")
        self.label_y.setText(f"{state.position[1]:.2f} m")
        self.label_z.setText(f"{state.position[2]:.2f} m")
        self.label_altitude.setText(f"{state.altitude:.2f} m")

        # 速度
        self.label_vx.setText(f"{state.velocity[0]:.2f} m/s")
        self.label_vy.setText(f"{state.velocity[1]:.2f} m/s")
        self.label_vz.setText(f"{state.velocity[2]:.2f} m/s")
        self.label_speed.setText(f"{state.speed:.2f} m/s")

        # 姿态（转换为度）
        euler = np.degrees(state.euler_angles)
        self.label_roll.setText(f"{euler[0]:.2f}°")
        self.label_pitch.setText(f"{euler[1]:.2f}°")
        self.label_yaw.setText(f"{euler[2]:.2f}°")

        # 电机
        for i, (bar, label) in enumerate(zip(self.motor_bars, self.motor_rpm_labels)):
            rpm = int(state.motor_speeds[i])
            bar.setValue(rpm)
            label.setText(str(rpm))

            # 根据转速设置颜色
            if rpm > 10000:
                bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            elif rpm > 8000:
                bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
            else:
                bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
