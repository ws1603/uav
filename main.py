# main.py（完整修复版）

"""
低空交通无人机教学演示系统 - 主程序入口
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """检查依赖"""
    missing = []

    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError:
        missing.append('numpy')

    try:
        import scipy
        print(f"✓ scipy {scipy.__version__}")
    except ImportError:
        missing.append('scipy')

    try:
        from PyQt5 import QtWidgets, QtCore
        print(f"✓ PyQt5 {QtCore.QT_VERSION_STR}")
    except ImportError:
        missing.append('PyQt5')

    try:
        import pyqtgraph
        print(f"✓ pyqtgraph {pyqtgraph.__version__}")
    except ImportError:
        missing.append('pyqtgraph')

    try:
        from loguru import logger
        print("✓ loguru")
    except ImportError:
        missing.append('loguru')

    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + ' '.join(missing))
        return False

    return True


def setup_environment():
    """设置环境"""
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception as e:
        print(f"设置高DPI支持失败: {e}")


def create_minimal_app():
    """创建最小化应用（用于测试核心功能）"""
    import numpy as np
    from loguru import logger

    logger.info("=" * 50)
    logger.info("低空交通无人机教学演示系统 - 控制台模式")
    logger.info("=" * 50)

    from core.physics.quadrotor_dynamics import QuadrotorDynamics
    from core.control.pid_controller import QuadrotorPIDController
    from simulation.engine.simulation_core import SimulationEngine

    engine = SimulationEngine()
    target_position = np.array([10.0, 5.0, -15.0])
    engine.set_target_position(target_position)

    logger.info(f"目标位置: {target_position}")
    logger.info("开始仿真...")

    import time
    engine.start()

    try:
        for i in range(100):
            time.sleep(0.1)
            state = engine.drone_state
            stats = engine.statistics

            if i % 10 == 0:
                logger.info(
                    f"时间: {stats.simulation_time:.1f}s | "
                    f"位置: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}] | "
                    f"高度: {state.altitude:.2f}m | "
                    f"速度: {state.speed:.2f}m/s"
                )
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        engine.stop()

    logger.info("仿真结束")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='低空交通无人机教学演示系统')
    parser.add_argument('--console', action='store_true', help='控制台模式（无GUI）')
    parser.add_argument('--test', action='store_true', help='运行测试')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--simple', action='store_true', help='简化GUI模式')

    args = parser.parse_args()

    print("=" * 50)
    print("低空交通无人机教学演示系统")
    print("=" * 50)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    print()

    # 设置日志
    from loguru import logger
    logger.remove()  # 移除默认处理器
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if args.debug:
        logger.add(
            "logs/debug_{time}.log",
            level="DEBUG",
            rotation="10 MB"
        )

    # 运行测试
    if args.test:
        run_tests()
        return

    # 控制台模式
    if args.console:
        create_minimal_app()
        return

    # GUI模式
    setup_environment()

    if args.simple:
        run_simple_gui()
    else:
        run_gui()


def run_gui():
    """运行GUI应用"""
    from PyQt5.QtWidgets import QApplication
    from loguru import logger
    import traceback

    logger.info("启动GUI应用...")

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("低空交通无人机教学演示系统")
        app.setOrganizationName("UAVTeaching")

        # 应用样式
        try:
            from ui.themes.dark_theme import get_dark_stylesheet
            app.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            logger.warning("无法加载主题，使用默认样式")

        # 尝试创建主窗口
        try:
            from ui.main_window.main_window import MainWindow
            window = MainWindow()
            window.show()
            logger.info("主窗口创建成功")
        except Exception as e:
            logger.error(f"创建主窗口失败: {e}")
            logger.info("尝试创建简化窗口...")
            traceback.print_exc()
            window = create_simple_window()
            window.show()

        # 进入事件循环
        sys.exit(app.exec_())

    except Exception as e:
        logger.error(f"GUI启动失败: {e}")
        traceback.print_exc()
        sys.exit(1)


def run_simple_gui():
    """运行简化GUI"""
    from PyQt5.QtWidgets import QApplication
    from loguru import logger

    logger.info("启动简化GUI...")

    app = QApplication(sys.argv)
    app.setApplicationName("低空交通无人机教学演示系统")

    # 应用样式
    try:
        from ui.themes.dark_theme import get_dark_stylesheet
        app.setStyleSheet(get_dark_stylesheet())
    except ImportError:
        pass

    window = create_simple_window()
    window.show()

    sys.exit(app.exec_())


def create_simple_window():
    """创建简化窗口"""
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QDoubleSpinBox, QGroupBox,
        QTextEdit, QSplitter, QFormLayout, QFrame
    )
    from PyQt5.QtCore import Qt, QTimer
    import numpy as np
    from loguru import logger

    # 安全导入仿真引擎
    try:
        from simulation.engine.simulation_core import SimulationEngine
    except Exception as e:
        logger.error(f"导入仿真引擎失败: {e}")
        SimulationEngine = None

    class SimpleMainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("低空交通无人机教学演示系统")
            self.setGeometry(100, 100, 1000, 700)

            # 创建仿真引擎
            self.engine = None
            if SimulationEngine is not None:
                try:
                    self.engine = SimulationEngine()
                    logger.info("仿真引擎初始化成功")
                except Exception as e:
                    logger.error(f"创建仿真引擎失败: {e}")

            # 设置UI
            self._setup_ui()

            # 更新定时器
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_display)
            self.timer.start(100)

            self._log("系统初始化完成")

        def _setup_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            layout = QHBoxLayout(central)

            # 左侧控制面板
            left_panel = QWidget()
            left_panel.setMaximumWidth(350)
            left_layout = QVBoxLayout(left_panel)

            # 控制按钮
            btn_group = QGroupBox("仿真控制")
            btn_layout = QHBoxLayout(btn_group)

            self.btn_start = QPushButton("开始")
            self.btn_start.clicked.connect(self._on_start)
            btn_layout.addWidget(self.btn_start)

            self.btn_pause = QPushButton("暂停")
            self.btn_pause.clicked.connect(self._on_pause)
            self.btn_pause.setEnabled(False)
            btn_layout.addWidget(self.btn_pause)

            self.btn_stop = QPushButton("停止")
            self.btn_stop.clicked.connect(self._on_stop)
            self.btn_stop.setEnabled(False)
            btn_layout.addWidget(self.btn_stop)

            self.btn_reset = QPushButton("重置")
            self.btn_reset.clicked.connect(self._on_reset)
            btn_layout.addWidget(self.btn_reset)

            left_layout.addWidget(btn_group)

            # 目标位置
            target_group = QGroupBox("目标位置 (NED坐标)")
            target_layout = QFormLayout(target_group)

            self.spin_x = QDoubleSpinBox()
            self.spin_x.setRange(-100, 100)
            self.spin_x.setValue(10)
            self.spin_x.setSingleStep(1)
            target_layout.addRow("X (北):", self.spin_x)

            self.spin_y = QDoubleSpinBox()
            self.spin_y.setRange(-100, 100)
            self.spin_y.setValue(5)
            self.spin_y.setSingleStep(1)
            target_layout.addRow("Y (东):", self.spin_y)

            self.spin_z = QDoubleSpinBox()
            self.spin_z.setRange(-100, 0)
            self.spin_z.setValue(-15)
            self.spin_z.setSingleStep(1)
            target_layout.addRow("Z (下):", self.spin_z)

            btn_apply = QPushButton("应用目标位置")
            btn_apply.clicked.connect(self._apply_target)
            target_layout.addRow(btn_apply)

            left_layout.addWidget(target_group)

            # 状态显示
            status_group = QGroupBox("飞行状态")
            status_layout = QFormLayout(status_group)

            self.label_state = QLabel("停止")
            self.label_state.setStyleSheet("font-weight: bold; color: #ff6b6b;")
            status_layout.addRow("仿真状态:", self.label_state)

            self.label_time = QLabel("0.00 s")
            status_layout.addRow("仿真时间:", self.label_time)

            self.label_pos = QLabel("[0.00, 0.00, 0.00]")
            status_layout.addRow("位置 (m):", self.label_pos)

            self.label_vel = QLabel("[0.00, 0.00, 0.00]")
            status_layout.addRow("速度 (m/s):", self.label_vel)

            self.label_altitude = QLabel("0.00 m")
            self.label_altitude.setStyleSheet("font-weight: bold; color: #4ecdc4;")
            status_layout.addRow("高度:", self.label_altitude)

            self.label_speed = QLabel("0.00 m/s")
            status_layout.addRow("速率:", self.label_speed)

            self.label_att = QLabel("[0.0°, 0.0°, 0.0°]")
            status_layout.addRow("姿态 (RPY):", self.label_att)

            self.label_fps = QLabel("0")
            status_layout.addRow("仿真帧率:", self.label_fps)

            left_layout.addWidget(status_group)
            left_layout.addStretch()

            # 右侧日志
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)

            log_label = QLabel("系统日志:")
            log_label.setStyleSheet("font-weight: bold;")
            right_layout.addWidget(log_label)

            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setStyleSheet("""
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: Consolas, monospace;
                    font-size: 12px;
                }
            """)
            right_layout.addWidget(self.log_text)

            # 分割器
            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setSizes([350, 650])

            layout.addWidget(splitter)

        def _log(self, message: str, level: str = "INFO"):
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            color_map = {
                "INFO": "#4ecdc4",
                "WARNING": "#f39c12",
                "ERROR": "#e74c3c",
                "SUCCESS": "#2ecc71"
            }
            color = color_map.get(level, "#d4d4d4")

            html = f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {color};">[{level}]</span> {message}'
            self.log_text.append(html)

            # 滚动到底部
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def _on_start(self):
            if self.engine is None:
                self._log("仿真引擎未初始化", "ERROR")
                return

            try:
                if hasattr(self.engine, 'is_paused') and self.engine.is_paused:
                    self.engine.resume()
                    self._log("仿真恢复", "SUCCESS")
                else:
                    self._apply_target()
                    self.engine.start()
                    self._log("仿真开始", "SUCCESS")

                self.btn_start.setEnabled(False)
                self.btn_pause.setEnabled(True)
                self.btn_stop.setEnabled(True)
                self.label_state.setText("运行中")
                self.label_state.setStyleSheet("font-weight: bold; color: #2ecc71;")
            except Exception as e:
                self._log(f"启动失败: {e}", "ERROR")

        def _on_pause(self):
            if self.engine is None:
                return

            try:
                self.engine.pause()
                self.btn_start.setEnabled(True)
                self.btn_pause.setEnabled(False)
                self.label_state.setText("暂停")
                self.label_state.setStyleSheet("font-weight: bold; color: #f39c12;")
                self._log("仿真暂停", "WARNING")
            except Exception as e:
                self._log(f"暂停失败: {e}", "ERROR")

        def _on_stop(self):
            if self.engine is None:
                return

            try:
                self.engine.stop()
                self.btn_start.setEnabled(True)
                self.btn_pause.setEnabled(False)
                self.btn_stop.setEnabled(False)
                self.label_state.setText("停止")
                self.label_state.setStyleSheet("font-weight: bold; color: #ff6b6b;")
                self._log("仿真停止")
            except Exception as e:
                self._log(f"停止失败: {e}", "ERROR")

        def _on_reset(self):
            if self.engine is None:
                return

            try:
                self.engine.reset()
                self.btn_start.setEnabled(True)
                self.btn_pause.setEnabled(False)
                self.btn_stop.setEnabled(False)
                self.label_state.setText("停止")
                self.label_state.setStyleSheet("font-weight: bold; color: #ff6b6b;")
                self._log("仿真已重置")
            except Exception as e:
                self._log(f"重置失败: {e}", "ERROR")

        def _apply_target(self):
            if self.engine is None:
                return

            target = np.array([
                self.spin_x.value(),
                self.spin_y.value(),
                self.spin_z.value()
            ])

            try:
                self.engine.set_target_position(target)
                self._log(f"目标位置: [{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}]")
            except Exception as e:
                self._log(f"设置目标失败: {e}", "ERROR")

        def _update_display(self):
            if self.engine is None:
                return

            try:
                if not self.engine.is_running and not self.engine.is_paused:
                    return

                state = self.engine.drone_state
                stats = self.engine.statistics

                self.label_time.setText(f"{stats.simulation_time:.2f} s")
                self.label_pos.setText(
                    f"[{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]"
                )
                self.label_vel.setText(
                    f"[{state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}]"
                )
                self.label_altitude.setText(f"{state.altitude:.2f} m")
                self.label_speed.setText(f"{state.speed:.2f} m/s")

                euler_deg = np.degrees(state.euler_angles)
                self.label_att.setText(
                    f"[{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°]"
                )

                self.label_fps.setText(f"{stats.average_fps:.0f}")

            except Exception as e:
                pass  # 静默处理更新错误

        def closeEvent(self, event):
            if self.engine is not None:
                try:
                    self.engine.stop()
                except Exception:
                    pass
            event.accept()

    return SimpleMainWindow()


# # main.py 中的 run_tests 函数修复
#
# def run_tests():
#     """运行测试"""
#     from loguru import logger
#     import numpy as np
#
#     logger.info("=" * 50)
#     logger.info("运行系统测试")
#     logger.info("=" * 50)
#
#     tests_passed = 0
#     tests_failed = 0
#
#     # 测试1: 四元数工具
#     try:
#         from utils.math.quaternion import Quaternion
#
#         q = np.array([1.0, 0.0, 0.0, 0.0])
#         R = Quaternion.to_rotation_matrix(q)
#         assert np.allclose(R, np.eye(3)), "单位四元数应产生单位矩阵"
#
#         euler = Quaternion.to_euler(q)
#         assert np.allclose(euler, np.zeros(3)), "单位四元数应产生零欧拉角"
#
#         logger.info("✓ 四元数工具测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 四元数工具测试失败: {e}")
#         tests_failed += 1
#
#     # 测试2: 无人机动力学
#     try:
#         from core.physics.quadrotor_dynamics import QuadrotorDynamics
#
#         dynamics = QuadrotorDynamics()
#         hover_speed = 5500
#         dynamics.set_motor_speeds(np.array([hover_speed] * 4))
#         dynamics.step(0.01)
#
#         assert dynamics.state is not None, "状态不应为空"
#
#         logger.info("✓ 无人机动力学测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 无人机动力学测试失败: {e}")
#         tests_failed += 1
#
#     # 测试3: PID控制器
#     try:
#         from core.control.pid_controller import PIDController, PIDGains
#
#         pid = PIDController(PIDGains(kp=1.0, ki=0.1, kd=0.05))
#         output = pid.update(target=10.0, current=0.0, dt=0.01)
#         assert output > 0, "PID输出应为正"
#
#         pid.reset()
#         assert pid.integral == 0, "重置后积分应为0"
#
#         logger.info("✓ PID控制器测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ PID控制器测试失败: {e}")
#         tests_failed += 1
#
#     # 测试4: 仿真引擎
#     try:
#         from simulation.engine.simulation_core import SimulationEngine, SimulationState
#
#         engine = SimulationEngine()
#         assert engine.state == SimulationState.STOPPED, "初始状态应为STOPPED"
#
#         engine.set_target_position(np.array([5.0, 0.0, -10.0]))
#         engine.start()
#
#         import time
#         time.sleep(0.5)
#
#         assert engine.is_running, "引擎应在运行中"
#         assert engine.statistics.simulation_time > 0, "仿真时间应大于0"
#
#         engine.stop()
#         assert not engine.is_running, "引擎应已停止"
#
#         logger.info("✓ 仿真引擎测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 仿真引擎测试失败: {e}")
#         tests_failed += 1
#
#     # 测试5: 路径规划
#     try:
#         from core.planning.astar_planner import OccupancyGrid3D, AStarPlanner, PathSmoother
#
#         grid = OccupancyGrid3D(size=(50, 50, 20), resolution=1.0)
#         planner = AStarPlanner(grid)
#
#         start = np.array([5.0, 5.0, 10.0])
#         goal = np.array([40.0, 40.0, 10.0])
#
#         path = planner.plan(start, goal)
#
#         assert path is not None, "应能找到路径"
#         assert len(path) > 0, "路径不应为空"
#
#         smoothed = PathSmoother.smooth_path(path)
#         assert len(smoothed) == len(path), "平滑路径长度应相同"
#
#         logger.info("✓ 路径规划测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 路径规划测试失败: {e}")
#         tests_failed += 1
#
#     # 测试6: 航点管理
#     try:
#         from core.planning.waypoint_manager import WaypointManager, Waypoint, Mission
#
#         manager = WaypointManager()
#         mission = manager.create_mission("测试任务")
#
#         mission.add_waypoint(Waypoint(position=[0, 0, -10]))
#         mission.add_waypoint(Waypoint(position=[10, 0, -15]))
#         mission.add_waypoint(Waypoint(position=[10, 10, -15]))
#
#         assert len(mission.waypoints) == 3, "应有3个航点"
#         assert mission.total_distance > 0, "总距离应大于0"
#
#         manager.load_mission(mission)
#         manager.start_mission()
#
#         wp = manager.get_current_waypoint()
#         assert wp is not None, "应有当前航点"
#
#         logger.info("✓ 航点管理测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 航点管理测试失败: {e}")
#         tests_failed += 1
#
#     # 测试结果
#     logger.info("=" * 50)
#     logger.info(f"测试完成: {tests_passed} 通过, {tests_failed} 失败")
#     logger.info("=" * 50)
#
#     return tests_failed == 0

# main.py 中 run_tests 函数的完整版本

def run_tests():
    """运行测试"""
    from loguru import logger
    import numpy as np

    logger.info("=" * 50)
    logger.info("运行系统测试")
    logger.info("=" * 50)

    tests_passed = 0
    tests_failed = 0

    # 测试1: 四元数工具
    try:
        from utils.math.quaternion import Quaternion

        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = Quaternion.to_rotation_matrix(q)
        assert np.allclose(R, np.eye(3)), "单位四元数应产生单位矩阵"

        euler = Quaternion.to_euler(q)
        assert np.allclose(euler, np.zeros(3)), "单位四元数应产生零欧拉角"

        # 测试积分
        omega = np.array([0.1, 0.0, 0.0])
        q_new = Quaternion.integrate(q, omega, 0.01)
        assert np.allclose(np.linalg.norm(q_new), 1.0), "积分后四元数应归一化"

        logger.info("✓ 四元数工具测试通过")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ 四元数工具测试失败: {e}")
        tests_failed += 1

    # 测试2: 无人机动力学
    try:
        from core.physics.quadrotor_dynamics import QuadrotorDynamics

        dynamics = QuadrotorDynamics()
        hover_speed = 5500
        dynamics.set_motor_speeds(np.array([hover_speed] * 4))
        dynamics.step(0.01)

        assert dynamics.state is not None, "状态不应为空"

        logger.info("✓ 无人机动力学测试通过")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ 无人机动力学测试失败: {e}")
        tests_failed += 1

    # 测试3: PID控制器
    try:
        from core.control.pid_controller import PIDController, PIDGains

        pid = PIDController(PIDGains(kp=1.0, ki=0.1, kd=0.05))
        output = pid.update(target=10.0, current=0.0, dt=0.01)
        assert output > 0, "PID输出应为正"

        pid.reset()
        assert pid.integral == 0, "重置后积分应为0"

        logger.info("✓ PID控制器测试通过")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ PID控制器测试失败: {e}")
        tests_failed += 1

    # 测试4: 仿真引擎
    try:
        from simulation.engine.simulation_core import SimulationEngine, SimulationState

        engine = SimulationEngine()
        assert engine.state == SimulationState.STOPPED, "初始状态应为STOPPED"

        engine.set_target_position(np.array([5.0, 0.0, -10.0]))
        engine.start()

        import time
        time.sleep(0.5)

        assert engine.is_running, "引擎应在运行中"
        assert engine.statistics.simulation_time > 0, "仿真时间应大于0"

        engine.stop()
        assert not engine.is_running, "引擎应已停止"

        logger.info("✓ 仿真引擎测试通过")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ 仿真引擎测试失败: {e}")
        tests_failed += 1

    # 测试5: 路径规划
    try:
        from core.planning.astar_planner import OccupancyGrid3D, AStarPlanner, PathSmoother

        grid = OccupancyGrid3D(size=(50, 50, 20), resolution=1.0)
        planner = AStarPlanner(grid)

        start = np.array([5.0, 5.0, 10.0])
        goal = np.array([40.0, 40.0, 10.0])

        path = planner.plan(start, goal)

        assert path is not None, "应能找到路径"
        assert len(path) > 0, "路径不应为空"

        smoothed = PathSmoother.smooth_path(path)
        assert len(smoothed) == len(path), "平滑路径长度应相同"

        logger.info("✓ 路径规划测试通过")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ 路径规划测试失败: {e}")
        tests_failed += 1

    # 测试6: 航点管理
    try:
        from core.planning.waypoint_manager import WaypointManager, Waypoint, Mission

        manager = WaypointManager()
        mission = manager.create_mission("测试任务")

        mission.add_waypoint(Waypoint(position=[0, 0, -10]))
        mission.add_waypoint(Waypoint(position=[10, 0, -15]))
        mission.add_waypoint(Waypoint(position=[10, 10, -15]))

        assert len(mission.waypoints) == 3, "应有3个航点"
        assert mission.total_distance > 0, "总距离应大于0"

        manager.load_mission(mission)
        manager.start_mission()

        wp = manager.get_current_waypoint()
        assert wp is not None, "应有当前航点"

        logger.info("✓ 航点管理测试通过")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ 航点管理测试失败: {e}")
        tests_failed += 1

    # 测试结果
    logger.info("=" * 50)
    logger.info(f"测试完成: {tests_passed} 通过, {tests_failed} 失败")
    logger.info("=" * 50)

    return tests_failed == 0


if __name__ == "__main__":
    main()

# main.py 中的 run_tests 函数修复

# def run_tests():
#     """运行测试"""
#     from loguru import logger
#     import numpy as np
#
#     logger.info("=" * 50)
#     logger.info("运行系统测试")
#     logger.info("=" * 50)
#
#     tests_passed = 0
#     tests_failed = 0
#
#     # 测试1: 四元数工具
#     try:
#         from utils.math.quaternion import Quaternion
#
#         q = np.array([1.0, 0.0, 0.0, 0.0])
#         R = Quaternion.to_rotation_matrix(q)
#         assert np.allclose(R, np.eye(3)), "单位四元数应产生单位矩阵"
#
#         euler = Quaternion.to_euler(q)
#         assert np.allclose(euler, np.zeros(3)), "单位四元数应产生零欧拉角"
#
#         logger.info("✓ 四元数工具测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 四元数工具测试失败: {e}")
#         tests_failed += 1
#
#     # 测试2: 无人机动力学
#     try:
#         from core.physics.quadrotor_dynamics import QuadrotorDynamics
#
#         dynamics = QuadrotorDynamics()
#         hover_speed = 5500
#         dynamics.set_motor_speeds(np.array([hover_speed] * 4))
#         dynamics.step(0.01)
#
#         assert dynamics.state is not None, "状态不应为空"
#
#         logger.info("✓ 无人机动力学测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 无人机动力学测试失败: {e}")
#         tests_failed += 1
#
#     # 测试3: PID控制器
#     try:
#         from core.control.pid_controller import PIDController, PIDGains
#
#         pid = PIDController(PIDGains(kp=1.0, ki=0.1, kd=0.05))
#         output = pid.update(target=10.0, current=0.0, dt=0.01)
#         assert output > 0, "PID输出应为正"
#
#         pid.reset()
#         assert pid.integral == 0, "重置后积分应为0"
#
#         logger.info("✓ PID控制器测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ PID控制器测试失败: {e}")
#         tests_failed += 1
#
#     # 测试4: 仿真引擎
#     try:
#         from simulation.engine.simulation_core import SimulationEngine, SimulationState
#
#         engine = SimulationEngine()
#         assert engine.state == SimulationState.STOPPED, "初始状态应为STOPPED"
#
#         engine.set_target_position(np.array([5.0, 0.0, -10.0]))
#         engine.start()
#
#         import time
#         time.sleep(0.5)
#
#         assert engine.is_running, "引擎应在运行中"
#         assert engine.statistics.simulation_time > 0, "仿真时间应大于0"
#
#         engine.stop()
#         assert not engine.is_running, "引擎应已停止"
#
#         logger.info("✓ 仿真引擎测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 仿真引擎测试失败: {e}")
#         tests_failed += 1
#
#     # 测试5: 路径规划
#     try:
#         from core.planning.astar_planner import OccupancyGrid3D, AStarPlanner, PathSmoother
#
#         grid = OccupancyGrid3D(size=(50, 50, 20), resolution=1.0)
#         planner = AStarPlanner(grid)
#
#         start = np.array([5.0, 5.0, 10.0])
#         goal = np.array([40.0, 40.0, 10.0])
#
#         path = planner.plan(start, goal)
#
#         assert path is not None, "应能找到路径"
#         assert len(path) > 0, "路径不应为空"
#
#         smoothed = PathSmoother.smooth_path(path)
#         assert len(smoothed) == len(path), "平滑路径长度应相同"
#
#         logger.info("✓ 路径规划测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 路径规划测试失败: {e}")
#         tests_failed += 1
#
#     # 测试6: 航点管理
#     try:
#         from core.planning.waypoint_manager import WaypointManager, Waypoint, Mission
#
#         manager = WaypointManager()
#         mission = manager.create_mission("测试任务")
#
#         mission.add_waypoint(Waypoint(position=[0, 0, -10]))
#         mission.add_waypoint(Waypoint(position=[10, 0, -15]))
#         mission.add_waypoint(Waypoint(position=[10, 10, -15]))
#
#         assert len(mission.waypoints) == 3, "应有3个航点"
#         assert mission.total_distance > 0, "总距离应大于0"
#
#         manager.load_mission(mission)
#         manager.start_mission()
#
#         wp = manager.get_current_waypoint()
#         assert wp is not None, "应有当前航点"
#
#         logger.info("✓ 航点管理测试通过")
#         tests_passed += 1
#     except Exception as e:
#         logger.error(f"✗ 航点管理测试失败: {e}")
#         tests_failed += 1
#
#     # 测试结果
#     logger.info("=" * 50)
#     logger.info(f"测试完成: {tests_passed} 通过, {tests_failed} 失败")
#     logger.info("=" * 50)
#
#     return tests_failed == 0
