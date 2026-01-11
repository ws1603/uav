# # # # ui/main_window/main_window.py
# # #
# # # # import sys
# # # # import numpy as np
# # # # from pathlib import Path
# # # # from typing import Optional
# # # # from PyQt5.QtWidgets import (
# # # #     QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
# # # #     QSplitter, QDockWidget, QAction, QActionGroup, QToolBar,
# # # #     QStatusBar, QLabel, QMessageBox, QFileDialog, QProgressBar
# # # # )
# # # # from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSettings
# # # # from PyQt5.QtGui import QIcon, QKeySequence
# # # # from loguru import logger
# # # #
# # # # from simulation.engine.simulation_core import SimulationEngine, SimulationState, SimulationEvent, EventType
# # # # from core.physics.quadrotor_dynamics import DroneState
# # # # from utils.config.config_manager import config_manager, get_config
# # # # from utils.logging.logger import setup_logging
# # #
# # #
# # # # ui/main_window/main_window.py（修复导入部分）
# # #
# # # # 修改开头的导入为：
# # #
# # # """
# # # 主窗口模块
# # # """
# # #
# # # import numpy as np
# # # from PyQt5.QtWidgets import (
# # #     QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
# # #     QDockWidget, QToolBar, QAction, QActionGroup,
# # #     QStatusBar, QLabel, QMenuBar, QMenu, QMessageBox,
# # #     QSplitter, QFrame
# # # )
# # # from PyQt5.QtCore import Qt, QTimer, pyqtSignal
# # # from PyQt5.QtGui import QIcon, QKeySequence
# # # from loguru import logger
# # #
# # # # 安全导入
# # # try:
# # #     from simulation.engine.simulation_core import SimulationEngine, SimulationState
# # # except ImportError as e:
# # #     logger.error(f"导入SimulationEngine失败: {e}")
# # #     SimulationEngine = None
# # #     SimulationState = None
# # #
# # # try:
# # #     from simulation.engine.simulation_core import SimulationEvent, EventType
# # # except ImportError:
# # #     SimulationEvent = None
# # #     EventType = None
# # #
# # # from utils.config.config_manager import get_config
# # #
# # # # 导入DroneState
# # # try:
# # #     from core.physics.quadrotor_dynamics import DroneState
# # # except ImportError:
# # #     from dataclasses import dataclass, field
# # #
# # # class MainWindow(QMainWindow):
# # #     """主窗口"""
# # #
# # #     # 信号定义
# # #     state_updated = pyqtSignal(object)  # DroneState
# # #     simulation_state_changed = pyqtSignal(str)
# # #
# # #     def __init__(self):
# # #         super().__init__()
# # #
# # #         # 初始化日志
# # #         setup_logging()
# # #
# # #         # 加载配置
# # #         config_manager.load()
# # #
# # #         # 创建仿真引擎
# # #         self.engine = SimulationEngine()
# # #
# # #         # 设置窗口
# # #         self._setup_window()
# # #         self._setup_menu_bar()
# # #         self._setup_tool_bar()
# # #         self._setup_status_bar()
# # #         self._setup_dock_widgets()
# # #         self._setup_central_widget()
# # #
# # #         # 连接信号
# # #         self._connect_signals()
# # #
# # #         # 更新定时器
# # #         self._update_timer = QTimer()
# # #         self._update_timer.timeout.connect(self._update_display)
# # #         self._update_timer.start(50)  # 20 FPS
# # #
# # #         # 加载设置
# # #         self._load_settings()
# # #
# # #         logger.info("主窗口初始化完成")
# # #
# # #     def _setup_window(self):
# # #         """设置窗口属性"""
# # #         config = get_config().visualization
# # #
# # #         self.setWindowTitle("低空交通无人机教学演示系统 v1.0")
# # #         self.setGeometry(100, 100, config.window_width, config.window_height)
# # #         self.setMinimumSize(800, 600)
# # #
# # #         # 设置窗口图标（如果有）
# # #         icon_path = Path("resources/icons/app_icon.png")
# # #         if icon_path.exists():
# # #             self.setWindowIcon(QIcon(str(icon_path)))
# # #
# # #     def _setup_menu_bar(self):
# # #         """设置菜单栏"""
# # #         menubar = self.menuBar()
# # #
# # #         # ===== 文件菜单 =====
# # #         file_menu = menubar.addMenu("文件(&F)")
# # #
# # #         # 新建场景
# # #         new_action = QAction("新建场景(&N)", self)
# # #         new_action.setShortcut(QKeySequence.New)
# # #         new_action.triggered.connect(self._on_new_scenario)
# # #         file_menu.addAction(new_action)
# # #
# # #         # 打开场景
# # #         open_action = QAction("打开场景(&O)...", self)
# # #         open_action.setShortcut(QKeySequence.Open)
# # #         open_action.triggered.connect(self._on_open_scenario)
# # #         file_menu.addAction(open_action)
# # #
# # #         # 保存场景
# # #         save_action = QAction("保存场景(&S)", self)
# # #         save_action.setShortcut(QKeySequence.Save)
# # #         save_action.triggered.connect(self._on_save_scenario)
# # #         file_menu.addAction(save_action)
# # #
# # #         file_menu.addSeparator()
# # #
# # #         # 导出数据
# # #         export_menu = file_menu.addMenu("导出(&E)")
# # #         export_csv_action = QAction("导出为 CSV...", self)
# # #         export_csv_action.triggered.connect(self._on_export_csv)
# # #         export_menu.addAction(export_csv_action)
# # #
# # #         export_pdf_action = QAction("导出报告 (PDF)...", self)
# # #         export_pdf_action.triggered.connect(self._on_export_pdf)
# # #         export_menu.addAction(export_pdf_action)
# # #
# # #         file_menu.addSeparator()
# # #
# # #         # 退出
# # #         exit_action = QAction("退出(&X)", self)
# # #         exit_action.setShortcut(QKeySequence.Quit)
# # #         exit_action.triggered.connect(self.close)
# # #         file_menu.addAction(exit_action)
# # #
# # #         # ===== 仿真菜单 =====
# # #         sim_menu = menubar.addMenu("仿真(&S)")
# # #
# # #         self.start_action = QAction("开始(&S)", self)
# # #         self.start_action.setShortcut("F5")
# # #         self.start_action.triggered.connect(self._on_start_simulation)
# # #         sim_menu.addAction(self.start_action)
# # #
# # #         self.pause_action = QAction("暂停(&P)", self)
# # #         self.pause_action.setShortcut("F6")
# # #         self.pause_action.setEnabled(False)
# # #         self.pause_action.triggered.connect(self._on_pause_simulation)
# # #         sim_menu.addAction(self.pause_action)
# # #
# # #         self.stop_action = QAction("停止(&T)", self)
# # #         self.stop_action.setShortcut("F7")
# # #         self.stop_action.setEnabled(False)
# # #         self.stop_action.triggered.connect(self._on_stop_simulation)
# # #         sim_menu.addAction(self.stop_action)
# # #
# # #         self.reset_action = QAction("重置(&R)", self)
# # #         self.reset_action.setShortcut("F8")
# # #         self.reset_action.triggered.connect(self._on_reset_simulation)
# # #         sim_menu.addAction(self.reset_action)
# # #
# # #         sim_menu.addSeparator()
# # #
# # #         self.step_action = QAction("单步执行", self)
# # #         self.step_action.setShortcut("F10")
# # #         self.step_action.triggered.connect(self._on_step_simulation)
# # #         sim_menu.addAction(self.step_action)
# # #
# # #         # ===== 视图菜单 =====
# # #         view_menu = menubar.addMenu("视图(&V)")
# # #
# # #         # 面板显示
# # #         self.panel_actions = {}
# # #         panels = ["控制面板", "参数面板", "信息面板", "日志面板", "图表面板"]
# # #         for panel_name in panels:
# # #             action = QAction(panel_name, self)
# # #             action.setCheckable(True)
# # #             action.setChecked(True)
# # #             self.panel_actions[panel_name] = action
# # #             view_menu.addAction(action)
# # #
# # #         view_menu.addSeparator()
# # #
# # #         # 视图模式
# # #         view_mode_menu = view_menu.addMenu("视图模式")
# # #         view_mode_group = QActionGroup(self)
# # #
# # #         for mode in ["自由视角", "跟随视角", "俯视图", "侧视图"]:
# # #             action = QAction(mode, self)
# # #             action.setCheckable(True)
# # #             action.setActionGroup(view_mode_group)
# # #             if mode == "跟随视角":
# # #                 action.setChecked(True)
# # #             view_mode_menu.addAction(action)
# # #
# # #         view_menu.addSeparator()
# # #
# # #         # 主题
# # #         theme_menu = view_menu.addMenu("主题")
# # #         theme_group = QActionGroup(self)
# # #
# # #         for theme in ["浅色主题", "深色主题"]:
# # #             action = QAction(theme, self)
# # #             action.setCheckable(True)
# # #             action.setActionGroup(theme_group)
# # #             action.triggered.connect(lambda checked, t=theme: self._on_theme_change(t))
# # #             if theme == "深色主题":
# # #                 action.setChecked(True)
# # #             theme_menu.addAction(action)
# # #
# # #         # ===== 工具菜单 =====
# # #         tools_menu = menubar.addMenu("工具(&T)")
# # #
# # #         settings_action = QAction("设置(&S)...", self)
# # #         settings_action.setShortcut("Ctrl+,")
# # #         settings_action.triggered.connect(self._on_open_settings)
# # #         tools_menu.addAction(settings_action)
# # #
# # #         # ===== 教程菜单 =====
# # #         tutorial_menu = menubar.addMenu("教程(&L)")
# # #
# # #         tutorials = [
# # #             ("基础飞行控制", "01_basic_flight"),
# # #             ("PID参数调节", "02_pid_tuning"),
# # #             ("路径规划入门", "03_path_planning"),
# # #             ("多机协同", "04_multi_uav")
# # #         ]
# # #
# # #         for name, tutorial_id in tutorials:
# # #             action = QAction(name, self)
# # #             action.triggered.connect(lambda checked, tid=tutorial_id: self._on_open_tutorial(tid))
# # #             tutorial_menu.addAction(action)
# # #
# # #         # ===== 帮助菜单 =====
# # #         help_menu = menubar.addMenu("帮助(&H)")
# # #
# # #         doc_action = QAction("文档(&D)", self)
# # #         doc_action.setShortcut("F1")
# # #         doc_action.triggered.connect(self._on_open_documentation)
# # #
# # #         help_menu.addAction(doc_action)
# # #
# # #         about_action = QAction("关于(&A)", self)
# # #         about_action.triggered.connect(self._on_about)
# # #         help_menu.addAction(about_action)
# # #
# # #     def _setup_tool_bar(self):
# # #         """设置工具栏"""
# # #         toolbar = QToolBar("主工具栏")
# # #         toolbar.setObjectName("main_toolbar")
# # #         toolbar.setMovable(True)
# # #         self.addToolBar(toolbar)
# # #
# # #         # 仿真控制按钮
# # #         self.btn_start = toolbar.addAction("▶ 开始")
# # #         self.btn_start.triggered.connect(self._on_start_simulation)
# # #
# # #         self.btn_pause = toolbar.addAction("⏸ 暂停")
# # #         self.btn_pause.setEnabled(False)
# # #         self.btn_pause.triggered.connect(self._on_pause_simulation)
# # #
# # #         self.btn_stop = toolbar.addAction("⏹ 停止")
# # #         self.btn_stop.setEnabled(False)
# # #         self.btn_stop.triggered.connect(self._on_stop_simulation)
# # #
# # #         self.btn_reset = toolbar.addAction("↺ 重置")
# # #         self.btn_reset.triggered.connect(self._on_reset_simulation)
# # #
# # #         toolbar.addSeparator()
# # #
# # #         # 速度控制
# # #         self.btn_speed_05x = toolbar.addAction("0.5x")
# # #         self.btn_speed_05x.triggered.connect(lambda: self._set_simulation_speed(0.5))
# # #
# # #         self.btn_speed_1x = toolbar.addAction("1x")
# # #         self.btn_speed_1x.triggered.connect(lambda: self._set_simulation_speed(1.0))
# # #
# # #         self.btn_speed_2x = toolbar.addAction("2x")
# # #         self.btn_speed_2x.triggered.connect(lambda: self._set_simulation_speed(2.0))
# # #
# # #         toolbar.addSeparator()
# # #
# # #         # 录制按钮
# # #         self.btn_record = toolbar.addAction("⏺ 录制")
# # #         self.btn_record.setCheckable(True)
# # #         self.btn_record.triggered.connect(self._on_toggle_recording)
# # #
# # #     def _setup_status_bar(self):
# # #         """设置状态栏"""
# # #         self.statusbar = QStatusBar()
# # #         self.setStatusBar(self.statusbar)
# # #
# # #         # 仿真状态
# # #         self.status_sim_state = QLabel("状态: 就绪")
# # #         self.status_sim_state.setMinimumWidth(100)
# # #         self.statusbar.addWidget(self.status_sim_state)
# # #
# # #         # 仿真时间
# # #         self.status_sim_time = QLabel("时间: 0.00s")
# # #         self.status_sim_time.setMinimumWidth(120)
# # #         self.statusbar.addWidget(self.status_sim_time)
# # #
# # #         # 实时性
# # #         self.status_realtime = QLabel("实时比: 1.00x")
# # #         self.status_realtime.setMinimumWidth(100)
# # #         self.statusbar.addWidget(self.status_realtime)
# # #
# # #         # 无人机状态
# # #         self.status_altitude = QLabel("高度: 0.0m")
# # #         self.status_altitude.setMinimumWidth(100)
# # #         self.statusbar.addWidget(self.status_altitude)
# # #
# # #         self.status_speed = QLabel("速度: 0.0m/s")
# # #         self.status_speed.setMinimumWidth(100)
# # #         self.statusbar.addWidget(self.status_speed)
# # #
# # #         # 进度条（用于加载等）
# # #         self.progress_bar = QProgressBar()
# # #         self.progress_bar.setMaximumWidth(150)
# # #         self.progress_bar.setVisible(False)
# # #         self.statusbar.addPermanentWidget(self.progress_bar)
# # #
# # #         # FPS显示
# # #         self.status_fps = QLabel("FPS: --")
# # #         self.status_fps.setMinimumWidth(80)
# # #         self.statusbar.addPermanentWidget(self.status_fps)
# # #
# # #     def _setup_dock_widgets(self):
# # #         """设置可停靠面板"""
# # #         # 控制面板
# # #         self.control_dock = QDockWidget("控制面板", self)
# # #         self.control_dock.setObjectName("control_dock")
# # #         self.control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
# # #         self.control_panel = ControlPanel(self.engine)
# # #         self.control_dock.setWidget(self.control_panel)
# # #         self.addDockWidget(Qt.LeftDockWidgetArea, self.control_dock)
# # #
# # #         # 参数面板
# # #         self.param_dock = QDockWidget("参数面板", self)
# # #         self.param_dock.setObjectName("param_dock")
# # #         self.param_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
# # #         self.param_panel = ParameterPanel()
# # #         self.param_dock.setWidget(self.param_panel)
# # #         self.addDockWidget(Qt.LeftDockWidgetArea, self.param_dock)
# # #
# # #         # 将控制和参数面板标签化
# # #         self.tabifyDockWidget(self.control_dock, self.param_dock)
# # #         self.control_dock.raise_()
# # #
# # #         # 信息面板
# # #         self.info_dock = QDockWidget("飞行信息", self)
# # #         self.info_dock.setObjectName("info_dock")
# # #         self.info_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
# # #         self.info_panel = InfoPanel()
# # #         self.info_dock.setWidget(self.info_panel)
# # #         self.addDockWidget(Qt.RightDockWidgetArea, self.info_dock)
# # #
# # #         # 图表面板
# # #         self.chart_dock = QDockWidget("实时图表", self)
# # #         self.chart_dock.setObjectName("chart_dock")
# # #         self.chart_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
# # #         self.chart_panel = ChartPanel()
# # #         self.chart_dock.setWidget(self.chart_panel)
# # #         self.addDockWidget(Qt.BottomDockWidgetArea, self.chart_dock)
# # #
# # #         # 日志面板
# # #         self.log_dock = QDockWidget("日志", self)
# # #         self.log_dock.setObjectName("log_dock")
# # #         self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
# # #         self.log_panel = LogPanel()
# # #         self.log_dock.setWidget(self.log_panel)
# # #         self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
# # #
# # #         # 标签化底部面板
# # #         self.tabifyDockWidget(self.chart_dock, self.log_dock)
# # #         self.chart_dock.raise_()
# # #
# # #     def _setup_central_widget(self):
# # #         """设置中央部件（3D视图）"""
# # #         self.central_widget = QWidget()
# # #         self.setCentralWidget(self.central_widget)
# # #
# # #         layout = QVBoxLayout(self.central_widget)
# # #         layout.setContentsMargins(0, 0, 0, 0)
# # #
# # #         # 3D视图
# # #         self.viewer_3d = Viewer3D(self.engine)
# # #         layout.addWidget(self.viewer_3d)
# # #
# # #     def _connect_signals(self):
# # #         """连接信号槽"""
# # #         # 仿真引擎事件
# # #         self.engine.add_event_listener(EventType.SIMULATION_START, self._on_sim_started)
# # #         self.engine.add_event_listener(EventType.SIMULATION_STOP, self._on_sim_stopped)
# # #         self.engine.add_event_listener(EventType.SIMULATION_PAUSE, self._on_sim_paused)
# # #         self.engine.add_event_listener(EventType.SIMULATION_RESUME, self._on_sim_resumed)
# # #         self.engine.add_event_listener(EventType.COLLISION, self._on_collision)
# # #
# # #         # 状态更新
# # #         self.engine.add_state_callback(self._on_state_update)
# # #
# # #         # 面板信号
# # #         self.control_panel.target_changed.connect(self._on_target_changed)
# # #         self.param_panel.params_changed.connect(self._on_params_changed)
# # #
# # #     # ==================== 事件处理 ====================
# # #
# # #     def _on_state_update(self, state: DroneState):
# # #         """处理状态更新"""
# # #         self.state_updated.emit(state)
# # #
# # #     def _update_display(self):
# # #         """定时更新显示"""
# # #         if self.engine.is_running or self.engine.is_paused:
# # #             state = self.engine.drone_state
# # #             stats = self.engine.statistics
# # #
# # #             # 更新状态栏
# # #             self.status_sim_time.setText(f"时间: {stats.simulation_time:.2f}s")
# # #             self.status_realtime.setText(f"实时比: {stats.realtime_ratio:.2f}x")
# # #             self.status_altitude.setText(f"高度: {state.altitude:.1f}m")
# # #             self.status_speed.setText(f"速度: {state.speed:.1f}m/s")
# # #
# # #             # 更新信息面板
# # #             self.info_panel.update_state(state)
# # #
# # #             # 更新图表
# # #             self.chart_panel.add_data_point(stats.simulation_time, state)
# # #
# # #             # 更新3D视图
# # #             self.viewer_3d.update_drone_state(state)
# # #
# # #     def _on_sim_started(self, event: SimulationEvent):
# # #         """仿真开始事件"""
# # #         self.status_sim_state.setText("状态: 运行中")
# # #         self.status_sim_state.setStyleSheet("color: green;")
# # #         self._update_button_states(running=True, paused=False)
# # #         logger.info("仿真已开始")
# # #
# # #     def _on_sim_stopped(self, event: SimulationEvent):
# # #         """仿真停止事件"""
# # #         self.status_sim_state.setText("状态: 已停止")
# # #         self.status_sim_state.setStyleSheet("color: red;")
# # #         self._update_button_states(running=False, paused=False)
# # #         logger.info("仿真已停止")
# # #
# # #     def _on_sim_paused(self, event: SimulationEvent):
# # #         """仿真暂停事件"""
# # #         self.status_sim_state.setText("状态: 已暂停")
# # #         self.status_sim_state.setStyleSheet("color: orange;")
# # #         self._update_button_states(running=True, paused=True)
# # #         logger.info("仿真已暂停")
# # #
# # #     def _on_sim_resumed(self, event: SimulationEvent):
# # #         """仿真恢复事件"""
# # #         self.status_sim_state.setText("状态: 运行中")
# # #         self.status_sim_state.setStyleSheet("color: green;")
# # #         self._update_button_states(running=True, paused=False)
# # #         logger.info("仿真已恢复")
# # #
# # #     def _on_collision(self, event: SimulationEvent):
# # #         """碰撞事件"""
# # #         collision_type = event.data.get('type', 'unknown')
# # #         logger.warning(f"发生碰撞: {collision_type}")
# # #
# # #         if collision_type == 'ground':
# # #             QMessageBox.warning(self, "警告", "无人机已触地！")
# # #
# # #     def _update_button_states(self, running: bool, paused: bool):
# # #         """更新按钮状态"""
# # #         self.start_action.setEnabled(not running or paused)
# # #         self.pause_action.setEnabled(running and not paused)
# # #         self.stop_action.setEnabled(running)
# # #         self.step_action.setEnabled(not running or paused)
# # #
# # #         self.btn_start.setEnabled(not running or paused)
# # #         self.btn_pause.setEnabled(running and not paused)
# # #         self.btn_stop.setEnabled(running)
# # #
# # #     # ==================== 菜单动作 ====================
# # #
# # #     def _on_start_simulation(self):
# # #         """开始仿真"""
# # #         if self.engine.is_paused:
# # #             self.engine.resume()
# # #         else:
# # #             self.engine.start()
# # #
# # #     def _on_pause_simulation(self):
# # #         """暂停仿真"""
# # #         self.engine.pause()
# # #
# # #     def _on_stop_simulation(self):
# # #         """停止仿真"""
# # #         self.engine.stop()
# # #
# # #     def _on_reset_simulation(self):
# # #         """重置仿真"""
# # #         self.engine.reset()
# # #         self.chart_panel.clear()
# # #         self.status_sim_state.setText("状态: 就绪")
# # #         self.status_sim_state.setStyleSheet("")
# # #
# # #     def _on_step_simulation(self):
# # #         """单步执行"""
# # #         self.engine.step(100)  # 执行100个物理步
# # #         self._update_display()
# # #
# # #     def _set_simulation_speed(self, speed: float):
# # #         """设置仿真速度"""
# # #         self.engine.config.realtime_factor = speed
# # #         logger.info(f"仿真速度设置为: {speed}x")
# # #
# # #     def _on_toggle_recording(self, checked: bool):
# # #         """切换录制状态"""
# # #         if checked:
# # #             self.engine.start_recording()
# # #             self.btn_record.setText("⏹ 停止录制")
# # #         else:
# # #             history = self.engine.stop_recording()
# # #             self.btn_record.setText("⏺ 录制")
# # #
# # #             # 保存录制
# # #             if history:
# # #                 reply = QMessageBox.question(
# # #                     self, "保存录制",
# # #                     f"已录制 {len(history)} 个数据点，是否保存？",
# # #                     QMessageBox.Yes | QMessageBox.No
# # #                 )
# # #                 if reply == QMessageBox.Yes:
# # #                     self._save_recording(history)
# # #
# # #     def _save_recording(self, history):
# # #         """保存录制数据"""
# # #         filename, _ = QFileDialog.getSaveFileName(
# # #             self, "保存录制", "", "CSV文件 (*.csv);;JSON文件 (*.json)"
# # #         )
# # #         if filename:
# # #             # TODO: 实现保存逻辑
# # #             logger.info(f"录制已保存到: {filename}")
# # #
# # #     def _on_new_scenario(self):
# # #         """新建场景"""
# # #         self.engine.reset()
# # #         logger.info("已创建新场景")
# # #
# # #     def _on_open_scenario(self):
# # #         """打开场景"""
# # #         filename, _ = QFileDialog.getOpenFileName(
# # #             self, "打开场景", "", "场景文件 (*.yaml *.json)"
# # #         )
# # #         if filename:
# # #             # TODO: 实现场景加载
# # #             logger.info(f"打开场景: {filename}")
# # #
# # #     def _on_save_scenario(self):
# # #         """保存场景"""
# # #         filename, _ = QFileDialog.getSaveFileName(
# # #             self, "保存场景", "", "场景文件 (*.yaml)"
# # #         )
# # #         if filename:
# # #             # TODO: 实现场景保存
# # #             logger.info(f"场景已保存: {filename}")
# # #
# # #     def _on_export_csv(self):
# # #         """导出CSV"""
# # #         filename, _ = QFileDialog.getSaveFileName(
# # #             self, "导出CSV", "", "CSV文件 (*.csv)"
# # #         )
# # #         if filename:
# # #             # TODO: 实现CSV导出
# # #             logger.info(f"数据已导出: {filename}")
# # #
# # #     def _on_export_pdf(self):
# # #         """导出PDF报告"""
# # #         filename, _ = QFileDialog.getSaveFileName(
# # #             self, "导出报告", "", "PDF文件 (*.pdf)"
# # #         )
# # #         if filename:
# # #             # TODO: 实现PDF导出
# # #             logger.info(f"报告已导出: {filename}")
# # #
# # #     def _on_target_changed(self, position: np.ndarray, yaw: float):
# # #         """目标改变"""
# # #         self.engine.set_target(position, yaw)
# # #
# # #     def _on_params_changed(self, params: dict):
# # #         """参数改变"""
# # #         # TODO: 更新参数
# # #         pass
# # #
# # #     def _on_theme_change(self, theme: str):
# # #         """主题切换"""
# # #         theme_file = "dark_theme.qss" if "深色" in theme else "light_theme.qss"
# # #         theme_path = Path(f"ui/themes/{theme_file}")
# # #
# # #         if theme_path.exists():
# # #             with open(theme_path, 'r', encoding='utf-8') as f:
# # #                 self.setStyleSheet(f.read())
# # #             logger.info(f"已切换到: {theme}")
# # #
# # #     def _on_open_settings(self):
# # #         """打开设置"""
# # #         from ui.dialogs.settings_dialog import SettingsDialog
# # #         dialog = SettingsDialog(self)
# # #         if dialog.exec_():
# # #             # 应用设置
# # #             config_manager.save()
# # #             logger.info("设置已保存")
# # #
# # #     def _on_open_tutorial(self, tutorial_id: str):
# # #         """打开教程"""
# # #         logger.info(f"打开教程: {tutorial_id}")
# # #         # TODO: 实现教程系统
# # #
# # #     def _on_open_documentation(self):
# # #         """打开文档"""
# # #         import webbrowser
# # #         webbrowser.open("https://docs.example.com/uav-teaching-system")
# # #
# # #     def _on_about(self):
# # #         """关于对话框"""
# # #         QMessageBox.about(
# # #             self,
# # #             "关于",
# # #             """<h3>低空交通无人机教学演示系统</h3>
# # #             <p>版本: 1.0.0</p>
# # #             <p>一个用于无人机飞行控制、路径规划和交通管理的教学平台。</p>
# # #             <p>Copyright © 2024</p>"""
# # #         )
# # #
# # #     # ==================== 设置保存/加载 ====================
# # #
# # #     def _load_settings(self):
# # #         """加载窗口设置"""
# # #         settings = QSettings("UAVTeaching", "MainWindow")
# # #
# # #         geometry = settings.value("geometry")
# # #         if geometry:
# # #             self.restoreGeometry(geometry)
# # #
# # #         state = settings.value("windowState")
# # #         if state:
# # #             self.restoreState(state)
# # #
# # #     def _save_settings(self):
# # #         """保存窗口设置"""
# # #         settings = QSettings("UAVTeaching", "MainWindow")
# # #         settings.setValue("geometry", self.saveGeometry())
# # #         settings.setValue("windowState", self.saveState())
# # #
# # #     def closeEvent(self, event):
# # #         """关闭事件"""
# # #         # 停止仿真
# # #         if self.engine.is_running:
# # #             reply = QMessageBox.question(
# # #                 self, "确认退出",
# # #                 "仿真正在运行，确定要退出吗？",
# # #                 QMessageBox.Yes | QMessageBox.No
# # #             )
# # #             if reply == QMessageBox.No:
# # #                 event.ignore()
# # #                 return
# # #             self.engine.stop()
# # #
# # #         # 保存设置
# # #         self._save_settings()
# # #         config_manager.save()
# # #
# # #         logger.info("应用程序关闭")
# # #         event.accept()
# #
# # # ui/main_window/main_window.py（完整修复版）
# #
# # """
# # 主窗口模块
# # """
# #
# # import numpy as np
# # from PyQt5.QtWidgets import (
# #     QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
# #     QDockWidget, QToolBar, QAction, QActionGroup,
# #     QStatusBar, QLabel, QMenuBar, QMenu, QMessageBox,
# #     QSplitter, QFrame
# # )
# # from PyQt5.QtCore import Qt, QTimer, pyqtSignal
# # from PyQt5.QtGui import QIcon, QKeySequence
# # from loguru import logger
# #
# # # 安全导入仿真引擎
# # try:
# #     from simulation.engine.simulation_core import SimulationEngine, SimulationState
# # except ImportError as e:
# #     logger.error(f"导入SimulationEngine失败: {e}")
# #     SimulationEngine = None
# #     SimulationState = None
# #
# # try:
# #     from simulation.engine.simulation_core import SimulationEvent, EventType
# # except ImportError:
# #     SimulationEvent = None
# #     EventType = None
# #
# # # 导入DroneState
# # try:
# #     from core.physics.quadrotor_dynamics import DroneState
# # except ImportError:
# #     from dataclasses import dataclass, field
# #
# #
# #     @dataclass
# #     class DroneState:
# #         """无人机状态（备用定义）"""
# #         position: np.ndarray = field(default_factory=lambda: np.zeros(3))
# #         velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
# #         quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
# #         angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
# #         motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
# #
# #         @property
# #         def euler_angles(self) -> np.ndarray:
# #             return np.zeros(3)
# #
# #         @property
# #         def altitude(self) -> float:
# #             return -self.position[2]
# #
# #         @property
# #         def speed(self) -> float:
# #             return float(np.linalg.norm(self.velocity))
# #
# # from utils.config.config_manager import get_config
# #
# # # 安全导入UI组件
# # try:
# #     from ui.visualization.view_3d import Drone3DView
# # except ImportError:
# #     Drone3DView = None
# #     logger.warning("无法导入3D视图组件")
# #
# # try:
# #     from ui.panels.control_panel import ControlPanel
# # except ImportError:
# #     ControlPanel = None
# #     logger.warning("无法导入控制面板")
# #
# # try:
# #     from ui.panels.status_panel import StatusPanel
# # except ImportError:
# #     StatusPanel = None
# #     logger.warning("无法导入状态面板")
# #
# # try:
# #     from ui.panels.parameter_panel import ParameterPanel
# # except ImportError:
# #     ParameterPanel = None
# #     logger.warning("无法导入参数面板")
# #
# # try:
# #     from ui.panels.log_panel import LogPanel
# # except ImportError:
# #     LogPanel = None
# #     logger.warning("无法导入日志面板")
# #
# # try:
# #     from ui.visualization.realtime_charts import RealtimeChartPanel
# # except ImportError:
# #     RealtimeChartPanel = None
# #     logger.warning("无法导入实时图表")
# #
# #
# # class MainWindow(QMainWindow):
# #     """主窗口"""
# #
# #     # 信号
# #     simulation_started = pyqtSignal()
# #     simulation_stopped = pyqtSignal()
# #
# #     def __init__(self):
# #         super().__init__()
# #
# #         self.setWindowTitle("低空交通无人机教学演示系统")
# #         self.setGeometry(100, 100, 1600, 900)
# #
# #         # 初始化仿真引擎
# #         self.engine = None
# #         if SimulationEngine is not None:
# #             try:
# #                 self.engine = SimulationEngine()
# #             except Exception as e:
# #                 logger.error(f"创建仿真引擎失败: {e}")
# #
# #         # 设置UI
# #         self._setup_ui()
# #         self._setup_menu()
# #         self._setup_toolbar()
# #         self._setup_statusbar()
# #         self._setup_connections()
# #
# #         # 更新定时器
# #         self.update_timer = QTimer()
# #         self.update_timer.timeout.connect(self._update_display)
# #         self.update_timer.start(33)  # 约30 FPS
# #
# #         logger.info("主窗口初始化完成")
# #
# #     def _setup_ui(self):
# #         """设置UI"""
# #         # 中央分割器
# #         central_splitter = QSplitter(Qt.Horizontal)
# #         self.setCentralWidget(central_splitter)
# #
# #         # 左侧面板区域
# #         left_panel = QWidget()
# #         left_layout = QVBoxLayout(left_panel)
# #         left_layout.setContentsMargins(0, 0, 0, 0)
# #
# #         # 控制面板
# #         if ControlPanel is not None:
# #             try:
# #                 self.control_panel = ControlPanel()
# #                 left_layout.addWidget(self.control_panel)
# #             except Exception as e:
# #                 logger.error(f"创建控制面板失败: {e}")
# #                 self.control_panel = None
# #                 left_layout.addWidget(QLabel("控制面板加载失败"))
# #         else:
# #             self.control_panel = None
# #             left_layout.addWidget(self._create_simple_control_panel())
# #
# #         # 状态面板
# #         if StatusPanel is not None:
# #             try:
# #                 self.status_panel = StatusPanel()
# #                 left_layout.addWidget(self.status_panel)
# #             except Exception as e:
# #                 logger.error(f"创建状态面板失败: {e}")
# #                 self.status_panel = None
# #         else:
# #             self.status_panel = None
# #
# #         left_layout.addStretch()
# #
# #         # 中央3D视图区域
# #         center_widget = QWidget()
# #         center_layout = QVBoxLayout(center_widget)
# #         center_layout.setContentsMargins(0, 0, 0, 0)
# #
# #         if Drone3DView is not None:
# #             try:
# #                 self.view_3d = Drone3DView()
# #                 center_layout.addWidget(self.view_3d)
# #             except Exception as e:
# #                 logger.error(f"创建3D视图失败: {e}")
# #                 self.view_3d = None
# #                 center_layout.addWidget(self._create_placeholder("3D视图"))
# #         else:
# #             self.view_3d = None
# #             center_layout.addWidget(self._create_placeholder("3D视图"))
# #
# #         # 右侧面板区域
# #         right_panel = QWidget()
# #         right_layout = QVBoxLayout(right_panel)
# #         right_layout.setContentsMargins(0, 0, 0, 0)
# #
# #         # 实时图表
# #         if RealtimeChartPanel is not None:
# #             try:
# #                 self.chart_panel = RealtimeChartPanel()
# #                 right_layout.addWidget(self.chart_panel)
# #             except Exception as e:
# #                 logger.error(f"创建图表面板失败: {e}")
# #                 self.chart_panel = None
# #         else:
# #             self.chart_panel = None
# #             right_layout.addWidget(self._create_placeholder("实时图表"))
# #
# #         # 添加到分割器
# #         central_splitter.addWidget(left_panel)
# #         central_splitter.addWidget(center_widget)
# #         central_splitter.addWidget(right_panel)
# #         central_splitter.setSizes([250, 900, 350])
# #
# #         # 底部停靠窗口（日志）
# #         if LogPanel is not None:
# #             try:
# #                 self.log_panel = LogPanel()
# #                 log_dock = QDockWidget("系统日志", self)
# #                 log_dock.setWidget(self.log_panel)
# #                 log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
# #                 self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
# #             except Exception as e:
# #                 logger.error(f"创建日志面板失败: {e}")
# #                 self.log_panel = None
# #         else:
# #             self.log_panel = None
# #
# #         # 右侧停靠窗口（参数）
# #         if ParameterPanel is not None:
# #             try:
# #                 self.param_panel = ParameterPanel()
# #                 param_dock = QDockWidget("参数设置", self)
# #                 param_dock.setWidget(self.param_panel)
# #                 param_dock.setAllowedAreas(Qt.RightDockWidgetArea)
# #                 self.addDockWidget(Qt.RightDockWidgetArea, param_dock)
# #                 param_dock.hide()  # 默认隐藏
# #                 self.param_dock = param_dock
# #             except Exception as e:
# #                 logger.error(f"创建参数面板失败: {e}")
# #                 self.param_panel = None
# #                 self.param_dock = None
# #         else:
# #             self.param_panel = None
# #             self.param_dock = None
# #
# #     def _create_placeholder(self, name: str) -> QWidget:
# #         """创建占位符控件"""
# #         widget = QFrame()
# #         widget.setFrameStyle(QFrame.StyledPanel)
# #         widget.setMinimumSize(200, 200)
# #
# #         layout = QVBoxLayout(widget)
# #         label = QLabel(f"{name}\n(组件未加载)")
# #         label.setAlignment(Qt.AlignCenter)
# #         label.setStyleSheet("color: #808080; font-size: 16px;")
# #         layout.addWidget(label)
# #
# #         return widget
# #
# #     def _create_simple_control_panel(self) -> QWidget:
# #         """创建简单控制面板"""
# #         from PyQt5.QtWidgets import QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox
# #
# #         panel = QWidget()
# #         layout = QVBoxLayout(panel)
# #
# #         # 控制按钮
# #         btn_group = QGroupBox("仿真控制")
# #         btn_layout = QHBoxLayout(btn_group)
# #
# #         self.btn_start = QPushButton("开始")
# #         self.btn_start.clicked.connect(self._on_start)
# #         btn_layout.addWidget(self.btn_start)
# #
# #         self.btn_pause = QPushButton("暂停")
# #         self.btn_pause.clicked.connect(self._on_pause)
# #         self.btn_pause.setEnabled(False)
# #         btn_layout.addWidget(self.btn_pause)
# #
# #         self.btn_stop = QPushButton("停止")
# #         self.btn_stop.clicked.connect(self._on_stop)
# #         self.btn_stop.setEnabled(False)
# #         btn_layout.addWidget(self.btn_stop)
# #
# #         self.btn_reset = QPushButton("重置")
# #         self.btn_reset.clicked.connect(self._on_reset)
# #         btn_layout.addWidget(self.btn_reset)
# #
# #         layout.addWidget(btn_group)
# #
# #         # 目标位置
# #         target_group = QGroupBox("目标位置")
# #         target_layout = QFormLayout(target_group)
# #
# #         self.spin_target_x = QDoubleSpinBox()
# #         self.spin_target_x.setRange(-100, 100)
# #         self.spin_target_x.setValue(10)
# #         target_layout.addRow("X:", self.spin_target_x)
# #
# #         self.spin_target_y = QDoubleSpinBox()
# #         self.spin_target_y.setRange(-100, 100)
# #         self.spin_target_y.setValue(5)
# #         target_layout.addRow("Y:", self.spin_target_y)
# #
# #         self.spin_target_z = QDoubleSpinBox()
# #         self.spin_target_z.setRange(-100, 0)
# #         self.spin_target_z.setValue(-15)
# #         target_layout.addRow("Z:", self.spin_target_z)
# #
# #         btn_apply = QPushButton("应用目标")
# #         btn_apply.clicked.connect(self._apply_target)
# #         target_layout.addRow(btn_apply)
# #
# #         layout.addWidget(target_group)
# #
# #         # 状态显示
# #         status_group = QGroupBox("飞行状态")
# #         status_layout = QFormLayout(status_group)
# #
# #         self.label_time = QLabel("0.00 s")
# #         status_layout.addRow("时间:", self.label_time)
# #
# #         self.label_pos = QLabel("[0, 0, 0]")
# #         status_layout.addRow("位置:", self.label_pos)
# #
# #         self.label_alt = QLabel("0.00 m")
# #         status_layout.addRow("高度:", self.label_alt)
# #
# #         self.label_speed = QLabel("0.00 m/s")
# #         status_layout.addRow("速度:", self.label_speed)
# #
# #         layout.addWidget(status_group)
# #
# #         return panel
# #
# #     def _setup_menu(self):
# #         """设置菜单"""
# #         menubar = self.menuBar()
# #
# #         # 文件菜单
# #         file_menu = menubar.addMenu("文件(&F)")
# #
# #         action_new = QAction("新建场景", self)
# #         action_new.setShortcut(QKeySequence.New)
# #         file_menu.addAction(action_new)
# #
# #         action_open = QAction("打开场景", self)
# #         action_open.setShortcut(QKeySequence.Open)
# #         file_menu.addAction(action_open)
# #
# #         action_save = QAction("保存场景", self)
# #         action_save.setShortcut(QKeySequence.Save)
# #         file_menu.addAction(action_save)
# #
# #         file_menu.addSeparator()
# #
# #         action_exit = QAction("退出", self)
# #         action_exit.setShortcut(QKeySequence.Quit)
# #         action_exit.triggered.connect(self.close)
# #         file_menu.addAction(action_exit)
# #
# #         # 仿真菜单
# #         sim_menu = menubar.addMenu("仿真(&S)")
# #
# #         self.action_start = QAction("开始", self)
# #         self.action_start.setShortcut("F5")
# #         self.action_start.triggered.connect(self._on_start)
# #         sim_menu.addAction(self.action_start)
# #
# #         self.action_pause = QAction("暂停", self)
# #         self.action_pause.setShortcut("F6")
# #         self.action_pause.triggered.connect(self._on_pause)
# #         self.action_pause.setEnabled(False)
# #         sim_menu.addAction(self.action_pause)
# #
# #         self.action_stop = QAction("停止", self)
# #         self.action_stop.setShortcut("F7")
# #         self.action_stop.triggered.connect(self._on_stop)
# #         self.action_stop.setEnabled(False)
# #         sim_menu.addAction(self.action_stop)
# #
# #         sim_menu.addSeparator()
# #
# #         action_reset = QAction("重置", self)
# #         action_reset.setShortcut("F8")
# #         action_reset.triggered.connect(self._on_reset)
# #         sim_menu.addAction(action_reset)
# #
# #         # 视图菜单
# #         view_menu = menubar.addMenu("视图(&V)")
# #
# #         if self.param_dock is not None:
# #             action_params = self.param_dock.toggleViewAction()
# #             action_params.setText("参数面板")
# #             view_menu.addAction(action_params)
# #
# #         view_menu.addSeparator()
# #
# #         action_reset_view = QAction("重置视图", self)
# #         action_reset_view.triggered.connect(self._reset_view)
# #         view_menu.addAction(action_reset_view)
# #
# #         # 帮助菜单
# #         help_menu = menubar.addMenu("帮助(&H)")
# #
# #         action_about = QAction("关于", self)
# #         action_about.triggered.connect(self._show_about)
# #         help_menu.addAction(action_about)
# #
# #     def _setup_toolbar(self):
# #         """设置工具栏"""
# #         toolbar = QToolBar("主工具栏")
# #         toolbar.setMovable(False)
# #         self.addToolBar(toolbar)
# #
# #         # 仿真控制按钮
# #         self.tb_start = QAction("▶ 开始", self)
# #         self.tb_start.triggered.connect(self._on_start)
# #         toolbar.addAction(self.tb_start)
# #
# #         self.tb_pause = QAction("⏸ 暂停", self)
# #         self.tb_pause.triggered.connect(self._on_pause)
# #         self.tb_pause.setEnabled(False)
# #         toolbar.addAction(self.tb_pause)
# #
# #         self.tb_stop = QAction("⏹ 停止", self)
# #         self.tb_stop.triggered.connect(self._on_stop)
# #         self.tb_stop.setEnabled(False)
# #         toolbar.addAction(self.tb_stop)
# #
# #         toolbar.addSeparator()
# #
# #         self.tb_reset = QAction("↺ 重置", self)
# #         self.tb_reset.triggered.connect(self._on_reset)
# #         toolbar.addAction(self.tb_reset)
# #
# #     def _setup_statusbar(self):
# #         """设置状态栏"""
# #         statusbar = self.statusBar()
# #
# #         self.status_sim_state = QLabel("状态: 停止")
# #         statusbar.addWidget(self.status_sim_state)
# #
# #         self.status_sim_time = QLabel("时间: 0.00s")
# #         statusbar.addWidget(self.status_sim_time)
# #
# #         self.status_fps = QLabel("FPS: 0")
# #         statusbar.addPermanentWidget(self.status_fps)
# #
# #     def _setup_connections(self):
# #         """设置信号连接"""
# #         if self.control_panel is not None:
# #             try:
# #                 self.control_panel.start_clicked.connect(self._on_start)
# #                 self.control_panel.pause_clicked.connect(self._on_pause)
# #                 self.control_panel.stop_clicked.connect(self._on_stop)
# #                 self.control_panel.reset_clicked.connect(self._on_reset)
# #             except AttributeError:
# #                 pass
# #
# #     def _on_start(self):
# #         """开始仿真"""
# #         if self.engine is None:
# #             logger.warning("仿真引擎未初始化")
# #             return
# #
# #         try:
# #             # 应用目标位置
# #             self._apply_target()
# #
# #             if hasattr(self.engine, 'is_paused') and self.engine.is_paused:
# #                 self.engine.resume()
# #             else:
# #                 self.engine.start()
# #
# #             self._update_button_states(running=True)
# #             self.simulation_started.emit()
# #             logger.info("仿真开始")
# #         except Exception as e:
# #             logger.error(f"启动仿真失败: {e}")
# #
# #     def _on_pause(self):
# #         """暂停仿真"""
# #         if self.engine is None:
# #             return
# #
# #         try:
# #             self.engine.pause()
# #             self._update_button_states(paused=True)
# #             logger.info("仿真暂停")
# #         except Exception as e:
# #             logger.error(f"暂停仿真失败: {e}")
# #
# #     def _on_stop(self):
# #         """停止仿真"""
# #         if self.engine is None:
# #             return
# #
# #         try:
# #             self.engine.stop()
# #             self._update_button_states(stopped=True)
# #             self.simulation_stopped.emit()
# #             logger.info("仿真停止")
# #         except Exception as e:
# #             logger.error(f"停止仿真失败: {e}")
# #
# #     def _on_reset(self):
# #         """重置仿真"""
# #         if self.engine is None:
# #             return
# #
# #         try:
# #             self.engine.reset()
# #             self._update_button_states(stopped=True)
# #             logger.info("仿真重置")
# #         except Exception as e:
# #             logger.error(f"重置仿真失败: {e}")
# #
# #     def _apply_target(self):
# #         """应用目标位置"""
# #         if self.engine is None:
# #             return
# #
# #         try:
# #             # 尝试从控制面板获取
# #             if self.control_panel is not None and hasattr(self.control_panel, 'get_target_position'):
# #                 target = self.control_panel.get_target_position()
# #             else:
# #                 # 从简单面板获取
# #                 target = np.array([
# #                     self.spin_target_x.value(),
# #                     self.spin_target_y.value(),
# #                     self.spin_target_z.value()
# #                 ])
# #
# #             self.engine.set_target_position(target)
# #             logger.info(f"目标位置设置为: {target}")
# #         except Exception as e:
# #             logger.error(f"设置目标位置失败: {e}")
# #
# #     def _update_button_states(self, running=False, paused=False, stopped=False):
# #         """更新按钮状态"""
# #         # 菜单动作
# #         self.action_start.setEnabled(not running or paused)
# #         self.action_pause.setEnabled(running and not paused)
# #         self.action_stop.setEnabled(running or paused)
# #
# #         # 工具栏
# #         self.tb_start.setEnabled(not running or paused)
# #         self.tb_pause.setEnabled(running and not paused)
# #         self.tb_stop.setEnabled(running or paused)
# #
# #         # 简单控制面板按钮
# #         if hasattr(self, 'btn_start'):
# #             self.btn_start.setEnabled(not running or paused)
# #             self.btn_pause.setEnabled(running and not paused)
# #             self.btn_stop.setEnabled(running or paused)
# #
# #         # 状态栏
# #         if stopped:
# #             self.status_sim_state.setText("状态: 停止")
# #         elif paused:
# #             self.status_sim_state.setText("状态: 暂停")
# #         elif running:
# #             self.status_sim_state.setText("状态: 运行中")
# #
# #     def _update_display(self):
# #         """更新显示"""
# #         if self.engine is None:
# #             return
# #
# #         try:
# #             state = self.engine.drone_state
# #             stats = self.engine.statistics
# #
# #             # 更新状态栏
# #             self.status_sim_time.setText(f"时间: {stats.simulation_time:.2f}s")
# #             self.status_fps.setText(f"FPS: {stats.average_fps:.0f}")
# #
# #             # 更新3D视图
# #             if self.view_3d is not None:
# #                 self.view_3d.update_drone_state(state)
# #
# #             # 更新状态面板
# #             if self.status_panel is not None:
# #                 self.status_panel.update_state(state)
# #
# #             # 更新图表
# #             if self.chart_panel is not None:
# #                 self.chart_panel.update_data(state, stats.simulation_time)
# #
# #             # 更新简单面板标签
# #             if hasattr(self, 'label_time'):
# #                 self.label_time.setText(f"{stats.simulation_time:.2f} s")
# #                 self.label_pos.setText(
# #                     f"[{state.position[0]:.1f}, {state.position[1]:.1f}, {state.position[2]:.1f}]"
# #                 )
# #                 self.label_alt.setText(f"{state.altitude:.2f} m")
# #                 self.label_speed.setText(f"{state.speed:.2f} m/s")
# #
# #         except Exception as e:
# #             logger.error(f"更新显示失败: {e}")
# #
# #     def _on_state_update(self, state: DroneState):
# #         """处理状态更新"""
# #         # 这个方法现在可以正常工作了
# #         pass
# #
# #     def _reset_view(self):
# #         """重置视图"""
# #         if self.view_3d is not None:
# #             try:
# #                 self.view_3d.reset_view()
# #             except Exception as e:
# #                 logger.error(f"重置视图失败: {e}")
# #
# #     def _show_about(self):
# #         """显示关于对话框"""
# #         QMessageBox.about(
# #             self,
# #             "关于",
# #             "低空交通无人机教学演示系统\n\n"
# #             "版本: 1.0.0\n"
# #             "用于无人机飞行控制、路径规划和\n"
# #             "低空交通管理的教学演示\n\n"
# #             "© 2024"
# #         )
# #
# #     def closeEvent(self, event):
# #         """关闭事件"""
# #         if self.engine is not None:
# #             try:
# #                 self.engine.stop()
# #             except Exception:
# #                 pass
# #
# #         event.accept()
#
#
# # ui/main_window/main_window.py（简化修复版）
#
# """
# 主窗口模块
# """
#
# import numpy as np
# from PyQt5.QtWidgets import (
#     QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#     QSplitter, QLabel, QDockWidget, QToolBar, QAction,
#     QStatusBar, QMessageBox, QMenuBar, QMenu, QGroupBox,
#     QFormLayout, QDoubleSpinBox, QPushButton, QTextEdit,
#     QFrame, QProgressBar
# )
# from PyQt5.QtCore import Qt, QTimer
# from PyQt5.QtGui import QFont, QIcon
# from loguru import logger
# # # 导入DroneState
# try:
#     from core.physics.quadrotor_dynamics import DroneState
# except ImportError:
#     from dataclasses import dataclass, field
#
# # ui/main_window/main_window.py（简化修复版）
#
# """
# 主窗口模块
# """
#
# import numpy as np
# from PyQt5.QtWidgets import (
#     QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
#     QSplitter, QLabel, QDockWidget, QToolBar, QAction,
#     QStatusBar, QMessageBox, QMenuBar, QMenu, QGroupBox,
#     QFormLayout, QDoubleSpinBox, QPushButton, QTextEdit,
#     QFrame, QProgressBar
# )
# from PyQt5.QtCore import Qt, QTimer
# from PyQt5.QtGui import QFont, QIcon
# from loguru import logger
#
#
# class MainWindow(QMainWindow):
#     """主窗口"""
#
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("低空交通无人机教学演示系统")
#         self.setGeometry(100, 100, 1400, 900)
#
#         # 初始化仿真引擎
#         self._init_engine()
#
#         # 设置UI
#         self._setup_ui()
#         self._setup_menu()
#         self._setup_toolbar()
#         self._setup_statusbar()
#
#         # 更新定时器
#         self.timer = QTimer()
#         self.timer.timeout.connect(self._update_display)
#         self.timer.start(50)  # 20 FPS
#
#         self._initial_distance = 0.0
#
#         logger.info("主窗口初始化完成")
#
#     def _init_engine(self):
#         """初始化仿真引擎"""
#         try:
#             from simulation.engine.simulation_core import SimulationEngine
#             self.engine = SimulationEngine()
#             logger.info("仿真引擎初始化成功")
#         except Exception as e:
#             logger.error(f"仿真引擎初始化失败: {e}")
#             self.engine = None
#
#     def _setup_ui(self):
#         """设置UI"""
#         central = QWidget()
#         self.setCentralWidget(central)
#         main_layout = QHBoxLayout(central)
#
#         # ========== 左侧控制面板 ==========
#         left_panel = QWidget()
#         left_panel.setMaximumWidth(350)
#         left_layout = QVBoxLayout(left_panel)
#
#         # 标题
#         title = QLabel("🚁 无人机仿真控制")
#         title.setFont(QFont("Arial", 14, QFont.Bold))
#         title.setAlignment(Qt.AlignCenter)
#         left_layout.addWidget(title)
#
#         # 控制按钮组
#         ctrl_group = QGroupBox("仿真控制")
#         ctrl_layout = QVBoxLayout(ctrl_group)
#
#         btn_row1 = QHBoxLayout()
#         self.btn_start = QPushButton("▶ 开始")
#         self.btn_start.clicked.connect(self._on_start)
#         btn_row1.addWidget(self.btn_start)
#
#         self.btn_pause = QPushButton("⏸ 暂停")
#         self.btn_pause.clicked.connect(self._on_pause)
#         self.btn_pause.setEnabled(False)
#         btn_row1.addWidget(self.btn_pause)
#         ctrl_layout.addLayout(btn_row1)
#
#         btn_row2 = QHBoxLayout()
#         self.btn_stop = QPushButton("⏹ 停止")
#         self.btn_stop.clicked.connect(self._on_stop)
#         self.btn_stop.setEnabled(False)
#         btn_row2.addWidget(self.btn_stop)
#
#         self.btn_reset = QPushButton("↺ 重置")
#         self.btn_reset.clicked.connect(self._on_reset)
#         btn_row2.addWidget(self.btn_reset)
#         ctrl_layout.addLayout(btn_row2)
#
#         left_layout.addWidget(ctrl_group)
#
#         # 目标位置组
#         target_group = QGroupBox("目标位置设置")
#         target_layout = QFormLayout(target_group)
#
#         self.spin_target_x = QDoubleSpinBox()
#         self.spin_target_x.setRange(-100, 100)
#         self.spin_target_x.setValue(10)
#         self.spin_target_x.setSingleStep(1)
#         self.spin_target_x.setDecimals(1)
#         target_layout.addRow("X (北) [m]:", self.spin_target_x)
#
#         self.spin_target_y = QDoubleSpinBox()
#         self.spin_target_y.setRange(-100, 100)
#         self.spin_target_y.setValue(5)
#         self.spin_target_y.setSingleStep(1)
#         self.spin_target_y.setDecimals(1)
#         target_layout.addRow("Y (东) [m]:", self.spin_target_y)
#
#         self.spin_target_z = QDoubleSpinBox()
#         self.spin_target_z.setRange(-100, 0)
#         self.spin_target_z.setValue(-15)
#         self.spin_target_z.setSingleStep(1)
#         self.spin_target_z.setDecimals(1)
#         target_layout.addRow("Z (高度) [m]:", self.spin_target_z)
#
#         btn_apply = QPushButton("📍 应用目标位置")
#         btn_apply.clicked.connect(self._apply_target)
#         target_layout.addRow(btn_apply)
#
#         left_layout.addWidget(target_group)
#
#         # 状态显示组
#         status_group = QGroupBox("飞行状态")
#         status_layout = QFormLayout(status_group)
#
#         self.label_state = QLabel("● 停止")
#         self.label_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
#         status_layout.addRow("仿真状态:", self.label_state)
#
#         self.label_time = QLabel("0.00 s")
#         status_layout.addRow("仿真时间:", self.label_time)
#
#         line1 = QFrame()
#         line1.setFrameShape(QFrame.HLine)
#         status_layout.addRow(line1)
#
#         self.label_pos = QLabel("[0.00, 0.00, 0.00]")
#         status_layout.addRow("位置 (m):", self.label_pos)
#
#         self.label_vel = QLabel("[0.00, 0.00, 0.00]")
#         status_layout.addRow("速度 (m/s):", self.label_vel)
#
#         self.label_altitude = QLabel("0.00 m")
#         self.label_altitude.setStyleSheet("font-weight: bold; color: #3498db; font-size: 14px;")
#         status_layout.addRow("高度:", self.label_altitude)
#
#         self.label_speed = QLabel("0.00 m/s")
#         status_layout.addRow("速率:", self.label_speed)
#
#         line2 = QFrame()
#         line2.setFrameShape(QFrame.HLine)
#         status_layout.addRow(line2)
#
#         self.label_att = QLabel("[0.0°, 0.0°, 0.0°]")
#         status_layout.addRow("姿态 (R/P/Y):", self.label_att)
#
#         self.label_motors = QLabel("[0, 0, 0, 0]")
#         status_layout.addRow("电机 (RPM):", self.label_motors)
#
#         self.label_fps = QLabel("0 Hz")
#         status_layout.addRow("仿真频率:", self.label_fps)
#
#         left_layout.addWidget(status_group)
#
#         # 距离进度
#         dist_group = QGroupBox("到目标距离")
#         dist_layout = QVBoxLayout(dist_group)
#
#         self.label_distance = QLabel("距离: 0.00 m")
#         dist_layout.addWidget(self.label_distance)
#
#         self.progress_distance = QProgressBar()
#         self.progress_distance.setRange(0, 100)
#         self.progress_distance.setValue(0)
#         dist_layout.addWidget(self.progress_distance)
#
#         left_layout.addWidget(dist_group)
#         left_layout.addStretch()
#
#         # ========== 中间视图区域 ==========
#         center_widget = QWidget()
#         center_layout = QVBoxLayout(center_widget)
#
#         # 3D视图占位
#         self.view_3d = QLabel("3D视图\n(需要 PyOpenGL)")
#         self.view_3d.setAlignment(Qt.AlignCenter)
#         self.view_3d.setMinimumSize(600, 500)
#         self.view_3d.setStyleSheet("""
#             background-color: #1a1a2e;
#             color: #888;
#             font-size: 18px;
#             border: 1px solid #333;
#             border-radius: 8px;
#         """)
#         center_layout.addWidget(self.view_3d, stretch=3)
#
#         # 日志区域
#         log_group = QGroupBox("系统日志")
#         log_layout = QVBoxLayout(log_group)
#
#         self.log_text = QTextEdit()
#         self.log_text.setReadOnly(True)
#         self.log_text.setMaximumHeight(200)
#         self.log_text.setStyleSheet("""
#             QTextEdit {
#                 background-color: #1a1a2e;
#                 color: #eaeaea;
#                 font-family: 'Consolas', 'Courier New', monospace;
#                 font-size: 11px;
#                 border: 1px solid #333;
#             }
#         """)
#         log_layout.addWidget(self.log_text)
#
#         center_layout.addWidget(log_group, stretch=1)
#
#         # ========== 分割器 ==========
#         splitter = QSplitter(Qt.Horizontal)
#         splitter.addWidget(left_panel)
#         splitter.addWidget(center_widget)
#         splitter.setSizes([350, 1050])
#
#         main_layout.addWidget(splitter)
#
#         # 初始日志
#         self._log("系统初始化完成", "SUCCESS")
#         if self.engine:
#             self._log("仿真引擎就绪", "INFO")
#         else:
#             self._log("警告: 仿真引擎未初始化", "WARNING")
#
#     def _setup_menu(self):
#         """设置菜单"""
#         menubar = self.menuBar()
#
#         # 文件菜单
#         file_menu = menubar.addMenu("文件(&F)")
#         file_menu.addAction("保存配置", self._save_config)
#         file_menu.addSeparator()
#         file_menu.addAction("退出", self.close)
#
#         # 仿真菜单
#         sim_menu = menubar.addMenu("仿真(&S)")
#         sim_menu.addAction("开始", self._on_start)
#         sim_menu.addAction("暂停", self._on_pause)
#         sim_menu.addAction("停止", self._on_stop)
#         sim_menu.addAction("重置", self._on_reset)
#
#         # 帮助菜单
#         help_menu = menubar.addMenu("帮助(&H)")
#         help_menu.addAction("使用说明", self._show_help)
#         help_menu.addAction("关于", self._show_about)
#
#     def _setup_toolbar(self):
#         """设置工具栏"""
#         toolbar = QToolBar("主工具栏")
#         toolbar.setMovable(False)
#         self.addToolBar(toolbar)
#
#         toolbar.addAction("▶ 开始", self._on_start)
#         toolbar.addAction("⏸ 暂停", self._on_pause)
#         toolbar.addAction("⏹ 停止", self._on_stop)
#         toolbar.addAction("↺ 重置", self._on_reset)
#
#     def _setup_statusbar(self):
#         """设置状态栏"""
#         self.statusbar = QStatusBar()
#         self.setStatusBar(self.statusbar)
#
#         self.status_label = QLabel("就绪")
#         self.statusbar.addWidget(self.status_label)
#
#         self.fps_label = QLabel("FPS: 0")
#         self.statusbar.addPermanentWidget(self.fps_label)
#
#     def _log(self, message: str, level: str = "INFO"):
#         """添加日志"""
#         from datetime import datetime
#         timestamp = datetime.now().strftime("%H:%M:%S")
#
#         colors = {
#             "INFO": "#3498db",
#             "WARNING": "#f39c12",
#             "ERROR": "#e74c3c",
#             "SUCCESS": "#2ecc71",
#         }
#         color = colors.get(level, "#eaeaea")
#
#         html = f'<span style="color:#888;">[{timestamp}]</span> ' \
#                f'<span style="color:{color};">[{level}]</span> {message}'
#         self.log_text.append(html)
#
#         # 滚动到底部
#         scrollbar = self.log_text.verticalScrollBar()
#         scrollbar.setValue(scrollbar.maximum())
#
#     def _on_start(self):
#         """开始仿真"""
#         if self.engine is None:
#             self._log("仿真引擎未初始化", "ERROR")
#             return
#
#         try:
#             if hasattr(self.engine, 'is_paused') and self.engine.is_paused:
#                 self.engine.resume()
#                 self._log("仿真恢复", "SUCCESS")
#             else:
#                 self._apply_target()
#                 self.engine.start()
#                 self._log("仿真开始", "SUCCESS")
#
#             self.btn_start.setEnabled(False)
#             self.btn_pause.setEnabled(True)
#             self.btn_stop.setEnabled(True)
#             self.label_state.setText("● 运行中")
#             self.label_state.setStyleSheet("font-weight: bold; color: #2ecc71;")
#             self.status_label.setText("状态: 运行中")
#         except Exception as e:
#             self._log(f"启动失败: {e}", "ERROR")
#
#     def _on_pause(self):
#         """暂停仿真"""
#         if self.engine is None:
#             return
#
#         try:
#             self.engine.pause()
#             self.btn_start.setEnabled(True)
#             self.btn_pause.setEnabled(False)
#             self.label_state.setText("● 暂停")
#             self.label_state.setStyleSheet("font-weight: bold; color: #f39c12;")
#             self.status_label.setText("状态: 暂停")
#             self._log("仿真暂停", "WARNING")
#         except Exception as e:
#             self._log(f"暂停失败: {e}", "ERROR")
#
#     def _on_stop(self):
#         """停止仿真"""
#         if self.engine is None:
#             return
#
#         try:
#             self.engine.stop()
#             self.btn_start.setEnabled(True)
#             self.btn_pause.setEnabled(False)
#             self.btn_stop.setEnabled(False)
#             self.label_state.setText("● 停止")
#             self.label_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
#             self.status_label.setText("状态: 停止")
#             self._log("仿真停止", "INFO")
#         except Exception as e:
#             self._log(f"停止失败: {e}", "ERROR")
#
#     def _on_reset(self):
#         """重置仿真"""
#         if self.engine is None:
#             return
#
#         try:
#             self.engine.reset()
#             self.btn_start.setEnabled(True)
#             self.btn_pause.setEnabled(False)
#             self.btn_stop.setEnabled(False)
#             self.label_state.setText("● 停止")
#             self.label_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
#             self.status_label.setText("状态: 就绪")
#             self._log("仿真已重置", "INFO")
#         except Exception as e:
#             self._log(f"重置失败: {e}", "ERROR")
#
#     def _apply_target(self):
#         """应用目标位置"""
#         if self.engine is None:
#             return
#
#         target = np.array([
#             self.spin_target_x.value(),
#             self.spin_target_y.value(),
#             self.spin_target_z.value()
#         ])
#
#         try:
#             self.engine.set_target_position(target)
#             self._log(f"目标位置: [{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}]", "INFO")
#
#             # 计算初始距离
#             state = self.engine.drone_state
#             self._initial_distance = np.linalg.norm(target - state.position)
#         except Exception as e:
#             self._log(f"设置目标失败: {e}", "ERROR")
#
#     def _update_display(self):
#         """更新显示"""
#         if self.engine is None:
#             return
#
#         try:
#             state = self.engine.drone_state
#             stats = self.engine.statistics
#
#             # 更新状态标签
#             self.label_time.setText(f"{stats.simulation_time:.2f} s")
#             self.label_pos.setText(
#                 f"[{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]"
#             )
#             self.label_vel.setText(
#                 f"[{state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}]"
#             )
#             self.label_altitude.setText(f"{state.altitude:.2f} m")
#             self.label_speed.setText(f"{state.speed:.2f} m/s")
#
#             euler_deg = np.degrees(state.euler_angles)
#             self.label_att.setText(
#                 f"[{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°]"
#             )
#
#             motors = state.motor_speeds
#             self.label_motors.setText(
#                 f"[{motors[0]:.0f}, {motors[1]:.0f}, {motors[2]:.0f}, {motors[3]:.0f}]"
#             )
#
#             self.label_fps.setText(f"{stats.average_fps:.0f} Hz")
#             self.fps_label.setText(f"FPS: {stats.average_fps:.0f}")
#
#             # 更新距离和进度
#             target = np.array([
#                 self.spin_target_x.value(),
#                 self.spin_target_y.value(),
#                 self.spin_target_z.value()
#             ])
#             distance = np.linalg.norm(target - state.position)
#             self.label_distance.setText(f"距离: {distance:.2f} m")
#
#             if self._initial_distance > 0:
#                 progress = max(0, min(100, int((1 - distance / self._initial_distance) * 100)))
#                 self.progress_distance.setValue(progress)
#
#                 if distance < 1.0:
#                     self.progress_distance.setFormat("已到达!")
#                 else:
#                     self.progress_distance.setFormat(f"{progress}%")
#         except Exception:
#             pass
#
#     def _save_config(self):
#         """保存配置"""
#         from utils.config.config_manager import get_config_manager
#         if get_config_manager().save_config():
#             self._log("配置已保存", "SUCCESS")
#         else:
#             self._log("配置保存失败", "ERROR")
#
#     def _show_help(self):
#         """显示帮助"""
#         help_text = """
#         <h2>🚁 低空交通无人机教学演示系统</h2>
#         <hr>
#         <h3>使用说明:</h3>
#         <ol>
#             <li>设置目标位置 (X, Y, Z)</li>
#             <li>点击"应用目标位置"</li>
#             <li>点击"开始"启动仿真</li>
#             <li>观察无人机飞行状态</li>
#         </ol>
#
#         <h3>坐标系 (NED):</h3>
#         <ul>
#             <li><b>X</b>: 北向为正</li>
#             <li><b>Y</b>: 东向为正</li>
#             <li><b>Z</b>: 向下为正 (高度取负值)</li>
#         </ul>
#
#         <h3>快捷键:</h3>
#         <ul>
#             <li>F5 - 开始仿真</li>
#             <li>F6 - 暂停仿真</li>
#             <li>F7 - 停止仿真</li>
#             <li>F8 - 重置仿真</li>
#         </ul>
#         """
#         QMessageBox.information(self, "使用说明", help_text)
#
#     def _show_about(self):
#         """显示关于"""
#         QMessageBox.about(self, "关于",
#                           "低空交通无人机教学演示系统\n\n"
#                           "版本: 1.0.0\n"
#                           "用于无人机飞行原理教学演示")
#
#     def closeEvent(self, event):
#         """关闭事件"""
#         if self.engine is not None:
#             try:
#                 self.engine.stop()
#             except Exception:
#                 pass
#         self._log("程序退出", "INFO")
#         event.accept()


# ui/main_window/main_window.py（修复版，添加3D视图）

"""
主窗口模块
"""

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QMessageBox, QGroupBox,
    QFormLayout, QDoubleSpinBox, QPushButton, QTextEdit,
    QFrame, QProgressBar, QToolBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from loguru import logger


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("低空交通无人机教学演示系统")
        self.setGeometry(100, 100, 1400, 900)

        # 初始化仿真引擎
        self._init_engine()

        # 设置UI
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        # 更新定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_display)
        self.timer.start(33)  # ~30 FPS

        self._initial_distance = 0.0

        logger.info("主窗口初始化完成")

    def _init_engine(self):
        """初始化仿真引擎"""
        try:
            from simulation.engine.simulation_core import SimulationEngine
            self.engine = SimulationEngine()
            logger.info("仿真引擎初始化成功")
        except Exception as e:
            logger.error(f"仿真引擎初始化失败: {e}")
            self.engine = None

    def _setup_ui(self):
        """设置UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ========== 左侧控制面板 ==========
        left_panel = QWidget()
        left_panel.setMaximumWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        # 标题
        title = QLabel("🚁 无人机仿真控制")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # 控制按钮组
        ctrl_group = QGroupBox("仿真控制")
        ctrl_layout = QVBoxLayout(ctrl_group)

        btn_row1 = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_start.setStyleSheet("background-color: #27ae60; font-weight: bold;")
        btn_row1.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ 暂停")
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet("background-color: #f39c12;")
        btn_row1.addWidget(self.btn_pause)
        ctrl_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #e74c3c;")
        btn_row2.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("↺ 重置")
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_reset.setStyleSheet("background-color: #3498db;")
        btn_row2.addWidget(self.btn_reset)
        ctrl_layout.addLayout(btn_row2)

        left_layout.addWidget(ctrl_group)

        # 目标位置组
        target_group = QGroupBox("目标位置设置")
        target_layout = QFormLayout(target_group)

        self.spin_target_x = QDoubleSpinBox()
        self.spin_target_x.setRange(-100, 100)
        self.spin_target_x.setValue(10)
        self.spin_target_x.setSingleStep(1)
        self.spin_target_x.setDecimals(1)
        target_layout.addRow("X (北) [m]:", self.spin_target_x)

        self.spin_target_y = QDoubleSpinBox()
        self.spin_target_y.setRange(-100, 100)
        self.spin_target_y.setValue(5)
        self.spin_target_y.setSingleStep(1)
        self.spin_target_y.setDecimals(1)
        target_layout.addRow("Y (东) [m]:", self.spin_target_y)

        self.spin_target_z = QDoubleSpinBox()
        self.spin_target_z.setRange(-100, 0)
        self.spin_target_z.setValue(-15)
        self.spin_target_z.setSingleStep(1)
        self.spin_target_z.setDecimals(1)
        target_layout.addRow("Z (高度) [m]:", self.spin_target_z)

        btn_apply = QPushButton("📍 应用目标位置")
        btn_apply.clicked.connect(self._apply_target)
        target_layout.addRow(btn_apply)

        left_layout.addWidget(target_group)

        # 状态显示组
        status_group = QGroupBox("飞行状态")
        status_layout = QFormLayout(status_group)

        self.label_state = QLabel("● 停止")
        self.label_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
        status_layout.addRow("仿真状态:", self.label_state)

        self.label_time = QLabel("0.00 s")
        status_layout.addRow("仿真时间:", self.label_time)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        status_layout.addRow(line1)

        self.label_pos = QLabel("[0.00, 0.00, 0.00]")
        status_layout.addRow("位置 (m):", self.label_pos)

        self.label_vel = QLabel("[0.00, 0.00, 0.00]")
        status_layout.addRow("速度 (m/s):", self.label_vel)

        self.label_altitude = QLabel("0.00 m")
        self.label_altitude.setStyleSheet("font-weight: bold; color: #3498db; font-size: 14px;")
        status_layout.addRow("高度:", self.label_altitude)

        self.label_speed = QLabel("0.00 m/s")
        status_layout.addRow("速率:", self.label_speed)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        status_layout.addRow(line2)

        self.label_att = QLabel("[0.0°, 0.0°, 0.0°]")
        status_layout.addRow("姿态 (R/P/Y):", self.label_att)

        self.label_motors = QLabel("[0, 0, 0, 0]")
        status_layout.addRow("电机 (RPM):", self.label_motors)

        self.label_fps = QLabel("0 Hz")
        status_layout.addRow("仿真频率:", self.label_fps)

        left_layout.addWidget(status_group)

        # 距离进度
        dist_group = QGroupBox("到目标距离")
        dist_layout = QVBoxLayout(dist_group)

        self.label_distance = QLabel("距离: 0.00 m")
        dist_layout.addWidget(self.label_distance)

        self.progress_distance = QProgressBar()
        self.progress_distance.setRange(0, 100)
        self.progress_distance.setValue(0)
        dist_layout.addWidget(self.progress_distance)

        left_layout.addWidget(dist_group)
        left_layout.addStretch()

        # ========== 中间视图区域 ==========
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setSpacing(5)

        # 3D视图
        try:
            from ui.widgets.simple_3d_view import Simple3DView
            self.view_3d = Simple3DView()
            center_layout.addWidget(self.view_3d, stretch=3)
            logger.info("3D视图加载成功")
        except Exception as e:
            logger.warning(f"3D视图加载失败: {e}")
            self.view_3d = QLabel("3D视图加载失败\n请安装 PyOpenGL:\npip install PyOpenGL PyOpenGL_accelerate")
            self.view_3d.setAlignment(Qt.AlignCenter)
            self.view_3d.setMinimumSize(600, 400)
            self.view_3d.setStyleSheet("""
                        background-color: #1a1a2e;
                        color: #888;
                        font-size: 16px;
                        border: 1px solid #333;
                        border-radius: 8px;
                    """)
            center_layout.addWidget(self.view_3d, stretch=3)

        # 日志区域
        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setStyleSheet("""
                    QTextEdit {
                        background-color: #1a1a2e;
                        color: #eaeaea;
                        font-family: 'Consolas', 'Courier New', monospace;
                        font-size: 11px;
                        border: 1px solid #333;
                    }
                """)
        log_layout.addWidget(self.log_text)

        center_layout.addWidget(log_group, stretch=1)

        # ========== 分割器 ==========
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_widget)
        splitter.setSizes([320, 1080])

        main_layout.addWidget(splitter)

        # 初始日志
        self._log("系统初始化完成", "SUCCESS")
        if self.engine:
            self._log("仿真引擎就绪", "INFO")
        else:
            self._log("警告: 仿真引擎未初始化", "WARNING")

    def _setup_menu(self):
        """设置菜单"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        file_menu.addAction("保存配置", self._save_config)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        # 仿真菜单
        sim_menu = menubar.addMenu("仿真(&S)")
        sim_menu.addAction("开始 (F5)", self._on_start)
        sim_menu.addAction("暂停 (F6)", self._on_pause)
        sim_menu.addAction("停止 (F7)", self._on_stop)
        sim_menu.addAction("重置 (F8)", self._on_reset)

        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        view_menu.addAction("重置视角", self._reset_view)
        view_menu.addAction("清除轨迹", self._clear_trajectory)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        help_menu.addAction("使用说明", self._show_help)
        help_menu.addAction("关于", self._show_about)

    def _setup_toolbar(self):
        """设置工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction("▶ 开始", self._on_start)
        toolbar.addAction("⏸ 暂停", self._on_pause)
        toolbar.addAction("⏹ 停止", self._on_stop)
        toolbar.addAction("↺ 重置", self._on_reset)
        toolbar.addSeparator()
        toolbar.addAction("🎯 重置视角", self._reset_view)
        toolbar.addAction("🗑 清除轨迹", self._clear_trajectory)

        # 添加快捷键
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence

        QShortcut(QKeySequence("F5"), self, self._on_start)
        QShortcut(QKeySequence("F6"), self, self._on_pause)
        QShortcut(QKeySequence("F7"), self, self._on_stop)
        QShortcut(QKeySequence("F8"), self, self._on_reset)

    def _setup_statusbar(self):
        """设置状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.status_label = QLabel("就绪")
        self.statusbar.addWidget(self.status_label)

        self.fps_label = QLabel("仿真: 0 Hz | 显示: 0 FPS")
        self.statusbar.addPermanentWidget(self.fps_label)

    def _log(self, message: str, level: str = "INFO"):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")

        colors = {
            "INFO": "#3498db",
            "WARNING": "#f39c12",
            "ERROR": "#e74c3c",
            "SUCCESS": "#2ecc71",
        }
        color = colors.get(level, "#eaeaea")

        html = f'<span style="color:#888;">[{timestamp}]</span> ' \
               f'<span style="color:{color};">[{level}]</span> {message}'
        self.log_text.append(html)

        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_start(self):
        """开始仿真"""
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
            self.label_state.setText("● 运行中")
            self.label_state.setStyleSheet("font-weight: bold; color: #2ecc71;")
            self.status_label.setText("状态: 运行中")
        except Exception as e:
            self._log(f"启动失败: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    def _on_pause(self):
        """暂停仿真"""
        if self.engine is None:
            return

        try:
            self.engine.pause()
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.label_state.setText("● 暂停")
            self.label_state.setStyleSheet("font-weight: bold; color: #f39c12;")
            self.status_label.setText("状态: 暂停")
            self._log("仿真暂停", "WARNING")
        except Exception as e:
            self._log(f"暂停失败: {e}", "ERROR")

    def _on_stop(self):
        """停止仿真"""
        if self.engine is None:
            return

        try:
            self.engine.stop()
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.label_state.setText("● 停止")
            self.label_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
            self.status_label.setText("状态: 停止")
            self._log("仿真停止", "INFO")
        except Exception as e:
            self._log(f"停止失败: {e}", "ERROR")

    def _on_reset(self):
        """重置仿真"""
        if self.engine is None:
            return

        try:
            self.engine.reset()
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.label_state.setText("● 停止")
            self.label_state.setStyleSheet("font-weight: bold; color: #e74c3c;")
            self.status_label.setText("状态: 就绪")
            self._clear_trajectory()
            self._log("仿真已重置", "INFO")
        except Exception as e:
            self._log(f"重置失败: {e}", "ERROR")

    def _apply_target(self):
        """应用目标位置"""
        if self.engine is None:
            return

        target = np.array([
            self.spin_target_x.value(),
            self.spin_target_y.value(),
            self.spin_target_z.value()
        ])

        try:
            self.engine.set_target_position(target)
            self._log(f"目标位置: [{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}]", "INFO")

            # 更新3D视图目标
            if hasattr(self.view_3d, 'set_target_position'):
                self.view_3d.set_target_position(target)

            # 计算初始距离
            state = self.engine.drone_state
            self._initial_distance = np.linalg.norm(target - state.position)
        except Exception as e:
            self._log(f"设置目标失败: {e}", "ERROR")

    def _update_display(self):
        """更新显示"""
        if self.engine is None:
            return

        try:
            state = self.engine.drone_state
            stats = self.engine.statistics

            # 更新状态标签
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

            motors = state.motor_speeds
            self.label_motors.setText(
                f"[{motors[0]:.0f}, {motors[1]:.0f}, {motors[2]:.0f}, {motors[3]:.0f}]"
            )

            self.label_fps.setText(f"{stats.average_fps:.0f} Hz")
            self.fps_label.setText(f"仿真: {stats.average_fps:.0f} Hz")

            # 更新距离和进度
            target = np.array([
                self.spin_target_x.value(),
                self.spin_target_y.value(),
                self.spin_target_z.value()
            ])
            distance = np.linalg.norm(target - state.position)
            self.label_distance.setText(f"距离: {distance:.2f} m")

            if self._initial_distance > 0:
                progress = max(0, min(100, int((1 - distance / self._initial_distance) * 100)))
                self.progress_distance.setValue(progress)

                if distance < 1.0:
                    self.progress_distance.setFormat("已到达!")
                else:
                    self.progress_distance.setFormat(f"{progress}%")

            # 更新3D视图
            if hasattr(self.view_3d, 'update_drone_state'):
                self.view_3d.update_drone_state(state.position, state.euler_angles)

        except Exception:
            pass

    def _reset_view(self):
        """重置视角"""
        if hasattr(self.view_3d, 'reset_view'):
            self.view_3d.reset_view()
            self._log("视角已重置", "INFO")

    def _clear_trajectory(self):
        """清除轨迹"""
        if hasattr(self.view_3d, 'clear_trajectory'):
            self.view_3d.clear_trajectory()
            self._log("轨迹已清除", "INFO")

    def _save_config(self):
        """保存配置"""
        try:
            from utils.config.config_manager import get_config_manager
            if get_config_manager().save_config():
                self._log("配置已保存", "SUCCESS")
            else:
                self._log("配置保存失败", "ERROR")
        except Exception as e:
            self._log(f"保存配置失败: {e}", "ERROR")

    def _show_help(self):
        """显示帮助"""
        help_text = """
                <h2>🚁 低空交通无人机教学演示系统</h2>
                <hr>
                <h3>使用说明:</h3>
                <ol>
                    <li>设置目标位置 (X, Y, Z)</li>
                    <li>点击"应用目标位置"</li>
                    <li>点击"开始"启动仿真</li>
                    <li>观察无人机飞行状态</li>
                </ol>

                <h3>坐标系 (NED):</h3>
                <ul>
                    <li><b>X</b>: 北向为正</li>
                    <li><b>Y</b>: 东向为正</li>
                    <li><b>Z</b>: 向下为正 (高度取负值，如-15表示15米高)</li>
                </ul>

                <h3>快捷键:</h3>
                <ul>
                    <li>F5 - 开始仿真</li>
                    <li>F6 - 暂停仿真</li>
                    <li>F7 - 停止仿真</li>
                    <li>F8 - 重置仿真</li>
                </ul>

                <h3>3D视图操作:</h3>
                <ul>
                    <li>鼠标左键拖动 - 旋转视角</li>
                    <li>鼠标滚轮 - 缩放</li>
                    <li>鼠标中键拖动 - 平移视角</li>
                </ul>
                """
        QMessageBox.information(self, "使用说明", help_text)

    def _show_about(self):
        """显示关于"""
        QMessageBox.about(self, "关于",
                          "低空交通无人机教学演示系统\n\n"
                          "版本: 1.0.0\n"
                          "用于无人机飞行原理教学演示\n\n"
                          "功能特点:\n"
                          "• 真实物理仿真\n"
                          "• PID级联控制\n"
                          "• 3D可视化\n"
                          "• 实时状态监控")

    def closeEvent(self, event):
        """关闭事件"""
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
        self._log("程序退出", "INFO")
        event.accept()

