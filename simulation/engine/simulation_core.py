# # # simulation/engine/simulation_core.py
# # """
# # 仿真引擎核心模块
# # """
# #
# # import time
# # import threading
# # import numpy as np
# # from enum import Enum, auto
# # from dataclasses import dataclass, field
# # from typing import Optional, Callable, List, Dict, Any
# # from loguru import logger
# #
# # from core.physics.quadrotor_dynamics import QuadrotorDynamics, DroneState
# # from core.control.pid_controller import QuadrotorPIDController
# # from utils.config.config_manager import get_config
# #
# # # 如果 MotorCommand 不存在，定义一个
# # try:
# #     from core.physics.quadrotor_dynamics import MotorCommand
# # except ImportError:
# #     @dataclass
# #     class MotorCommand:
# #         """电机命令"""
# #         speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
# #
# #         def __post_init__(self):
# #             self.speeds = np.array(self.speeds)
# #
# #
# # class SimulationState(Enum):
# #     """仿真状态枚举"""
# #     STOPPED = auto()
# #     RUNNING = auto()
# #     PAUSED = auto()
# #     STEPPING = auto()
# #
# #
# # @dataclass
# # class SimulationConfig:
# #     """仿真配置"""
# #     dt: float = 0.001  # 物理仿真步长 (s)
# #     realtime_factor: float = 1.0  # 实时因子
# #     max_time: float = 3600.0  # 最大仿真时间 (s)
# #     visualization_dt: float = 0.02  # 可视化更新步长 (s)
# #     control_dt: float = 0.01  # 控制更新步长 (s)
# #     enable_gravity: bool = True
# #     enable_wind: bool = False
# #     enable_ground_effect: bool = False
# #
# #
# # @dataclass
# # class SimulationStatistics:
# #     """仿真统计信息"""
# #     total_steps: int = 0
# #     simulation_time: float = 0.0
# #     real_time: float = 0.0
# #     average_step_time: float = 0.0
# #     physics_time: float = 0.0
# #     control_time: float = 0.0
# #     render_time: float = 0.0
# #
# #     @property
# #     def realtime_ratio(self) -> float:
# #         """实时性比率"""
# #         if self.real_time > 0:
# #             return self.simulation_time / self.real_time
# #         return 0.0
# #
# #
# # class EventType(Enum):
# #     """事件类型"""
# #     SIMULATION_START = auto()
# #     SIMULATION_STOP = auto()
# #     SIMULATION_PAUSE = auto()
# #     SIMULATION_RESUME = auto()
# #     STATE_UPDATE = auto()
# #     TARGET_CHANGED = auto()
# #     COLLISION = auto()
# #     BOUNDARY_VIOLATION = auto()
# #     LOW_BATTERY = auto()
# #     CUSTOM = auto()
# #
# #
# # @dataclass
# # class SimulationEvent:
# #     """仿真事件"""
# #     event_type: EventType
# #     timestamp: float
# #     data: Dict[str, Any] = field(default_factory=dict)
# #
# #
# # class SimulationEngine:
# #     """
# #     仿真引擎核心
# #
# #     负责:
# #     - 物理仿真循环
# #     - 时间管理
# #     - 状态同步
# #     - 事件分发
# #     """
# #
# #     def __init__(self, config: Optional[SimulationConfig] = None):
# #         self.config = config or SimulationConfig()
# #
# #         # 动力学模型
# #         self.dynamics = QuadrotorDynamics()
# #
# #         # 控制器
# #         self.controller = QuadrotorPIDController()
# #
# #         # 仿真状态
# #         self._state = SimulationState.STOPPED
# #         self._lock = threading.RLock()
# #
# #         # 时间管理
# #         self._sim_time = 0.0
# #         self._last_control_time = 0.0
# #         self._last_viz_time = 0.0
# #         self._start_real_time = 0.0
# #
# #         # 目标状态
# #         self._target_position = np.array([0.0, 0.0, -10.0])  # NED
# #         self._target_yaw = 0.0
# #
# #         # 事件系统
# #         self._event_listeners: Dict[EventType, List[Callable]] = {
# #             event_type: [] for event_type in EventType
# #         }
# #         self._event_queue: Queue = Queue()
# #
# #         # 状态历史（用于回放）
# #         self._state_history: List[DroneState] = []
# #         self._max_history_length = 10000
# #         self._record_history = False
# #
# #         # 统计信息
# #         self.statistics = SimulationStatistics()
# #
# #         # 仿真线程
# #         self._sim_thread: Optional[threading.Thread] = None
# #         self._stop_flag = threading.Event()
# #
# #         # 回调函数
# #         self._state_callbacks: List[Callable[[DroneState], None]] = []
# #
# #         logger.info("仿真引擎初始化完成")
# #
# #     # ==================== 状态管理 ====================
# #
# #     @property
# #     def state(self) -> SimulationState:
# #         """获取仿真状态"""
# #         with self._lock:
# #             return self._state
# #
# #     @property
# #     def simulation_time(self) -> float:
# #         """获取仿真时间"""
# #         with self._lock:
# #             return self._sim_time
# #
# #     @property
# #     def drone_state(self) -> DroneState:
# #         """获取无人机状态"""
# #         with self._lock:
# #             return self.dynamics.state.copy()
# #
# #     @property
# #     def is_running(self) -> bool:
# #         return self._state == SimulationState.RUNNING
# #
# #     @property
# #     def is_paused(self) -> bool:
# #         return self._state == SimulationState.PAUSED
# #
# #     # ==================== 仿真控制 ====================
# #
# #     def start(self):
# #         """启动仿真"""
# #         with self._lock:
# #             if self._state == SimulationState.RUNNING:
# #                 logger.warning("仿真已在运行中")
# #                 return
# #
# #             self._state = SimulationState.RUNNING
# #             self._stop_flag.clear()
# #             self._start_real_time = time.perf_counter()
# #
# #             # 启动仿真线程
# #             self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
# #             self._sim_thread.start()
# #
# #             self._emit_event(SimulationEvent(
# #                 EventType.SIMULATION_START,
# #                 self._sim_time
# #             ))
# #
# #             logger.info("仿真已启动")
# #
# #     def stop(self):
# #         """停止仿真"""
# #         with self._lock:
# #             if self._state == SimulationState.STOPPED:
# #                 return
# #
# #             self._state = SimulationState.STOPPED
# #             self._stop_flag.set()
# #
# #         # 等待线程结束
# #         if self._sim_thread and self._sim_thread.is_alive():
# #             self._sim_thread.join(timeout=2.0)
# #
# #         self._emit_event(SimulationEvent(
# #             EventType.SIMULATION_STOP,
# #             self._sim_time
# #         ))
# #
# #         logger.info("仿真已停止")
# #
# #     def pause(self):
# #         """暂停仿真"""
# #         with self._lock:
# #             if self._state == SimulationState.RUNNING:
# #                 self._state = SimulationState.PAUSED
# #                 self._emit_event(SimulationEvent(
# #                     EventType.SIMULATION_PAUSE,
# #                     self._sim_time
# #                 ))
# #                 logger.info("仿真已暂停")
# #
# #     def resume(self):
# #         """恢复仿真"""
# #         with self._lock:
# #             if self._state == SimulationState.PAUSED:
# #                 self._state = SimulationState.RUNNING
# #                 self._emit_event(SimulationEvent(
# #                     EventType.SIMULATION_RESUME,
# #                     self._sim_time
# #                 ))
# #                 logger.info("仿真已恢复")
# #
# #     def reset(self, initial_state: Optional[DroneState] = None):
# #         """重置仿真"""
# #         was_running = self._state == SimulationState.RUNNING
# #
# #         if was_running:
# #             self.stop()
# #
# #         with self._lock:
# #             self._sim_time = 0.0
# #             self._last_control_time = 0.0
# #             self._last_viz_time = 0.0
# #
# #             self.dynamics.reset(initial_state)
# #             self.controller.reset()
# #
# #             self._state_history.clear()
# #             self.statistics = SimulationStatistics()
# #
# #         logger.info("仿真已重置")
# #
# #         if was_running:
# #             self.start()
# #
# #     def step(self, n_steps: int = 1):
# #         """单步执行"""
# #         with self._lock:
# #             if self._state not in [SimulationState.PAUSED, SimulationState.STOPPED]:
# #                 return
# #
# #             for _ in range(n_steps):
# #                 self._simulation_step()
# #
# #     # ==================== 目标设置 ====================
# #
# #     def set_target_position(self, position: np.ndarray):
# #         """设置目标位置"""
# #         with self._lock:
# #             self._target_position = np.array(position)
# #             self._emit_event(SimulationEvent(
# #                 EventType.TARGET_CHANGED,
# #                 self._sim_time,
# #                 {'position': position.tolist()}
# #             ))
# #
# #     def set_target_yaw(self, yaw: float):
# #         """设置目标偏航角"""
# #         with self._lock:
# #             self._target_yaw = yaw
# #
# #     def set_target(self, position: np.ndarray, yaw: float):
# #         """设置目标位置和偏航角"""
# #         with self._lock:
# #             self._target_position = np.array(position)
# #             self._target_yaw = yaw
# #
# #     # ==================== 仿真循环 ====================
# #
# #     def _simulation_loop(self):
# #         """仿真主循环"""
# #         logger.debug("仿真循环开始")
# #
# #         last_time = time.perf_counter()
# #         accumulated_time = 0.0
# #
# #         while not self._stop_flag.is_set():
# #             current_time = time.perf_counter()
# #
# #             with self._lock:
# #                 if self._state != SimulationState.RUNNING:
# #                     time.sleep(0.01)
# #                     last_time = current_time
# #                     continue
# #
# #                 # 计算需要仿真的时间
# #                 delta_real_time = current_time - last_time
# #                 delta_sim_time = delta_real_time * self.config.realtime_factor
# #                 accumulated_time += delta_sim_time
# #
# #                 # 执行仿真步
# #                 step_start = time.perf_counter()
# #                 steps_executed = 0
# #
# #                 while accumulated_time >= self.config.dt:
# #                     self._simulation_step()
# #                     accumulated_time -= self.config.dt
# #                     steps_executed += 1
# #
# #                     # 防止累积过多
# #                     if steps_executed > 1000:
# #                         accumulated_time = 0.0
# #                         logger.warning("仿真落后过多，跳过部分步骤")
# #                         break
# #
# #                 # 更新统计
# #                 step_time = time.perf_counter() - step_start
# #                 if steps_executed > 0:
# #                     self.statistics.average_step_time = step_time / steps_executed
# #                 self.statistics.real_time = current_time - self._start_real_time
# #
# #             last_time = current_time
# #
# #             # 控制循环速率
# #             sleep_time = self.config.dt / self.config.realtime_factor - (time.perf_counter() - current_time)
# #             if sleep_time > 0:
# #                 time.sleep(sleep_time * 0.5)
# #
# #         logger.debug("仿真循环结束")
# #
# #     def _simulation_step(self):
# #         """执行单个仿真步"""
# #         dt = self.config.dt
# #
# #         # 控制更新
# #         if self._sim_time - self._last_control_time >= self.config.control_dt:
# #             self._update_control()
# #             self._last_control_time = self._sim_time
# #
# #         # 物理仿真
# #         physics_start = time.perf_counter()
# #         self.dynamics.step(dt)
# #         self.statistics.physics_time += time.perf_counter() - physics_start
# #
# #         # 更新时间
# #         self._sim_time += dt
# #         self.statistics.total_steps += 1
# #         self.statistics.simulation_time = self._sim_time
# #
# #         # 记录历史
# #         if self._record_history:
# #             self._record_state()
# #
# #         # 发送状态更新
# #         if self._sim_time - self._last_viz_time >= self.config.visualization_dt:
# #             self._notify_state_callbacks()
# #             self._last_viz_time = self._sim_time
# #
# #         # 边界检查
# #         self._check_boundaries()
# #
# #     def _update_control(self):
# #         """更新控制"""
# #         control_start = time.perf_counter()
# #
# #         state = self.dynamics.state
# #         config = get_config()
# #
# #         # 计算控制输入
# #         thrust, torques = self.controller.compute_control(
# #             target_position=self._target_position,
# #             target_yaw=self._target_yaw,
# #             current_position=state.position,
# #             current_velocity=state.velocity,
# #             current_attitude=state.euler_angles,
# #             current_angular_velocity=state.angular_velocity,
# #             mass=config.drone.mass,
# #             gravity=config.simulation.gravity,
# #             dt=self.config.control_dt
# #         )
# #
# #         # 转换为电机指令
# #         motor_cmd = MotorCommand.from_thrust_torques(thrust, torques, config.drone)
# #         self.dynamics.set_motor_speeds(motor_cmd.speeds)
# #
# #         self.statistics.control_time += time.perf_counter() - control_start
# #
# #     def _check_boundaries(self):
# #         """检查边界条件"""
# #         state = self.dynamics.state
# #
# #         # 高度限制
# #         if state.altitude < 0:
# #             self._emit_event(SimulationEvent(
# #                 EventType.COLLISION,
# #                 self._sim_time,
# #                 {'type': 'ground', 'position': state.position.tolist()}
# #             ))
# #
# #         # 水平距离限制
# #         horizontal_distance = np.linalg.norm(state.position[:2])
# #         if horizontal_distance > 500:  # 500m边界
# #             self._emit_event(SimulationEvent(
# #                 EventType.BOUNDARY_VIOLATION,
# #                 self._sim_time,
# #                 {'distance': horizontal_distance}
# #             ))
# #
# #     def _record_state(self):
# #         """记录状态历史"""
# #         self._state_history.append(self.dynamics.state.copy())
# #         if len(self._state_history) > self._max_history_length:
# #             self._state_history.pop(0)
# #
# #     # ==================== 事件系统 ====================
# #
# #     def add_event_listener(self, event_type: EventType, callback: Callable):
# #         """添加事件监听器"""
# #         if callback not in self._event_listeners[event_type]:
# #             self._event_listeners[event_type].append(callback)
# #
# #     def remove_event_listener(self, event_type: EventType, callback: Callable):
# #         """移除事件监听器"""
# #         if callback in self._event_listeners[event_type]:
# #             self._event_listeners[event_type].remove(callback)
# #
# #     def _emit_event(self, event: SimulationEvent):
# #         """发送事件"""
# #         self._event_queue.put(event)
# #         for callback in self._event_listeners[event.event_type]:
# #             try:
# #                 callback(event)
# #             except Exception as e:
# #                 logger.error(f"事件回调错误: {e}")
# #
# #     # ==================== 状态回调 ====================
# #
# #     def add_state_callback(self, callback: Callable[[DroneState], None]):
# #         """添加状态更新回调"""
# #         if callback not in self._state_callbacks:
# #             self._state_callbacks.append(callback)
# #
# #     def remove_state_callback(self, callback: Callable[[DroneState], None]):
# #         """移除状态更新回调"""
# #         if callback in self._state_callbacks:
# #             self._state_callbacks.remove(callback)
# #
# #     def _notify_state_callbacks(self):
# #         """通知状态回调"""
# #         state = self.dynamics.state.copy()
# #         for callback in self._state_callbacks:
# #             try:
# #                 callback(state)
# #             except Exception as e:
# #                 logger.error(f"状态回调错误: {e}")
# #
# #     # ==================== 历史与回放 ====================
# #
# #     def start_recording(self):
# #         """开始记录历史"""
# #         self._record_history = True
# #         self._state_history.clear()
# #         logger.info("开始记录仿真历史")
# #
# #     def stop_recording(self) -> List[DroneState]:
# #         """停止记录并返回历史"""
# #         self._record_history = False
# #         history = self._state_history.copy()
# #         logger.info(f"停止记录，共 {len(history)} 个状态点")
# #         return history
# #
# #     def get_history(self) -> List[DroneState]:
# #         """获取状态历史"""
# #         return self._state_history.copy()
# #
# #     # ==================== 环境控制 ====================
# #
# #     def set_wind(self, wind_velocity: np.ndarray):
# #         """设置风速（惯性系）"""
# #         # 简化处理：风力 = 0.5 * rho * Cd * A * v^2
# #         air_density = 1.225
# #         drag_area = 0.1  # m^2
# #
# #         wind_force = 0.5 * air_density * drag_area * wind_velocity * np.abs(wind_velocity)
# #         self.dynamics.external_force = wind_force
# #         logger.debug(f"设置风速: {wind_velocity}, 风力: {wind_force}")
# #
# #     def set_external_force(self, force: np.ndarray):
# #         """设置外部力"""
# #         self.dynamics.external_force = force
# #
# #     def set_external_torque(self, torque: np.ndarray):
# #         """设置外部力矩"""
# #         self.dynamics.external_torque = torque
#
#
# # simulation/engine/simulation_core.py（修复版开头）
#
# """
# 仿真引擎核心模块
# """
#
# import time
# import threading
# from queue import Queue
# import numpy as np
# from enum import Enum, auto
# from dataclasses import dataclass, field
# from typing import Optional, Callable, List, Any
# from loguru import logger
#
# from core.physics.quadrotor_dynamics import QuadrotorDynamics, DroneState
# from core.control.pid_controller import QuadrotorPIDController
# from utils.config.config_manager import get_config  # 确保这行正确
#
#
# class SimulationState(Enum):
#     """仿真状态"""
#     STOPPED = auto()
#     RUNNING = auto()
#     PAUSED = auto()
#
#
# class EventType(Enum):
#     """事件类型"""
#     STATE_UPDATE = auto()
#     SIMULATION_START = auto()
#     SIMULATION_STOP = auto()
#     SIMULATION_PAUSE = auto()
#     SIMULATION_RESET = auto()
#     TARGET_REACHED = auto()
#     COLLISION = auto()
#     WARNING = auto()
#     ERROR = auto()
#
#
# @dataclass
# class SimulationEvent:
#     """仿真事件"""
#     event_type: EventType
#     timestamp: float
#     data: Any = None
#
#
# @dataclass
# class SimulationStatistics:
#     """仿真统计"""
#     simulation_time: float = 0.0
#     real_time: float = 0.0
#     total_frames: int = 0
#     average_fps: float = 0.0
#     min_fps: float = float('inf')
#     max_fps: float = 0.0
#
#
# class SimulationEngine:
#     """仿真引擎"""
#
#     def __init__(self):
#         """初始化仿真引擎"""
#         config = get_config()
#
#         # 仿真参数
#         self.dt = config.simulation.dt
#         self.realtime_factor = config.simulation.realtime_factor
#
#         # 组件
#         self.dynamics = QuadrotorDynamics()
#         self.controller = QuadrotorPIDController()
#
#         # 状态
#         self._state = SimulationState.STOPPED
#         self._lock = threading.Lock()
#
#         # 目标
#         self.target_position = np.array([0.0, 0.0, -10.0])
#         self.target_yaw = 0.0
#
#         # 统计
#         self._statistics = SimulationStatistics()
#
#         # 仿真线程
#         self._thread: Optional[threading.Thread] = None
#         self._stop_event = threading.Event()
#
#         # 事件队列
#         self._event_queue = Queue()
#
#         # 回调
#         self._callbacks: List[Callable[[SimulationEvent], None]] = []
#
#         logger.info("仿真引擎初始化完成")
#
#     @property
#     def state(self) -> SimulationState:
#         """获取仿真状态"""
#         return self._state
#
#     @property
#     def is_running(self) -> bool:
#         """是否正在运行"""
#         return self._state == SimulationState.RUNNING
#
#     @property
#     def is_paused(self) -> bool:
#         """是否暂停"""
#         return self._state == SimulationState.PAUSED
#
#     @property
#     def drone_state(self) -> DroneState:
#         """获取无人机状态"""
#         with self._lock:
#             return self.dynamics.state.copy()
#
#     @property
#     def statistics(self) -> SimulationStatistics:
#         """获取统计信息"""
#         with self._lock:
#             return SimulationStatistics(
#                 simulation_time=self._statistics.simulation_time,
#                 real_time=self._statistics.real_time,
#                 total_frames=self._statistics.total_frames,
#                 average_fps=self._statistics.average_fps,
#                 min_fps=self._statistics.min_fps,
#                 max_fps=self._statistics.max_fps
#             )
#
#     def set_target_position(self, position: np.ndarray):
#         """设置目标位置"""
#         self.target_position = np.array(position, dtype=float)
#         logger.info(f"目标位置设置为: {self.target_position}")
#
#     def set_target_yaw(self, yaw: float):
#         """设置目标偏航角"""
#         self.target_yaw = float(yaw)
#
#     def start(self):
#         """开始仿真"""
#         if self._state == SimulationState.RUNNING:
#             return
#
#         self._stop_event.clear()
#         self._state = SimulationState.RUNNING
#
#         # 启动仿真线程
#         self._thread = threading.Thread(target=self._simulation_loop, daemon=True)
#         self._thread.start()
#
#         self._emit_event(EventType.SIMULATION_START)
#         logger.info("仿真开始")
#
#     def stop(self):
#         """停止仿真"""
#         if self._state == SimulationState.STOPPED:
#             return
#
#         self._stop_event.set()
#         self._state = SimulationState.STOPPED
#
#         if self._thread is not None:
#             self._thread.join(timeout=1.0)
#             self._thread = None
#
#         self._emit_event(EventType.SIMULATION_STOP)
#         logger.info("仿真停止")
#
#     def pause(self):
#         """暂停仿真"""
#         if self._state == SimulationState.RUNNING:
#             self._state = SimulationState.PAUSED
#             self._emit_event(EventType.SIMULATION_PAUSE)
#             logger.info("仿真暂停")
#
#     def resume(self):
#         """恢复仿真"""
#         if self._state == SimulationState.PAUSED:
#             self._state = SimulationState.RUNNING
#             logger.info("仿真恢复")
#
#     def reset(self):
#         """重置仿真"""
#         was_running = self._state == SimulationState.RUNNING
#
#         self.stop()
#
#         with self._lock:
#             self.dynamics.reset()
#             self.controller.reset()
#             self._statistics = SimulationStatistics()
#
#         self._emit_event(EventType.SIMULATION_RESET)
#         logger.info("仿真重置")
#
#     def _simulation_loop(self):
#         """仿真主循环"""
#         last_time = time.perf_counter()
#         frame_times = []
#
#         while not self._stop_event.is_set():
#             # 检查暂停
#             if self._state == SimulationState.PAUSED:
#                 time.sleep(0.01)
#                 last_time = time.perf_counter()
#                 continue
#
#             loop_start = time.perf_counter()
#
#             with self._lock:
#                 # 计算控制输入
#                 motor_commands = self.controller.compute_control(
#                     state=self.dynamics.state,
#                     target_position=self.target_position,
#                     target_yaw=self.target_yaw,
#                     dt=self.dt
#                 )
#
#                 # 更新动力学
#                 self.dynamics.set_motor_speeds(motor_commands)
#                 self.dynamics.step(self.dt)
#
#                 # 更新统计
#                 self._statistics.simulation_time += self.dt
#                 self._statistics.total_frames += 1
#
#             # 实时同步
#             elapsed = time.perf_counter() - loop_start
#             target_dt = self.dt / self.realtime_factor
#
#             if elapsed < target_dt:
#                 time.sleep(target_dt - elapsed)
#
#             # 计算帧率
#             current_time = time.perf_counter()
#             frame_time = current_time - last_time
#             last_time = current_time
#
#             if frame_time > 0:
#                 fps = 1.0 / frame_time
#                 frame_times.append(fps)
#                 if len(frame_times) > 100:
#                     frame_times.pop(0)
#
#                 with self._lock:
#                     self._statistics.average_fps = np.mean(frame_times)
#                     self._statistics.min_fps = min(self._statistics.min_fps, fps)
#                     self._statistics.max_fps = max(self._statistics.max_fps, fps)
#                     self._statistics.real_time += frame_time
#
#             # 发送状态更新事件
#             self._emit_event(EventType.STATE_UPDATE, self.dynamics.state.copy())
#
#     def _emit_event(self, event_type: EventType, data: Any = None):
#         """发送事件"""
#         event = SimulationEvent(
#             event_type=event_type,
#             timestamp=self._statistics.simulation_time,
#             data=data
#         )
#
#         self._event_queue.put(event)
#
#         for callback in self._callbacks:
#             try:
#                 callback(event)
#             except Exception as e:
#                 logger.error(f"事件回调错误: {e}")
#
#     def add_callback(self, callback: Callable[[SimulationEvent], None]):
#         """添加事件回调"""
#         self._callbacks.append(callback)
#
#     def remove_callback(self, callback: Callable[[SimulationEvent], None]):
#         """移除事件回调"""
#         if callback in self._callbacks:
#             self._callbacks.remove(callback)


# simulation/engine/simulation_core.py 开头导入部分和控制器初始化

"""
仿真引擎核心模块
"""

import time
import threading
from queue import Queue
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from loguru import logger

from core.physics.quadrotor_dynamics import QuadrotorDynamics, DroneState
from core.control.pid_controller import SimpleQuadrotorController  # 使用简化控制器
from utils.config.config_manager import get_config


class SimulationState(Enum):
    """仿真状态枚举"""
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


@dataclass
class SimulationStatistics:
    """仿真统计信息"""
    simulation_time: float = 0.0
    real_time: float = 0.0
    step_count: int = 0
    average_fps: float = 0.0
    real_time_factor: float = 1.0


class SimulationEngine:
    """仿真引擎"""

    def __init__(self):
        """初始化仿真引擎"""
        self._config = get_config()
        self._dt = self._config.simulation.dt

        # 动力学模型
        self._dynamics = QuadrotorDynamics()

        # 控制器
        self._controller = SimpleQuadrotorController()  # 使用简化控制器

        # 目标位置
        self._target_position = np.array([0.0, 0.0, -10.0])  # 默认目标：10米高
        self._target_yaw = 0.0

        # 仿真状态
        self._state = SimulationState.STOPPED
        self._statistics = SimulationStatistics()

        # 线程控制
        self._sim_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # FPS计算
        self._fps_counter = 0
        self._fps_timer = time.time()

        logger.info("仿真引擎初始化完成")

    @property
    def state(self) -> SimulationState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == SimulationState.RUNNING

    @property
    def is_paused(self) -> bool:
        return self._state == SimulationState.PAUSED

    @property
    def drone_state(self) -> DroneState:
        return self._dynamics.state

    @property
    def statistics(self) -> SimulationStatistics:
        return self._statistics

    def set_target_position(self, position: np.ndarray, yaw: float = None):
        """设置目标位置"""
        self._target_position = np.array(position, dtype=float)
        if yaw is not None:
            self._target_yaw = float(yaw)
        logger.info(f"目标位置设置为: {position}")

    def start(self):
        """开始仿真"""
        if self._state == SimulationState.RUNNING:
            return

        self._stop_event.clear()
        self._pause_event.clear()

        self._state = SimulationState.RUNNING
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()

        logger.info("仿真开始")

    def pause(self):
        """暂停仿真"""
        if self._state == SimulationState.RUNNING:
            self._pause_event.set()
            self._state = SimulationState.PAUSED
            logger.info("仿真暂停")

    def resume(self):
        """恢复仿真"""
        if self._state == SimulationState.PAUSED:
            self._pause_event.clear()
            self._state = SimulationState.RUNNING
            logger.info("仿真恢复")

    def stop(self):
        """停止仿真"""
        self._stop_event.set()
        self._pause_event.clear()

        if self._sim_thread is not None:
            self._sim_thread.join(timeout=1.0)

        self._state = SimulationState.STOPPED
        logger.info("仿真停止")

    def reset(self):
        """重置仿真"""
        was_running = self._state == SimulationState.RUNNING

        if was_running:
            self.stop()

        # 重置动力学
        self._dynamics.reset()

        # 重置控制器
        self._controller.reset()

        # 重置统计
        self._statistics = SimulationStatistics()

        self._state = SimulationState.STOPPED
        logger.info("仿真已重置")

    def _simulation_loop(self):
        """仿真主循环"""
        last_time = time.time()

        while not self._stop_event.is_set():
            # 检查暂停
            if self._pause_event.is_set():
                time.sleep(0.01)
                last_time = time.time()
                continue

            current_time = time.time()

            # 执行仿真步
            self._step()

            # 更新统计
            self._statistics.simulation_time += self._dt
            self._statistics.step_count += 1

            # 计算FPS
            self._fps_counter += 1
            if current_time - self._fps_timer >= 1.0:
                self._statistics.average_fps = self._fps_counter / (current_time - self._fps_timer)
                self._fps_counter = 0
                self._fps_timer = current_time

            # 实时同步
            elapsed = time.time() - last_time
            sleep_time = self._dt / self._config.simulation.realtime_factor - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            last_time = time.time()

    def _step(self):
        """执行一步仿真"""
        # 计算控制
        motor_speeds = self._controller.compute_control(
            self._dynamics.state,
            self._target_position,
            self._target_yaw,
            self._dt
        )

        # 应用控制
        self._dynamics.set_motor_speeds(motor_speeds)

        # 物理步进
        self._dynamics.step(self._dt)
