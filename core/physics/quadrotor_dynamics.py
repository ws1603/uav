# # # # # core/physics/quadrotor_dynamics.py
# # # #
# # # # import numpy as np
# # # # from dataclasses import dataclass, field
# # # # from typing import Tuple, Optional
# # # # from enum import Enum
# # # # from loguru import logger
# # # #
# # # # from utils.config.config_manager import get_config, DroneParams
# # # # from utils.math.quaternion import Quaternion
# # # # from utils.math.coordinate_transform import CoordinateTransform
# # # #
# # # #
# # # # class MotorPosition(Enum):
# # # #     """电机位置枚举（X型配置）"""
# # # #     FRONT_RIGHT = 0  # 前右 - 逆时针
# # # #     REAR_LEFT = 1  # 后左 - 逆时针
# # # #     FRONT_LEFT = 2  # 前左 - 顺时针
# # # #     REAR_RIGHT = 3  # 后右 - 顺时针
# # # #
# # # #
# # # # @dataclass
# # # # class DroneState:
# # # #     """无人机状态"""
# # # #     # 位置 (NED坐标系, m)
# # # #     position: np.ndarray = field(default_factory=lambda: np.zeros(3))
# # # #
# # # #     # 速度 (NED坐标系, m/s)
# # # #     velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
# # # #
# # # #     # 姿态四元数 [w, x, y, z]
# # # #     quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
# # # #
# # # #     # 角速度 (机体坐标系, rad/s)
# # # #     angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
# # # #
# # # #     # 电机转速 (RPM)
# # # #     motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
# # # #
# # # #     # 时间戳
# # # #     timestamp: float = 0.0
# # # #
# # # #     @property
# # # #     def euler_angles(self) -> np.ndarray:
# # # #         """获取欧拉角 [roll, pitch, yaw] (rad)"""
# # # #         return Quaternion.to_euler(self.quaternion)
# # # #
# # # #     @property
# # # #     def roll(self) -> float:
# # # #         return self.euler_angles[0]
# # # #
# # # #     @property
# # # #     def pitch(self) -> float:
# # # #         return self.euler_angles[1]
# # # #
# # # #     @property
# # # #     def yaw(self) -> float:
# # # #         return self.euler_angles[2]
# # # #
# # # #     @property
# # # #     def altitude(self) -> float:
# # # #         """获取高度（向上为正）"""
# # # #         return -self.position[2]
# # # #
# # # #     @property
# # # #     def speed(self) -> float:
# # # #         """获取速度大小"""
# # # #         return np.linalg.norm(self.velocity)
# # # #
# # # #     def copy(self) -> 'DroneState':
# # # #         """创建状态副本"""
# # # #         return DroneState(
# # # #             position=self.position.copy(),
# # # #             velocity=self.velocity.copy(),
# # # #             quaternion=self.quaternion.copy(),
# # # #             angular_velocity=self.angular_velocity.copy(),
# # # #             motor_speeds=self.motor_speeds.copy(),
# # # #             timestamp=self.timestamp
# # # #         )
# # # #
# # # #     def to_dict(self) -> dict:
# # # #         """转换为字典"""
# # # #         return {
# # # #             'position': self.position.tolist(),
# # # #             'velocity': self.velocity.tolist(),
# # # #             'quaternion': self.quaternion.tolist(),
# # # #             'euler_angles': self.euler_angles.tolist(),
# # # #             'angular_velocity': self.angular_velocity.tolist(),
# # # #             'motor_speeds': self.motor_speeds.tolist(),
# # # #             'altitude': self.altitude,
# # # #             'speed': self.speed,
# # # #             'timestamp': self.timestamp
# # # #         }
# # # #
# # # #
# # # # @dataclass
# # # # class MotorCommand:
# # # #     """电机控制指令"""
# # # #     speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))  # RPM
# # # #
# # # #     @classmethod
# # # #     def from_thrust_torques(cls, thrust: float, torques: np.ndarray,
# # # #                             params: DroneParams) -> 'MotorCommand':
# # # #         """
# # # #         从总推力和力矩计算电机转速
# # # #
# # # #         Args:
# # # #             thrust: 总推力 (N)
# # # #             torques: [tau_x, tau_y, tau_z] 力矩 (N·m)
# # # #             params: 无人机参数
# # # #         """
# # # #         kf = params.motor_constant
# # # #         km = params.moment_constant
# # # #         L = params.arm_length
# # # #
# # # #         # 混控矩阵 (X型配置)
# # # #         # [F]     [1    1    1    1  ] [F1]
# # # #         # [τx] =  [L   -L   -L    L  ] [F2]
# # # #         # [τy]    [L    L   -L   -L  ] [F3]
# # # #         # [τz]    [-km  km  -km   km ] [F4]
# # # #
# # # #         # 逆混控矩阵计算各电机推力
# # # #         L_sqrt2 = L * np.sqrt(2) / 2  # X型配置
# # # #
# # # #         A = np.array([
# # # #             [1, 1, 1, 1],
# # # #             [L_sqrt2, -L_sqrt2, -L_sqrt2, L_sqrt2],
# # # #             [L_sqrt2, L_sqrt2, -L_sqrt2, -L_sqrt2],
# # # #             [-km / kf, km / kf, -km / kf, km / kf]
# # # #         ])
# # # #
# # # #         b = np.array([thrust, torques[0], torques[1], torques[2]])
# # # #
# # # #         try:
# # # #             motor_forces = np.linalg.solve(A, b)
# # # #         except np.linalg.LinAlgError:
# # # #             motor_forces = np.ones(4) * thrust / 4
# # # #
# # # #         # 推力转换为转速
# # # #         motor_forces = np.clip(motor_forces, 0, None)  # 确保非负
# # # #         motor_speeds_squared = motor_forces / kf
# # # #         motor_speeds = np.sqrt(motor_speeds_squared)
# # # #
# # # #         # 限幅
# # # #         motor_speeds = np.clip(motor_speeds, params.min_rpm, params.max_rpm)
# # # #
# # # #         return cls(speeds=motor_speeds)
# # # #
# # # #
# # # # class QuadrotorDynamics:
# # # #     """
# # # #     四旋翼动力学模型
# # # #
# # # #     坐标系约定：
# # # #     - NED (North-East-Down) 惯性坐标系
# # # #     - 机体坐标系: x前, y右, z下
# # # #     - X型电机配置
# # # #     """
# # # #
# # # #     def __init__(self, params: Optional[DroneParams] = None):
# # # #         """
# # # #         初始化动力学模型
# # # #
# # # #         Args:
# # # #             params: 无人机参数，为None时使用默认配置
# # # #         """
# # # #         self.params = params or get_config().drone
# # # #         self._update_derived_params()
# # # #
# # # #         # 状态
# # # #         self.state = DroneState()
# # # #
# # # #         # 环境参数
# # # #         self.gravity = get_config().simulation.gravity
# # # #         self.air_density = get_config().simulation.air_density
# # # #
# # # #         # 外部力和力矩（用于风等干扰）
# # # #         self.external_force = np.zeros(3)
# # # #         self.external_torque = np.zeros(3)
# # # #
# # # #         logger.debug("四旋翼动力学模型初始化完成")
# # # #
# # # #     def _update_derived_params(self):
# # # #         """更新派生参数"""
# # # #         self.mass = self.params.mass
# # # #         self.inertia = self.params.get_inertia_matrix()
# # # #         self.inertia_inv = np.linalg.inv(self.inertia)
# # # #         self.kf = self.params.motor_constant
# # # #         self.km = self.params.moment_constant
# # # #         self.L = self.params.arm_length
# # # #
# # # #         # 混控矩阵（从电机转速到力/力矩）
# # # #         L_sqrt2 = self.L * np.sqrt(2) / 2
# # # #         self.mixer_matrix = np.array([
# # # #             [self.kf, self.kf, self.kf, self.kf],
# # # #             [L_sqrt2 * self.kf, -L_sqrt2 * self.kf, -L_sqrt2 * self.kf, L_sqrt2 * self.kf],
# # # #             [L_sqrt2 * self.kf, L_sqrt2 * self.kf, -L_sqrt2 * self.kf, -L_sqrt2 * self.kf],
# # # #             [-self.km, self.km, -self.km, self.km]
# # # #         ])
# # # #
# # # #     def reset(self, initial_state: Optional[DroneState] = None):
# # # #         """重置状态"""
# # # #         if initial_state:
# # # #             self.state = initial_state.copy()
# # # #         else:
# # # #             self.state = DroneState()
# # # #             # 默认悬停高度10m
# # # #             self.state.position = np.array([0.0, 0.0, -10.0])
# # # #
# # # #         self.external_force = np.zeros(3)
# # # #         self.external_torque = np.zeros(3)
# # # #
# # # #         logger.debug("动力学模型已重置")
# # # #
# # # #     def set_motor_speeds(self, speeds: np.ndarray):
# # # #         """设置电机转速 (RPM)"""
# # # #         self.state.motor_speeds = np.clip(
# # # #             speeds,
# # # #             self.params.min_rpm,
# # # #             self.params.max_rpm
# # # #         )
# # # #
# # # #     def compute_forces_and_torques(self) -> Tuple[np.ndarray, np.ndarray]:
# # # #         """
# # # #         计算作用在机体上的力和力矩
# # # #
# # # #         Returns:
# # # #             force: 机体坐标系下的力 [Fx, Fy, Fz]
# # # #             torque: 机体坐标系下的力矩 [τx, τy, τz]
# # # #         """
# # # #         # 电机推力和力矩
# # # #         omega_squared = (self.state.motor_speeds ** 2)
# # # #         motor_output = self.mixer_matrix @ omega_squared
# # # #
# # # #         total_thrust = motor_output[0]
# # # #         torques = motor_output[1:4]
# # # #
# # # #         # 推力在机体z轴负方向（向上）
# # # #         thrust_force = np.array([0, 0, -total_thrust])
# # # #
# # # #         # 气动阻力（机体坐标系）
# # # #         R = Quaternion.to_rotation_matrix(self.state.quaternion)
# # # #         velocity_body = R.T @ self.state.velocity
# # # #
# # # #         drag_force = -np.array([
# # # #             self.params.drag_coefficient_xy,
# # # #             self.params.drag_coefficient_xy,
# # # #             self.params.drag_coefficient_z
# # # #         ]) * velocity_body * np.abs(velocity_body)
# # # #
# # # #         # 陀螺效应力矩
# # # #         omega = self.state.angular_velocity
# # # #         gyro_torque = -np.cross(omega, self.inertia @ omega)
# # # #
# # # #         # 总力和力矩
# # # #         total_force = thrust_force + drag_force
# # # #         total_torque = torques + gyro_torque
# # # #
# # # #         return total_force, total_torque
# # # #
# # # #     def step(self, dt: float) -> DroneState:
# # # #         """
# # # #         执行一步仿真
# # # #
# # # #         Args:
# # # #             dt: 时间步长 (s)
# # # #
# # # #         Returns:
# # # #             更新后的状态
# # # #         """
# # # #         # 计算力和力矩
# # # #         force_body, torque_body = self.compute_forces_and_torques()
# # # #
# # # #         # 添加外部干扰
# # # #         R = Quaternion.to_rotation_matrix(self.state.quaternion)
# # # #         force_body += R.T @ self.external_force
# # # #         torque_body += self.external_torque
# # # #
# # # #         # 转换推力到惯性坐标系
# # # #         force_inertial = R @ force_body
# # # #
# # # #         # 添加重力
# # # #         gravity_force = np.array([0, 0, self.mass * self.gravity])
# # # #         force_inertial += gravity_force
# # # #
# # # #         # 使用RK4积分
# # # #         self._rk4_step(force_inertial, torque_body, dt)
# # # #
# # # #         # 更新时间戳
# # # #         self.state.timestamp += dt
# # # #
# # # #         # 地面碰撞检测
# # # #         self._check_ground_collision()
# # # #
# # # #         return self.state
# # # #
# # # #     def _rk4_step(self, force: np.ndarray, torque: np.ndarray, dt: float):
# # # #         """RK4积分"""
# # # #         # 保存当前状态
# # # #         pos = self.state.position.copy()
# # # #         vel = self.state.velocity.copy()
# # # #         quat = self.state.quaternion.copy()
# # # #         omega = self.state.angular_velocity.copy()
# # # #
# # # #         # k1
# # # #         k1_pos = vel
# # # #         k1_vel = force / self.mass
# # # #         k1_quat = Quaternion.derivative(quat, omega)
# # # #         k1_omega = self.inertia_inv @ (torque - np.cross(omega, self.inertia @ omega))
# # # #
# # # #         # k2
# # # #         vel_k2 = vel + 0.5 * dt * k1_vel
# # # #         omega_k2 = omega + 0.5 * dt * k1_omega
# # # #         quat_k2 = Quaternion.normalize(quat + 0.5 * dt * k1_quat)
# # # #
# # # #         k2_pos = vel_k2
# # # #         k2_vel = force / self.mass
# # # #         k2_quat = Quaternion.derivative(quat_k2, omega_k2)
# # # #         k2_omega = self.inertia_inv @ (torque - np.cross(omega_k2, self.inertia @ omega_k2))
# # # #
# # # #         # k3
# # # #         vel_k3 = vel + 0.5 * dt * k2_vel
# # # #         omega_k3 = omega + 0.5 * dt * k2_omega
# # # #         quat_k3 = Quaternion.normalize(quat + 0.5 * dt * k2_quat)
# # # #
# # # #         k3_pos = vel_k3
# # # #         k3_vel = force / self.mass
# # # #         k3_quat = Quaternion.derivative(quat_k3, omega_k3)
# # # #         k3_omega = self.inertia_inv @ (torque - np.cross(omega_k3, self.inertia @ omega_k3))
# # # #
# # # #         # k4
# # # #         vel_k4 = vel + dt * k3_vel
# # # #         omega_k4 = omega + dt * k3_omega
# # # #         quat_k4 = Quaternion.normalize(quat + dt * k3_quat)
# # # #
# # # #         k4_pos = vel_k4
# # # #         k4_vel = force / self.mass
# # # #         k4_quat = Quaternion.derivative(quat_k4, omega_k4)
# # # #         k4_omega = self.inertia_inv @ (torque - np.cross(omega_k4, self.inertia @ omega_k4))
# # # #
# # # #         # 更新状态
# # # #         self.state.position = pos + (dt / 6) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
# # # #         self.state.velocity = vel + (dt / 6) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)
# # # #         self.state.quaternion = Quaternion.normalize(
# # # #             quat + (dt / 6) * (k1_quat + 2 * k2_quat + 2 * k3_quat + k4_quat)
# # # #         )
# # # #         self.state.angular_velocity = omega + (dt / 6) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)
# # # #
# # # #     def _check_ground_collision(self):
# # # #         """检测地面碰撞"""
# # # #         if self.state.position[2] > 0:  # NED坐标系，z>0表示在地面以下
# # # #             self.state.position[2] = 0
# # # #             self.state.velocity[2] = min(0, self.state.velocity[2])  # 阻止向下运动
# # # #             # 可以添加碰撞回调
# # # #             logger.warning("检测到地面碰撞！")
# # #
# # # # core/physics/quadrotor_dynamics.py（修复版开头）
# # #
# # # """
# # # 四旋翼无人机动力学模型
# # # """
# # #
# # # import numpy as np
# # # from dataclasses import dataclass, field
# # # from typing import Optional, Tuple
# # # from loguru import logger
# # #
# # # from utils.math.quaternion import Quaternion
# # # from utils.config.config_manager import get_config
# # #
# # # # 尝试导入 DroneParams，如果失败则使用本地定义
# # # try:
# # #     from utils.config.config_manager import DroneParams
# # # except ImportError:
# # #     @dataclass
# # #     class DroneParams:
# # #         """无人机参数"""
# # #         mass: float = 1.5
# # #         arm_length: float = 0.225
# # #         max_rpm: float = 12000
# # #         motor_constant: float = 8.54858e-06
# # #         moment_constant: float = 0.016
# # #         inertia_xx: float = 0.029125
# # #         inertia_yy: float = 0.029125
# # #         inertia_zz: float = 0.055225
# # #         drag_coefficient: float = 0.1
# # #
# # #         @property
# # #         def inertia(self) -> np.ndarray:
# # #             return np.diag([self.inertia_xx, self.inertia_yy, self.inertia_zz])
# # #
# # #
# # # @dataclass
# # # class MotorCommand:
# # #     """电机命令"""
# # #     speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
# # #
# # #     def __post_init__(self):
# # #         self.speeds = np.array(self.speeds, dtype=float)
# # #
# # #
# # # @dataclass
# # # class DroneState:
# # #     """无人机状态"""
# # #     position: np.ndarray = field(default_factory=lambda: np.zeros(3))
# # #     velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
# # #     quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
# # #     angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
# # #     motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
# # #
# # #     def __post_init__(self):
# # #         self.position = np.array(self.position, dtype=float)
# # #         self.velocity = np.array(self.velocity, dtype=float)
# # #         self.quaternion = np.array(self.quaternion, dtype=float)
# # #         self.angular_velocity = np.array(self.angular_velocity, dtype=float)
# # #         self.motor_speeds = np.array(self.motor_speeds, dtype=float)
# # #
# # #     @property
# # #     def euler_angles(self) -> np.ndarray:
# # #         """获取欧拉角 [roll, pitch, yaw]"""
# # #         return Quaternion.to_euler(self.quaternion)
# # #
# # #     @property
# # #     def altitude(self) -> float:
# # #         """获取高度（NED坐标系中为-z）"""
# # #         return -self.position[2]
# # #
# # #     @property
# # #     def speed(self) -> float:
# # #         """获取速率"""
# # #         return float(np.linalg.norm(self.velocity))
# # #
# # #     def copy(self) -> 'DroneState':
# # #         """复制状态"""
# # #         return DroneState(
# # #             position=self.position.copy(),
# # #             velocity=self.velocity.copy(),
# # #             quaternion=self.quaternion.copy(),
# # #             angular_velocity=self.angular_velocity.copy(),
# # #             motor_speeds=self.motor_speeds.copy()
# # #         )
# # #
# # #
# # # class QuadrotorDynamics:
# # #     """四旋翼动力学模型"""
# # #
# # #     def __init__(self, params: Optional[DroneParams] = None):
# # #         """
# # #         初始化动力学模型
# # #
# # #         Args:
# # #             params: 无人机参数，如果为None则从配置加载
# # #         """
# # #         if params is None:
# # #             config = get_config()
# # #             self.params = config.drone
# # #         else:
# # #             self.params = params
# # #
# # #         # 状态
# # #         self.state = DroneState()
# # #
# # #         # 物理常数
# # #         self.gravity = 9.81
# # #
# # #         # 电机布局（X型配置）
# # #         L = self.params.arm_length
# # #         self.motor_positions = np.array([
# # #             [L / np.sqrt(2), L / np.sqrt(2), 0],  # 前右
# # #             [-L / np.sqrt(2), L / np.sqrt(2), 0],  # 后右
# # #             [-L / np.sqrt(2), -L / np.sqrt(2), 0],  # 后左
# # #             [L / np.sqrt(2), -L / np.sqrt(2), 0],  # 前左
# # #         ])
# # #
# # #         # 电机旋转方向 (1=CW, -1=CCW)
# # #         self.motor_directions = np.array([1, -1, 1, -1])
# # #
# # #         # 计算混控矩阵
# # #         self._compute_mixer_matrix()
# # #
# # #         logger.debug("四旋翼动力学模型初始化完成")
# # #
# # #     def _compute_mixer_matrix(self):
# # #         """计算混控矩阵"""
# # #         L = self.params.arm_length
# # #         kf = self.params.motor_constant
# # #         km = self.params.moment_constant
# # #
# # #         # 从电机转速到力和力矩的映射
# # #         # [F, Mx, My, Mz] = A * [w1^2, w2^2, w3^2, w4^2]
# # #         self.mixer_matrix = np.array([
# # #             [kf, kf, kf, kf],  # 总推力
# # #             [L / np.sqrt(2) * kf, -L / np.sqrt(2) * kf, -L / np.sqrt(2) * kf, L / np.sqrt(2) * kf],  # Roll力矩
# # #             [L / np.sqrt(2) * kf, L / np.sqrt(2) * kf, -L / np.sqrt(2) * kf, -L / np.sqrt(2) * kf],  # Pitch力矩
# # #             [km * kf, -km * kf, km * kf, -km * kf],  # Yaw力矩
# # #         ])
# # #
# # #         # 逆混控矩阵（从期望力/力矩到电机转速）
# # #         try:
# # #             self.mixer_matrix_inv = np.linalg.pinv(self.mixer_matrix)
# # #         except np.linalg.LinAlgError:
# # #             logger.warning("混控矩阵求逆失败，使用伪逆")
# # #             self.mixer_matrix_inv = np.linalg.pinv(self.mixer_matrix)
# # #
# # #     def set_motor_speeds(self, speeds: np.ndarray):
# # #         """设置电机转速（RPM）"""
# # #         speeds = np.clip(speeds, 0, self.params.max_rpm)
# # #         self.state.motor_speeds = speeds.copy()
# # #
# # #     def compute_forces_and_moments(self) -> Tuple[np.ndarray, np.ndarray]:
# # #         """计算作用力和力矩"""
# # #         # 电机角速度（rad/s）
# # #         omega = self.state.motor_speeds * 2 * np.pi / 60
# # #         omega_squared = omega ** 2
# # #
# # #         # 使用混控矩阵计算
# # #         wrench = self.mixer_matrix @ omega_squared
# # #         thrust = wrench[0]
# # #         moments = wrench[1:4]
# # #
# # #         # 机体坐标系下的推力（向上为负z方向）
# # #         thrust_body = np.array([0, 0, -thrust])
# # #
# # #         # 转换到世界坐标系
# # #         R = Quaternion.to_rotation_matrix(self.state.quaternion)
# # #         thrust_world = R @ thrust_body
# # #
# # #         # 重力
# # #         gravity_force = np.array([0, 0, self.params.mass * self.gravity])
# # #
# # #         # 空气阻力
# # #         drag = -self.params.drag_coefficient * self.state.velocity
# # #
# # #         # 总力
# # #         total_force = thrust_world + gravity_force + drag
# # #
# # #         return total_force, moments
# # #
# # #     def step(self, dt: float):
# # #         """
# # #         仿真一步
# # #
# # #         Args:
# # #             dt: 时间步长
# # #         """
# # #         # 计算力和力矩
# # #         forces, moments = self.compute_forces_and_moments()
# # #
# # #         # 线性加速度
# # #         acceleration = forces / self.params.mass
# # #
# # #         # 角加速度
# # #         I = self.params.inertia
# # #         I_inv = np.linalg.inv(I)
# # #         omega = self.state.angular_velocity
# # #
# # #         # 欧拉方程: I * omega_dot = M - omega x (I * omega)
# # #         angular_acceleration = I_inv @ (moments - np.cross(omega, I @ omega))
# # #
# # #         # 积分更新
# # #         # 位置和速度（欧拉积分）
# # #         self.state.velocity = self.state.velocity + acceleration * dt
# # #         self.state.position = self.state.position + self.state.velocity * dt
# # #
# # #         # 角速度
# # #         self.state.angular_velocity = self.state.angular_velocity + angular_acceleration * dt
# # #
# # #         # 四元数更新
# # #         q = self.state.quaternion
# # #         omega_quat = np.array([0, omega[0], omega[1], omega[2]])
# # #         q_dot = 0.5 * Quaternion.multiply(q, omega_quat)
# # #         q_new = q + q_dot * dt
# # #         self.state.quaternion = Quaternion.normalize(q_new)
# # #
# # #         # 地面碰撞检测
# # #         if self.state.position[2] > 0:
# # #             self.state.position[2] = 0
# # #             self.state.velocity[2] = 0
# # #             if self.state.velocity[2] > 0:
# # #                 self.state.velocity[2] = 0
# # #
# # #     def reset(self, position: Optional[np.ndarray] = None):
# # #         """重置状态"""
# # #         self.state = DroneState()
# # #         if position is not None:
# # #             self.state.position = np.array(position, dtype=float)
# # #
# # #     def get_state(self) -> DroneState:
# # #         """获取当前状态"""
# # #         return self.state.copy()
# #
# #
# # # core/physics/quadrotor_dynamics.py（修复导入部分）
# #
# # """
# # 四旋翼无人机动力学模型
# # """
# #
# # import numpy as np
# # from dataclasses import dataclass, field
# # from typing import Optional
# # from loguru import logger
# #
# # from utils.math.quaternion import Quaternion
# # from utils.config.config_manager import get_config  # 修复导入
# #
# #
# # @dataclass
# # class DroneState:
# #     """无人机状态"""
# #     position: np.ndarray = field(default_factory=lambda: np.zeros(3))
# #     velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
# #     quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
# #     angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
# #     motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
# #
# #     @property
# #     def euler_angles(self) -> np.ndarray:
# #         """获取欧拉角 [roll, pitch, yaw]"""
# #         return Quaternion.to_euler(self.quaternion)
# #
# #     @property
# #     def rotation_matrix(self) -> np.ndarray:
# #         """获取旋转矩阵"""
# #         return Quaternion.to_rotation_matrix(self.quaternion)
# #
# #     @property
# #     def altitude(self) -> float:
# #         """获取高度（向上为正）"""
# #         return -self.position[2]
# #
# #     @property
# #     def speed(self) -> float:
# #         """获取速率"""
# #         return float(np.linalg.norm(self.velocity))
# #
# #     def copy(self) -> 'DroneState':
# #         """深拷贝"""
# #         return DroneState(
# #             position=self.position.copy(),
# #             velocity=self.velocity.copy(),
# #             quaternion=self.quaternion.copy(),
# #             angular_velocity=self.angular_velocity.copy(),
# #             motor_speeds=self.motor_speeds.copy()
# #         )
# #
# #
# # class QuadrotorDynamics:
# #     """四旋翼动力学模型"""
# #
# #     def __init__(self):
# #         """初始化动力学模型"""
# #         config = get_config()
# #
# #         # 物理参数
# #         self.mass = config.drone.mass
# #         self.arm_length = config.drone.arm_length
# #         self.inertia = np.diag(config.drone.inertia)
# #         self.inertia_inv = np.linalg.inv(self.inertia)
# #
# #         # 电机参数
# #         self.kf = config.drone.motor_constant
# #         self.km = config.drone.moment_constant
# #         self.max_rpm = config.drone.max_rpm
# #         self.min_rpm = config.drone.min_rpm
# #
# #         # 阻力系数
# #         self.drag_coeff = config.drone.drag_coefficient
# #         self.rotor_drag_coeff = config.drone.rotor_drag_coefficient
# #
# #         # 重力
# #         self.gravity = config.simulation.gravity
# #
# #         # 状态
# #         self._state = DroneState()
# #
# #         # 电机命令
# #         self._motor_commands = np.zeros(4)
# #
# #         logger.debug("四旋翼动力学模型初始化完成")
# #
# #     @property
# #     def state(self) -> DroneState:
# #         """获取当前状态"""
# #         return self._state
# #
# #     def set_motor_speeds(self, speeds: np.ndarray):
# #         """设置电机转速命令 (RPM)"""
# #         self._motor_commands = np.clip(speeds, self.min_rpm, self.max_rpm)
# #         self._state.motor_speeds = self._motor_commands.copy()
# #
# #     def reset(self, position: np.ndarray = None, yaw: float = 0.0):
# #         """重置状态"""
# #         self._state = DroneState()
# #         if position is not None:
# #             self._state.position = np.array(position, dtype=float)
# #
# #         # 设置初始偏航角
# #         if yaw != 0.0:
# #             self._state.quaternion = Quaternion.from_euler(np.array([0.0, 0.0, yaw]))
# #
# #         self._motor_commands = np.zeros(4)
# #
# #     def step(self, dt: float):
# #         """仿真步进"""
# #         # 转换RPM到rad/s
# #         omega = self._motor_commands * 2 * np.pi / 60
# #
# #         # 计算各电机推力 (N)
# #         thrust_per_motor = self.kf * omega ** 2
# #         total_thrust = np.sum(thrust_per_motor)
# #
# #         # 计算力矩 (Nm) - X型布局
# #         L = self.arm_length / np.sqrt(2)
# #
# #         tau_x = L * (thrust_per_motor[0] - thrust_per_motor[1] - thrust_per_motor[2] + thrust_per_motor[3])
# #         tau_y = L * (thrust_per_motor[0] + thrust_per_motor[1] - thrust_per_motor[2] - thrust_per_motor[3])
# #         tau_z = self.km * (-thrust_per_motor[0] + thrust_per_motor[1] - thrust_per_motor[2] + thrust_per_motor[3])
# #
# #         torques = np.array([tau_x, tau_y, tau_z])
# #
# #         # 获取旋转矩阵
# #         R = self._state.rotation_matrix
# #
# #         # 计算体坐标系下的推力 (沿z轴负方向)
# #         thrust_body = np.array([0, 0, -total_thrust])
# #
# #         # 转换到世界坐标系
# #         thrust_world = R @ thrust_body
# #
# #         # 重力 (NED坐标系，向下为正)
# #         gravity_world = np.array([0, 0, self.mass * self.gravity])
# #
# #         # 空气阻力
# #         drag = -self.drag_coeff * self._state.velocity
# #
# #         # 总加速度
# #         acceleration = (thrust_world + gravity_world + drag) / self.mass
# #
# #         # 更新速度和位置
# #         self._state.velocity += acceleration * dt
# #         self._state.position += self._state.velocity * dt
# #
# #         # 角加速度
# #         omega_body = self._state.angular_velocity
# #         gyroscopic = np.cross(omega_body, self.inertia @ omega_body)
# #         rotor_drag = -self.rotor_drag_coeff * omega_body
# #
# #         angular_acceleration = self.inertia_inv @ (torques - gyroscopic + rotor_drag)
# #
# #         # 更新角速度
# #         self._state.angular_velocity += angular_acceleration * dt
# #
# #         # 更新四元数
# #         self._state.quaternion = Quaternion.integrate(
# #             self._state.quaternion,
# #             self._state.angular_velocity,
# #             dt
# #         )
# #         self._state.quaternion = Quaternion.normalize(self._state.quaternion)
# #
# #         # 地面碰撞检测
# #         if self._state.position[2] > 0:
# #             self._state.position[2] = 0
# #             self._state.velocity[2] = min(0, self._state.velocity[2])
#
#
# # core/physics/quadrotor_dynamics.py（修复版）
#
# """
# 四旋翼无人机动力学模型
# """
#
# import numpy as np
# from dataclasses import dataclass, field
# from typing import Optional
# from loguru import logger
#
# from utils.math.quaternion import Quaternion
# from utils.config.config_manager import get_config
#
#
# @dataclass
# class DroneState:
#     """无人机状态"""
#     position: np.ndarray = field(default_factory=lambda: np.zeros(3))
#     velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
#     quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
#     angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
#     motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
#
#     @property
#     def euler_angles(self) -> np.ndarray:
#         """获取欧拉角 [roll, pitch, yaw]"""
#         return Quaternion.to_euler(self.quaternion)
#
#     @property
#     def rotation_matrix(self) -> np.ndarray:
#         """获取旋转矩阵（体坐标系到世界坐标系）"""
#         return Quaternion.to_rotation_matrix(self.quaternion)
#
#     @property
#     def altitude(self) -> float:
#         """获取高度（向上为正）- NED坐标系z向下"""
#         return -self.position[2]
#
#     @property
#     def speed(self) -> float:
#         """获取速率"""
#         return float(np.linalg.norm(self.velocity))
#
#     def copy(self) -> 'DroneState':
#         """深拷贝"""
#         return DroneState(
#             position=self.position.copy(),
#             velocity=self.velocity.copy(),
#             quaternion=self.quaternion.copy(),
#             angular_velocity=self.angular_velocity.copy(),
#             motor_speeds=self.motor_speeds.copy()
#         )
#
#
# class QuadrotorDynamics:
#     """四旋翼动力学模型"""
#
#     def __init__(self):
#         """初始化动力学模型"""
#         config = get_config()
#
#         # 物理参数
#         self.mass = config.drone.mass
#         self.arm_length = config.drone.arm_length
#         self.inertia = np.diag(config.drone.inertia)
#         self.inertia_inv = np.linalg.inv(self.inertia)
#
#         # 电机参数
#         self.kf = config.drone.motor_constant
#         self.km = config.drone.moment_constant
#         self.max_rpm = config.drone.max_rpm
#         self.min_rpm = config.drone.min_rpm
#
#         # 阻力系数
#         self.drag_coeff = config.drone.drag_coefficient
#         self.rotor_drag_coeff = config.drone.rotor_drag_coefficient
#
#         # 重力
#         self.gravity = config.simulation.gravity
#
#         # 状态
#         self._state = DroneState()
#
#         # 电机命令
#         self._motor_commands = np.zeros(4)
#
#         logger.debug("四旋翼动力学模型初始化完成")
#
#     @property
#     def state(self) -> DroneState:
#         """获取当前状态"""
#         return self._state
#
#     def set_motor_speeds(self, speeds: np.ndarray):
#         """设置电机转速命令 (RPM)"""
#         self._motor_commands = np.clip(speeds, self.min_rpm, self.max_rpm)
#         self._state.motor_speeds = self._motor_commands.copy()
#
#     def reset(self, position: np.ndarray = None, yaw: float = 0.0):
#         """重置状态"""
#         self._state = DroneState()
#         if position is not None:
#             self._state.position = np.array(position, dtype=float)
#
#         # 设置初始偏航角
#         if yaw != 0.0:
#             self._state.quaternion = Quaternion.from_euler(np.array([0.0, 0.0, yaw]))
#
#         self._motor_commands = np.zeros(4)
#
#     def step(self, dt: float):
#         """仿真步进"""
#         # 转换RPM到rad/s
#         omega_motors = self._motor_commands * 2 * np.pi / 60
#
#         # 计算各电机推力 (N): F = kf * omega^2
#         thrust_per_motor = self.kf * omega_motors ** 2
#         total_thrust = np.sum(thrust_per_motor)
#
#         # 计算力矩 (Nm) - X型布局
#         L = self.arm_length * 0.707  # cos(45°)
#
#         # 力矩计算（匹配控制器的混控矩阵）
#         # tau_roll: 电机0,3产生正力矩，电机1,2产生负力矩
#         # tau_pitch: 电机0,1产生正力矩，电机2,3产生负力矩
#         # tau_yaw: CCW电机(0,3)产生正力矩，CW电机(1,2)产生负力矩
#
#         tau_roll = L * (thrust_per_motor[0] - thrust_per_motor[1]
#                         - thrust_per_motor[2] + thrust_per_motor[3])
#         tau_pitch = L * (thrust_per_motor[0] + thrust_per_motor[1]
#                          - thrust_per_motor[2] - thrust_per_motor[3])
#         tau_yaw = self.km * (-thrust_per_motor[0] + thrust_per_motor[1]
#                              - thrust_per_motor[2] + thrust_per_motor[3])
#
#         torques = np.array([tau_roll, tau_pitch, tau_yaw])
#
#         # 获取旋转矩阵（体到世界）
#         R = self._state.rotation_matrix
#
#         # 推力在体坐标系中沿z轴负方向（向上）
#         thrust_body = np.array([0, 0, -total_thrust])
#
#         # 转换到世界坐标系
#         thrust_world = R @ thrust_body
#
#         # 重力 (NED坐标系，向下为正)
#         gravity_world = np.array([0, 0, self.mass * self.gravity])
#
#         # 空气阻力
#         drag = -self.drag_coeff * self._state.velocity * np.abs(self._state.velocity)
#
#         # 总加速度
#         acceleration = (thrust_world + gravity_world + drag) / self.mass
#
#         # 更新速度和位置（使用半隐式欧拉法）
#         self._state.velocity = self._state.velocity + acceleration * dt
#         self._state.position = self._state.position + self._state.velocity * dt
#
#         # 地面碰撞检测 (z > 0 表示在地面以下)
#         if self._state.position[2] > 0:
#             self._state.position[2] = 0
#             if self._state.velocity[2] > 0:
#                 self._state.velocity[2] = 0
#             # 地面摩擦
#             self._state.velocity[0] *= 0.9
#             self._state.velocity[1] *= 0.9
#
#         # 角加速度
#         omega_body = self._state.angular_velocity
#         # 欧拉方程: I * alpha = tau - omega x (I * omega)
#         gyroscopic = np.cross(omega_body, self.inertia @ omega_body)
#         # 旋翼阻尼
#         rotor_damping = -self.rotor_drag_coeff * omega_body
#
#         angular_acceleration = self.inertia_inv @ (torques - gyroscopic + rotor_damping)
#
#         # 更新角速度
#         self._state.angular_velocity = self._state.angular_velocity + angular_acceleration * dt
#
#         # 角速度限制（防止失控）
#         max_angular_vel = 10.0  # rad/s
#         self._state.angular_velocity = np.clip(
#             self._state.angular_velocity, -max_angular_vel, max_angular_vel
#         )
#
#         # 更新四元数
#         self._state.quaternion = Quaternion.integrate(
#             self._state.quaternion,
#             self._state.angular_velocity,
#             dt
#         )
#         self._state.quaternion = Quaternion.normalize(self._state.quaternion)


# core/physics/quadrotor_dynamics.py（力矩方向修正）

"""
四旋翼无人机动力学模型 - 修正版
"""

import numpy as np
from dataclasses import dataclass, field
from loguru import logger

from utils.math.quaternion import Quaternion
from utils.config.config_manager import get_config


@dataclass
class DroneState:
    """无人机状态"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))

    @property
    def euler_angles(self) -> np.ndarray:
        return Quaternion.to_euler(self.quaternion)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return Quaternion.to_rotation_matrix(self.quaternion)

    @property
    def altitude(self) -> float:
        return -self.position[2]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def copy(self) -> 'DroneState':
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
            motor_speeds=self.motor_speeds.copy()
        )


class QuadrotorDynamics:
    """四旋翼动力学模型"""

    def __init__(self):
        config = get_config()

        self.mass = config.drone.mass
        self.arm_length = config.drone.arm_length
        self.inertia = np.diag(config.drone.inertia)
        self.inertia_inv = np.linalg.inv(self.inertia)

        self.kf = config.drone.motor_constant
        self.km = config.drone.moment_constant
        self.max_rpm = config.drone.max_rpm
        self.min_rpm = config.drone.min_rpm

        self.drag_coeff = config.drone.drag_coefficient
        self.rotor_drag_coeff = config.drone.rotor_drag_coefficient

        self.gravity = config.simulation.gravity

        self._state = DroneState()
        self._motor_commands = np.zeros(4)

        logger.debug("四旋翼动力学模型初始化完成")

    @property
    def state(self) -> DroneState:
        return self._state

    def set_motor_speeds(self, speeds: np.ndarray):
        self._motor_commands = np.clip(speeds, self.min_rpm, self.max_rpm)
        self._state.motor_speeds = self._motor_commands.copy()

    def reset(self, position: np.ndarray = None, yaw: float = 0.0):
        self._state = DroneState()
        if position is not None:
            self._state.position = np.array(position, dtype=float)
        if yaw != 0.0:
            self._state.quaternion = Quaternion.from_euler(np.array([0.0, 0.0, yaw]))
        self._motor_commands = np.zeros(4)

    def step(self, dt: float):
        """仿真步进"""
        # RPM转rad/s
        omega_motors = self._motor_commands * 2 * np.pi / 60

        # 各电机推力
        thrust = self.kf * omega_motors ** 2
        total_thrust = np.sum(thrust)

        # 力矩计算 (与控制器混控器匹配)
        # 电机布局 (X型):
        #    0(FR)  1(FL)
        #    3(BL)  2(BR)
        #
        # tau_x (roll): 0,2增加 -> 右翼下沉 -> +roll
        # tau_y (pitch): 0,1增加 -> 机头上仰 -> +pitch
        # tau_z (yaw): 0,3(CCW)增加 -> 顺时针 -> +yaw

        L = self.arm_length * 0.707

        tau_x = L * (thrust[0] - thrust[1] - thrust[2] + thrust[3])
        tau_y = L * (thrust[0] + thrust[1] - thrust[2] - thrust[3])
        tau_z = self.km * (thrust[0] - thrust[1] + thrust[2] - thrust[3])

        torques = np.array([tau_x, tau_y, tau_z])

        # 旋转矩阵
        R = self._state.rotation_matrix

        # 推力在体坐标系沿-Z轴（向上）
        thrust_body = np.array([0, 0, -total_thrust])
        thrust_world = R @ thrust_body

        # 重力 (NED: 向下为正)
        gravity_world = np.array([0, 0, self.mass * self.gravity])

        # 阻力
        drag = -self.drag_coeff * self._state.velocity

        # 加速度
        acceleration = (thrust_world + gravity_world + drag) / self.mass

        # 更新速度和位置
        self._state.velocity += acceleration * dt
        self._state.position += self._state.velocity * dt

        # 地面碰撞
        if self._state.position[2] > 0:
            self._state.position[2] = 0
            if self._state.velocity[2] > 0:
                self._state.velocity[2] = 0

        # 角加速度
        omega = self._state.angular_velocity
        gyroscopic = np.cross(omega, self.inertia @ omega)
        damping = -self.rotor_drag_coeff * omega

        angular_acc = self.inertia_inv @ (torques - gyroscopic + damping)

        # 更新角速度
        self._state.angular_velocity += angular_acc * dt
        self._state.angular_velocity = np.clip(self._state.angular_velocity, -10, 10)

        # 更新四元数
        self._state.quaternion = Quaternion.integrate(
            self._state.quaternion,
            self._state.angular_velocity,
            dt
        )
        self._state.quaternion = Quaternion.normalize(self._state.quaternion)
