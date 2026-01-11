# # # # # # import numpy as np
# # # # # # from dataclasses import dataclass, field
# # # # # # from typing import Optional, Tuple
# # # # # # from loguru import logger
# # # # # #
# # # # # #
# # # # # # @dataclass
# # # # # # class PIDGains:
# # # # # #     """PID增益参数"""
# # # # # #     kp: float = 1.0  # 比例增益
# # # # # #     ki: float = 0.0  # 积分增益
# # # # # #     kd: float = 0.0  # 微分增益
# # # # # #
# # # # # #     # 限幅
# # # # # #     output_min: float = -float('inf')
# # # # # #     output_max: float = float('inf')
# # # # # #     integral_min: float = -float('inf')
# # # # # #     integral_max: float = float('inf')
# # # # # #
# # # # # #
# # # # # # class PIDController:
# # # # # #     """单轴PID控制器"""
# # # # # #
# # # # # #     def __init__(self, gains: Optional[PIDGains] = None):
# # # # # #         self.gains = gains or PIDGains()
# # # # # #         self.reset()
# # # # # #
# # # # # #     def reset(self):
# # # # # #         """重置控制器状态"""
# # # # # #         self._integral = 0.0
# # # # # #         self._last_error = 0.0
# # # # # #         self._last_derivative = 0.0
# # # # # #         self._first_run = True
# # # # # #
# # # # # #     def update(self, setpoint: float, measurement: float, dt: float) -> float:
# # # # # #         """
# # # # # #         更新PID控制器
# # # # # #
# # # # # #         Args:
# # # # # #             setpoint: 设定值
# # # # # #             measurement: 测量值
# # # # # #             dt: 时间步长 (s)
# # # # # #
# # # # # #         Returns:
# # # # # #             控制输出
# # # # # #         """
# # # # # #         error = setpoint - measurement
# # # # # #
# # # # # #         # 比例项
# # # # # #         p_term = self.gains.kp * error
# # # # # #
# # # # # #         # 积分项（带抗饱和）
# # # # # #         self._integral += error * dt
# # # # # #         self._integral = np.clip(
# # # # # #             self._integral,
# # # # # #             self.gains.integral_min,
# # # # # #             self.gains.integral_max
# # # # # #         )
# # # # # #         i_term = self.gains.ki * self._integral
# # # # # #
# # # # # #         # 微分项（带滤波）
# # # # # #         if self._first_run:
# # # # # #             derivative = 0.0
# # # # # #             self._first_run = False
# # # # # #         else:
# # # # # #             derivative = (error - self._last_error) / dt
# # # # # #             # 低通滤波
# # # # # #             alpha = 0.8
# # # # # #             derivative = alpha * derivative + (1 - alpha) * self._last_derivative
# # # # # #
# # # # # #         d_term = self.gains.kd * derivative
# # # # # #         self._last_derivative = derivative
# # # # # #         self._last_error = error
# # # # # #
# # # # # #         # 总输出
# # # # # #         output = p_term + i_term + d_term
# # # # # #
# # # # # #         # 输出限幅
# # # # # #         output = np.clip(output, self.gains.output_min, self.gains.output_max)
# # # # # #
# # # # # #         return output
# # # # # #
# # # # # #     def set_gains(self, kp: float, ki: float, kd: float):
# # # # # #         """设置PID增益"""
# # # # # #         self.gains.kp = kp
# # # # # #         self.gains.ki = ki
# # # # # #         self.gains.kd = kd
# # # # # #
# # # # # #
# # # # # # class PIDController3D:
# # # # # #     """三轴PID控制器"""
# # # # # #
# # # # # #     def __init__(self,
# # # # # #                  kp: Tuple[float, float, float] = (1.0, 1.0, 1.0),
# # # # # #                  ki: Tuple[float, float, float] = (0.0, 0.0, 0.0),
# # # # # #                  kd: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
# # # # # #
# # # # # #         self.controllers = [
# # # # # #             PIDController(PIDGains(kp=kp[i], ki=ki[i], kd=kd[i]))
# # # # # #             for i in range(3)
# # # # # #         ]
# # # # # #
# # # # # #     def reset(self):
# # # # # #         """重置所有控制器"""
# # # # # #         for controller in self.controllers:
# # # # # #             controller.reset()
# # # # # #
# # # # # #     def update(self, setpoint: np.ndarray, measurement: np.ndarray,
# # # # # #                dt: float) -> np.ndarray:
# # # # # #         """更新控制器"""
# # # # # #         output = np.zeros(3)
# # # # # #         for i, controller in enumerate(self.controllers):
# # # # # #             output[i] = controller.update(setpoint[i], measurement[i], dt)
# # # # # #         return output
# # # # # #
# # # # # #     def set_gains(self, axis: int, kp: float, ki: float, kd: float):
# # # # # #         """设置指定轴的增益"""
# # # # # #         if 0 <= axis < 3:
# # # # # #             self.controllers[axis].set_gains(kp, ki, kd)
# # # # # #
# # # # # #     def set_output_limits(self, min_val: float, max_val: float):
# # # # # #         """设置输出限幅"""
# # # # # #         for controller in self.controllers:
# # # # # #             controller.gains.output_min = min_val
# # # # # #             controller.gains.output_max = max_val
# # # # # #
# # # # # #
# # # # # # @dataclass
# # # # # # class CascadePIDConfig:
# # # # # #     """串级PID配置"""
# # # # # #     # 外环（位置）
# # # # # #     position_kp: Tuple[float, float, float] = (2.0, 2.0, 4.0)
# # # # # #     position_ki: Tuple[float, float, float] = (0.0, 0.0, 0.2)
# # # # # #     position_kd: Tuple[float, float, float] = (1.5, 1.5, 2.0)
# # # # # #
# # # # # #     # 内环（姿态）
# # # # # #     attitude_kp: Tuple[float, float, float] = (8.0, 8.0, 4.0)
# # # # # #     attitude_ki: Tuple[float, float, float] = (0.0, 0.0, 0.0)
# # # # # #     attitude_kd: Tuple[float, float, float] = (3.5, 3.5, 1.5)
# # # # # #
# # # # # #     # 限制
# # # # # #     max_velocity: float = 10.0
# # # # # #     max_tilt_angle: float = 0.5  # rad (~28.6°)
# # # # # #     max_yaw_rate: float = 2.0  # rad/s
# # # # # #
# # # # # #
# # # # # # class QuadrotorPIDController:
# # # # # #     """
# # # # # #     四旋翼串级PID控制器
# # # # # #
# # # # # #     控制结构:
# # # # # #     位置设定 -> 位置控制器 -> 期望速度 -> 速度控制器 -> 期望加速度
# # # # # #             -> 姿态解算 -> 期望姿态 -> 姿态控制器 -> 期望力矩
# # # # # #     """
# # # # # #
# # # # # #     def __init__(self, config: Optional[CascadePIDConfig] = None):
# # # # # #         self.config = config or CascadePIDConfig()
# # # # # #
# # # # # #         # 位置控制器
# # # # # #         self.position_controller = PIDController3D(
# # # # # #             kp=self.config.position_kp,
# # # # # #             ki=self.config.position_ki,
# # # # # #             kd=self.config.position_kd
# # # # # #         )
# # # # # #
# # # # # #         # 姿态控制器
# # # # # #         self.attitude_controller = PIDController3D(
# # # # # #             kp=self.config.attitude_kp,
# # # # # #             ki=self.config.attitude_ki,
# # # # # #             kd=self.config.attitude_kd
# # # # # #         )
# # # # # #
# # # # # #         # 偏航控制器（单独）
# # # # # #         self.yaw_controller = PIDController(
# # # # # #             PIDGains(kp=4.0, ki=0.0, kd=1.5)
# # # # # #         )
# # # # # #
# # # # # #         logger.info("四旋翼PID控制器初始化完成")
# # # # # #
# # # # # #     def reset(self):
# # # # # #         """重置控制器"""
# # # # # #         self.position_controller.reset()
# # # # # #         self.attitude_controller.reset()
# # # # # #         self.yaw_controller.reset()
# # # # # #
# # # # # #     def compute_control(self,
# # # # # #                         target_position: np.ndarray,
# # # # # #                         target_yaw: float,
# # # # # #                         current_position: np.ndarray,
# # # # # #                         current_velocity: np.ndarray,
# # # # # #                         current_attitude: np.ndarray,  # [roll, pitch, yaw]
# # # # # #                         current_angular_velocity: np.ndarray,
# # # # # #                         mass: float,
# # # # # #                         gravity: float,
# # # # # #                         dt: float) -> Tuple[float, np.ndarray]:
# # # # # #         """
# # # # # #         计算控制输出
# # # # # #
# # # # # #         Args:
# # # # # #             target_position: 目标位置 [x, y, z] (NED)
# # # # # #             target_yaw: 目标偏航角 (rad)
# # # # # #             current_position: 当前位置 (NED)
# # # # # #             current_velocity: 当前速度 (NED)
# # # # # #             current_attitude: 当前姿态 [roll, pitch, yaw]
# # # # # #             current_angular_velocity: 当前角速度
# # # # # #             mass: 质量 (kg)
# # # # # #             gravity: 重力加速度 (m/s²)
# # # # # #             dt: 时间步长 (s)
# # # # # #
# # # # # #         Returns:
# # # # # #             (thrust, torques): 总推力(N)和力矩[τx, τy, τz](N·m)
# # # # # #         """
# # # # # #         # ========== 外环：位置控制 ==========
# # # # # #         # 计算期望加速度
# # # # # #         desired_acceleration = self.position_controller.update(
# # # # # #             target_position, current_position, dt
# # # # # #         )
# # # # # #
# # # # # #         # 速度前馈（可选）
# # # # # #         velocity_error = -current_velocity  # 目标速度为0时
# # # # # #         desired_acceleration += 0.5 * velocity_error
# # # # # #
# # # # # #         # 限制加速度
# # # # # #         accel_magnitude = np.linalg.norm(desired_acceleration[:2])
# # # # # #         if accel_magnitude > self.config.max_velocity:
# # # # # #             desired_acceleration[:2] *= self.config.max_velocity / accel_magnitude
# # # # # #
# # # # # #         # ========== 推力计算 ==========
# # # # # #         # 期望推力向量（惯性系）
# # # # # #         # F = m * (a_desired - g)，其中g在NED系为[0, 0, g]
# # # # # #         thrust_vector = mass * (desired_acceleration - np.array([0, 0, gravity]))
# # # # # #
# # # # # #         # 总推力大小（负z方向）
# # # # # #         thrust = np.linalg.norm(thrust_vector)
# # # # # #
# # # # # #         # ========== 姿态解算 ==========
# # # # # #         # 从期望推力向量计算期望姿态
# # # # # #         if thrust > 0.1:  # 避免除零
# # # # # #             # 期望的z轴方向（归一化推力向量的反方向）
# # # # # #             z_des = -thrust_vector / thrust
# # # # # #
# # # # # #             # 期望偏航方向
# # # # # #             yaw_vec = np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
# # # # # #
# # # # # #             # 计算期望的x轴方向
# # # # # #             y_des = np.cross(z_des, yaw_vec)
# # # # # #             y_des_norm = np.linalg.norm(y_des)
# # # # # #             if y_des_norm > 0.01:
# # # # # #                 y_des /= y_des_norm
# # # # # #             else:
# # # # # #                 y_des = np.array([0, 1, 0])
# # # # # #
# # # # # #             x_des = np.cross(y_des, z_des)
# # # # # #             x_des /= np.linalg.norm(x_des)
# # # # # #
# # # # # #             # 从旋转矩阵提取欧拉角
# # # # # #             R_des = np.column_stack([x_des, y_des, z_des])
# # # # # #             desired_roll = np.arctan2(R_des[2, 1], R_des[2, 2])
# # # # # #             desired_pitch = -np.arcsin(np.clip(R_des[2, 0], -1, 1))
# # # # # #         else:
# # # # # #             desired_roll = 0.0
# # # # # #             desired_pitch = 0.0
# # # # # #
# # # # # #         # 限制倾斜角
# # # # # #         desired_roll = np.clip(desired_roll, -self.config.max_tilt_angle,
# # # # # #                                self.config.max_tilt_angle)
# # # # # #         desired_pitch = np.clip(desired_pitch, -self.config.max_tilt_angle,
# # # # # #                                 self.config.max_tilt_angle)
# # # # # #
# # # # # #         desired_attitude = np.array([desired_roll, desired_pitch, target_yaw])
# # # # # #
# # # # # #         # ========== 内环：姿态控制 ==========
# # # # # #         # 计算姿态误差
# # # # # #         attitude_error = desired_attitude - current_attitude
# # # # # #
# # # # # #         # 处理偏航角跨越±π的情况
# # # # # #         attitude_error[2] = self._normalize_angle(attitude_error[2])
# # # # # #
# # # # # #         # 计算期望力矩
# # # # # #         torques = self.attitude_controller.update(
# # # # # #             desired_attitude, current_attitude, dt
# # # # # #         )
# # # # # #
# # # # # #         # 添加角速度阻尼
# # # # # #         torques -= 0.5 * current_angular_velocity
# # # # # #
# # # # # #         return thrust, torques
# # # # # #
# # # # # #     @staticmethod
# # # # # #     def _normalize_angle(angle: float) -> float:
# # # # # #         """将角度归一化到 [-π, π]"""
# # # # # #         while angle > np.pi:
# # # # # #             angle -= 2 * np.pi
# # # # # #         while angle < -np.pi:
# # # # # #             angle += 2 * np.pi
# # # # # #         return angle
# # # # # #
# # # # # #
# # # # # # class AttitudeRateController:
# # # # # #     """
# # # # # #     姿态角速度控制器（内环）
# # # # # #     用于更精确的姿态控制
# # # # # #     """
# # # # # #
# # # # # #     def __init__(self):
# # # # # #         self.roll_rate_pid = PIDController(PIDGains(kp=0.15, ki=0.0, kd=0.01))
# # # # # #         self.pitch_rate_pid = PIDController(PIDGains(kp=0.15, ki=0.0, kd=0.01))
# # # # # #         self.yaw_rate_pid = PIDController(PIDGains(kp=0.3, ki=0.0, kd=0.0))
# # # # # #
# # # # # #     def reset(self):
# # # # # #         self.roll_rate_pid.reset()
# # # # # #         self.pitch_rate_pid.reset()
# # # # # #         self.yaw_rate_pid.reset()
# # # # # #
# # # # # #     def update(self,
# # # # # #                desired_rates: np.ndarray,
# # # # # #                current_rates: np.ndarray,
# # # # # #                dt: float) -> np.ndarray:
# # # # # #         """计算力矩输出"""
# # # # # #         torques = np.array([
# # # # # #             self.roll_rate_pid.update(desired_rates[0], current_rates[0], dt),
# # # # # #             self.pitch_rate_pid.update(desired_rates[1], current_rates[1], dt),
# # # # # #             self.yaw_rate_pid.update(desired_rates[2], current_rates[2], dt)
# # # # # #         ])
# # # # # #         return torques
# # # # #
# # # # # core/control/pid_controller.py（修复版）
# # # #
# # # # # """
# # # # # PID控制器模块
# # # # # """
# # # # #
# # # # # import numpy as np
# # # # # from dataclasses import dataclass
# # # # # from typing import Optional, Tuple
# # # # # from loguru import logger
# # # # #
# # # # #
# # # # # @dataclass
# # # # # class PIDGains:
# # # # #     """PID增益参数"""
# # # # #     kp: float = 1.0
# # # # #     ki: float = 0.0
# # # # #     kd: float = 0.0
# # # # #
# # # # #
# # # # # class PIDController:
# # # # #     """单轴PID控制器"""
# # # # #
# # # # #     def __init__(self, gains: Optional[PIDGains] = None):
# # # # #         """
# # # # #         初始化PID控制器
# # # # #
# # # # #         Args:
# # # # #             gains: PID增益参数
# # # # #         """
# # # # #         self.gains = gains or PIDGains()
# # # # #
# # # # #         self.integral = 0.0
# # # # #         self.last_error = 0.0
# # # # #         self.last_derivative = 0.0
# # # # #
# # # # #         # 积分限幅
# # # # #         self.integral_limit = 100.0
# # # # #
# # # # #         # 输出限幅
# # # # #         self.output_limit = float('inf')
# # # # #
# # # # #         # 微分滤波系数
# # # # #         self.derivative_filter = 0.1
# # # # #
# # # # #     def update(self, target: float = None, current: float = None,
# # # # #                error: float = None, dt: float = 0.01) -> float:
# # # # #         """
# # # # #         更新PID控制器
# # # # #
# # # # #         Args:
# # # # #             target: 目标值（如果提供error则忽略）
# # # # #             current: 当前值（如果提供error则忽略）
# # # # #             error: 直接提供误差值
# # # # #             dt: 时间步长
# # # # #
# # # # #         Returns:
# # # # #             控制输出
# # # # #         """
# # # # #         # 计算误差
# # # # #         if error is None:
# # # # #             if target is None or current is None:
# # # # #                 return 0.0
# # # # #             error = target - current
# # # # #
# # # # #         # 比例项
# # # # #         p_term = self.gains.kp * error
# # # # #
# # # # #         # 积分项
# # # # #         self.integral += error * dt
# # # # #         self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
# # # # #         i_term = self.gains.ki * self.integral
# # # # #
# # # # #         # 微分项（带滤波）
# # # # #         if dt > 0:
# # # # #             raw_derivative = (error - self.last_error) / dt
# # # # #             derivative = (1 - self.derivative_filter) * self.last_derivative + \
# # # # #                          self.derivative_filter * raw_derivative
# # # # #             self.last_derivative = derivative
# # # # #         else:
# # # # #             derivative = 0.0
# # # # #         d_term = self.gains.kd * derivative
# # # # #
# # # # #         self.last_error = error
# # # # #
# # # # #         # 总输出
# # # # #         output = p_term + i_term + d_term
# # # # #
# # # # #         # 输出限幅
# # # # #         output = np.clip(output, -self.output_limit, self.output_limit)
# # # # #
# # # # #         return float(output)
# # # # #
# # # # #     def reset(self):
# # # # #         """重置控制器状态"""
# # # # #         self.integral = 0.0
# # # # #         self.last_error = 0.0
# # # # #         self.last_derivative = 0.0
# # # # #
# # # # #     def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
# # # # #         """设置PID增益"""
# # # # #         if kp is not None:
# # # # #             self.gains.kp = kp
# # # # #         if ki is not None:
# # # # #             self.gains.ki = ki
# # # # #         if kd is not None:
# # # # #             self.gains.kd = kd
# # # # #
# # # # #
# # # # # class PIDController3D:
# # # # #     """三轴PID控制器"""
# # # # #
# # # # #     def __init__(self, gains_x: PIDGains = None, gains_y: PIDGains = None,
# # # # #                  gains_z: PIDGains = None):
# # # # #         """
# # # # #         初始化三轴PID控制器
# # # # #
# # # # #         Args:
# # # # #             gains_x: X轴增益
# # # # #             gains_y: Y轴增益
# # # # #             gains_z: Z轴增益
# # # # #         """
# # # # #         self.controllers = [
# # # # #             PIDController(gains_x or PIDGains()),
# # # # #             PIDController(gains_y or PIDGains()),
# # # # #             PIDController(gains_z or PIDGains())
# # # # #         ]
# # # # #
# # # # #     def update(self, target: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
# # # # #         """
# # # # #         更新三轴控制器
# # # # #
# # # # #         Args:
# # # # #             target: 目标值 [x, y, z]
# # # # #             current: 当前值 [x, y, z]
# # # # #             dt: 时间步长
# # # # #
# # # # #         Returns:
# # # # #             控制输出 [x, y, z]
# # # # #         """
# # # # #         output = np.zeros(3)
# # # # #         for i in range(3):
# # # # #             output[i] = self.controllers[i].update(
# # # # #                 target=target[i],
# # # # #                 current=current[i],
# # # # #                 dt=dt
# # # # #             )
# # # # #         return output
# # # # #
# # # # #     def reset(self):
# # # # #         """重置所有控制器"""
# # # # #         for controller in self.controllers:
# # # # #             controller.reset()
# # # # #
# # # # #
# # # # # class QuadrotorPIDController:
# # # # #     """四旋翼无人机完整PID控制器"""
# # # # #
# # # # #     def __init__(self):
# # # # #         """初始化四旋翼控制器"""
# # # # #         from utils.config.config_manager import get_config
# # # # #         config = get_config()
# # # # #
# # # # #         # 位置控制器
# # # # #         self.pos_controller = PIDController3D(
# # # # #             PIDGains(*config.control.position_kp[:1], *config.control.position_ki[:1], *config.control.position_kd[:1]),
# # # # #             PIDGains(config.control.position_kp[1], config.control.position_ki[1], config.control.position_kd[1]),
# # # # #             PIDGains(config.control.position_kp[2], config.control.position_ki[2], config.control.position_kd[2])
# # # # #         )
# # # # #
# # # # #         # 速度控制器
# # # # #         self.vel_controller = PIDController3D(
# # # # #             PIDGains(config.control.velocity_kp[0], config.control.velocity_ki[0], config.control.velocity_kd[0]),
# # # # #             PIDGains(config.control.velocity_kp[1], config.control.velocity_ki[1], config.control.velocity_kd[1]),
# # # # #             PIDGains(config.control.velocity_kp[2], config.control.velocity_ki[2], config.control.velocity_kd[2])
# # # # #         )
# # # # #
# # # # #         # 姿态控制器
# # # # #         self.att_controller = PIDController3D(
# # # # #             PIDGains(config.control.attitude_kp[0], config.control.attitude_ki[0], config.control.attitude_kd[0]),
# # # # #             PIDGains(config.control.attitude_kp[1], config.control.attitude_ki[1], config.control.attitude_kd[1]),
# # # # #             PIDGains(config.control.attitude_kp[2], config.control.attitude_ki[2], config.control.attitude_kd[2])
# # # # #         )
# # # # #
# # # # #         # 角速度控制器
# # # # #         self.rate_controller = PIDController3D(
# # # # #             PIDGains(config.control.rate_kp[0], config.control.rate_ki[0], config.control.rate_kd[0]),
# # # # #             PIDGains(config.control.rate_kp[1], config.control.rate_ki[1], config.control.rate_kd[1]),
# # # # #             PIDGains(config.control.rate_kp[2], config.control.rate_ki[2], config.control.rate_kd[2])
# # # # #         )
# # # # #
# # # # #         # 控制限制
# # # # #         self.max_velocity = config.control.max_velocity
# # # # #         self.max_tilt = config.control.max_tilt_angle
# # # # #         self.max_yaw_rate = config.control.max_yaw_rate
# # # # #
# # # # #         # 无人机参数
# # # # #         self.mass = config.drone.mass
# # # # #         self.gravity = 9.81
# # # # #
# # # # #         logger.info("四旋翼PID控制器初始化完成")
# # # # #
# # # # #     def compute_control(self, state, target_position: np.ndarray,
# # # # #                         target_yaw: float = 0.0, dt: float = 0.01) -> np.ndarray:
# # # # #         """
# # # # #         计算控制输入
# # # # #
# # # # #         Args:
# # # # #             state: 无人机状态
# # # # #             target_position: 目标位置
# # # # #             target_yaw: 目标偏航角
# # # # #             dt: 时间步长
# # # # #
# # # # #         Returns:
# # # # #             电机转速命令 (4,)
# # # # #         """
# # # # #         # 位置控制 -> 期望速度
# # # # #         pos_error = target_position - state.position
# # # # #         desired_velocity = self.pos_controller.update(
# # # # #             target=target_position,
# # # # #             current=state.position,
# # # # #             dt=dt
# # # # #         )
# # # # #
# # # # #         # 限制速度
# # # # #         vel_norm = np.linalg.norm(desired_velocity)
# # # # #         if vel_norm > self.max_velocity:
# # # # #             desired_velocity = desired_velocity / vel_norm * self.max_velocity
# # # # #
# # # # #         # 速度控制 -> 期望加速度
# # # # #         desired_acceleration = self.vel_controller.update(
# # # # #             target=desired_velocity,
# # # # #             current=state.velocity,
# # # # #             dt=dt
# # # # #         )
# # # # #
# # # # #         # 计算期望姿态
# # # # #         # 加上重力补偿
# # # # #         desired_acceleration[2] -= self.gravity
# # # # #
# # # # #         # 计算期望推力
# # # # #         thrust = self.mass * np.linalg.norm(desired_acceleration)
# # # # #
# # # # #         # 计算期望姿态角
# # # # #         if thrust > 0.1:
# # # # #             # 期望Roll和Pitch
# # # # #             desired_roll = np.arcsin(
# # # # #                 np.clip((desired_acceleration[1] * np.cos(target_yaw) -
# # # # #                          desired_acceleration[0] * np.sin(target_yaw)) *
# # # # #                         self.mass / thrust, -1, 1)
# # # # #             )
# # # # #             desired_pitch = np.arctan2(
# # # # #                 desired_acceleration[0] * np.cos(target_yaw) +
# # # # #                 desired_acceleration[1] * np.sin(target_yaw),
# # # # #                 -desired_acceleration[2]
# # # # #             )
# # # # #         else:
# # # # #             desired_roll = 0.0
# # # # #             desired_pitch = 0.0
# # # # #
# # # # #         # 限制倾斜角
# # # # #         desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
# # # # #         desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)
# # # # #
# # # # #         # 姿态控制 -> 期望角速度
# # # # #         current_euler = state.euler_angles
# # # # #         target_attitude = np.array([desired_roll, desired_pitch, target_yaw])
# # # # #
# # # # #         desired_rates = self.att_controller.update(
# # # # #             target=target_attitude,
# # # # #             current=current_euler,
# # # # #             dt=dt
# # # # #         )
# # # # #
# # # # #         # 限制偏航角速度
# # # # #         desired_rates[2] = np.clip(desired_rates[2], -self.max_yaw_rate, self.max_yaw_rate)
# # # # #
# # # # #         # 角速度控制 -> 力矩
# # # # #         torques = self.rate_controller.update(
# # # # #             target=desired_rates,
# # # # #             current=state.angular_velocity,
# # # # #             dt=dt
# # # # #         )
# # # # #
# # # # #         # 转换为电机转速
# # # # #         motor_speeds = self._torques_to_motor_speeds(thrust, torques)
# # # # #
# # # # #         return motor_speeds
# # # # #
# # # # #     def _torques_to_motor_speeds(self, thrust: float, torques: np.ndarray) -> np.ndarray:
# # # # #         """将推力和力矩转换为电机转速"""
# # # # #         from utils.config.config_manager import get_config
# # # # #         config = get_config()
# # # # #
# # # # #         L = config.drone.arm_length
# # # # #         kf = config.drone.motor_constant
# # # # #         km = config.drone.moment_constant
# # # # #
# # # # #         # 混控矩阵求解
# # # # #         # [F]     [1    1    1    1  ] [w1^2]
# # # # #         # [Mx] = L[1   -1   -1    1  ] [w2^2] * kf
# # # # #         # [My]   L[1    1   -1   -1  ] [w3^2]
# # # # #         # [Mz]   km[1   -1    1   -1 ] [w4^2]
# # # # #
# # # # #         wrench = np.array([thrust, torques[0], torques[1], torques[2]])
# # # # #
# # # # #         # 简化的逆混控
# # # # #         L_sqrt2 = L / np.sqrt(2)
# # # # #
# # # # #         w1_sq = (wrench[0] / 4 + wrench[1] / (4 * L_sqrt2) +
# # # # #                  wrench[2] / (4 * L_sqrt2) + wrench[3] / (4 * km)) / kf
# # # # #         w2_sq = (wrench[0] / 4 - wrench[1] / (4 * L_sqrt2) +
# # # # #                  wrench[2] / (4 * L_sqrt2) - wrench[3] / (4 * km)) / kf
# # # # #         w3_sq = (wrench[0] / 4 - wrench[1] / (4 * L_sqrt2) -
# # # # #                  wrench[2] / (4 * L_sqrt2) + wrench[3] / (4 * km)) / kf
# # # # #         w4_sq = (wrench[0] / 4 + wrench[1] / (4 * L_sqrt2) -
# # # # #                  wrench[2] / (4 * L_sqrt2) - wrench[3] / (4 * km)) / kf
# # # # #
# # # # #         # 转换为RPM
# # # # #         omega_squared = np.array([w1_sq, w2_sq, w3_sq, w4_sq])
# # # # #         omega_squared = np.maximum(omega_squared, 0)
# # # # #         omega = np.sqrt(omega_squared)
# # # # #         rpm = omega * 60 / (2 * np.pi)
# # # # #
# # # # #         # 限制RPM
# # # # #         rpm = np.clip(rpm, 0, config.drone.max_rpm)
# # # # #
# # # # #         return rpm
# # # # #
# # # # #     def reset(self):
# # # # #         """重置所有控制器"""
# # # # #         self.pos_controller.reset()
# # # # #         self.vel_controller.reset()
# # # # #         self.att_controller.reset()
# # # # #         self.rate_controller.reset()
# # # #
# # # #
# # # # # core/control/pid_controller.py（完整修复版）
# # # #
# # # # """
# # # # PID控制器模块
# # # # """
# # # #
# # # # import numpy as np
# # # # from dataclasses import dataclass
# # # # from typing import Optional
# # # # from loguru import logger
# # # #
# # # #
# # # # @dataclass
# # # # class PIDGains:
# # # #     """PID增益参数"""
# # # #     kp: float = 1.0
# # # #     ki: float = 0.0
# # # #     kd: float = 0.0
# # # #
# # # #
# # # # class PIDController:
# # # #     """单轴PID控制器"""
# # # #
# # # #     def __init__(self, gains: Optional[PIDGains] = None):
# # # #         self.gains = gains or PIDGains()
# # # #
# # # #         self.integral = 0.0
# # # #         self.last_error = 0.0
# # # #         self.last_derivative = 0.0
# # # #         self.first_update = True
# # # #
# # # #         # 积分限幅
# # # #         self.integral_limit = 10.0
# # # #
# # # #         # 输出限幅
# # # #         self.output_limit = float('inf')
# # # #
# # # #         # 微分滤波系数
# # # #         self.derivative_filter = 0.2
# # # #
# # # #     def update(self, target: float = None, current: float = None,
# # # #                error: float = None, dt: float = 0.01) -> float:
# # # #         """更新PID控制器"""
# # # #         if error is None:
# # # #             if target is None or current is None:
# # # #                 return 0.0
# # # #             error = target - current
# # # #
# # # #         # 比例项
# # # #         p_term = self.gains.kp * error
# # # #
# # # #         # 积分项（带限幅）
# # # #         self.integral += error * dt
# # # #         self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
# # # #         i_term = self.gains.ki * self.integral
# # # #
# # # #         # 微分项（带滤波）
# # # #         if self.first_update:
# # # #             derivative = 0.0
# # # #             self.first_update = False
# # # #         elif dt > 0:
# # # #             raw_derivative = (error - self.last_error) / dt
# # # #             derivative = (1 - self.derivative_filter) * self.last_derivative + \
# # # #                          self.derivative_filter * raw_derivative
# # # #             self.last_derivative = derivative
# # # #         else:
# # # #             derivative = 0.0
# # # #         d_term = self.gains.kd * derivative
# # # #
# # # #         self.last_error = error
# # # #
# # # #         # 总输出
# # # #         output = p_term + i_term + d_term
# # # #
# # # #         # 输出限幅
# # # #         output = np.clip(output, -self.output_limit, self.output_limit)
# # # #
# # # #         return float(output)
# # # #
# # # #     def reset(self):
# # # #         """重置控制器状态"""
# # # #         self.integral = 0.0
# # # #         self.last_error = 0.0
# # # #         self.last_derivative = 0.0
# # # #         self.first_update = True
# # # #
# # # #     def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
# # # #         """设置PID增益"""
# # # #         if kp is not None:
# # # #             self.gains.kp = kp
# # # #         if ki is not None:
# # # #             self.gains.ki = ki
# # # #         if kd is not None:
# # # #             self.gains.kd = kd
# # # #
# # # #
# # # # class PIDController3D:
# # # #     """三轴PID控制器"""
# # # #
# # # #     def __init__(self, gains_x: PIDGains = None, gains_y: PIDGains = None,
# # # #                  gains_z: PIDGains = None):
# # # #         self.controllers = [
# # # #             PIDController(gains_x or PIDGains()),
# # # #             PIDController(gains_y or PIDGains()),
# # # #             PIDController(gains_z or PIDGains())
# # # #         ]
# # # #
# # # #     def update(self, target: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
# # # #         """更新三轴控制器"""
# # # #         output = np.zeros(3)
# # # #         for i in range(3):
# # # #             output[i] = self.controllers[i].update(
# # # #                 target=float(target[i]),
# # # #                 current=float(current[i]),
# # # #                 dt=dt
# # # #             )
# # # #         return output
# # # #
# # # #     def reset(self):
# # # #         """重置所有控制器"""
# # # #         for controller in self.controllers:
# # # #             controller.reset()
# # # #
# # # #     def set_output_limits(self, limit: float):
# # # #         """设置输出限幅"""
# # # #         for controller in self.controllers:
# # # #             controller.output_limit = limit
# # # #
# # # #
# # # # class QuadrotorPIDController:
# # # #     """四旋翼无人机完整PID控制器（级联控制）"""
# # # #
# # # #     def __init__(self):
# # # #         """初始化四旋翼控制器"""
# # # #         from utils.config.config_manager import get_config
# # # #         config = get_config()
# # # #
# # # #         # 位置控制器 (外环) -> 输出期望速度
# # # #         self.pos_controller = PIDController3D(
# # # #             PIDGains(config.control.position_kp[0], config.control.position_ki[0], config.control.position_kd[0]),
# # # #             PIDGains(config.control.position_kp[1], config.control.position_ki[1], config.control.position_kd[1]),
# # # #             PIDGains(config.control.position_kp[2], config.control.position_ki[2], config.control.position_kd[2])
# # # #         )
# # # #         self.pos_controller.set_output_limits(config.control.max_velocity)
# # # #
# # # #         # 速度控制器 (中环) -> 输出期望加速度/推力
# # # #         self.vel_controller = PIDController3D(
# # # #             PIDGains(config.control.velocity_kp[0], config.control.velocity_ki[0], config.control.velocity_kd[0]),
# # # #             PIDGains(config.control.velocity_kp[1], config.control.velocity_ki[1], config.control.velocity_kd[1]),
# # # #             PIDGains(config.control.velocity_kp[2], config.control.velocity_ki[2], config.control.velocity_kd[2])
# # # #         )
# # # #         self.vel_controller.set_output_limits(config.control.max_acceleration)
# # # #
# # # #         # 姿态控制器 (内环) -> 输出期望角速度
# # # #         self.att_controller = PIDController3D(
# # # #             PIDGains(config.control.attitude_kp[0], config.control.attitude_ki[0], config.control.attitude_kd[0]),
# # # #             PIDGains(config.control.attitude_kp[1], config.control.attitude_ki[1], config.control.attitude_kd[1]),
# # # #             PIDGains(config.control.attitude_kp[2], config.control.attitude_ki[2], config.control.attitude_kd[2])
# # # #         )
# # # #         self.att_controller.set_output_limits(5.0)  # rad/s
# # # #
# # # #         # 角速度控制器 (最内环) -> 输出力矩
# # # #         self.rate_controller = PIDController3D(
# # # #             PIDGains(config.control.rate_kp[0], config.control.rate_ki[0], config.control.rate_kd[0]),
# # # #             PIDGains(config.control.rate_kp[1], config.control.rate_ki[1], config.control.rate_kd[1]),
# # # #             PIDGains(config.control.rate_kp[2], config.control.rate_ki[2], config.control.rate_kd[2])
# # # #         )
# # # #
# # # #         # 控制限制
# # # #         self.max_velocity = config.control.max_velocity
# # # #         self.max_acceleration = config.control.max_acceleration
# # # #         self.max_tilt = config.control.max_tilt_angle
# # # #         self.max_yaw_rate = config.control.max_yaw_rate
# # # #
# # # #         # 无人机参数
# # # #         self.mass = config.drone.mass
# # # #         self.gravity = 9.81
# # # #         self.arm_length = config.drone.arm_length
# # # #         self.kf = config.drone.motor_constant
# # # #         self.km = config.drone.moment_constant
# # # #         self.max_rpm = config.drone.max_rpm
# # # #         self.min_rpm = config.drone.min_rpm
# # # #
# # # #         # 悬停油门（预计算）
# # # #         self.hover_thrust = self.mass * self.gravity
# # # #         self.hover_rpm = np.sqrt(self.hover_thrust / (4 * self.kf)) * 60 / (2 * np.pi)
# # # #
# # # #         logger.info(f"四旋翼PID控制器初始化完成 (悬停转速: {self.hover_rpm:.0f} RPM)")
# # # #
# # # #     def compute_control(self, state, target_position: np.ndarray,
# # # #                         target_yaw: float = 0.0, dt: float = 0.01) -> np.ndarray:
# # # #         """
# # # #         计算控制输入
# # # #
# # # #         Args:
# # # #             state: 无人机状态
# # # #             target_position: 目标位置 (NED坐标)
# # # #             target_yaw: 目标偏航角
# # # #             dt: 时间步长
# # # #
# # # #         Returns:
# # # #             电机转速命令 (4,) RPM
# # # #         """
# # # #         # ========== 位置控制 -> 期望速度 ==========
# # # #         desired_velocity = self.pos_controller.update(
# # # #             target=target_position,
# # # #             current=state.position,
# # # #             dt=dt
# # # #         )
# # # #
# # # #         # 限制水平速度
# # # #         vel_xy_norm = np.linalg.norm(desired_velocity[:2])
# # # #         if vel_xy_norm > self.max_velocity:
# # # #             desired_velocity[:2] = desired_velocity[:2] / vel_xy_norm * self.max_velocity
# # # #
# # # #         # ========== 速度控制 -> 期望加速度 ==========
# # # #         desired_acceleration = self.vel_controller.update(
# # # #             target=desired_velocity,
# # # #             current=state.velocity,
# # # #             dt=dt
# # # #         )
# # # #
# # # #         # ========== 计算推力和期望姿态 ==========
# # # #         # 期望加速度 + 重力补偿 (NED坐标系，向下为正)
# # # #         # F/m = a_desired - g (g在NED中为[0, 0, 9.81])
# # # #         thrust_acc = desired_acceleration.copy()
# # # #         thrust_acc[2] -= self.gravity  # 补偿重力
# # # #
# # # #         # 总推力
# # # #         thrust = self.mass * np.linalg.norm(thrust_acc)
# # # #         thrust = np.clip(thrust, 0.1, 4 * self.mass * self.gravity)  # 限制推力范围
# # # #
# # # #         # 从期望加速度计算期望姿态
# # # #         if thrust > 0.1:
# # # #             # 期望的推力方向（归一化）
# # # #             thrust_dir = thrust_acc / np.linalg.norm(thrust_acc) if np.linalg.norm(thrust_acc) > 0.01 else np.array(
# # # #                 [0, 0, -1])
# # # #
# # # #             # 从推力方向计算roll和pitch
# # # #             # 在NED坐标系中，向下推力对应水平飞行
# # # #             desired_roll = np.arcsin(np.clip(
# # # #                 -thrust_dir[1] * np.cos(target_yaw) + thrust_dir[0] * np.sin(target_yaw),
# # # #                 -1, 1
# # # #             ))
# # # #
# # # #             desired_pitch = np.arctan2(
# # # #                 thrust_dir[0] * np.cos(target_yaw) + thrust_dir[1] * np.sin(target_yaw),
# # # #                 -thrust_dir[2]
# # # #             )
# # # #         else:
# # # #             desired_roll = 0.0
# # # #             desired_pitch = 0.0
# # # #
# # # #         # 限制倾斜角
# # # #         desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
# # # #         desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)
# # # #
# # # #         # ========== 姿态控制 -> 期望角速度 ==========
# # # #         current_euler = state.euler_angles
# # # #         target_attitude = np.array([desired_roll, desired_pitch, target_yaw])
# # # #
# # # #         # 计算姿态误差（处理角度跨越）
# # # #         attitude_error = target_attitude - current_euler
# # # #         # 处理yaw角度跨越
# # # #         if attitude_error[2] > np.pi:
# # # #             attitude_error[2] -= 2 * np.pi
# # # #         elif attitude_error[2] < -np.pi:
# # # #             attitude_error[2] += 2 * np.pi
# # # #
# # # #         desired_rates = self.att_controller.update(
# # # #             target=target_attitude,
# # # #             current=current_euler,
# # # #             dt=dt
# # # #         )
# # # #
# # # #         # 限制偏航角速度
# # # #         desired_rates[2] = np.clip(desired_rates[2], -self.max_yaw_rate, self.max_yaw_rate)
# # # #
# # # #         # ========== 角速度控制 -> 力矩 ==========
# # # #         torques = self.rate_controller.update(
# # # #             target=desired_rates,
# # # #             current=state.angular_velocity,
# # # #             dt=dt
# # # #         )
# # # #
# # # #         # ========== 混控：推力和力矩 -> 电机转速 ==========
# # # #         motor_speeds = self._mixer(thrust, torques)
# # # #
# # # #         return motor_speeds
# # # #
# # # #     def _mixer(self, thrust: float, torques: np.ndarray) -> np.ndarray:
# # # #         """
# # # #         混控器：将推力和力矩转换为电机转速
# # # #
# # # #         电机布局 (X型):
# # # #             1 (前右, CCW)    2 (前左, CW)
# # # #                     \    /
# # # #                      \  /
# # # #                       \/
# # # #                       /\
# # # #                      /  \
# # # #             4 (后左, CCW)    3 (后右, CW)
# # # #         """
# # # #         L = self.arm_length / np.sqrt(2)  # X型布局的有效力臂
# # # #
# # # #         # 混控矩阵求解
# # # #         # F = kf * (w1^2 + w2^2 + w3^2 + w4^2)
# # # #         # Mx = kf * L * (w1^2 - w2^2 - w3^2 + w4^2)
# # # #         # My = kf * L * (w1^2 + w2^2 - w3^2 - w4^2)
# # # #         # Mz = km * kf * (-w1^2 + w2^2 - w3^2 + w4^2)
# # # #
# # # #         tau_x = torques[0]
# # # #         tau_y = torques[1]
# # # #         tau_z = torques[2]
# # # #
# # # #         # 求解各电机推力
# # # #         f1 = thrust / 4 + tau_x / (4 * L) + tau_y / (4 * L) - tau_z / (4 * self.km)
# # # #         f2 = thrust / 4 - tau_x / (4 * L) + tau_y / (4 * L) + tau_z / (4 * self.km)
# # # #         f3 = thrust / 4 - tau_x / (4 * L) - tau_y / (4 * L) - tau_z / (4 * self.km)
# # # #         f4 = thrust / 4 + tau_x / (4 * L) - tau_y / (4 * L) + tau_z / (4 * self.km)
# # # #
# # # #         # 推力转转速 (f = kf * omega^2)
# # # #         forces = np.array([f1, f2, f3, f4])
# # # #         forces = np.maximum(forces, 0)  # 推力不能为负
# # # #
# # # #         omega = np.sqrt(forces / self.kf)  # rad/s
# # # #         rpm = omega * 60 / (2 * np.pi)  # 转换为RPM
# # # #
# # # #         # 限制RPM
# # # #         rpm = np.clip(rpm, self.min_rpm, self.max_rpm)
# # # #
# # # #         return rpm
# # # #
# # # #     def reset(self):
# # # #         """重置所有控制器"""
# # # #         self.pos_controller.reset()
# # # #         self.vel_controller.reset()
# # # #         self.att_controller.reset()
# # # #         self.rate_controller.reset()
# # #
# # #
# # # # core/control/pid_controller.py（完全重写修复版）
# # #
# # # """
# # # PID控制器模块
# # # """
# # #
# # # import numpy as np
# # # from dataclasses import dataclass
# # # from typing import Optional
# # # from loguru import logger
# # #
# # #
# # # @dataclass
# # # class PIDGains:
# # #     """PID增益参数"""
# # #     kp: float = 1.0
# # #     ki: float = 0.0
# # #     kd: float = 0.0
# # #
# # #
# # # class PIDController:
# # #     """单轴PID控制器"""
# # #
# # #     def __init__(self, gains: Optional[PIDGains] = None):
# # #         self.gains = gains or PIDGains()
# # #         self.integral = 0.0
# # #         self.last_error = 0.0
# # #         self.last_derivative = 0.0
# # #         self.first_update = True
# # #         self.integral_limit = 10.0
# # #         self.output_limit = float('inf')
# # #         self.derivative_filter = 0.2
# # #
# # #     def update(self, target: float = None, current: float = None,
# # #                error: float = None, dt: float = 0.01) -> float:
# # #         """更新PID控制器"""
# # #         if error is None:
# # #             if target is None or current is None:
# # #                 return 0.0
# # #             error = target - current
# # #
# # #         # 比例项
# # #         p_term = self.gains.kp * error
# # #
# # #         # 积分项（带限幅和抗饱和）
# # #         self.integral += error * dt
# # #         self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
# # #         i_term = self.gains.ki * self.integral
# # #
# # #         # 微分项（带滤波）
# # #         if self.first_update:
# # #             derivative = 0.0
# # #             self.first_update = False
# # #         elif dt > 0:
# # #             raw_derivative = (error - self.last_error) / dt
# # #             derivative = (1 - self.derivative_filter) * self.last_derivative + \
# # #                          self.derivative_filter * raw_derivative
# # #             self.last_derivative = derivative
# # #         else:
# # #             derivative = 0.0
# # #         d_term = self.gains.kd * derivative
# # #
# # #         self.last_error = error
# # #
# # #         # 总输出
# # #         output = p_term + i_term + d_term
# # #         output = np.clip(output, -self.output_limit, self.output_limit)
# # #
# # #         return float(output)
# # #
# # #     def reset(self):
# # #         """重置控制器状态"""
# # #         self.integral = 0.0
# # #         self.last_error = 0.0
# # #         self.last_derivative = 0.0
# # #         self.first_update = True
# # #
# # #     def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
# # #         """设置PID增益"""
# # #         if kp is not None:
# # #             self.gains.kp = kp
# # #         if ki is not None:
# # #             self.gains.ki = ki
# # #         if kd is not None:
# # #             self.gains.kd = kd
# # #
# # #
# # # class PIDController3D:
# # #     """三轴PID控制器"""
# # #
# # #     def __init__(self, gains_x: PIDGains = None, gains_y: PIDGains = None,
# # #                  gains_z: PIDGains = None):
# # #         self.controllers = [
# # #             PIDController(gains_x or PIDGains()),
# # #             PIDController(gains_y or PIDGains()),
# # #             PIDController(gains_z or PIDGains())
# # #         ]
# # #
# # #     def update(self, target: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
# # #         """更新三轴控制器"""
# # #         output = np.zeros(3)
# # #         for i in range(3):
# # #             output[i] = self.controllers[i].update(
# # #                 target=float(target[i]),
# # #                 current=float(current[i]),
# # #                 dt=dt
# # #             )
# # #         return output
# # #
# # #     def reset(self):
# # #         """重置所有控制器"""
# # #         for controller in self.controllers:
# # #             controller.reset()
# # #
# # #     def set_output_limits(self, limit: float):
# # #         """设置输出限幅"""
# # #         for controller in self.controllers:
# # #             controller.output_limit = limit
# # #
# # #     def set_integral_limits(self, limit: float):
# # #         """设置积分限幅"""
# # #         for controller in self.controllers:
# # #             controller.integral_limit = limit
# # #
# # #
# # # class QuadrotorPIDController:
# # #     """四旋翼无人机完整PID控制器（级联控制）- 修复版"""
# # #
# # #     def __init__(self):
# # #         """初始化四旋翼控制器"""
# # #         from utils.config.config_manager import get_config
# # #         config = get_config()
# # #
# # #         # 无人机参数
# # #         self.mass = config.drone.mass
# # #         self.gravity = 9.81
# # #         self.arm_length = config.drone.arm_length
# # #         self.kf = config.drone.motor_constant
# # #         self.km = config.drone.moment_constant
# # #         self.max_rpm = config.drone.max_rpm
# # #         self.min_rpm = config.drone.min_rpm
# # #
# # #         # 控制限制
# # #         self.max_velocity = config.control.max_velocity
# # #         self.max_tilt = config.control.max_tilt_angle
# # #
# # #         # ====== 简化的控制器参数（经过调试） ======
# # #         # 位置 -> 速度
# # #         self.kp_pos = np.array([0.8, 0.8, 1.0])
# # #
# # #         # 速度 -> 加速度
# # #         self.kp_vel = np.array([2.0, 2.0, 2.5])
# # #         self.ki_vel = np.array([0.1, 0.1, 0.2])
# # #         self.kd_vel = np.array([0.5, 0.5, 0.3])
# # #
# # #         # 姿态 -> 角速度
# # #         self.kp_att = np.array([8.0, 8.0, 4.0])
# # #
# # #         # 角速度 -> 力矩
# # #         self.kp_rate = np.array([0.15, 0.15, 0.08])
# # #         self.ki_rate = np.array([0.02, 0.02, 0.01])
# # #         self.kd_rate = np.array([0.01, 0.01, 0.005])
# # #
# # #         # 积分器状态
# # #         self.vel_integral = np.zeros(3)
# # #         self.rate_integral = np.zeros(3)
# # #
# # #         # 上一次误差（用于微分）
# # #         self.last_vel_error = np.zeros(3)
# # #         self.last_rate_error = np.zeros(3)
# # #
# # #         # 积分限制
# # #         self.vel_integral_limit = 5.0
# # #         self.rate_integral_limit = 1.0
# # #
# # #         # 悬停油门
# # #         self.hover_thrust = self.mass * self.gravity
# # #
# # #         # 计算悬停转速
# # #         thrust_per_motor = self.hover_thrust / 4
# # #         omega_hover = np.sqrt(thrust_per_motor / self.kf)  # rad/s
# # #         self.hover_rpm = omega_hover * 60 / (2 * np.pi)
# # #
# # #         logger.info(f"四旋翼PID控制器初始化完成 (悬停转速: {self.hover_rpm:.0f} RPM)")
# # #
# # #     def compute_control(self, state, target_position: np.ndarray,
# # #                         target_yaw: float = 0.0, dt: float = 0.01) -> np.ndarray:
# # #         """
# # #         计算控制输入
# # #
# # #         Args:
# # #             state: 无人机状态
# # #             target_position: 目标位置 (NED坐标)
# # #             target_yaw: 目标偏航角
# # #             dt: 时间步长
# # #
# # #         Returns:
# # #             电机转速命令 (4,) RPM
# # #         """
# # #         # ========== 1. 位置误差 -> 期望速度 ==========
# # #         pos_error = target_position - state.position
# # #         desired_velocity = self.kp_pos * pos_error
# # #
# # #         # 限制水平速度
# # #         vel_xy_norm = np.linalg.norm(desired_velocity[:2])
# # #         if vel_xy_norm > self.max_velocity:
# # #             desired_velocity[:2] *= self.max_velocity / vel_xy_norm
# # #
# # #         # 限制垂直速度
# # #         desired_velocity[2] = np.clip(desired_velocity[2], -self.max_velocity, self.max_velocity)
# # #
# # #         # ========== 2. 速度误差 -> 期望加速度 ==========
# # #         vel_error = desired_velocity - state.velocity
# # #
# # #         # PID计算
# # #         self.vel_integral += vel_error * dt
# # #         self.vel_integral = np.clip(self.vel_integral, -self.vel_integral_limit, self.vel_integral_limit)
# # #
# # #         vel_derivative = (vel_error - self.last_vel_error) / dt if dt > 0 else np.zeros(3)
# # #         self.last_vel_error = vel_error.copy()
# # #
# # #         desired_acceleration = (
# # #                 self.kp_vel * vel_error +
# # #                 self.ki_vel * self.vel_integral +
# # #                 self.kd_vel * vel_derivative
# # #         )
# # #
# # #         # ========== 3. 计算推力和期望姿态 ==========
# # #         # NED坐标系：Z向下为正，重力加速度为 +9.81
# # #         # 需要的推力加速度 = 期望加速度 - 重力加速度
# # #         # 在NED中，向上飞需要产生向上的力（即Z方向的负加速度来抵消重力）
# # #
# # #         # 推力向量 (在世界坐标系)
# # #         thrust_acc = desired_acceleration.copy()
# # #         thrust_acc[2] -= self.gravity  # 补偿重力（在NED中重力向下为正）
# # #
# # #         # 获取当前旋转矩阵（体到世界）
# # #         R = state.rotation_matrix
# # #
# # #         # 计算总推力（推力沿机体Z轴负方向）
# # #         # 世界坐标系中推力向量投影到机体Z轴
# # #         body_z_world = R[:, 2]  # 机体Z轴在世界坐标系中的方向
# # #         thrust = -self.mass * np.dot(thrust_acc, body_z_world)
# # #         thrust = np.clip(thrust, 0.1 * self.hover_thrust, 2.5 * self.hover_thrust)
# # #
# # #         # 计算期望姿态
# # #         if np.linalg.norm(thrust_acc) > 0.1:
# # #             # 期望的推力方向（归一化，指向推力方向的反方向）
# # #             thrust_dir = -thrust_acc / np.linalg.norm(thrust_acc)
# # #
# # #             # 使用简化的姿态计算
# # #             # 假设yaw=0时，期望的roll和pitch
# # #             desired_pitch = np.arcsin(np.clip(-thrust_dir[0], -1, 1))
# # #             desired_roll = np.arctan2(thrust_dir[1], -thrust_dir[2])
# # #         else:
# # #             desired_roll = 0.0
# # #             desired_pitch = 0.0
# # #
# # #         # 限制倾斜角
# # #         desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
# # #         desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)
# # #
# # #         # ========== 4. 姿态误差 -> 期望角速度 ==========
# # #         current_euler = state.euler_angles
# # #
# # #         # 姿态误差
# # #         roll_error = desired_roll - current_euler[0]
# # #         pitch_error = desired_pitch - current_euler[1]
# # #         yaw_error = target_yaw - current_euler[2]
# # #
# # #         # 处理yaw角度跨越
# # #         while yaw_error > np.pi:
# # #             yaw_error -= 2 * np.pi
# # #         while yaw_error < -np.pi:
# # #             yaw_error += 2 * np.pi
# # #
# # #         attitude_error = np.array([roll_error, pitch_error, yaw_error])
# # #         desired_rates = self.kp_att * attitude_error
# # #
# # #         # 限制角速度
# # #         max_rate = 3.0  # rad/s
# # #         desired_rates = np.clip(desired_rates, -max_rate, max_rate)
# # #
# # #         # ========== 5. 角速度误差 -> 力矩 ==========
# # #         rate_error = desired_rates - state.angular_velocity
# # #
# # #         self.rate_integral += rate_error * dt
# # #         self.rate_integral = np.clip(self.rate_integral, -self.rate_integral_limit, self.rate_integral_limit)
# # #
# # #         rate_derivative = (rate_error - self.last_rate_error) / dt if dt > 0 else np.zeros(3)
# # #         self.last_rate_error = rate_error.copy()
# # #
# # #         torques = (
# # #                 self.kp_rate * rate_error +
# # #                 self.ki_rate * self.rate_integral +
# # #                 self.kd_rate * rate_derivative
# # #         )
# # #
# # #         # ========== 6. 混控：推力和力矩 -> 电机转速 ==========
# # #         motor_speeds = self._mixer(thrust, torques)
# # #
# # #         return motor_speeds
# # #
# # #     def _mixer(self, thrust: float, torques: np.ndarray) -> np.ndarray:
# # #         """
# # #         混控器：将推力和力矩转换为电机转速
# # #
# # #         电机布局 (X型，从上方看):
# # #             前
# # #           1   2      (1: 前右CW, 2: 前左CCW)
# # #             X
# # #           4   3      (4: 后左CW, 3: 后右CCW)
# # #             后
# # #
# # #         注意：这里的CW/CCW会影响偏航力矩方向
# # #         """
# # #         L = self.arm_length * 0.707  # X型布局的有效力臂 (cos45°)
# # #
# # #         tau_x = torques[0]  # roll力矩
# # #         tau_y = torques[1]  # pitch力矩
# # #         tau_z = torques[2]  # yaw力矩
# # #
# # #         # 混控矩阵求解各电机推力
# # #         # 推力分配：
# # #         # F1 (前右): thrust/4 + roll/(4L) + pitch/(4L) - yaw/(4km)
# # #         # F2 (前左): thrust/4 - roll/(4L) + pitch/(4L) + yaw/(4km)
# # #         # F3 (后右): thrust/4 + roll/(4L) - pitch/(4L) + yaw/(4km)
# # #         # F4 (后左): thrust/4 - roll/(4L) - pitch/(4L) - yaw/(4km)
# # #
# # #         f1 = thrust / 4 + tau_x / (4 * L) + tau_y / (4 * L) - tau_z / (4 * self.km)
# # #         f2 = thrust / 4 - tau_x / (4 * L) + tau_y / (4 * L) + tau_z / (4 * self.km)
# # #         f3 = thrust / 4 + tau_x / (4 * L) - tau_y / (4 * L) + tau_z / (4 * self.km)
# # #         f4 = thrust / 4 - tau_x / (4 * L) - tau_y / (4 * L) - tau_z / (4 * self.km)
# # #
# # #         forces = np.array([f1, f2, f3, f4])
# # #         forces = np.maximum(forces, 0)  # 推力不能为负
# # #
# # #         # 推力转转速: F = kf * omega^2
# # #         omega = np.sqrt(forces / self.kf)  # rad/s
# # #         rpm = omega * 60 / (2 * np.pi)  # 转为RPM
# # #
# # #         # 限制RPM
# # #         rpm = np.clip(rpm, self.min_rpm, self.max_rpm)
# # #
# # #         return rpm
# # #
# # #     def reset(self):
# # #         """重置控制器状态"""
# # #         self.vel_integral = np.zeros(3)
# # #         self.rate_integral = np.zeros(3)
# # #         self.last_vel_error = np.zeros(3)
# # #         self.last_rate_error = np.zeros(3)
# #
# #
# # # core/control/pid_controller.py（彻底重写 - 简化稳定版）
# #
# # """
# # PID控制器模块 - 简化稳定版
# # """
# #
# # import numpy as np
# # from dataclasses import dataclass
# # from typing import Optional
# # from loguru import logger
# #
# #
# # @dataclass
# # class PIDGains:
# #     """PID增益参数"""
# #     kp: float = 1.0
# #     ki: float = 0.0
# #     kd: float = 0.0
# #
# #
# # class PIDController:
# #     """单轴PID控制器"""
# #
# #     def __init__(self, kp=1.0, ki=0.0, kd=0.0):
# #         self.kp = kp
# #         self.ki = ki
# #         self.kd = kd
# #         self.integral = 0.0
# #         self.last_error = 0.0
# #         self.integral_limit = 10.0
# #         self.output_limit = float('inf')
# #
# #     def update(self, error: float, dt: float) -> float:
# #         """更新PID"""
# #         # 比例
# #         p = self.kp * error
# #
# #         # 积分
# #         self.integral += error * dt
# #         self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
# #         i = self.ki * self.integral
# #
# #         # 微分
# #         d = self.kd * (error - self.last_error) / dt if dt > 0 else 0
# #         self.last_error = error
# #
# #         output = p + i + d
# #         return np.clip(output, -self.output_limit, self.output_limit)
# #
# #     def reset(self):
# #         self.integral = 0.0
# #         self.last_error = 0.0
# #
# #
# # class SimpleQuadrotorController:
# #     """
# #     简化的四旋翼控制器
# #     使用最基本的控制策略确保稳定
# #     """
# #
# #     def __init__(self):
# #         from utils.config.config_manager import get_config
# #         config = get_config()
# #
# #         # 物理参数
# #         self.mass = config.drone.mass
# #         self.gravity = 9.81
# #         self.arm_length = config.drone.arm_length
# #         self.kf = config.drone.motor_constant
# #         self.km = config.drone.moment_constant
# #         self.max_rpm = config.drone.max_rpm
# #         self.min_rpm = config.drone.min_rpm
# #
# #         # 悬停推力和转速
# #         self.hover_thrust = self.mass * self.gravity
# #         thrust_per_motor = self.hover_thrust / 4
# #         omega_hover = np.sqrt(thrust_per_motor / self.kf)
# #         self.hover_rpm = omega_hover * 60 / (2 * np.pi)
# #
# #         # ============ 简单的PID控制器 ============
# #         # 高度控制 (z方向，NED坐标系z向下为正)
# #         self.alt_pid = PIDController(kp=2.0, ki=0.5, kd=1.5)
# #         self.alt_pid.integral_limit = 5.0
# #
# #         # 水平位置控制
# #         self.pos_x_pid = PIDController(kp=0.5, ki=0.0, kd=0.3)
# #         self.pos_y_pid = PIDController(kp=0.5, ki=0.0, kd=0.3)
# #
# #         # 水平速度控制
# #         self.vel_x_pid = PIDController(kp=1.5, ki=0.1, kd=0.2)
# #         self.vel_y_pid = PIDController(kp=1.5, ki=0.1, kd=0.2)
# #
# #         # 姿态控制
# #         self.roll_pid = PIDController(kp=6.0, ki=0.0, kd=0.5)
# #         self.pitch_pid = PIDController(kp=6.0, ki=0.0, kd=0.5)
# #         self.yaw_pid = PIDController(kp=3.0, ki=0.0, kd=0.3)
# #
# #         # 角速度控制
# #         self.roll_rate_pid = PIDController(kp=0.15, ki=0.0, kd=0.01)
# #         self.pitch_rate_pid = PIDController(kp=0.15, ki=0.0, kd=0.01)
# #         self.yaw_rate_pid = PIDController(kp=0.1, ki=0.0, kd=0.005)
# #
# #         # 控制限制
# #         self.max_tilt = 0.3  # ~17度
# #         self.max_velocity = 3.0  # m/s
# #
# #         logger.info(f"简化控制器初始化 (悬停RPM: {self.hover_rpm:.0f})")
# #
# #     def compute_control(self, state, target_position: np.ndarray,
# #                         target_yaw: float = 0.0, dt: float = 0.01) -> np.ndarray:
# #         """计算控制输入"""
# #
# #         pos = state.position
# #         vel = state.velocity
# #         euler = state.euler_angles  # [roll, pitch, yaw]
# #         omega = state.angular_velocity
# #
# #         # ============ 1. 高度控制 ============
# #         # NED坐标系: z向下为正，所以目标z是负的（如-10表示10米高）
# #         # 误差 = 目标 - 当前，如果目标z=-10，当前z=0，误差=-10
# #         # 我们需要向上飞（减小z），所以需要增加推力
# #         alt_error = target_position[2] - pos[2]  # 负值表示需要向上
# #
# #         # 高度PID输出一个推力调整量
# #         thrust_adjustment = -self.alt_pid.update(alt_error, dt)  # 取负因为z向下
# #
# #         # 计算总推力
# #         thrust = self.hover_thrust + thrust_adjustment * self.mass
# #         thrust = np.clip(thrust, 0.2 * self.hover_thrust, 2.0 * self.hover_thrust)
# #
# #         # ============ 2. 水平位置控制 -> 期望速度 ============
# #         pos_error_x = target_position[0] - pos[0]
# #         pos_error_y = target_position[1] - pos[1]
# #
# #         desired_vel_x = self.pos_x_pid.update(pos_error_x, dt)
# #         desired_vel_y = self.pos_y_pid.update(pos_error_y, dt)
# #
# #         # 限制速度
# #         desired_vel_x = np.clip(desired_vel_x, -self.max_velocity, self.max_velocity)
# #         desired_vel_y = np.clip(desired_vel_y, -self.max_velocity, self.max_velocity)
# #
# #         # ============ 3. 速度控制 -> 期望倾斜角 ============
# #         vel_error_x = desired_vel_x - vel[0]
# #         vel_error_y = desired_vel_y - vel[1]
# #
# #         # 速度误差转换为期望加速度，再转为期望倾斜角
# #         # 在NED坐标系中：
# #         # - 向北飞（+x）需要向前倾斜（+pitch）
# #         # - 向东飞（+y）需要向右倾斜（+roll）
# #         accel_cmd_x = self.vel_x_pid.update(vel_error_x, dt)
# #         accel_cmd_y = self.vel_y_pid.update(vel_error_y, dt)
# #
# #         # 加速度到倾斜角的转换（小角度近似）
# #         # pitch产生x方向加速度，roll产生y方向加速度
# #         desired_pitch = np.arctan2(accel_cmd_x, self.gravity)
# #         desired_roll = np.arctan2(-accel_cmd_y, self.gravity)  # 注意符号
# #
# #         # 限制倾斜角
# #         desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
# #         desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)
# #
# #         # ============ 4. 姿态控制 -> 期望角速度 ============
# #         roll_error = desired_roll - euler[0]
# #         pitch_error = desired_pitch - euler[1]
# #         yaw_error = target_yaw - euler[2]
# #
# #         # 处理yaw角度跨越
# #         while yaw_error > np.pi:
# #             yaw_error -= 2 * np.pi
# #         while yaw_error < -np.pi:
# #             yaw_error += 2 * np.pi
# #
# #         desired_roll_rate = self.roll_pid.update(roll_error, dt)
# #         desired_pitch_rate = self.pitch_pid.update(pitch_error, dt)
# #         desired_yaw_rate = self.yaw_pid.update(yaw_error, dt)
# #
# #         # ============ 5. 角速度控制 -> 力矩 ============
# #         roll_rate_error = desired_roll_rate - omega[0]
# #         pitch_rate_error = desired_pitch_rate - omega[1]
# #         yaw_rate_error = desired_yaw_rate - omega[2]
# #
# #         tau_roll = self.roll_rate_pid.update(roll_rate_error, dt)
# #         tau_pitch = self.pitch_rate_pid.update(pitch_rate_error, dt)
# #         tau_yaw = self.yaw_rate_pid.update(yaw_rate_error, dt)
# #
# #         # ============ 6. 混控 ============
# #         motor_speeds = self._mix(thrust, tau_roll, tau_pitch, tau_yaw)
# #
# #         return motor_speeds
# #
# #     def _mix(self, thrust: float, tau_roll: float, tau_pitch: float, tau_yaw: float) -> np.ndarray:
# #         """
# #         混控器
# #
# #         电机布局 (从上方看, X型):
# #                前
# #             0     1
# #               \ /
# #                X
# #               / \
# #             3     2
# #                后
# #
# #         0: 前右 (CCW, 产生+yaw力矩)
# #         1: 前左 (CW, 产生-yaw力矩)
# #         2: 后右 (CW, 产生-yaw力矩)
# #         3: 后左 (CCW, 产生+yaw力矩)
# #         """
# #         L = self.arm_length * 0.707  # cos(45°)
# #
# #         # 计算每个电机的推力
# #         # 推力分配矩阵的逆
# #         f0 = thrust / 4 + tau_roll / (4 * L) + tau_pitch / (4 * L) - tau_yaw / (4 * self.km)
# #         f1 = thrust / 4 - tau_roll / (4 * L) + tau_pitch / (4 * L) + tau_yaw / (4 * self.km)
# #         f2 = thrust / 4 - tau_roll / (4 * L) - tau_pitch / (4 * L) - tau_yaw / (4 * self.km)
# #         f3 = thrust / 4 + tau_roll / (4 * L) - tau_pitch / (4 * L) + tau_yaw / (4 * self.km)
# #
# #         forces = np.array([f0, f1, f2, f3])
# #         forces = np.maximum(forces, 0)
# #
# #         # 推力转RPM: F = kf * (omega)^2, omega in rad/s
# #         omega_rad = np.sqrt(forces / self.kf)
# #         rpm = omega_rad * 60 / (2 * np.pi)
# #
# #         # 限制
# #         rpm = np.clip(rpm, self.min_rpm, self.max_rpm)
# #
# #         return rpm
# #
# #     def reset(self):
# #         """重置所有控制器"""
# #         self.alt_pid.reset()
# #         self.pos_x_pid.reset()
# #         self.pos_y_pid.reset()
# #         self.vel_x_pid.reset()
# #         self.vel_y_pid.reset()
# #         self.roll_pid.reset()
# #         self.pitch_pid.reset()
# #         self.yaw_pid.reset()
# #         self.roll_rate_pid.reset()
# #         self.pitch_rate_pid.reset()
# #         self.yaw_rate_pid.reset()
# #
# #
# # # 保持兼容性的别名
# # QuadrotorPIDController = SimpleQuadrotorController
#
#
# # core/control/pid_controller.py（完全重写 - 方向修正版）
#
# """
# PID控制器模块 - 方向修正版
# """
#
# import numpy as np
# from dataclasses import dataclass
# from loguru import logger
#
#
# @dataclass
# class PIDGains:
#     """PID增益参数"""
#     kp: float = 1.0
#     ki: float = 0.0
#     kd: float = 0.0
#
#
# class PIDController:
#     """单轴PID控制器"""
#
#     def __init__(self, kp=1.0, ki=0.0, kd=0.0):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.integral = 0.0
#         self.last_error = 0.0
#         self.integral_limit = 10.0
#
#     def update(self, error: float, dt: float) -> float:
#         p = self.kp * error
#
#         self.integral += error * dt
#         self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
#         i = self.ki * self.integral
#
#         d = self.kd * (error - self.last_error) / dt if dt > 1e-6 else 0
#         self.last_error = error
#
#         return p + i + d
#
#     def reset(self):
#         self.integral = 0.0
#         self.last_error = 0.0
#
#
# class SimpleQuadrotorController:
#     """
#     简化四旋翼控制器 - 方向修正版
#
#     坐标系约定 (NED):
#     - X: 北向为正
#     - Y: 东向为正
#     - Z: 向下为正（高度为负值）
#
#     姿态约定:
#     - Roll (绕X轴): 右倾为正 -> 产生+Y方向加速度
#     - Pitch (绕Y轴): 抬头为正 -> 产生-X方向加速度
#     - Yaw (绕Z轴): 顺时针为正（从上看）
#     """
#
#     def __init__(self):
#         from utils.config.config_manager import get_config
#         config = get_config()
#
#         # 物理参数
#         self.mass = config.drone.mass
#         self.gravity = 9.81
#         self.arm_length = config.drone.arm_length
#         self.kf = config.drone.motor_constant
#         self.km = config.drone.moment_constant
#         self.max_rpm = config.drone.max_rpm
#         self.min_rpm = config.drone.min_rpm
#
#         # 悬停转速计算
#         self.hover_thrust = self.mass * self.gravity
#         thrust_per_motor = self.hover_thrust / 4
#         omega_hover = np.sqrt(thrust_per_motor / self.kf)
#         self.hover_rpm = omega_hover * 60 / (2 * np.pi)
#
#         # ============ 控制参数（经过调整） ============
#         # 高度控制
#         self.z_pid = PIDController(kp=3.0, ki=0.8, kd=2.0)
#         self.z_pid.integral_limit = 3.0
#
#         # 水平位置控制 (位置误差 -> 期望速度)
#         self.pos_gain = 0.6  # 简单比例控制
#         self.max_vel = 2.0  # 最大速度 m/s
#
#         # 水平速度控制 (速度误差 -> 期望倾斜角)
#         self.vx_pid = PIDController(kp=0.15, ki=0.02, kd=0.05)
#         self.vy_pid = PIDController(kp=0.15, ki=0.02, kd=0.05)
#         self.vx_pid.integral_limit = 0.3
#         self.vy_pid.integral_limit = 0.3
#
#         # 姿态控制 (姿态误差 -> 期望角速度)
#         self.roll_pid = PIDController(kp=8.0, ki=0.0, kd=1.0)
#         self.pitch_pid = PIDController(kp=8.0, ki=0.0, kd=1.0)
#         self.yaw_pid = PIDController(kp=4.0, ki=0.0, kd=0.5)
#
#         # 角速度控制 (角速度误差 -> 力矩)
#         self.roll_rate_pid = PIDController(kp=0.12, ki=0.01, kd=0.005)
#         self.pitch_rate_pid = PIDController(kp=0.12, ki=0.01, kd=0.005)
#         self.yaw_rate_pid = PIDController(kp=0.08, ki=0.005, kd=0.002)
#
#         # 限制
#         self.max_tilt = 0.25  # ~14度，保守值
#         self.max_rate = 2.0  # rad/s
#
#         logger.info(f"控制器初始化完成 (悬停RPM: {self.hover_rpm:.0f})")
#
#     def compute_control(self, state, target_position: np.ndarray,
#                         target_yaw: float = 0.0, dt: float = 0.01) -> np.ndarray:
#         """计算控制输入"""
#
#         pos = state.position  # [x, y, z] NED
#         vel = state.velocity  # [vx, vy, vz]
#         euler = state.euler_angles  # [roll, pitch, yaw]
#         omega = state.angular_velocity  # [p, q, r]
#
#         # ============ 1. 高度控制（Z轴） ============
#         # NED坐标系: z向下为正，目标z<0表示在空中
#         # 误差 = 目标z - 当前z
#         # 如果目标z=-15(15米高)，当前z=0(地面)，误差=-15
#         # 需要向上飞(减小z)，所以需要增加推力
#         z_error = target_position[2] - pos[2]
#
#         # PID输出: 负误差应该产生正的推力调整（向上）
#         # z_error < 0 时需要向上飞，thrust_adj应该 > 0
#         thrust_adj = -self.z_pid.update(z_error, dt)
#
#         # 总推力
#         thrust = self.hover_thrust + thrust_adj * self.mass
#         thrust = np.clip(thrust, 0.3 * self.hover_thrust, 2.0 * self.hover_thrust)
#
#         # ============ 2. 水平位置控制 ============
#         # 位置误差
#         x_error = target_position[0] - pos[0]  # 北向误差
#         y_error = target_position[1] - pos[1]  # 东向误差
#
#         # 期望速度 = 位置误差 × 增益
#         desired_vx = np.clip(self.pos_gain * x_error, -self.max_vel, self.max_vel)
#         desired_vy = np.clip(self.pos_gain * y_error, -self.max_vel, self.max_vel)
#
#         # ============ 3. 速度控制 -> 期望倾斜角 ============
#         vx_error = desired_vx - vel[0]
#         vy_error = desired_vy - vel[1]
#
#         # 速度PID输出期望倾斜角
#         # 关键修正：
#         # - 要向+X(北)飞，需要+pitch(机头抬起会让机体向后倾，推力向前)
#         #   不对！pitch正表示绕Y轴正转，右手定则：机头向上
#         #   机头向上时，推力向后，所以向北飞需要pitch为负（低头）
#         # - 要向+Y(东)飞，需要+roll(向右倾斜)
#
#         # 重新分析：
#         # 在NED坐标系，推力沿机体-Z轴（向上）
#         # pitch > 0: 绕Y轴正转 = 机头上仰 = 推力向后（-X）
#         # 所以向+X飞需要 pitch < 0
#         # roll > 0: 绕X轴正转 = 右翼下沉 = 推力向左（-Y）
#         # 所以向+Y飞需要 roll < 0
#
#         # 这样分析更简单：
#         # 期望加速度 ax = g * tan(pitch) （小角度下 ≈ g * pitch）
#         # 但pitch正导致向-X加速，所以 ax = -g * pitch
#         # 因此 pitch = -ax/g
#         # 类似地 ay = -g * roll（roll正导致-Y加速）
#         # 因此 roll = -ay/g
#
#         # 速度控制器输出期望加速度
#         ax_cmd = self.vx_pid.update(vx_error, dt)
#         ay_cmd = self.vy_pid.update(vy_error, dt)
#
#         # 转换为期望姿态角
#         desired_pitch = -ax_cmd  # 已经是角度量级
#         desired_roll = -ay_cmd
#
#         # 限制倾斜角
#         desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
#         desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)
#
#         # ============ 4. 姿态控制 ============
#         roll_error = desired_roll - euler[0]
#         pitch_error = desired_pitch - euler[1]
#         yaw_error = target_yaw - euler[2]
#
#         # 处理yaw角度跨越
#         while yaw_error > np.pi:
#             yaw_error -= 2 * np.pi
#         while yaw_error < -np.pi:
#             yaw_error += 2 * np.pi
#
#         desired_p = self.roll_pid.update(roll_error, dt)
#         desired_q = self.pitch_pid.update(pitch_error, dt)
#         desired_r = self.yaw_pid.update(yaw_error, dt)
#
#         # 限制角速度
#         desired_p = np.clip(desired_p, -self.max_rate, self.max_rate)
#         desired_q = np.clip(desired_q, -self.max_rate, self.max_rate)
#         desired_r = np.clip(desired_r, -self.max_rate, self.max_rate)
#
#         # ============ 5. 角速度控制 ============
#         p_error = desired_p - omega[0]
#         q_error = desired_q - omega[1]
#         r_error = desired_r - omega[2]
#
#         tau_x = self.roll_rate_pid.update(p_error, dt)
#         tau_y = self.pitch_rate_pid.update(q_error, dt)
#         tau_z = self.yaw_rate_pid.update(r_error, dt)
#
#         # ============ 6. 混控 ============
#         motor_speeds = self._mix(thrust, tau_x, tau_y, tau_z)
#
#         return motor_speeds
#
#     def _mix(self, thrust: float, tau_x: float, tau_y: float, tau_z: float) -> np.ndarray:
#         """
#         混控器: 推力和力矩 -> 电机转速
#
#         电机布局 (X型, 从上往下看):
#
#                前(+X)
#             0     1
#               \ /
#                X
#               / \
#             3     2
#                后
#
#         0: 前右, CCW (产生+yaw力矩)
#         1: 前左, CW  (产生-yaw力矩)
#         2: 后右, CW  (产生-yaw力矩)
#         3: 后左, CCW (产生+yaw力矩)
#
#         力矩方向:
#         - tau_x (roll): +时右翼下沉 -> 电机0,2推力增加, 电机1,3减少
#         - tau_y (pitch): +时机头上仰 -> 电机0,1推力增加, 电机2,3减少
#         - tau_z (yaw): +时顺时针 -> CCW电机(0,3)增加, CW电机(1,2)减少
#         """
#         L = self.arm_length * 0.707  # 有效力臂
#
#         # 各电机推力分配
#         f0 = thrust / 4 + tau_x / (4 * L) + tau_y / (4 * L) + tau_z / (4 * self.km)
#         f1 = thrust / 4 - tau_x / (4 * L) + tau_y / (4 * L) - tau_z / (4 * self.km)
#         f2 = thrust / 4 - tau_x / (4 * L) - tau_y / (4 * L) + tau_z / (4 * self.km)
#         f3 = thrust / 4 + tau_x / (4 * L) - tau_y / (4 * L) - tau_z / (4 * self.km)
#
#         forces = np.array([f0, f1, f2, f3])
#         forces = np.maximum(forces, 0)
#
#         # 推力转RPM
#         omega_rad = np.sqrt(forces / self.kf)
#         rpm = omega_rad * 60 / (2 * np.pi)
#         rpm = np.clip(rpm, self.min_rpm, self.max_rpm)
#
#         return rpm
#
#     def reset(self):
#         """重置控制器"""
#         self.z_pid.reset()
#         self.vx_pid.reset()
#         self.vy_pid.reset()
#         self.roll_pid.reset()
#         self.pitch_pid.reset()
#         self.yaw_pid.reset()
#         self.roll_rate_pid.reset()
#         self.pitch_rate_pid.reset()
#         self.yaw_rate_pid.reset()
#
#
# # 兼容别名
# QuadrotorPIDController = SimpleQuadrotorController


# core/control/pid_controller.py（Y轴方向修正）

"""
PID控制器模块 - Y轴方向修正版
"""

import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class PIDGains:
    """PID增益参数"""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0


class PIDController:
    """单轴PID控制器"""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0
        self.integral_limit = 10.0

    def update(self, error: float, dt: float) -> float:
        p = self.kp * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self.integral

        d = self.kd * (error - self.last_error) / dt if dt > 1e-6 else 0
        self.last_error = error

        return p + i + d

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0


class SimpleQuadrotorController:
    """
    简化四旋翼控制器 - 方向完全修正版

    NED坐标系:
    - X: 北向为正
    - Y: 东向为正
    - Z: 向下为正

    姿态定义 (右手定则):
    - Roll (φ): 绕X轴旋转，正值=右翼下沉
    - Pitch (θ): 绕Y轴旋转，正值=机头上仰
    - Yaw (ψ): 绕Z轴旋转，正值=顺时针(从上看)

    关键物理关系:
    - Roll正 -> 右翼下沉 -> 推力向左倾斜 -> 向+Y(东)方向加速
    - Pitch正 -> 机头上仰 -> 推力向后倾斜 -> 向-X(南)方向加速
    """

    def __init__(self):
        from utils.config.config_manager import get_config
        config = get_config()

        # 物理参数
        self.mass = config.drone.mass
        self.gravity = 9.81
        self.arm_length = config.drone.arm_length
        self.kf = config.drone.motor_constant
        self.km = config.drone.moment_constant
        self.max_rpm = config.drone.max_rpm
        self.min_rpm = config.drone.min_rpm

        # 悬停转速
        self.hover_thrust = self.mass * self.gravity
        thrust_per_motor = self.hover_thrust / 4
        omega_hover = np.sqrt(thrust_per_motor / self.kf)
        self.hover_rpm = omega_hover * 60 / (2 * np.pi)

        # ============ 控制参数 ============
        # 高度控制
        self.z_pid = PIDController(kp=2.5, ki=0.5, kd=1.5)
        self.z_pid.integral_limit = 3.0

        # 水平位置 -> 速度
        self.pos_gain = 0.5
        self.max_vel = 3.0

        # 水平速度 -> 倾斜角
        self.vx_pid = PIDController(kp=0.12, ki=0.015, kd=0.04)
        self.vy_pid = PIDController(kp=0.12, ki=0.015, kd=0.04)
        self.vx_pid.integral_limit = 0.2
        self.vy_pid.integral_limit = 0.2

        # 姿态 -> 角速度
        self.roll_pid = PIDController(kp=6.0, ki=0.0, kd=0.8)
        self.pitch_pid = PIDController(kp=6.0, ki=0.0, kd=0.8)
        self.yaw_pid = PIDController(kp=3.0, ki=0.0, kd=0.3)

        # 角速度 -> 力矩
        self.p_pid = PIDController(kp=0.1, ki=0.01, kd=0.003)
        self.q_pid = PIDController(kp=0.1, ki=0.01, kd=0.003)
        self.r_pid = PIDController(kp=0.06, ki=0.005, kd=0.002)

        # 限制
        self.max_tilt = 0.3  # ~17度
        self.max_rate = 3.0  # rad/s

        logger.info(f"控制器初始化 (悬停RPM: {self.hover_rpm:.0f})")

    def compute_control(self, state, target_position: np.ndarray,
                        target_yaw: float = 0.0, dt: float = 0.01) -> np.ndarray:
        """计算控制输入"""

        pos = state.position
        vel = state.velocity
        euler = state.euler_angles  # [roll, pitch, yaw]
        omega = state.angular_velocity  # [p, q, r]

        # ============ 1. 高度控制 ============
        z_error = target_position[2] - pos[2]
        # z_error < 0 表示需要向上飞（z变小）
        # 需要增加推力，所以取负
        thrust_adj = -self.z_pid.update(z_error, dt)

        thrust = self.hover_thrust + thrust_adj * self.mass
        thrust = np.clip(thrust, 0.3 * self.hover_thrust, 2.0 * self.hover_thrust)

        # ============ 2. 水平位置 -> 期望速度 ============
        x_error = target_position[0] - pos[0]
        y_error = target_position[1] - pos[1]

        desired_vx = np.clip(self.pos_gain * x_error, -self.max_vel, self.max_vel)
        desired_vy = np.clip(self.pos_gain * y_error, -self.max_vel, self.max_vel)

        # ============ 3. 速度误差 -> 期望姿态角 ============
        vx_error = desired_vx - vel[0]
        vy_error = desired_vy - vel[1]

        # 速度控制器输出
        ax_cmd = self.vx_pid.update(vx_error, dt)
        ay_cmd = self.vy_pid.update(vy_error, dt)

        # 关键：姿态角与加速度的正确关系
        #
        # 1. 要向+X(北)飞，需要产生+X方向的加速度
        #    pitch负(机头低头) -> 推力向前 -> +X加速
        #    所以: desired_pitch = -ax_cmd
        #
        # 2. 要向+Y(东)飞，需要产生+Y方向的加速度
        #    roll正(右翼下沉) -> 推力向右 -> +Y加速
        #    所以: desired_roll = +ay_cmd

        desired_pitch = -ax_cmd  # 向北飞需要低头（负pitch）
        desired_roll = ay_cmd  # 向东飞需要右倾（正roll）

        # 限制倾斜角
        desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)

        # ============ 4. 姿态控制 ============
        roll_error = desired_roll - euler[0]
        pitch_error = desired_pitch - euler[1]
        yaw_error = target_yaw - euler[2]

        # Yaw角度归一化
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi

        desired_p = np.clip(self.roll_pid.update(roll_error, dt), -self.max_rate, self.max_rate)
        desired_q = np.clip(self.pitch_pid.update(pitch_error, dt), -self.max_rate, self.max_rate)
        desired_r = np.clip(self.yaw_pid.update(yaw_error, dt), -self.max_rate, self.max_rate)

        # ============ 5. 角速度控制 ============
        p_error = desired_p - omega[0]
        q_error = desired_q - omega[1]
        r_error = desired_r - omega[2]

        tau_x = self.p_pid.update(p_error, dt)
        tau_y = self.q_pid.update(q_error, dt)
        tau_z = self.r_pid.update(r_error, dt)

        # ============ 6. 混控 ============
        motor_speeds = self._mix(thrust, tau_x, tau_y, tau_z)

        return motor_speeds

    def _mix(self, thrust: float, tau_x: float, tau_y: float, tau_z: float) -> np.ndarray:
        """
        混控器

        电机布局 (X型, 从上看):
               前(+X/北)
            0     1
              \ /
               X
              / \
            3     2
               后

        0: 前右 CCW
        1: 前左 CW
        2: 后右 CW
        3: 后左 CCW
        """
        L = self.arm_length * 0.707

        # 推力分配
        f0 = thrust / 4 + tau_x / (4 * L) + tau_y / (4 * L) + tau_z / (4 * self.km)
        f1 = thrust / 4 - tau_x / (4 * L) + tau_y / (4 * L) - tau_z / (4 * self.km)
        f2 = thrust / 4 - tau_x / (4 * L) - tau_y / (4 * L) + tau_z / (4 * self.km)
        f3 = thrust / 4 + tau_x / (4 * L) - tau_y / (4 * L) - tau_z / (4 * self.km)

        forces = np.maximum(np.array([f0, f1, f2, f3]), 0)

        omega_rad = np.sqrt(forces / self.kf)
        rpm = omega_rad * 60 / (2 * np.pi)
        rpm = np.clip(rpm, self.min_rpm, self.max_rpm)

        return rpm

    def reset(self):
        """重置"""
        for pid in [self.z_pid, self.vx_pid, self.vy_pid,
                    self.roll_pid, self.pitch_pid, self.yaw_pid,
                    self.p_pid, self.q_pid, self.r_pid]:
            pid.reset()


# 兼容别名
QuadrotorPIDController = SimpleQuadrotorController
