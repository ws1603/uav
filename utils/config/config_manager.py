# # utils/config/config_manager.py（完整修复版）
#
# """
# 配置管理模块
# """
#
# import os
# from pathlib import Path
# from dataclasses import dataclass, field
# from typing import Tuple, Optional, List
# import numpy as np
#
# try:
#     import yaml
# except ImportError:
#     yaml = None
#
# from loguru import logger
#
#
# @dataclass
# class DroneParams:
#     """无人机参数（兼容旧接口）"""
#     mass: float = 1.5
#     arm_length: float = 0.225
#     max_rpm: float = 12000
#     motor_constant: float = 8.54858e-06
#     moment_constant: float = 0.016
#     inertia_xx: float = 0.029125
#     inertia_yy: float = 0.029125
#     inertia_zz: float = 0.055225
#     drag_coefficient: float = 0.1
#
#     @property
#     def inertia(self) -> np.ndarray:
#         """转动惯量矩阵"""
#         return np.diag([self.inertia_xx, self.inertia_yy, self.inertia_zz])
#
#
# # 别名，保持兼容性
# DroneConfig = DroneParams
#
#
# @dataclass
# class ControlConfig:
#     """控制配置"""
#     # 位置控制
#     position_kp: Tuple[float, float, float] = (2.0, 2.0, 4.0)
#     position_ki: Tuple[float, float, float] = (0.1, 0.1, 0.2)
#     position_kd: Tuple[float, float, float] = (1.5, 1.5, 2.0)
#
#     # 速度控制
#     velocity_kp: Tuple[float, float, float] = (4.0, 4.0, 6.0)
#     velocity_ki: Tuple[float, float, float] = (0.5, 0.5, 0.8)
#     velocity_kd: Tuple[float, float, float] = (0.5, 0.5, 0.5)
#
#     # 姿态控制
#     attitude_kp: Tuple[float, float, float] = (8.0, 8.0, 4.0)
#     attitude_ki: Tuple[float, float, float] = (0.5, 0.5, 0.2)
#     attitude_kd: Tuple[float, float, float] = (2.0, 2.0, 1.0)
#
#     # 角速度控制
#     rate_kp: Tuple[float, float, float] = (0.15, 0.15, 0.1)
#     rate_ki: Tuple[float, float, float] = (0.01, 0.01, 0.01)
#     rate_kd: Tuple[float, float, float] = (0.01, 0.01, 0.01)
#
#     # 限制
#     max_velocity: float = 15.0
#     max_acceleration: float = 5.0
#     max_tilt_angle: float = 0.5236
#     max_yaw_rate: float = 1.5708
#
#
# @dataclass
# class SimulationConfig:
#     """仿真配置"""
#     dt: float = 0.002
#     realtime_factor: float = 1.0
#     gravity: float = 9.81
#     air_density: float = 1.225
#
#
# @dataclass
# class VisualizationConfig:
#     """可视化配置"""
#     update_rate: int = 30
#     trail_length: int = 500
#     grid_size: int = 100
#     show_axes: bool = True
#     show_grid: bool = True
#
#
# @dataclass
# class SystemConfig:
#     """系统总配置"""
#     drone: DroneParams = field(default_factory=DroneParams)
#     control: ControlConfig = field(default_factory=ControlConfig)
#     simulation: SimulationConfig = field(default_factory=SimulationConfig)
#     visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
#
#
# class ConfigManager:
#     """配置管理器"""
#
#     _instance = None
#
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance
#
#     def __init__(self):
#         if self._initialized:
#             return
#
#         self._initialized = True
#         self.config = SystemConfig()
#         self._config_path: Optional[Path] = None
#
#         # 查找配置文件
#         self._find_and_load_config()
#
#         logger.info("配置管理器初始化完成")
#
#     def _find_and_load_config(self):
#         """查找并加载配置文件"""
#         possible_paths = [
#             Path("config/default_config.yaml"),
#             Path("config/config.yaml"),
#             Path("resources/default_config.yaml"),
#             Path(__file__).parent.parent.parent / "config" / "default_config.yaml",
#         ]
#
#         for path in possible_paths:
#             if path.exists():
#                 self.load(path)
#                 return
#
#         logger.warning("未找到配置文件，使用默认配置")
#
#     def load(self, filepath: Path):
#         """加载配置文件"""
#         if yaml is None:
#             logger.warning("PyYAML未安装，无法加载配置文件")
#             return
#
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 data = yaml.safe_load(f)
#
#             if data is None:
#                 logger.warning(f"配置文件为空: {filepath}")
#                 return
#
#             self._apply_config(data)
#             self._config_path = filepath
#             logger.info(f"已加载配置文件: {filepath}")
#
#         except Exception as e:
#             logger.error(f"加载配置文件失败: {e}")
#
#     def _apply_config(self, data: dict):
#         """应用配置数据"""
#         if 'drone' in data:
#             drone_data = data['drone']
#             for key, value in drone_data.items():
#                 if hasattr(self.config.drone, key):
#                     setattr(self.config.drone, key, value)
#
#         if 'control' in data:
#             control_data = data['control']
#             for key, value in control_data.items():
#                 if hasattr(self.config.control, key):
#                     if isinstance(value, list):
#                         value = tuple(value)
#                     setattr(self.config.control, key, value)
#
#         if 'simulation' in data:
#             sim_data = data['simulation']
#             for key, value in sim_data.items():
#                 if hasattr(self.config.simulation, key):
#                     setattr(self.config.simulation, key, value)
#
#         if 'visualization' in data:
#             vis_data = data['visualization']
#             for key, value in vis_data.items():
#                 if hasattr(self.config.visualization, key):
#                     setattr(self.config.visualization, key, value)
#
#     def save(self, filepath: Optional[Path] = None):
#         """保存配置到文件"""
#         if yaml is None:
#             logger.warning("PyYAML未安装，无法保存配置文件")
#             return
#
#         filepath = filepath or self._config_path
#         if filepath is None:
#             filepath = Path("config/config.yaml")
#
#         filepath.parent.mkdir(parents=True, exist_ok=True)
#
#         data = {
#             'drone': {
#                 'mass': self.config.drone.mass,
#                 'arm_length': self.config.drone.arm_length,
#                 'max_rpm': self.config.drone.max_rpm,
#                 'motor_constant': self.config.drone.motor_constant,
#                 'moment_constant': self.config.drone.moment_constant,
#                 'inertia_xx': self.config.drone.inertia_xx,
#                 'inertia_yy': self.config.drone.inertia_yy,
#                 'inertia_zz': self.config.drone.inertia_zz,
#                 'drag_coefficient': self.config.drone.drag_coefficient,
#             },
#             'control': {
#                 'position_kp': list(self.config.control.position_kp),
#                 'position_ki': list(self.config.control.position_ki),
#                 'position_kd': list(self.config.control.position_kd),
#                 'velocity_kp': list(self.config.control.velocity_kp),
#                 'velocity_ki': list(self.config.control.velocity_ki),
#                 'velocity_kd': list(self.config.control.velocity_kd),
#                 'attitude_kp': list(self.config.control.attitude_kp),
#                 'attitude_ki': list(self.config.control.attitude_ki),
#                 'attitude_kd': list(self.config.control.attitude_kd),
#                 'rate_kp': list(self.config.control.rate_kp),
#                 'rate_ki': list(self.config.control.rate_ki),
#                 'rate_kd': list(self.config.control.rate_kd),
#                 'max_velocity': self.config.control.max_velocity,
#                 'max_acceleration': self.config.control.max_acceleration,
#                 'max_tilt_angle': self.config.control.max_tilt_angle,
#                 'max_yaw_rate': self.config.control.max_yaw_rate,
#             },
#             'simulation': {
#                 'dt': self.config.simulation.dt,
#                 'realtime_factor': self.config.simulation.realtime_factor,
#                 'gravity': self.config.simulation.gravity,
#                 'air_density': self.config.simulation.air_density,
#             },
#             'visualization': {
#                 'update_rate': self.config.visualization.update_rate,
#                 'trail_length': self.config.visualization.trail_length,
#                 'grid_size': self.config.visualization.grid_size,
#                 'show_axes': self.config.visualization.show_axes,
#                 'show_grid': self.config.visualization.show_grid,
#             }
#         }
#
#         with open(filepath, 'w', encoding='utf-8') as f:
#             yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
#
#         logger.info(f"配置已保存到: {filepath}")
#
#     def reset_to_default(self):
#         """重置为默认配置"""
#         self.config = SystemConfig()
#         logger.info("配置已重置为默认值")
#
#     def get(self, key: str, default=None):
#         """获取配置值"""
#         parts = key.split('.')
#         obj = self.config
#
#         for part in parts:
#             if hasattr(obj, part):
#                 obj = getattr(obj, part)
#             else:
#                 return default
#
#         return obj
#
#
# # 全局配置管理器实例
# config_manager = ConfigManager()
#
#
# def get_config() -> SystemConfig:
#     """获取系统配置"""
#     return config_manager.config

# utils/config/config_manager.py（修复版）

"""
配置管理模块
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger


@dataclass
class SimulationConfig:
    """仿真配置"""
    dt: float = 0.01
    realtime_factor: float = 1.0
    max_duration: float = 300.0
    gravity: float = 9.81


@dataclass
class DroneParams:
    """无人机参数"""
    mass: float = 1.5
    arm_length: float = 0.25
    inertia: List[float] = field(default_factory=lambda: [0.0165, 0.0165, 0.0293])
    motor_constant: float = 8.54858e-06
    moment_constant: float = 0.016
    max_rpm: float = 10000
    min_rpm: float = 0  # 添加这个属性
    drag_coefficient: float = 0.1
    rotor_drag_coefficient: float = 0.01


@dataclass
class ControlConfig:
    """控制器配置"""
    position_kp: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.5])
    position_ki: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position_kd: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    velocity_kp: List[float] = field(default_factory=lambda: [2.0, 2.0, 3.0])
    velocity_ki: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.2])
    velocity_kd: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])

    attitude_kp: List[float] = field(default_factory=lambda: [6.0, 6.0, 4.0])
    attitude_ki: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    attitude_kd: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    rate_kp: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.05])
    rate_ki: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.01])
    rate_kd: List[float] = field(default_factory=lambda: [0.005, 0.005, 0.002])

    max_velocity: float = 5.0
    max_acceleration: float = 3.0
    max_tilt_angle: float = 0.35
    max_yaw_rate: float = 1.0


@dataclass
class IMUConfig:
    """IMU配置"""
    gyro_noise: float = 0.001
    accel_noise: float = 0.01
    update_rate: int = 400


@dataclass
class GPSConfig:
    """GPS配置"""
    position_noise: float = 0.5
    velocity_noise: float = 0.1
    update_rate: int = 10


@dataclass
class BarometerConfig:
    """气压计配置"""
    altitude_noise: float = 0.5
    update_rate: int = 50


@dataclass
class SensorConfig:
    """传感器配置"""
    imu: IMUConfig = field(default_factory=IMUConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    barometer: BarometerConfig = field(default_factory=BarometerConfig)


@dataclass
class WindConfig:
    """风场配置"""
    enabled: bool = False
    base_speed: float = 0.0
    direction: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    turbulence: float = 0.0


@dataclass
class EnvironmentConfig:
    """环境配置"""
    wind: WindConfig = field(default_factory=WindConfig)


@dataclass
class VisualizationConfig:
    """可视化配置"""
    update_rate: int = 30
    trail_length: int = 500
    show_axes: bool = True
    show_grid: bool = True


@dataclass
class SystemConfig:
    """系统总配置"""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    drone: DroneParams = field(default_factory=DroneParams)
    control: ControlConfig = field(default_factory=ControlConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


class ConfigManager:
    """配置管理器"""

    _instance: Optional['ConfigManager'] = None
    _config: SystemConfig = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config = SystemConfig()
        self._config_path: Optional[Path] = None

        # 尝试加载默认配置
        default_paths = [
            Path("config/default_config.yaml"),
            Path(__file__).parent.parent.parent / "config" / "default_config.yaml",
        ]

        for path in default_paths:
            if path.exists():
                self.load_config(path)
                break

        self._initialized = True
        logger.info("配置管理器初始化完成")

    @property
    def config(self) -> SystemConfig:
        """获取配置"""
        return self._config

    def load_config(self, path: Path) -> bool:
        """加载配置文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if data:
                self._apply_config_data(data)

            self._config_path = path
            logger.info(f"已加载配置文件: {path}")
            return True

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return False

    def _apply_config_data(self, data: dict):
        """应用配置数据"""
        # 仿真配置
        if 'simulation' in data:
            sim_data = data['simulation']
            self._config.simulation = SimulationConfig(
                dt=sim_data.get('dt', 0.01),
                realtime_factor=sim_data.get('realtime_factor', 1.0),
                max_duration=sim_data.get('max_duration', 300.0),
                gravity=sim_data.get('gravity', 9.81)
            )

        # 无人机参数
        if 'drone' in data:
            drone_data = data['drone']
            self._config.drone = DroneParams(
                mass=drone_data.get('mass', 1.5),
                arm_length=drone_data.get('arm_length', 0.25),
                inertia=drone_data.get('inertia', [0.0165, 0.0165, 0.0293]),
                motor_constant=drone_data.get('motor_constant', 8.54858e-06),
                moment_constant=drone_data.get('moment_constant', 0.016),
                max_rpm=drone_data.get('max_rpm', 10000),
                min_rpm=drone_data.get('min_rpm', 0),
                drag_coefficient=drone_data.get('drag_coefficient', 0.1),
                rotor_drag_coefficient=drone_data.get('rotor_drag_coefficient', 0.01)
            )

        # 控制器配置
        if 'control' in data:
            ctrl_data = data['control']
            self._config.control = ControlConfig(
                position_kp=ctrl_data.get('position_kp', [1.0, 1.0, 1.5]),
                position_ki=ctrl_data.get('position_ki', [0.0, 0.0, 0.0]),
                position_kd=ctrl_data.get('position_kd', [0.0, 0.0, 0.0]),
                velocity_kp=ctrl_data.get('velocity_kp', [2.0, 2.0, 3.0]),
                velocity_ki=ctrl_data.get('velocity_ki', [0.1, 0.1, 0.2]),
                velocity_kd=ctrl_data.get('velocity_kd', [0.1, 0.1, 0.1]),
                attitude_kp=ctrl_data.get('attitude_kp', [6.0, 6.0, 4.0]),
                attitude_ki=ctrl_data.get('attitude_ki', [0.0, 0.0, 0.0]),
                attitude_kd=ctrl_data.get('attitude_kd', [0.0, 0.0, 0.0]),
                rate_kp=ctrl_data.get('rate_kp', [0.1, 0.1, 0.05]),
                rate_ki=ctrl_data.get('rate_ki', [0.01, 0.01, 0.01]),
                rate_kd=ctrl_data.get('rate_kd', [0.005, 0.005, 0.002]),
                max_velocity=ctrl_data.get('max_velocity', 5.0),
                max_acceleration=ctrl_data.get('max_acceleration', 3.0),
                max_tilt_angle=ctrl_data.get('max_tilt_angle', 0.35),
                max_yaw_rate=ctrl_data.get('max_yaw_rate', 1.0)
            )

        # 传感器配置
        if 'sensors' in data:
            sens_data = data['sensors']

            imu_data = sens_data.get('imu', {})
            imu_config = IMUConfig(
                gyro_noise=imu_data.get('gyro_noise', 0.001),
                accel_noise=imu_data.get('accel_noise', 0.01),
                update_rate=imu_data.get('update_rate', 400)
            )

            gps_data = sens_data.get('gps', {})
            gps_config = GPSConfig(
                position_noise=gps_data.get('position_noise', 0.5),
                velocity_noise=gps_data.get('velocity_noise', 0.1),
                update_rate=gps_data.get('update_rate', 10)
            )

            baro_data = sens_data.get('barometer', {})
            baro_config = BarometerConfig(
                altitude_noise=baro_data.get('altitude_noise', 0.5),
                update_rate=baro_data.get('update_rate', 50)
            )

            self._config.sensors = SensorConfig(
                imu=imu_config,
                gps=gps_config,
                barometer=baro_config
            )

        # 环境配置
        if 'environment' in data:
            env_data = data['environment']
            wind_data = env_data.get('wind', {})

            self._config.environment = EnvironmentConfig(
                wind=WindConfig(
                    enabled=wind_data.get('enabled', False),
                    base_speed=wind_data.get('base_speed', 0.0),
                    direction=wind_data.get('direction', [1.0, 0.0, 0.0]),
                    turbulence=wind_data.get('turbulence', 0.0)
                )
            )

        # 可视化配置
        if 'visualization' in data:
            vis_data = data['visualization']
            self._config.visualization = VisualizationConfig(
                update_rate=vis_data.get('update_rate', 30),
                trail_length=vis_data.get('trail_length', 500),
                show_axes=vis_data.get('show_axes', True),
                show_grid=vis_data.get('show_grid', True)
            )

    def save_config(self, path: Path = None) -> bool:
        """保存配置文件"""
        save_path = path or self._config_path
        if save_path is None:
            save_path = Path("config/user_config.yaml")

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'simulation': {
                    'dt': self._config.simulation.dt,
                    'realtime_factor': self._config.simulation.realtime_factor,
                    'max_duration': self._config.simulation.max_duration,
                    'gravity': self._config.simulation.gravity,
                },
                'drone': {
                    'mass': self._config.drone.mass,
                    'arm_length': self._config.drone.arm_length,
                    'inertia': self._config.drone.inertia,
                    'motor_constant': self._config.drone.motor_constant,
                    'moment_constant': self._config.drone.moment_constant,
                    'max_rpm': self._config.drone.max_rpm,
                    'min_rpm': self._config.drone.min_rpm,
                    'drag_coefficient': self._config.drone.drag_coefficient,
                    'rotor_drag_coefficient': self._config.drone.rotor_drag_coefficient,
                },
                'control': {
                    'position_kp': self._config.control.position_kp,
                    'position_ki': self._config.control.position_ki,
                    'position_kd': self._config.control.position_kd,
                    'velocity_kp': self._config.control.velocity_kp,
                    'velocity_ki': self._config.control.velocity_ki,
                    'velocity_kd': self._config.control.velocity_kd,
                    'attitude_kp': self._config.control.attitude_kp,
                    'attitude_ki': self._config.control.attitude_ki,
                    'attitude_kd': self._config.control.attitude_kd,
                    'rate_kp': self._config.control.rate_kp,
                    'rate_ki': self._config.control.rate_ki,
                    'rate_kd': self._config.control.rate_kd,
                    'max_velocity': self._config.control.max_velocity,
                    'max_acceleration': self._config.control.max_acceleration,
                    'max_tilt_angle': self._config.control.max_tilt_angle,
                    'max_yaw_rate': self._config.control.max_yaw_rate,
                }
            }

            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"配置已保存到: {save_path}")
            return True

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False

    def reset_to_defaults(self):
        """重置为默认配置"""
        self._config = SystemConfig()
        logger.info("配置已重置为默认值")


# 全局访问函数
def get_config() -> SystemConfig:
    """获取配置"""
    return ConfigManager().config


def get_config_manager() -> ConfigManager:
    """获取配置管理器"""
    return ConfigManager()
