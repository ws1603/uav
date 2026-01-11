# # utils/config/__init__.py
#
# """配置管理模块"""
#
# from pathlib import Path
#
# # 确保配置目录存在
# config_dir = Path(__file__).parent.parent.parent / "config"
# config_dir.mkdir(exist_ok=True)
#
# # 如果配置文件不存在，创建默认配置
# default_config_path = config_dir / "default_config.yaml"
# if not default_config_path.exists():
#     default_config_content = """# 低空交通无人机教学演示系统 - 默认配置
#
# drone:
#   mass: 1.5
#   arm_length: 0.225
#   max_rpm: 12000
#   motor_constant: 8.54858e-06
#   moment_constant: 0.016
#   inertia_xx: 0.029125
#   inertia_yy: 0.029125
#   inertia_zz: 0.055225
#   drag_coefficient: 0.1
#
# control:
#   position_kp: [2.0, 2.0, 4.0]
#   position_ki: [0.1, 0.1, 0.2]
#   position_kd: [1.5, 1.5, 2.0]
#   velocity_kp: [4.0, 4.0, 6.0]
#   velocity_ki: [0.5, 0.5, 0.8]
#   velocity_kd: [0.5, 0.5, 0.5]
#   attitude_kp: [8.0, 8.0, 4.0]
#   attitude_ki: [0.5, 0.5, 0.2]
#   attitude_kd: [2.0, 2.0, 1.0]
#   rate_kp: [0.15, 0.15, 0.1]
#   rate_ki: [0.01, 0.01, 0.01]
#   rate_kd: [0.01, 0.01, 0.01]
#   max_velocity: 15.0
#   max_acceleration: 5.0
#   max_tilt_angle: 0.5236
#   max_yaw_rate: 1.5708
#
# simulation:
#   dt: 0.002
#   realtime_factor: 1.0
#   gravity: 9.81
#   air_density: 1.225
#
# visualization:
#   update_rate: 30
#   trail_length: 500
#   grid_size: 100
#   show_axes: true
#   show_grid: true
# """
#     default_config_path.write_text(default_config_content, encoding='utf-8')
#
# from utils.config.config_manager import config_manager, get_config, SystemConfig


# utils/config/__init__.py

"""配置管理模块"""

from utils.config.config_manager import (
    ConfigManager,
    SystemConfig,
    SimulationConfig,
    DroneParams,
    ControlConfig,
    SensorConfig,
    EnvironmentConfig,
    VisualizationConfig,
    get_config,
    get_config_manager
)

__all__ = [
    'ConfigManager',
    'SystemConfig',
    'SimulationConfig',
    'DroneParams',
    'ControlConfig',
    'SensorConfig',
    'EnvironmentConfig',
    'VisualizationConfig',
    'get_config',
    'get_config_manager'
]
