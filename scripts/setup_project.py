# scripts/setup_project.py

"""
项目初始化脚本 - 创建必要的目录结构和文件
"""

import os
from pathlib import Path


def create_directory_structure():
    """创建目录结构"""

    directories = [
        # 核心模块
        "core",
        "core/physics",
        "core/control",
        "core/planning",
        "core/traffic",

        # 仿真模块
        "simulation",
        "simulation/engine",
        "simulation/scenarios",
        "simulation/environment",

        # UI模块
        "ui",
        "ui/main_window",
        "ui/panels",
        "ui/dialogs",
        "ui/visualization",
        "ui/themes",
        "ui/widgets",

        # 工具模块
        "utils",
        "utils/math",
        "utils/config",
        "utils/io",

        # 资源目录
        "resources",
        "resources/models",
        "resources/textures",
        "resources/icons",
        "resources/scenarios",

        # 其他目录
        "tests",
        "docs",
        "logs",
        "scripts",
    ]

    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # 创建__init__.py
        init_file = path / "__init__.py"
        if not init_file.exists() and not directory.startswith(("resources", "docs", "logs", "scripts")):
            init_file.write_text('"""{}"""\n'.format(directory.replace("/", ".")))
            print(f"Created: {init_file}")

    print("Directory structure created successfully!")


def create_init_files():
    """创建各模块的__init__.py文件"""

    init_contents = {
        "core/__init__.py": '''"""核心模块"""
from core.physics.quadrotor_dynamics import QuadrotorDynamics, DroneState
from core.control.pid_controller import PIDController, QuadrotorPIDController
''',

        "core/physics/__init__.py": '''"""物理仿真模块"""
from core.physics.quadrotor_dynamics import QuadrotorDynamics, DroneState, DroneParams
''',

        "core/control/__init__.py": '''"""控制模块"""
from core.control.pid_controller import PIDController, PIDGains, QuadrotorPIDController
''',

        "core/planning/__init__.py": '''"""路径规划模块"""
from core.planning.astar_planner import AStarPlanner, OccupancyGrid3D, PathSmoother
from core.planning.waypoint_manager import WaypointManager, Waypoint, Mission
''',

        "simulation/__init__.py": '''"""仿真模块"""
from simulation.engine.simulation_core import SimulationEngine
''',

        "simulation/engine/__init__.py": '''"""仿真引擎模块"""
from simulation.engine.simulation_core import SimulationEngine, SimulationState, SimulationConfig
''',

        "ui/__init__.py": '''"""用户界面模块"""
''',

        "ui/themes/__init__.py": '''"""主题模块"""
from ui.themes.dark_theme import get_dark_stylesheet
''',

        "utils/__init__.py": '''"""工具模块"""
''',

        "utils/math/__init__.py": '''"""数学工具模块"""
from utils.math.quaternion import Quaternion
from utils.math.coordinate_transforms import CoordinateTransforms
''',

        "utils/config/__init__.py": '''"""配置管理模块"""
from utils.config.config_manager import config_manager, get_config
''',
    }

    for filepath, content in init_contents.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        print(f"Created: {filepath}")


def create_requirements():
    """创建requirements.txt"""

    requirements = """# 低空交通无人机教学演示系统 - 依赖列表

# 核心依赖
numpy>=1.21.0
scipy>=1.7.0

# GUI框架
PyQt5>=5.15.0
pyqtgraph>=0.12.0

# 3D可视化（可选）
PyOpenGL>=3.1.0

# 日志
loguru>=0.6.0

# 配置文件
pyyaml>=6.0

# 开发依赖
pytest>=7.0.0
pytest-qt>=4.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950

# 文档（可选）
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
"""

    Path("requirements.txt").write_text(requirements.strip(), encoding='utf-8')
    print("Created: requirements.txt")


def create_readme():
    """创建README.md"""

    readme = """# 低空交通无人机教学演示系统
    
## 简介

本系统是一个用于无人机飞行控制、路径规划和低空交通管理的教学演示平台。

## 功能特性

- 🚁 **无人机飞行仿真**: 基于物理的四旋翼动力学仿真
- 🎮 **PID控制器**: 位置和姿态控制
- 🗺️ **路径规划**: A*算法和路径平滑
- 📊 **实时可视化**: 3D视图和实时数据图表
- 🎓 **教学模式**: 逐步演示和参数调节

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

"""