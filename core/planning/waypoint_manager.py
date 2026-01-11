# core/planning/waypoint_manager.py

"""
航点管理模块
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger


class WaypointType(Enum):
    """航点类型"""
    NORMAL = auto()
    TAKEOFF = auto()
    LANDING = auto()
    HOVER = auto()
    LOITER = auto()
    RTL = auto()  # Return to Launch


class MissionState(Enum):
    """任务状态"""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ABORTED = auto()


@dataclass
class Waypoint:
    """航点"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    waypoint_type: WaypointType = WaypointType.NORMAL
    speed: float = 5.0
    heading: Optional[float] = None
    loiter_time: float = 0.0
    acceptance_radius: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.position, (list, tuple)):
            self.position = np.array(self.position, dtype=float)


@dataclass
class Mission:
    """飞行任务"""
    name: str = "未命名任务"
    waypoints: List[Waypoint] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""

    def add_waypoint(self, waypoint: Waypoint):
        """添加航点"""
        self.waypoints.append(waypoint)

    def insert_waypoint(self, index: int, waypoint: Waypoint):
        """插入航点"""
        self.waypoints.insert(index, waypoint)

    def remove_waypoint(self, index: int):
        """移除航点"""
        if 0 <= index < len(self.waypoints):
            self.waypoints.pop(index)

    def clear_waypoints(self):
        """清空航点"""
        self.waypoints.clear()

    @property
    def total_distance(self) -> float:
        """计算总飞行距离"""
        if len(self.waypoints) < 2:
            return 0.0

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += np.linalg.norm(
                self.waypoints[i + 1].position - self.waypoints[i].position
            )
        return total

    @property
    def estimated_time(self) -> float:
        """估计飞行时间"""
        if len(self.waypoints) < 2:
            return 0.0

        total_time = 0.0
        for i in range(len(self.waypoints) - 1):
            distance = np.linalg.norm(
                self.waypoints[i + 1].position - self.waypoints[i].position
            )
            speed = self.waypoints[i].speed
            if speed > 0:
                total_time += distance / speed
            total_time += self.waypoints[i].loiter_time

        # 最后一个航点的悬停时间
        if self.waypoints:
            total_time += self.waypoints[-1].loiter_time

        return total_time


class WaypointManager:
    """航点管理器"""

    def __init__(self):
        """初始化航点管理器"""
        self._missions: Dict[str, Mission] = {}
        self._current_mission: Optional[Mission] = None
        self._current_waypoint_index: int = 0
        self._state = MissionState.IDLE
        self._loiter_start_time: Optional[float] = None

        logger.debug("航点管理器初始化完成")

    @property
    def state(self) -> MissionState:
        """获取任务状态"""
        return self._state

    @property
    def current_mission(self) -> Optional[Mission]:
        """获取当前任务"""
        return self._current_mission

    @property
    def current_waypoint_index(self) -> int:
        """获取当前航点索引"""
        return self._current_waypoint_index

    def create_mission(self, name: str, description: str = "") -> Mission:
        """创建新任务"""
        mission = Mission(name=name, description=description)
        self._missions[name] = mission
        logger.info(f"创建任务: {name}")
        return mission

    def get_mission(self, name: str) -> Optional[Mission]:
        """获取任务"""
        return self._missions.get(name)

    def delete_mission(self, name: str):
        """删除任务"""
        if name in self._missions:
            if self._current_mission and self._current_mission.name == name:
                self.abort_mission()
            del self._missions[name]
            logger.info(f"删除任务: {name}")

    def list_missions(self) -> List[str]:
        """列出所有任务"""
        return list(self._missions.keys())

    def load_mission(self, mission: Mission):
        """加载任务"""
        self._current_mission = mission
        self._current_waypoint_index = 0
        self._state = MissionState.IDLE
        logger.info(f"加载任务: {mission.name}, 共{len(mission.waypoints)}个航点")

    def start_mission(self):
        """开始执行任务"""
        if self._current_mission is None:
            logger.warning("没有加载任务")
            return

        if len(self._current_mission.waypoints) == 0:
            logger.warning("任务没有航点")
            return

        self._current_waypoint_index = 0
        self._state = MissionState.RUNNING
        self._loiter_start_time = None
        logger.info("开始执行任务")

    def pause_mission(self):
        """暂停任务"""
        if self._state == MissionState.RUNNING:
            self._state = MissionState.PAUSED
            logger.info("任务暂停")

    def resume_mission(self):
        """恢复任务"""
        if self._state == MissionState.PAUSED:
            self._state = MissionState.RUNNING
            logger.info("任务恢复")

    def abort_mission(self):
        """中止任务"""
        self._state = MissionState.ABORTED
        logger.info("任务中止")

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """获取当前航点"""
        if self._current_mission is None:
            return None
        if self._current_waypoint_index >= len(self._current_mission.waypoints):
            return None
        return self._current_mission.waypoints[self._current_waypoint_index]

    def get_next_waypoint(self) -> Optional[Waypoint]:
        """获取下一个航点"""
        if self._current_mission is None:
            return None
        next_index = self._current_waypoint_index + 1
        if next_index >= len(self._current_mission.waypoints):
            return None
        return self._current_mission.waypoints[next_index]

    def update(self, current_position: np.ndarray, current_time: float) -> Optional[Waypoint]:
        """
        更新航点管理器

        Args:
            current_position: 当前位置
            current_time: 当前时间

        Returns:
            当前目标航点
        """
        if self._state != MissionState.RUNNING:
            return None

        current_wp = self.get_current_waypoint()
        if current_wp is None:
            self._state = MissionState.COMPLETED
            logger.info("任务完成")
            return None

        # 检查是否到达当前航点
        distance = np.linalg.norm(current_position - current_wp.position)

        if distance <= current_wp.acceptance_radius:
            # 处理悬停
            if current_wp.loiter_time > 0:
                if self._loiter_start_time is None:
                    self._loiter_start_time = current_time
                    logger.info(f"到达航点{self._current_waypoint_index}, 开始悬停{current_wp.loiter_time}秒")

                elapsed = current_time - self._loiter_start_time
                if elapsed < current_wp.loiter_time:
                    return current_wp  # 继续悬停

            # 前进到下一个航点
            self._current_waypoint_index += 1
            self._loiter_start_time = None

            if self._current_waypoint_index >= len(self._current_mission.waypoints):
                self._state = MissionState.COMPLETED
                logger.info("任务完成")
                return None

            logger.info(f"前进到航点{self._current_waypoint_index}")
            return self.get_current_waypoint()

        return current_wp

    def get_progress(self) -> float:
        """获取任务进度 (0-1)"""
        if self._current_mission is None:
            return 0.0

        total = len(self._current_mission.waypoints)
        if total == 0:
            return 0.0

        return self._current_waypoint_index / total

    def get_remaining_distance(self, current_position: np.ndarray) -> float:
        """获取剩余飞行距离"""
        if self._current_mission is None:
            return 0.0

        waypoints = self._current_mission.waypoints
        if self._current_waypoint_index >= len(waypoints):
            return 0.0

        # 到当前航点的距离
        current_wp = waypoints[self._current_waypoint_index]
        remaining = np.linalg.norm(current_position - current_wp.position)

        # 剩余航点间的距离
        for i in range(self._current_waypoint_index, len(waypoints) - 1):
            remaining += np.linalg.norm(
                waypoints[i + 1].position - waypoints[i].position
            )

        return remaining
