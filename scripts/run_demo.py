# scripts/run_demo.py（完整版）

"""
演示脚本 - 展示系统核心功能
"""

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_physics():
    """演示物理仿真"""
    from core.physics.quadrotor_dynamics import QuadrotorDynamics

    print("\n" + "=" * 40)
    print("演示1: 物理仿真")
    print("=" * 40)

    dynamics = QuadrotorDynamics()

    # 悬停测试
    hover_rpm = 5500
    dynamics.set_motor_speeds(np.array([hover_rpm] * 4))

    print(f"\n设置悬停转速: {hover_rpm} RPM")
    print("开始仿真...")

    for i in range(100):
        dynamics.step(0.01)

        if i % 20 == 0:
            state = dynamics.state
            print(f"  t={i * 0.01:.2f}s: z={-state.position[2]:.3f}m, vz={-state.velocity[2]:.3f}m/s")

    print("物理仿真演示完成")


def demo_control():
    """演示控制器"""
    from core.physics.quadrotor_dynamics import QuadrotorDynamics
    from core.control.pid_controller import QuadrotorPIDController

    print("\n" + "=" * 40)
    print("演示2: PID控制")
    print("=" * 40)

    dynamics = QuadrotorDynamics()
    controller = QuadrotorPIDController()

    # 设置目标
    target_position = np.array([0.0, 0.0, -10.0])
    target_yaw = 0.0

    print(f"\n目标高度: {-target_position[2]}m")
    print("开始控制...")

    dt = 0.01

    for i in range(500):
        state = dynamics.state

        # 计算控制输入
        motor_commands = controller.compute_control(
            state=state,
            target_position=target_position,
            target_yaw=target_yaw,
            dt=dt
        )

        # 应用控制
        dynamics.set_motor_speeds(motor_commands)
        dynamics.step(dt)

        if i % 100 == 0:
            altitude = -state.position[2]
            error = -target_position[2] - altitude
            print(f"  t={i * dt:.2f}s: 高度={altitude:.2f}m, 误差={error:.2f}m")

    final_altitude = -dynamics.state.position[2]
    print(f"\n最终高度: {final_altitude:.2f}m (目标: {-target_position[2]}m)")
    print("控制演示完成")


def demo_path_planning():
    """演示路径规划"""
    from core.planning.astar_planner import OccupancyGrid3D, AStarPlanner, PathSmoother

    print("\n" + "=" * 40)
    print("演示3: 路径规划")
    print("=" * 40)

    # 创建地图
    grid = OccupancyGrid3D(size=(50, 50, 20), resolution=1.0)

    # 添加障碍物
    grid.add_box_obstacle(
        min_corner=np.array([20, 20, 5]),
        max_corner=np.array([25, 25, 15])
    )

    print("\n地图大小: 50x50x20")
    print("添加了一个障碍物: [20-25, 20-25, 5-15]")

    # 创建规划器
    planner = AStarPlanner(grid)

    start = np.array([5.0, 5.0, 10.0])
    goal = np.array([40.0, 40.0, 10.0])

    print(f"起点: {start}")
    print(f"终点: {goal}")
    print("开始规划...")

    start_time = time.time()
    path = planner.plan(start, goal)
    elapsed = time.time() - start_time

    if path:
        print(f"\n找到路径! 共 {len(path)} 个点, 耗时 {elapsed * 1000:.1f}ms")

        # 计算路径长度
        total_length = 0
        for i in range(len(path) - 1):
            total_length += np.linalg.norm(path[i + 1] - path[i])
        print(f"路径长度: {total_length:.2f}m")

        # 平滑路径
        smoothed = PathSmoother.smooth_path(path)
        smooth_length = 0
        for i in range(len(smoothed) - 1):
            smooth_length += np.linalg.norm(smoothed[i + 1] - smoothed[i])
        print(f"平滑后路径长度: {smooth_length:.2f}m")
    else:
        print("未找到路径!")

    print("路径规划演示完成")


def demo_waypoint_mission():
    """演示航点任务"""
    from core.planning.waypoint_manager import WaypointManager, Waypoint, WaypointType

    print("\n" + "=" * 40)
    print("演示4: 航点任务")
    print("=" * 40)

    manager = WaypointManager()

    # 创建任务
    mission = manager.create_mission("测试任务")

    # 添加航点
    waypoints = [
        Waypoint(position=[0, 0, -10], waypoint_type=WaypointType.TAKEOFF),
        Waypoint(position=[20, 0, -15], speed=5.0),
        Waypoint(position=[20, 20, -15], speed=5.0),
        Waypoint(position=[0, 20, -10], speed=3.0, loiter_time=5.0),
        Waypoint(position=[0, 0, -5], waypoint_type=WaypointType.LANDING),
    ]

    for wp in waypoints:
        mission.add_waypoint(wp)

    print(f"\n任务名称: {mission.name}")
    print(f"航点数量: {len(mission.waypoints)}")
    print(f"总距离: {mission.total_distance:.2f}m")
    print(f"预计时间: {mission.estimated_time:.1f}s")

    print("\n航点列表:")
    for i, wp in enumerate(mission.waypoints):
        print(f"  {i}: {wp.position} - {wp.waypoint_type.name}")

    # 模拟执行
    manager.load_mission(mission)
    manager.start_mission()

    print("\n模拟执行任务...")
    current_pos = np.array([0.0, 0.0, 0.0])
    sim_time = 0.0

    while manager.state.name == 'RUNNING':
        target_wp = manager.update(current_pos, sim_time)

        if target_wp is None:
            break

        # 简单模拟移动
        direction = target_wp.position - current_pos
        dist = np.linalg.norm(direction)

        if dist > 0.1:
            move = direction / dist * min(target_wp.speed * 0.1, dist)
            current_pos = current_pos + move

        sim_time += 0.1

        if sim_time > 100:  # 超时保护
            break

    print(f"任务状态: {manager.state.name}")
    print("航点任务演示完成")


def demo_simulation_engine():
    """演示仿真引擎"""
    from simulation.engine.simulation_core import SimulationEngine

    print("\n" + "=" * 40)
    print("演示5: 仿真引擎")
    print("=" * 40)

    engine = SimulationEngine()

    # 设置目标
    target = np.array([10.0, 5.0, -15.0])
    engine.set_target_position(target)

    print(f"\n目标位置: {target}")
    print("启动仿真引擎...")

    engine.start()

    try:
        for i in range(50):
            time.sleep(0.1)

            state = engine.drone_state
            stats = engine.statistics

            if i % 10 == 0:
                print(
                    f"  t={stats.simulation_time:.1f}s: "
                    f"位置=[{state.position[0]:.1f}, {state.position[1]:.1f}, {state.position[2]:.1f}] "
                    f"高度={state.altitude:.1f}m"
                )
    finally:
        engine.stop()

    print(f"\n仿真统计:")
    print(f"  总帧数: {stats.total_frames}")
    print(f"  仿真时间: {stats.simulation_time:.2f}s")
    print(f"  平均帧率: {stats.average_fps:.1f} FPS")

    print("仿真引擎演示完成")


def main():
    """主函数"""
    print("=" * 50)
    print("低空交通无人机教学演示系统 - 功能演示")
    print("=" * 50)

    try:
        demo_physics()
    except Exception as e:
        print(f"物理仿真演示失败: {e}")

    try:
        demo_control()
    except Exception as e:
        print(f"控制演示失败: {e}")

    try:
        demo_path_planning()
    except Exception as e:
        print(f"路径规划演示失败: {e}")

    try:
        demo_waypoint_mission()
    except Exception as e:
        print(f"航点任务演示失败: {e}")

    try:
        demo_simulation_engine()
    except Exception as e:
        print(f"仿真引擎演示失败: {e}")

    print("\n" + "=" * 50)
    print("所有演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()