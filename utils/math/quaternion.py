# # utils/math/quaternion.py
#
# import numpy as np
# from typing import Tuple, Union
#
#
# class Quaternion:
#     """
#     四元数工具类
#
#     约定: q = [w, x, y, z]，其中 w 是标量部分
#     """
#
#     @staticmethod
#     def normalize(q: np.ndarray) -> np.ndarray:
#         """归一化四元数"""
#         norm = np.linalg.norm(q)
#         if norm < 1e-10:
#             return np.array([1.0, 0.0, 0.0, 0.0])
#         return q / norm
#
#     @staticmethod
#     def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
#         """
#         四元数乘法 q1 * q2
#
#         Args:
#             q1, q2: 四元数 [w, x, y, z]
#
#         Returns:
#             乘积四元数
#         """
#         w1, x1, y1, z1 = q1
#         w2, x2, y2, z2 = q2
#
#         return np.array([
#             w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
#             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
#             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
#             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
#         ])
#
#     @staticmethod
#     def conjugate(q: np.ndarray) -> np.ndarray:
#         """四元数共轭"""
#         return np.array([q[0], -q[1], -q[2], -q[3]])
#
#     @staticmethod
#     def inverse(q: np.ndarray) -> np.ndarray:
#         """四元数逆"""
#         return Quaternion.conjugate(q) / np.dot(q, q)
#
#     @staticmethod
#     def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
#         """
#         使用四元数旋转向量
#
#         Args:
#             q: 四元数 [w, x, y, z]
#             v: 向量 [x, y, z]
#
#         Returns:
#             旋转后的向量
#         """
#         # 将向量扩展为纯四元数
#         v_quat = np.array([0, v[0], v[1], v[2]])
#
#         # v' = q * v * q^(-1)
#         q_inv = Quaternion.inverse(q)
#         result = Quaternion.multiply(Quaternion.multiply(q, v_quat), q_inv)
#
#         return result[1:4]
#
#     @staticmethod
#     def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
#         """
#         四元数转旋转矩阵
#
#         Args:
#             q: 四元数 [w, x, y, z]
#
#         Returns:
#             3x3 旋转矩阵
#         """
#         q = Quaternion.normalize(q)
#         w, x, y, z = q
#
#         return np.array([
#             [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
#             [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
#             [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
#         ])
#
#     @staticmethod
#     def from_rotation_matrix(R: np.ndarray) -> np.ndarray:
#         """旋转矩阵转四元数"""
#         trace = np.trace(R)
#
#         if trace > 0:
#             s = 0.5 / np.sqrt(trace + 1.0)
#             w = 0.25 / s
#             x = (R[2, 1] - R[1, 2]) * s
#             y = (R[0, 2] - R[2, 0]) * s
#             z = (R[1, 0] - R[0, 1]) * s
#         elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
#             s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
#             w = (R[2, 1] - R[1, 2]) / s
#             x = 0.25 * s
#             y = (R[0, 1] + R[1, 0]) / s
#             z = (R[0, 2] + R[2, 0]) / s
#         elif R[1, 1] > R[2, 2]:
#             s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
#             w = (R[0, 2] - R[2, 0]) / s
#             x = (R[0, 1] + R[1, 0]) / s
#             y = 0.25 * s
#             z = (R[1, 2] + R[2, 1]) / s
#         else:
#             s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
#             w = (R[1, 0] - R[0, 1]) / s
#             x = (R[0, 2] + R[2, 0]) / s
#             y = (R[1, 2] + R[2, 1]) / s
#             z = 0.25 * s
#
#         return Quaternion.normalize(np.array([w, x, y, z]))
#
#     @staticmethod
#     def from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
#         """
#         欧拉角转四元数 (ZYX顺序，即yaw-pitch-roll)
#
#         Args:
#             roll: 横滚角 (rad)
#             pitch: 俯仰角 (rad)
#             yaw: 偏航角 (rad)
#
#         Returns:
#             四元数 [w, x, y, z]
#         """
#         cr = np.cos(roll / 2)
#         sr = np.sin(roll / 2)
#         cp = np.cos(pitch / 2)
#         sp = np.sin(pitch / 2)
#         cy = np.cos(yaw / 2)
#         sy = np.sin(yaw / 2)
#
#         w = cr * cp * cy + sr * sp * sy
#         x = sr * cp * cy - cr * sp * sy
#         y = cr * sp * cy + sr * cp * sy
#         z = cr * cp * sy - sr * sp * cy
#
#         return np.array([w, x, y, z])
#
#     @staticmethod
#     def to_euler(q: np.ndarray) -> np.ndarray:
#         """
#         四元数转欧拉角 (ZYX顺序)
#
#         Args:
#             q: 四元数 [w, x, y, z]
#
#         Returns:
#             欧拉角 [roll, pitch, yaw] (rad)
#         """
#         q = Quaternion.normalize(q)
#         w, x, y, z = q
#
#         # Roll (x-axis rotation)
#         sinr_cosp = 2 * (w * x + y * z)
#         cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
#         roll = np.arctan2(sinr_cosp, cosr_cosp)
#
#         # Pitch (y-axis rotation)
#         sinp = 2 * (w * y - z * x)
#         if abs(sinp) >= 1:
#             pitch = np.copysign(np.pi / 2, sinp)  # 万向锁
#         else:
#             pitch = np.arcsin(sinp)
#
#         # Yaw (z-axis rotation)
#         siny_cosp = 2 * (w * z + x * y)
#         cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
#         yaw = np.arctan2(siny_cosp, cosy_cosp)
#
#         return np.array([roll, pitch, yaw])
#
#     @staticmethod
#     def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
#         """从轴角表示创建四元数"""
#         axis = axis / np.linalg.norm(axis)
#         half_angle = angle / 2
#
#         w = np.cos(half_angle)
#         xyz = axis * np.sin(half_angle)
#
#         return np.array([w, xyz[0], xyz[1], xyz[2]])
#
#     @staticmethod
#     def to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
#         """四元数转轴角表示"""
#         q = Quaternion.normalize(q)
#         angle = 2 * np.arccos(np.clip(q[0], -1, 1))
#
#         if angle < 1e-10:
#             return np.array([0, 0, 1]), 0.0
#
#         axis = q[1:4] / np.sin(angle / 2)
#         return axis, angle
#
#     @staticmethod
#     def derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
#         """
#         计算四元数导数
#
#         Args:
#             q: 当前四元数 [w, x, y, z]
#             omega: 角速度 [wx, wy, wz] (机体坐标系)
#
#         Returns:
#             四元数导数 dq/dt
#         """
#         omega_quat = np.array([0, omega[0], omega[1], omega[2]])
#         return 0.5 * Quaternion.multiply(q, omega_quat)
#
#     @staticmethod
#     def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
#         """
#         球面线性插值
#
#         Args:
#             q1: 起始四元数
#             q2: 终止四元数
#             t: 插值参数 [0, 1]
#
#         Returns:
#             插值结果四元数
#         """
#         q1 = Quaternion.normalize(q1)
#         q2 = Quaternion.normalize(q2)
#
#         dot = np.dot(q1, q2)
#
#         # 选择短路径
#         if dot < 0:
#             q2 = -q2
#             dot = -dot
#
#         # 点积接近1时使用线性插值
#         if dot > 0.9995:
#             result = q1 + t * (q2 - q1)
#             return Quaternion.normalize(result)
#
#         theta = np.arccos(np.clip(dot, -1, 1))
#         sin_theta = np.sin(theta)
#
#         s1 = np.sin((1 - t) * theta) / sin_theta
#         s2 = np.sin(t * theta) / sin_theta
#
#         return s1 * q1 + s2 * q2


# utils/math/quaternion.py（完整修复版）

"""
四元数工具类
"""

import numpy as np
from typing import Tuple


class Quaternion:
    """四元数工具类 (静态方法集合)

    四元数格式: [w, x, y, z] (标量在前)
    """

    @staticmethod
    def identity() -> np.ndarray:
        """返回单位四元数"""
        return np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """归一化四元数"""
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return Quaternion.identity()
        return q / norm

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """四元数乘法 q1 * q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """四元数共轭"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def inverse(q: np.ndarray) -> np.ndarray:
        """四元数逆"""
        return Quaternion.conjugate(q) / np.dot(q, q)

    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        q = Quaternion.normalize(q)
        w, x, y, z = q

        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])

    @staticmethod
    def from_rotation_matrix(R: np.ndarray) -> np.ndarray:
        """旋转矩阵转四元数"""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return Quaternion.normalize(np.array([w, x, y, z]))

    @staticmethod
    def to_euler(q: np.ndarray) -> np.ndarray:
        """四元数转欧拉角 (ZYX顺序) -> [roll, pitch, yaw]"""
        q = Quaternion.normalize(q)
        w, x, y, z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @staticmethod
    def from_euler(euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数 (ZYX顺序) [roll, pitch, yaw]"""
        roll, pitch, yaw = euler

        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return Quaternion.normalize(np.array([w, x, y, z]))

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """轴角表示转四元数"""
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        half_angle = angle / 2

        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)

        return np.array([w, xyz[0], xyz[1], xyz[2]])

    @staticmethod
    def to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
        """四元数转轴角表示"""
        q = Quaternion.normalize(q)

        angle = 2 * np.arccos(np.clip(q[0], -1, 1))

        s = np.sqrt(1 - q[0] ** 2)
        if s < 1e-10:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = q[1:4] / s

        return axis, angle

    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """使用四元数旋转向量"""
        qv = np.array([0, v[0], v[1], v[2]])
        q_conj = Quaternion.conjugate(q)

        result = Quaternion.multiply(Quaternion.multiply(q, qv), q_conj)
        return result[1:4]

    @staticmethod
    def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """球面线性插值"""
        q1 = Quaternion.normalize(q1)
        q2 = Quaternion.normalize(q2)

        dot = np.dot(q1, q2)

        if dot < 0:
            q2 = -q2
            dot = -dot

        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return Quaternion.normalize(result)

        theta_0 = np.arccos(np.clip(dot, -1, 1))
        theta = theta_0 * t

        q2_orthogonal = q2 - q1 * dot
        q2_orthogonal = q2_orthogonal / (np.linalg.norm(q2_orthogonal) + 1e-10)

        return q1 * np.cos(theta) + q2_orthogonal * np.sin(theta)

    @staticmethod
    def integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        使用角速度积分更新四元数

        Args:
            q: 当前四元数 [w, x, y, z]
            omega: 角速度 [wx, wy, wz] (rad/s)
            dt: 时间步长

        Returns:
            更新后的四元数
        """
        # 构造角速度四元数
        omega_norm = np.linalg.norm(omega)

        if omega_norm < 1e-10:
            return q

        # 方法1: 使用四元数微分方程
        # q_dot = 0.5 * q * omega_quat
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * Quaternion.multiply(q, omega_quat)

        # 一阶积分
        q_new = q + q_dot * dt

        return Quaternion.normalize(q_new)

    @staticmethod
    def angular_velocity_to_quaternion_rate(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        将角速度转换为四元数变化率

        Args:
            q: 当前四元数
            omega: 角速度 [wx, wy, wz]

        Returns:
            四元数变化率
        """
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        return 0.5 * Quaternion.multiply(q, omega_quat)
