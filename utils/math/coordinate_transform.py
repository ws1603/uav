import numpy as np
from typing import Tuple


class CoordinateTransform:
    """坐标变换工具类"""

    # WGS84椭球参数
    WGS84_A = 6378137.0  # 长半轴 (m)
    WGS84_B = 6356752.314245  # 短半轴 (m)
    WGS84_E2 = 0.00669437999014  # 第一偏心率平方

    @staticmethod
    def ned_to_enu(ned: np.ndarray) -> np.ndarray:
        """NED坐标转ENU坐标"""
        return np.array([ned[1], ned[0], -ned[2]])

    @staticmethod
    def enu_to_ned(enu: np.ndarray) -> np.ndarray:
        """ENU坐标转NED坐标"""
        return np.array([enu[1], enu[0], -enu[2]])

    @staticmethod
    def body_to_inertial(v_body: np.ndarray, R: np.ndarray) -> np.ndarray:
        """机体坐标系转惯性坐标系"""
        return R @ v_body

    @staticmethod
    def inertial_to_body(v_inertial: np.ndarray, R: np.ndarray) -> np.ndarray:
        """惯性坐标系转机体坐标系"""
        return R.T @ v_inertial

    @staticmethod
    def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
        """
        大地坐标转地心地固坐标 (ECEF)

        Args:
            lat: 纬度 (rad)
            lon: 经度 (rad)
            alt: 高度 (m)

        Returns:
            ECEF坐标 [x, y, z] (m)
        """
        a = CoordinateTransform.WGS84_A
        e2 = CoordinateTransform.WGS84_E2

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)

        N = a / np.sqrt(1 - e2 * sin_lat ** 2)

        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1 - e2) + alt) * sin_lat

        return np.array([x, y, z])

    @staticmethod
    def ecef_to_geodetic(ecef: np.ndarray) -> Tuple[float, float, float]:
        """
        地心地固坐标转大地坐标

        Args:
            ecef: ECEF坐标 [x, y, z] (m)

        Returns:
            (lat, lon, alt) - 纬度(rad), 经度(rad), 高度(m)
        """
        x, y, z = ecef
        a = CoordinateTransform.WGS84_A
        b = CoordinateTransform.WGS84_B
        e2 = CoordinateTransform.WGS84_E2

        lon = np.arctan2(y, x)

        p = np.sqrt(x ** 2 + y ** 2)
        lat = np.arctan2(z, p * (1 - e2))

        # 迭代求解
        for _ in range(10):
            sin_lat = np.sin(lat)
            N = a / np.sqrt(1 - e2 * sin_lat ** 2)
            lat_new = np.arctan2(z + e2 * N * sin_lat, p)

            if abs(lat_new - lat) < 1e-12:
                break
            lat = lat_new

        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        N = a / np.sqrt(1 - e2 * sin_lat ** 2)

        alt = p / cos_lat - N if abs(cos_lat) > 1e-10 else abs(z) / sin_lat - N * (1 - e2)

        return lat, lon, alt

    @staticmethod
    def geodetic_to_ned(lat: float, lon: float, alt: float,
                        lat_ref: float, lon_ref: float, alt_ref: float) -> np.ndarray:
        """
        大地坐标转NED局部坐标

        Args:
            lat, lon, alt: 目标点大地坐标
            lat_ref, lon_ref, alt_ref: 参考点大地坐标

        Returns:
            NED坐标 [n, e, d] (m)
        """
        # 简化计算（适用于小范围）
        a = CoordinateTransform.WGS84_A
        e2 = CoordinateTransform.WGS84_E2

        sin_lat_ref = np.sin(lat_ref)
        cos_lat_ref = np.cos(lat_ref)

        R_n = a * (1 - e2) / (1 - e2 * sin_lat_ref ** 2) ** 1.5
        R_e = a / np.sqrt(1 - e2 * sin_lat_ref ** 2)

        d_lat = lat - lat_ref
        d_lon = lon - lon_ref
        d_alt = alt - alt_ref

        n = d_lat * R_n
        e = d_lon * R_e * cos_lat_ref
        d = -d_alt

        return np.array([n, e, d])

    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """绕X轴旋转矩阵"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        """绕Y轴旋转矩阵"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """绕Z轴旋转矩阵"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """欧拉角转旋转矩阵 (ZYX顺序)"""
        Rx = CoordinateTransform.rotation_matrix_x(roll)
        Ry = CoordinateTransform.rotation_matrix_y(pitch)
        Rz = CoordinateTransform.rotation_matrix_z(yaw)
        return Rz @ Ry @ Rx