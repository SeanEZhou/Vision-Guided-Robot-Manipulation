"""Camera geometry: depth conversion, projection, back-projection, frame transforms.

Conventions (checked by scripts/localize_eval.py):
- Images are in standard image orientation: row 0 at the top. robosuite renders bottom-up, so
  callers flip RGB and depth with `[::-1]` before using these functions.
- Pixels are continuous (u, v) = (column, row) coordinates with the principal point at (W/2, H/2),
  so the centre of integer pixel index (i, j) is (i + 0.5, j + 0.5).
- Camera frame follows OpenCV: x right, y down, z forward along the optical axis.
- World frame is robosuite's world frame (the same frame as the robot base controller targets).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray  # 3x3 intrinsics
    T_world_cam: np.ndarray  # 4x4 pose of the camera in the world (OpenCV axes)
    near: float  # clip planes (m), needed to linearize MuJoCo's depth buffer
    far: float
    height: int
    width: int

    @classmethod
    def from_sim(cls, sim, camera_name, height, width):
        """Calibration from the simulator's camera model (not object state)."""
        cam_id = sim.model.camera_name2id(camera_name)
        fovy = np.deg2rad(sim.model.cam_fovy[cam_id])
        f = 0.5 * height / np.tan(fovy / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])

        R_mj = sim.data.cam_xmat[cam_id].reshape(3, 3)
        # MuJoCo cameras look down -z with y up; flip y and z to get OpenCV axes.
        T = np.eye(4)
        T[:3, :3] = R_mj @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = sim.data.cam_xpos[cam_id]

        extent = sim.model.stat.extent
        return cls(K, T, sim.model.vis.map.znear * extent, sim.model.vis.map.zfar * extent, height, width)

    def metric_depth(self, depth_normalized):
        """MuJoCo's [0, 1] depth buffer -> optical-axis depth Z in metres."""
        d = np.asarray(depth_normalized, dtype=np.float64)
        if d.ndim == 3:
            d = d[..., 0]
        return self.near / (1.0 - d * (1.0 - self.near / self.far))

    def backproject(self, uv, Z):
        """Pixels (N, 2) with optical-axis depth Z (N,) -> world points (N, 3)."""
        uv = np.asarray(uv, dtype=np.float64)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        p_cam = np.stack([(uv[:, 0] - cx) * Z / fx, (uv[:, 1] - cy) * Z / fy, Z], axis=1)
        return p_cam @ self.T_world_cam[:3, :3].T + self.T_world_cam[:3, 3]

    def project(self, p_world):
        """World points (N, 3) -> pixels (N, 2) as float (u, v), and optical-axis depth (N,)."""
        T_cam_world = np.linalg.inv(self.T_world_cam)
        p_cam = np.asarray(p_world) @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
        uvw = p_cam @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3], p_cam[:, 2]
