# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixture and peg initialization for peg-in-hole."""

import concurrent.futures
import copy
import os
import time
from typing import Any

from absl import logging
import mujoco
from mujoco_playground.robots.fr3_robotiq.sim import constant
from mujoco_playground.tasks.manipulation.peg_in_hole import world_assets
import numpy as np
from reaf.simulation.torch import environment
import torch

_FIXTURE_POS_RANGE = ((0.54, 0.76), (-0.29, 0.17), (-0.20, 0.29))
_FIXTURE_TILT_RANGE = (-5.0, 5.0)
RANDOMIZE_FIXTURE_POSITION = True
RANDOMIZE_FIXTURE_TILT = True
_PEG_OFFSET_RANGE = (-0.0125, 0.0125)
_GRASP_OFFSET_RANGE = (-0.0055, 0.0055)
_GRASP_TILT_RANGE = (-10.0, 10.0)
_GRIPPER_CLOSED_CTRL = 255.0
_PINCH_DISTANCE_BELOW_PEG_TOP = (
    0.014  # seems to land right in the middle of the finger pads
)


def _body_mesh_z_bounds(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> tuple[float, float]:
  """Returns mesh geometry z bounds in the body's local frame."""
  body_pos = data.xpos[body_id]
  body_rot = data.xmat[body_id].reshape(3, 3)
  body_z_values = []
  for geom_id in range(model.ngeom):
    if model.geom_bodyid[geom_id] != body_id:
      continue
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    end = start + model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start:end]
    geom_rot = data.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = data.geom_xpos[geom_id] + vertices @ geom_rot.T
    body_vertices = (world_vertices - body_pos) @ body_rot
    body_z_values.extend(body_vertices[:, 2])
  if not body_z_values:
    raise ValueError("Plate body has no mesh geometry to measure thickness")
  return float(np.min(body_z_values)), float(np.max(body_z_values))


def _quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
  roll, pitch, yaw = rpy / 2
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  return np.array([
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
  ])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return np.array([
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
  ])


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
  relative = target @ current.T
  cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
  angle = np.arccos(cosine)
  if angle < 1e-8:
    return np.zeros(3)
  axis = np.array([
      relative[2, 1] - relative[1, 2],
      relative[0, 2] - relative[2, 0],
      relative[1, 0] - relative[0, 1],
  ]) / (2.0 * np.sin(angle))
  return angle * axis


def _align_gripper_yaw(
    home_ee_rot: np.ndarray, finger_axis_ee: np.ndarray
) -> np.ndarray:
  """Aligns jaw closure with peg-frame +Y, preserving the home approach tilt.

  The peg and hole share a nominal frame; +Y spans the narrow dimension of
  the canonical peg. The FR3 home pose puts the jaws diagonally across that
  frame, which can twist the initially aligned peg as the grasp settles.
  """
  finger_axis = home_ee_rot @ finger_axis_ee
  if np.linalg.norm(finger_axis[:2]) < 1e-8:
    raise ValueError("Home jaw-closing axis must have a horizontal component")
  yaw = np.pi / 2 - np.arctan2(finger_axis[1], finger_axis[0])
  cosine, sine = np.cos(yaw), np.sin(yaw)
  yaw_rotation = np.array([
      [cosine, -sine, 0.0],
      [sine, cosine, 0.0],
      [0.0, 0.0, 1.0],
  ])
  return yaw_rotation @ home_ee_rot


def _solve_gripper_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
) -> np.ndarray:
  """Solves full-pose damped least-squares IK for the FR3 arm."""
  arm_joints = [model.joint(f"robot/{name}").id for name in constant.ARM_JOINTS]
  qpos_ids = model.jnt_qposadr[arm_joints]
  dof_ids = model.jnt_dofadr[arm_joints]
  lower = model.jnt_range[arm_joints, 0] + 1e-5
  upper = model.jnt_range[arm_joints, 1] - 1e-5
  ee_id = model.site(constant.EE_SITE).id
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))

  for _ in range(400):
    mujoco.mj_forward(model, data)
    current_rot = data.site(ee_id).xmat.reshape(3, 3)
    error = np.concatenate([
        target_pos - data.site(ee_id).xpos,
        _rotation_error(target_rot, current_rot),
    ])
    if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
      return data.qpos[qpos_ids].copy()
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
    jacobian = np.concatenate([jacp[:, dof_ids], jacr[:, dof_ids]], axis=0)
    delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + 1e-4 * np.eye(6), error
    )
    norm = np.linalg.norm(delta)
    if norm > 0.15:
      delta *= 0.15 / norm
    data.qpos[qpos_ids] = np.clip(data.qpos[qpos_ids] + delta, lower, upper)

  mujoco.mj_forward(model, data)
  pos_error = np.linalg.norm(target_pos - data.site(ee_id).xpos)
  rot_error = np.linalg.norm(
      _rotation_error(target_rot, data.site(ee_id).xmat.reshape(3, 3))
  )
  if pos_error > 0.01 or rot_error > 0.1:
    raise RuntimeError(
        "Peg grasp pose is outside the FR3 workspace: position"
        f" error={pos_error:.3f} m, orientation error={rot_error:.3f} rad"
    )
  return data.qpos[qpos_ids].copy()


class PegInHoleResetter(environment.EnvironmentReset):
  """Randomizes the fixture and peg, then initializes a closed grasp."""

  def __init__(
      self,
      fixture_pos_range: tuple[tuple[float, float], ...] = _FIXTURE_POS_RANGE,
      fixture_tilt_range: tuple[float, float] = _FIXTURE_TILT_RANGE,
      peg_offset_range: tuple[float, float] = _PEG_OFFSET_RANGE,
      grasp_offset_range: tuple[float, float] = _GRASP_OFFSET_RANGE,
      grasp_tilt_range: tuple[float, float] = _GRASP_TILT_RANGE,
      gripper_peg_z_offset: float | None = None,
      world_xml_paths: list[str] | None = None,
      robot_spec_fn: Any = None,
  ):
    self._fixture_pos_range = fixture_pos_range
    self._fixture_tilt_range = fixture_tilt_range
    self._peg_offset_range = peg_offset_range
    self._grasp_offset_range = grasp_offset_range
    self._grasp_tilt_range = grasp_tilt_range
    self._gripper_peg_z_offset_override = gripper_peg_z_offset
    self._world_xml_paths = world_xml_paths
    self._robot_spec_fn = robot_spec_fn

  def _world_model(self, path: str) -> mujoco.MjModel:
    from mujoco_playground.tasks.manipulation.peg_in_hole import task

    spec = task.peg_in_hole_env(path).base_scene
    spec.copy_during_attach = True
    frame = spec.worldbody.add_frame()
    robot = self._robot_spec_fn()
    for field in ("impratio", "integrator", "cone"):
      setattr(robot.option, field, getattr(spec.option, field))
    spec.attach(robot, prefix=constant.ATTACHMENT_PREFIX, frame=frame)
    return spec.compile()

  def on_startup(self) -> None:
    started = time.monotonic()
    world_count = len(self._world_xml_paths) if self._world_xml_paths else 1
    unique_count = (
        len(set(self._world_xml_paths)) if self._world_xml_paths else 1
    )
    logging.info(
        "Peg-in-hole: calibrating grasps for %d worlds (%d unique assets)",
        world_count,
        unique_count,
    )
    sim = self._env.device_layer.sim
    plate_body_id = sim.mj_model.body("plate").id
    self._default_fixture_pos = sim.mj_model.body_pos[plate_body_id].copy()
    self._default_fixture_quat = sim.mj_model.body_quat[plate_body_id].copy()
    self._plate_mocap_id = int(sim.mj_model.body_mocapid[plate_body_id])
    if self._plate_mocap_id < 0:
      raise ValueError("Plate body must be configured as a mocap body")
    reference_model = (
        self._world_model(self._world_xml_paths[0])
        if self._world_xml_paths
        else sim.mj_model
    )
    plate_data = mujoco.MjData(reference_model)
    mujoco.mj_forward(reference_model, plate_data)
    plate_bottom_z, plate_top_z = _body_mesh_z_bounds(
        reference_model, plate_data, reference_model.body("plate").id
    )
    self._plate_thickness = plate_top_z - plate_bottom_z
    # Place the peg's lowest site half a plate thickness above the top face.
    self._plate_z_offset = self._plate_thickness / 2
    self._peg_joint_id = sim.mj_model.joint("peg_joint").id
    self._peg_qpos = int(sim.mj_model.jnt_qposadr[self._peg_joint_id])
    self._peg_dof = int(sim.mj_model.jnt_dofadr[self._peg_joint_id])
    self._plate_top_z = float(reference_model.site("plate_edge_0_0").pos[2])
    self._peg_site_pos = reference_model.site_pos[
        [reference_model.site(f"peg_{i}").id for i in range(64)]
    ].copy()
    gripper_joint_names = [
        sim.mj_model.joint(i).name
        for i in range(sim.mj_model.njnt)
        if sim.mj_model.joint(i).name.startswith("robot/robotiq/")
    ]
    self._gripper_qpos_ids = np.array([
        sim.mj_model.jnt_qposadr[sim.mj_model.joint(name).id]
        for name in gripper_joint_names
    ])
    self._gripper_dof_ids = np.array([
        sim.mj_model.jnt_dofadr[sim.mj_model.joint(name).id]
        for name in gripper_joint_names
    ])
    self._ik_model = (
        reference_model if self._world_xml_paths else sim.mj_spec.compile()
    )
    self._ik_data = mujoco.MjData(self._ik_model)
    self._set_home_state(self._ik_model, self._ik_data)
    mujoco.mj_forward(self._ik_model, self._ik_data)
    ee_id = self._ik_model.site(constant.EE_SITE).id
    self._home_ee_rot = self._ik_data.site(ee_id).xmat.reshape(3, 3).copy()
    pad_r = self._ik_data.geom("robot/robotiq/right_pad1").xpos.copy()
    pad_r_tip = self._ik_data.geom("robot/robotiq/right_pad2").xpos.copy()
    pad_l = self._ik_data.geom("robot/robotiq/left_pad1").xpos.copy()
    ee_rot_t = self._home_ee_rot.T
    self._pad_axis_ee = ee_rot_t @ (pad_r_tip - pad_r)
    self._pad_axis_ee /= np.linalg.norm(self._pad_axis_ee)
    self._finger_axis_ee = ee_rot_t @ (pad_r - pad_l)
    self._finger_axis_ee /= np.linalg.norm(self._finger_axis_ee)
    self._aligned_ee_rot = _align_gripper_yaw(
        self._home_ee_rot, self._finger_axis_ee
    )
    # Keep the pinch point a fixed distance below the actual peg top.
    peg_id = self._ik_model.body("peg").id
    peg_bottom, peg_top = _body_mesh_z_bounds(
        self._ik_model, self._ik_data, peg_id
    )
    if self._gripper_peg_z_offset_override is None:
      self._gripper_peg_z_offset = peg_top - _PINCH_DISTANCE_BELOW_PEG_TOP
      if self._gripper_peg_z_offset < peg_bottom:
        raise ValueError(
            "Peg is too short for the calibrated pinch distance: "
            f"length={peg_top - peg_bottom:.4f} m, required="
            f"{_PINCH_DISTANCE_BELOW_PEG_TOP:.4f} m. "
            "Use a longer peg or specify gripper_peg_z_offset."
        )
    else:
      self._gripper_peg_z_offset = self._gripper_peg_z_offset_override
    self._grasp_offsets = np.linspace(*self._grasp_offset_range, 3)
    self._grasp_tilts = np.deg2rad(np.linspace(*self._grasp_tilt_range, 3))
    self._grasp_grid = np.stack([
        np.stack([
            self._measure_closed_gripper_pose(offset, tilt)
            for tilt in self._grasp_tilts
        ])
        for offset in self._grasp_offsets
    ])
    driver_qpos = int(self._ik_model.joint(constant.JOINTS[-1]).qposadr[0])
    self._driver_index = int(
        np.flatnonzero(self._gripper_qpos_ids == driver_qpos)[0]
    )
    if self._world_xml_paths:
      self._cache_world_grasps(started)
    else:
      world_assets.log_progress("grasp calibration", 1, 1, started)

  def _cache_world_grasps(self, started: float) -> None:
    """Calibrate each distinct asset once and keep its data across resets."""
    cache = {
        self._world_xml_paths[0]: (
            self._plate_top_z,
            self._plate_z_offset,
            self._peg_site_pos,
            self._gripper_peg_z_offset,
            self._grasp_grid,
        )
    }
    paths = list(dict.fromkeys(self._world_xml_paths[1:]))
    paths = [path for path in paths if path not in cache]
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    workers = min(len(paths), available_cpus)
    logging.info("Peg-in-hole: using %d calibration workers", workers)
    world_assets.log_progress("grasp calibration", 1, len(paths) + 1, started)
    if workers:
      with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(self._calibrate_world_grasp, path): path
            for path in paths
        }
        for future in concurrent.futures.as_completed(futures):
          path = futures[future]
          try:
            cache[path] = future.result()
          except Exception as exc:
            raise RuntimeError(f"Grasp calibration failed for {path}") from exc
          world_assets.log_progress(
              "grasp calibration", len(cache), len(paths) + 1, started
          )
    self._world_grasps = [cache[path] for path in self._world_xml_paths]

  def _calibrate_world_grasp(self, path: str) -> tuple:
    self = copy.copy(self)
    model = self._world_model(path)
    data = mujoco.MjData(model)
    self._ik_model, self._ik_data = model, data
    mujoco.mj_forward(model, data)
    plate = model.body("plate").id
    bottom, top = _body_mesh_z_bounds(model, data, plate)
    self._plate_top_z = float(model.site("plate_edge_0_0").pos[2])
    self._plate_z_offset = (top - bottom) / 2
    self._peg_site_pos = model.site_pos[
        [model.site(f"peg_{i}").id for i in range(64)]
    ].copy()
    peg_bottom, peg_top = _body_mesh_z_bounds(model, data, model.body("peg").id)
    self._gripper_peg_z_offset = (
        peg_top - _PINCH_DISTANCE_BELOW_PEG_TOP
        if self._gripper_peg_z_offset_override is None
        else self._gripper_peg_z_offset_override
    )
    if self._gripper_peg_z_offset < peg_bottom:
      raise ValueError(f"Peg is too short for the calibrated grasp: {path}")
    self._grasp_grid = np.stack([
        np.stack([
            self._measure_closed_gripper_pose(offset, tilt)
            for tilt in self._grasp_tilts
        ])
        for offset in self._grasp_offsets
    ])
    return (
        self._plate_top_z,
        self._plate_z_offset,
        self._peg_site_pos,
        self._gripper_peg_z_offset,
        self._grasp_grid,
    )

  def _measure_closed_gripper_pose(
      self, grasp_offset: float, grasp_tilt: float
  ) -> np.ndarray:
    """Close around the peg once with the arm and peg held in place."""
    mujoco.mj_resetData(self._ik_model, self._ik_data)
    self._set_home_state(self._ik_model, self._ik_data)
    self._ik_data.qpos[self._gripper_qpos_ids] = 0.0
    # Keep the fixture away while solving the gripper--peg contact.
    self._ik_data.mocap_pos[self._plate_mocap_id] = (5.0, 5.0, 5.0)
    mujoco.mj_forward(self._ik_model, self._ik_data)
    ee = self._ik_data.site(constant.EE_SITE)
    frame_rot = ee.xmat.reshape(3, 3) @ self._aligned_ee_rot.T
    axis = self._aligned_ee_rot @ self._finger_axis_ee
    tilt_quat = np.concatenate(
        [[np.cos(grasp_tilt / 2)], axis * np.sin(grasp_tilt / 2)]
    )
    tilt_rot = np.empty(9)
    mujoco.mju_quat2Mat(tilt_rot, tilt_quat)
    tilt_rot = tilt_rot.reshape(3, 3)
    peg_rot = frame_rot @ tilt_rot
    peg_quat = np.empty(4)
    mujoco.mju_mat2Quat(peg_quat, peg_rot.flatten())
    peg_pose = np.concatenate([
        ee.xpos
        + frame_rot
        @ (
            -np.array([0.0, 0.0, self._gripper_peg_z_offset])
            + (self._aligned_ee_rot @ self._pad_axis_ee) * grasp_offset
            + np.array([
                0.0,
                0.0,
                self._peg_site_pos[:, 2].min()
                - (self._peg_site_pos @ tilt_rot[2]).min(),
            ])
        ),
        peg_quat,
    ])
    arm_ids = [
        int(self._ik_model.joint(name).qposadr[0])
        for name in constant.JOINTS[:7]
    ]
    arm_dofs = [
        int(self._ik_model.joint(name).dofadr[0])
        for name in constant.JOINTS[:7]
    ]
    arm_pose = self._ik_data.qpos[arm_ids].copy()
    for actuator, target in zip(constant.ACTUATORS, constant.HOME_CTRL):
      self._ik_data.ctrl[self._ik_model.actuator(actuator).id] = target
    self._ik_data.ctrl[self._ik_model.actuator(constant.ACTUATORS[-1]).id] = (
        _GRIPPER_CLOSED_CTRL
    )
    for _ in range(600):
      self._ik_data.qpos[arm_ids] = arm_pose
      self._ik_data.qvel[arm_dofs] = 0.0
      self._ik_data.qpos[self._peg_qpos : self._peg_qpos + 7] = peg_pose
      self._ik_data.qvel[self._peg_dof : self._peg_dof + 6] = 0.0
      mujoco.mj_step(self._ik_model, self._ik_data)
    self._ik_data.qpos[arm_ids] = arm_pose
    self._ik_data.qpos[self._peg_qpos : self._peg_qpos + 7] = peg_pose
    mujoco.mj_forward(self._ik_model, self._ik_data)
    peg_body = self._ik_model.body("peg").id
    sides = set()
    deepest = 0.0
    for contact in self._ik_data.contact:
      geom1, geom2 = map(int, contact.geom)
      for pad, other in ((geom1, geom2), (geom2, geom1)):
        if self._ik_model.geom_bodyid[other] != peg_body:
          continue
        name = self._ik_model.geom(pad).name or ""
        for side in ("left", "right"):
          if name in (
              f"robot/robotiq/{side}_pad1",
              f"robot/robotiq/{side}_pad2",
          ):
            sides.add(side)
            deepest = min(deepest, float(contact.dist))
    if sides != {"left", "right"} or deepest < -0.002:
      raise RuntimeError(
          "Could not initialize a two-sided peg grasp: "
          f"contact sides={sorted(sides)}, penetration={-deepest:.4f} m. "
          "Check peg geometry and gripper pinch height."
      )
    return self._ik_data.qpos[self._gripper_qpos_ids].copy()

  def _interpolate_grasp(self, offset: float, tilt: float) -> np.ndarray:
    """Bilinearly interpolate the cached finger positions."""

    def weights(grid: np.ndarray, value: float) -> tuple[int, float]:
      if grid[-1] == grid[0]:
        return 0, 0.0
      i = min(
          max(int(np.searchsorted(grid, value, side="right")) - 1, 0),
          len(grid) - 2,
      )
      return i, float(
          np.clip((value - grid[i]) / (grid[i + 1] - grid[i]), 0, 1)
      )

    i, a = weights(self._grasp_offsets, offset)
    j, b = weights(self._grasp_tilts, tilt)
    return (
        (1 - a) * (1 - b) * self._grasp_grid[i, j]
        + a * (1 - b) * self._grasp_grid[i + 1, j]
        + (1 - a) * b * self._grasp_grid[i, j + 1]
        + a * b * self._grasp_grid[i + 1, j + 1]
    )

  @staticmethod
  def _set_home_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name, value in zip(constant.JOINTS, constant.HOME_QPOS):
      data.qpos[model.joint(name).qposadr[0]] = value

  def on_reset(self, config: Any, env_mask: torch.Tensor) -> None:
    del config
    sim = self._env.device_layer.sim
    device = torch.device(self._env.xpu_device)
    nworld = self._env.nworld

    fixture_pos_np = np.zeros((nworld, 3))
    fixture_quat_np = np.zeros((nworld, 4))
    peg_pos_np = np.zeros((nworld, 3))
    peg_quat_np = np.zeros((nworld, 4))
    robot_qpos_np = np.zeros((nworld, len(constant.JOINTS)))
    gripper_qpos_np = np.zeros((nworld, len(self._gripper_qpos_ids)))
    for i in torch.nonzero(env_mask).flatten().cpu().tolist():
      if self._world_xml_paths:
        (
            self._plate_top_z,
            self._plate_z_offset,
            self._peg_site_pos,
            self._gripper_peg_z_offset,
            self._grasp_grid,
        ) = self._world_grasps[i]
      while True:
        # Draw a new complete randomization candidate for this world. IK is
        # checked on the CPU model before any candidate is committed to sim.
        sample = torch.rand(10, device=device).cpu().numpy()
        fixture_pos_np[i] = (
            [
                low + sample[j] * (high - low)
                for j, (low, high) in enumerate(self._fixture_pos_range)
            ]
            if RANDOMIZE_FIXTURE_POSITION
            else self._default_fixture_pos
        )
        if RANDOMIZE_FIXTURE_TILT:
          fixture_rpy = (
              sample[3:6]
              * (self._fixture_tilt_range[1] - self._fixture_tilt_range[0])
              + self._fixture_tilt_range[0]
          ) * (np.pi / 180.0)
          fixture_quat_np[i] = _quat_from_rpy(fixture_rpy)
        else:
          fixture_quat_np[i] = self._default_fixture_quat
        candidate_peg_offset = (
            sample[6:8]
            * (self._peg_offset_range[1] - self._peg_offset_range[0])
            + self._peg_offset_range[0]
        )
        grasp_offset = (
            sample[8]
            * (self._grasp_offset_range[1] - self._grasp_offset_range[0])
            + self._grasp_offset_range[0]
        )
        grasp_tilt = (
            sample[9] * (self._grasp_tilt_range[1] - self._grasp_tilt_range[0])
            + self._grasp_tilt_range[0]
        ) * (np.pi / 180.0)
        plate_rot = np.empty(9)
        mujoco.mju_quat2Mat(plate_rot, fixture_quat_np[i])
        plate_rot = plate_rot.reshape(3, 3)

        # Solve the arm pose for the nominal grasp first. Random grasp offset
        # and peg tilt are then applied to the object, not the gripper target.
        mujoco.mj_resetData(self._ik_model, self._ik_data)
        self._ik_data.mocap_pos[self._plate_mocap_id] = fixture_pos_np[i]
        self._ik_data.mocap_quat[self._plate_mocap_id] = fixture_quat_np[i]
        self._set_home_state(self._ik_model, self._ik_data)
        gripper_qpos_np[i] = self._interpolate_grasp(grasp_offset, grasp_tilt)
        self._ik_data.qpos[self._gripper_qpos_ids] = gripper_qpos_np[i]
        mujoco.mj_forward(self._ik_model, self._ik_data)
        target_rot = plate_rot @ self._aligned_ee_rot
        peg_pos = fixture_pos_np[i] + plate_rot @ np.array([
            candidate_peg_offset[0],
            candidate_peg_offset[1],
            self._plate_top_z
            + self._plate_z_offset
            - self._peg_site_pos[:, 2].min(),
        ])
        target_pos = peg_pos + plate_rot[:, 2] * self._gripper_peg_z_offset
        try:
          arm_qpos = _solve_gripper_pose(
              self._ik_model, self._ik_data, target_pos, target_rot
          )
        except RuntimeError:
          continue

        robot_qpos_np[i, :7] = arm_qpos
        robot_qpos_np[i, 7] = gripper_qpos_np[i, self._driver_index]
        tilt_axis_local = self._aligned_ee_rot @ self._finger_axis_ee
        tilt_quat = np.concatenate([
            [np.cos(grasp_tilt / 2)],
            tilt_axis_local * np.sin(grasp_tilt / 2),
        ])
        tilt_rot = np.empty(9)
        mujoco.mju_quat2Mat(tilt_rot, tilt_quat)
        tilt_rot = tilt_rot.reshape(3, 3)
        min_site_z = (self._peg_site_pos @ tilt_rot[2]).min()
        grasp_axis_local = self._aligned_ee_rot @ self._pad_axis_ee
        peg_pos_np[i] = (
            peg_pos
            + plate_rot @ (grasp_axis_local * grasp_offset)
            + plate_rot
            @ np.array([
                0.0,
                0.0,
                self._peg_site_pos[:, 2].min() - min_site_z,
            ])
        )
        peg_quat_np[i] = _quat_mul(fixture_quat_np[i], tilt_quat)
        break

    mocap_pos = sim.data.mocap_pos[:, self._plate_mocap_id]
    mocap_quat = sim.data.mocap_quat[:, self._plate_mocap_id]
    mocap_pos[:] = torch.where(
        env_mask[:, None],
        torch.as_tensor(fixture_pos_np, dtype=mocap_pos.dtype, device=device),
        mocap_pos,
    )
    mocap_quat[:] = torch.where(
        env_mask[:, None],
        torch.as_tensor(fixture_quat_np, dtype=mocap_quat.dtype, device=device),
        mocap_quat,
    )
    qpos = sim.data.qpos
    qvel = sim.data.qvel
    peg_qpos_slice = slice(self._peg_qpos, self._peg_qpos + 7)
    peg_dof_slice = slice(self._peg_dof, self._peg_dof + 6)
    peg_qpos_batch = torch.tensor(
        np.concatenate([peg_pos_np, peg_quat_np], axis=-1),
        dtype=qpos.dtype,
        device=device,
    )
    qpos[:, peg_qpos_slice] = torch.where(
        env_mask[:, None], peg_qpos_batch, qpos[:, peg_qpos_slice]
    )
    qvel[:, peg_dof_slice] = torch.where(
        env_mask[:, None],
        torch.zeros_like(qvel[:, peg_dof_slice]),
        qvel[:, peg_dof_slice],
    )
    gripper_qpos = torch.as_tensor(
        gripper_qpos_np, dtype=qpos.dtype, device=device
    )
    qpos[:, self._gripper_qpos_ids] = torch.where(
        env_mask[:, None], gripper_qpos, qpos[:, self._gripper_qpos_ids]
    )
    qvel[:, self._gripper_dof_ids] = torch.where(
        env_mask[:, None],
        torch.zeros_like(qvel[:, self._gripper_dof_ids]),
        qvel[:, self._gripper_dof_ids],
    )

    robot_state = self._env.device_layer.get_device("robot_state")
    robot_state.set_qpos(
        env_mask,
        torch.tensor(robot_qpos_np, dtype=torch.float32, device=device),
    )
    robot_state.set_qvel(
        env_mask,
        torch.zeros(
            (nworld, len(constant.JOINTS)), dtype=torch.float32, device=device
        ),
    )
    robot = self._env.device_layer.get_device("robot")
    closed_ctrl = torch.tensor(
        robot_qpos_np, dtype=torch.float32, device=device
    )
    closed_ctrl[:, -1] = _GRIPPER_CLOSED_CTRL
    robot.set_ctrl(env_mask, closed_ctrl)
