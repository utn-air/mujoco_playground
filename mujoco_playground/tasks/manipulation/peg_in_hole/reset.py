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

from typing import Any

import mujoco
from mujoco_playground.robots.fr3_robotiq.sim import constant
import numpy as np
from reaf.simulation.torch import environment
import torch

_FIXTURE_POS_RANGE = ((0.54, 0.76), (-0.29, 0.17), (-0.20, 0.29))
_FIXTURE_TILT_RANGE = (-5.0, 5.0)
RANDOMIZE_FIXTURE_POSITION = False
RANDOMIZE_FIXTURE_TILT = False
_PEG_OFFSET_RANGE = (-0.0125, 0.0125)
_GRASP_OFFSET_RANGE = (-0.0055, 0.0055)
_GRASP_TILT_RANGE = (-10.0, 10.0)
_CLOSED_GRIPPER = 0.8
_GRIPPER_CLOSED_CTRL = 255.0


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
  ):
    self._fixture_pos_range = fixture_pos_range
    self._fixture_tilt_range = fixture_tilt_range
    self._peg_offset_range = peg_offset_range
    self._grasp_offset_range = grasp_offset_range
    self._grasp_tilt_range = grasp_tilt_range
    self._gripper_peg_z_offset_override = gripper_peg_z_offset

  def on_startup(self) -> None:
    sim = self._env.device_layer.sim
    plate_body_id = sim.mj_model.body("plate").id
    self._default_fixture_pos = sim.mj_model.body_pos[plate_body_id].copy()
    self._default_fixture_quat = sim.mj_model.body_quat[plate_body_id].copy()
    self._plate_mocap_id = int(sim.mj_model.body_mocapid[plate_body_id])
    if self._plate_mocap_id < 0:
      raise ValueError("Plate body must be configured as a mocap body")
    plate_data = mujoco.MjData(sim.mj_model)
    mujoco.mj_forward(sim.mj_model, plate_data)
    plate_bottom_z, plate_top_z = _body_mesh_z_bounds(
        sim.mj_model, plate_data, plate_body_id
    )
    self._plate_thickness = plate_top_z - plate_bottom_z
    # Place the peg's lowest site half a plate thickness above the top face.
    self._plate_z_offset = self._plate_thickness / 2
    self._gripper_peg_z_offset = (
        self._plate_z_offset * 2
        if self._gripper_peg_z_offset_override is None
        else self._gripper_peg_z_offset_override
    )
    self._peg_joint_id = sim.mj_model.joint("peg_joint").id
    self._peg_qpos = int(sim.mj_model.jnt_qposadr[self._peg_joint_id])
    self._peg_dof = int(sim.mj_model.jnt_dofadr[self._peg_joint_id])
    self._plate_top_z = float(sim.mj_model.site("plate_edge_0_0").pos[2])
    self._peg_site_pos = sim.mj_model.site_pos[
        [sim.mj_model.site(f"peg_{i}").id for i in range(64)]
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
    self._ik_model = sim.mj_spec.compile()
    self._ik_data = mujoco.MjData(self._ik_model)
    self._closed_gripper_qpos = self._measure_closed_gripper_pose()
    self._set_home_state(self._ik_model, self._ik_data)
    self._ik_data.qpos[self._gripper_qpos_ids] = self._closed_gripper_qpos
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

  def _measure_closed_gripper_pose(self) -> np.ndarray:
    """Settles coupled Robotiq joints to capture the closed pose."""
    self._set_home_state(self._ik_model, self._ik_data)
    for actuator, target in zip(constant.ACTUATORS, constant.HOME_CTRL):
      self._ik_data.ctrl[self._ik_model.actuator(actuator).id] = target
    self._ik_data.ctrl[self._ik_model.actuator(constant.ACTUATORS[-1]).id] = (
        _GRIPPER_CLOSED_CTRL
    )
    for _ in range(200):
      mujoco.mj_step(self._ik_model, self._ik_data)
    return self._ik_data.qpos[self._gripper_qpos_ids].copy()

  @staticmethod
  def _set_home_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name, value in zip(constant.JOINTS, constant.HOME_QPOS):
      data.qpos[model.joint(name).qposadr[0]] = value
    data.qpos[
        model.joint(f"robot/robotiq/{constant.GRIPPER_JOINT}").qposadr[0]
    ] = _CLOSED_GRIPPER

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
    for i in torch.nonzero(env_mask).flatten().cpu().tolist():
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
        self._ik_data.qpos[self._gripper_qpos_ids] = self._closed_gripper_qpos
        mujoco.mj_forward(self._ik_model, self._ik_data)
        target_rot = plate_rot @ self._home_ee_rot
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
        robot_qpos_np[i, 7] = _CLOSED_GRIPPER
        tilt_axis_local = self._home_ee_rot @ self._finger_axis_ee
        tilt_quat = np.concatenate([
            [np.cos(grasp_tilt / 2)],
            tilt_axis_local * np.sin(grasp_tilt / 2),
        ])
        tilt_rot = np.empty(9)
        mujoco.mju_quat2Mat(tilt_rot, tilt_quat)
        tilt_rot = tilt_rot.reshape(3, 3)
        min_site_z = (self._peg_site_pos @ tilt_rot[2]).min()
        grasp_axis_local = self._home_ee_rot @ self._pad_axis_ee
        peg_pos_np[i] = (
            peg_pos
            + plate_rot @ (grasp_axis_local * grasp_offset)
            + plate_rot
            @ np.array([
                0.0,
                0.0,
                min_site_z - self._peg_site_pos[:, 2].min(),
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
        self._closed_gripper_qpos, dtype=qpos.dtype, device=device
    ).expand(nworld, -1)
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
