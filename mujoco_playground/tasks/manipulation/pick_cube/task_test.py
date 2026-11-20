# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic SO101 vs FR3 Robotiq pick-reward comparison."""

from absl import app
from absl import flags
from absl import logging
import mujoco
from mujoco_playground._src.rl import env_registry
from mujoco_playground.robots.fr3_robotiq.sim import constant as fr3_constant
from mujoco_playground.robots.so101.real import constant as so101_constant
import numpy as np
import torch

_TOLERANCE = flags.DEFINE_float(
    "tolerance",
    0.015,
    "Maximum absolute difference allowed for shared reward terms.",
)

_SHARED_TERMS = (
    "reaching",
    "grasp",
    "lift",
    "bringing",
    "success",
    "action_rate",
    "joint_vel_hinge",
)
_PHASES = (
    # name, ee target, cube center, goal, gripper closed fraction
    (
        "approach",
        (0.26, 0.06, 0.14),
        (0.26, 0.06, 0.018),
        (0.34, 0.06, 0.15),
        0.0,
    ),
    (
        "grasp",
        (0.26, 0.06, 0.055),
        (0.26, 0.06, 0.018),
        (0.34, 0.06, 0.15),
        1.0,
    ),
    (
        "lift",
        (0.26, 0.06, 0.16),
        (0.26, 0.06, 0.12),
        (0.34, 0.06, 0.15),
        1.0,
    ),
    (
        "place",
        (0.34, 0.06, 0.20),
        (0.34, 0.06, 0.15),
        (0.34, 0.06, 0.15),
        0.0,
    ),
)


def _make_env(env_id: str):
  env = env_registry.get(env_id).env_builder(nworld=1, device="cpu", impl="cpp")
  env.reset()
  return env


def _robot_joint_names(env_id: str) -> tuple[list[str], str, str]:
  if env_id == "SO101PickCube":
    return (
        [f"robot/{name}" for name in so101_constant.JOINTS[:5]],
        "robot/gripper",
        "robot/gripperframe",
    )
  return (
      [f"robot/{name}" for name in fr3_constant.ARM_JOINTS],
      f"robot/robotiq/{fr3_constant.GRIPPER_JOINT}",
      fr3_constant.EE_SITE,
  )


def _set_joint_position(model, data, joint_name: str, value: float) -> None:
  joint_id = model.joint(joint_name).id
  data.qpos[model.jnt_qposadr[joint_id]] = value


def _solve_position_ik(
    model,
    data,
    site_name: str,
    arm_joint_names: list[str],
    target: np.ndarray,
) -> float:
  """Solves damped least-squares position IK; returns final position error."""
  site_id = model.site(site_name).id
  joint_ids = [model.joint(name).id for name in arm_joint_names]
  qpos_ids = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
  dof_ids = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
  lower = model.jnt_range[joint_ids, 0] + 1e-5
  upper = model.jnt_range[joint_ids, 1] - 1e-5
  jacp = np.zeros((3, model.nv))
  jacr = np.zeros((3, model.nv))

  for _ in range(600):
    mujoco.mj_forward(model, data)
    error = target - data.site(site_id).xpos
    error_norm = float(np.linalg.norm(error))
    if error_norm < 5e-5:
      return error_norm

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jacobian = jacp[:, dof_ids]
    damping = 1e-4
    delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping * np.eye(3), error
    )
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > 0.15:
      delta *= 0.15 / delta_norm
    data.qpos[qpos_ids] = np.clip(data.qpos[qpos_ids] + delta, lower, upper)

  mujoco.mj_forward(model, data)
  error_norm = float(np.linalg.norm(target - data.site(site_id).xpos))
  if error_norm > 0.002:
    raise RuntimeError(
        f"IK failed for site {site_name} at {target.tolist()}; "
        f"residual={error_norm:.5f} m"
    )
  return error_norm


def _set_cube_and_goal(model, data, cube_pos, goal_pos) -> None:
  cube_joint = model.joint("cube/cube_freejoint").id
  cube_qpos = model.jnt_qposadr[cube_joint]
  cube_qvel = model.jnt_dofadr[cube_joint]
  data.qpos[cube_qpos : cube_qpos + 7] = [*cube_pos, 1.0, 0.0, 0.0, 0.0]
  data.qvel[cube_qvel : cube_qvel + 6] = 0.0

  target_body = model.body("target/target").id
  target_mocap = model.body_mocapid[target_body]
  if target_mocap < 0:
    raise RuntimeError("Expected target/target to be a mocap body")
  data.mocap_pos[target_mocap] = goal_pos


def _set_state(
    env, env_id: str, phase
) -> tuple[dict[str, torch.Tensor], float]:
  model = env.device_layer.sim.mj_model
  data = env.device_layer.sim.mj_data
  arm_joint_names, gripper_joint_name, ee_site_name = _robot_joint_names(env_id)
  phase_name, ee_target, cube_pos, goal_pos, closed_fraction = phase

  if phase_name == "approach":
    if env_id == "SO101PickCube":
      for name, value in zip(arm_joint_names, so101_constant.HOVER_POSE[:5]):
        _set_joint_position(model, data, name, value)
    else:
      for name, value in zip(arm_joint_names, fr3_constant.HOME_QPOS[:7]):
        _set_joint_position(model, data, name, value)

  data.qvel[:] = 0.0
  residual = _solve_position_ik(
      model,
      data,
      ee_site_name,
      arm_joint_names,
      np.asarray(ee_target, dtype=np.float64),
  )

  if env_id == "SO101PickCube":
    open_pos, closed_pos = 0.80, 0.15
  else:
    open_pos, closed_pos = 0.0, 0.8
  gripper_pos = open_pos + closed_fraction * (closed_pos - open_pos)
  _set_joint_position(model, data, gripper_joint_name, gripper_pos)
  _set_cube_and_goal(model, data, cube_pos, goal_pos)
  mujoco.mj_forward(model, data)

  env.device_layer.update_measurements()
  measurements = env.device_layer.get_measurements()
  features = env.task_layer.compute_all_features(measurements)
  reward = env.task_layer.compute_reward(features)
  if not isinstance(reward, dict) or "terms" not in reward:
    raise RuntimeError("Configured reward provider did not return named terms")
  for term_name, term_value in reward["terms"].items():
    if not torch.isfinite(term_value).all():
      raise RuntimeError(
          f"{env_id} produced a non-finite {term_name} reward "
          f"during the {phase_name} phase"
      )
  return reward["terms"], residual


def main(argv: list[str]) -> None:
  del argv
  envs = {
      "SO101PickCube": _make_env("SO101PickCube"),
      "FR3RobotiqPickCube": _make_env("FR3RobotiqPickCube"),
  }
  failed = False
  logging.info(
      "Shared raw reward terms compared at deterministic privileged states "
      "(tolerance %.4f).",
      _TOLERANCE.value,
  )
  logging.info(
      "Phase       Term           SO101       FR3       |diff|     IK residuals"
  )

  for phase in _PHASES:
    phase_name = phase[0]
    terms_by_env = {}
    residuals = {}
    for env_id, env in envs.items():
      terms_by_env[env_id], residuals[env_id] = _set_state(env, env_id, phase)

    for term_name in _SHARED_TERMS:
      so_value = float(terms_by_env["SO101PickCube"][term_name][0])
      fr_value = float(terms_by_env["FR3RobotiqPickCube"][term_name][0])
      difference = abs(so_value - fr_value)
      passed = difference <= _TOLERANCE.value
      failed |= not passed
      logging.info(
          "%-11s %-14s %9.5f  %9.5f  %8.5f  %7.5f / %7.5f m %s",
          phase_name,
          term_name,
          so_value,
          fr_value,
          difference,
          residuals["SO101PickCube"],
          residuals["FR3RobotiqPickCube"],
          "PASS" if passed else "FAIL",
      )

    # These terms intentionally depend on robot geometry or embodiment.
    for term_name in (
        "joint_pos_limits",
        "floor_contact",
        "arm_nominal_pose",
        "ee_nominal_orientation",
    ):
      logging.info(
          "%-11s %-14s %9.5f  %9.5f  (robot-specific)",
          phase_name,
          term_name,
          float(terms_by_env["SO101PickCube"][term_name][0]),
          float(terms_by_env["FR3RobotiqPickCube"][term_name][0]),
      )

  for env in envs.values():
    env.close()
  if failed:
    raise RuntimeError(
        "SO101 and FR3 shared task rewards differ beyond tolerance; "
        "see the per-phase report above."
    )
  logging.info("All shared task reward terms are comparable.")


if __name__ == "__main__":
  app.run(main)
