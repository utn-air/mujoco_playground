"""FR3 + Robotiq peg-in-hole task configuration."""

from pathlib import Path

import mujoco
from mujoco_playground.robots.fr3_robotiq.sim import constant
from mujoco_playground.robots.fr3_robotiq.sim import scene
from mujoco_playground.tasks.manipulation.peg_in_hole import mdp
from mujoco_playground.tasks.manipulation.peg_in_hole import process_mjcf
from mujoco_playground.tasks.manipulation.peg_in_hole import reset
from mujoco_playground.tasks.manipulation.peg_in_hole import task as peg_task
import numpy as np
from reaf.simulation import torch as sim_device
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import processors

_FINGER_GEOMS = (
    "robotiq/right_pad1",
    "robotiq/right_pad2",
    "robotiq/left_pad1",
    "robotiq/left_pad2",
)
_FINGER_FRICTION_RANGE = (1.41, 2.0)
_FINGER_CONTACT_TIMECONST_RANGE = (0.78, 1.34)
_GRIPPER_FORCE_CAP_RANGE = (1.41, 2.83)
_ARM_KP = 80.0
_ARM_KD = 20.0


def _log_uniform(bounds: tuple[float, float]) -> float:
  return float(np.exp(np.random.uniform(np.log(bounds[0]), np.log(bounds[1]))))


def _robot_spec_with_contact_scales(
    scales: dict[str, float],
) -> mujoco.MjSpec:
  """Builds the FR3 spec with joint impedance and sampled contact parameters."""
  spec = scene.fr3_robotiq()
  for name in constant.ARM_ACTUATORS:
    actuator = spec.actuator(name)
    if actuator is None:
      raise ValueError(f"FR3/Robotiq spec is missing arm actuator {name!r}.")
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    actuator.gainprm[0] = _ARM_KP
    actuator.biasprm[1] = -_ARM_KP
    actuator.biasprm[2] = -_ARM_KD

  for name in _FINGER_GEOMS:
    geom = spec.geom(name)
    if geom is None:
      raise ValueError(f"FR3/Robotiq spec is missing finger geom {name!r}.")
    friction = np.asarray(geom.friction).copy()
    friction[0] *= scales["finger_friction"]
    geom.friction = friction
    solref = np.asarray(geom.solref).copy()
    solref[0] *= scales["finger_contact_timeconst"]
    geom.solref = solref

  actuator = spec.actuator(constant.GRIPPER_ACTUATOR)
  if actuator is None:
    raise ValueError("FR3/Robotiq spec is missing its gripper actuator.")
  actuator.forcerange = (
      np.asarray(actuator.forcerange).copy() * scales["gripper_force_cap"]
  )
  return spec


def fr3_robotiq_peg_in_hole_env(
    xml_path: str | None = None,
    plate_pos: tuple[float, float, float] | None = None,
    peg_pos: tuple[float, float, float] | None = None,
    peg_quat: tuple[float, float, float, float] | None = None,
    generate_mjfit_object: bool = False,
    mjfit_output_dir: str | None = None,
    mjfit_parameters: dict[str, float] | None = None,
    robot_contact_scales: dict[str, float] | None = None,
) -> config.Environment:
  """Builds the peg-in-hole task around the FR3 + Robotiq model.

  When ``generate_mjfit_object`` is true, one randomized object is generated
  (or loaded from the parameter-based cache) before compiling the environment.
  That object remains fixed across resets. Otherwise, ``xml_path`` must point
  to a preprocessed peg/plate MJCF.
  """
  if generate_mjfit_object:
    output_dir = (
        Path(mjfit_output_dir).expanduser()
        if mjfit_output_dir is not None
        else Path.home() / ".cache" / "mujoco_playground" / "peg_in_hole"
    )
    xml_path = str(
        process_mjcf.generate_randomized_mjfit_xml(
            output_dir,
            parameters=mjfit_parameters,
        )
    )
  elif xml_path is None:
    raise ValueError(
        "xml_path is required unless generate_mjfit_object is enabled."
    )

  if robot_contact_scales is None:
    robot_contact_scales = {
        "finger_friction": _log_uniform(_FINGER_FRICTION_RANGE),
        "finger_contact_timeconst": _log_uniform(
            _FINGER_CONTACT_TIMECONST_RANGE
        ),
        "gripper_force_cap": _log_uniform(_GRIPPER_FORCE_CAP_RANGE),
    }
  else:
    robot_contact_scales = dict(robot_contact_scales)

  def robot_spec_fn() -> mujoco.MjSpec:
    return _robot_spec_with_contact_scales(robot_contact_scales)

  cfg = peg_task.peg_in_hole_env(
      xml_path=xml_path,
      plate_pos=plate_pos,
      peg_pos=peg_pos,
      peg_quat=peg_quat,
  )
  peg_site_names, plate_edge_rings, plate_top_z = mdp.site_names_from_spec(
      cfg.base_scene
  )
  observations = {
      "joint_pos": (
          config.Obs("robot_state/joint_pos").configure(
              output_shape=(len(constant.JOINTS),)
          )
      ),
      "joint_vel": (
          config.Obs("robot_state/joint_vel").configure(
              output_shape=(len(constant.JOINTS),)
          )
      ),
      "ee_pos": config.Obs("ee/xpos").configure(output_shape=(3,)),
  }

  cfg.device_layer.devices = {
      "robot_state": (
          config.Component(
              sim_device.ArticulatedDevice,
              joint_names=constant.JOINTS,
              home_qpos=constant.HOME_QPOS,
              measurements=["home_qpos", "joint_pos", "joint_vel"],
          ).attach(robot_spec_fn, prefix=constant.ATTACHMENT_PREFIX)
      ),
      "robot": config.Component(
          sim_device.ActuatorDevice,
          targets=constant.ACTUATORS,
          home_ctrl=constant.HOME_CTRL,
      ),
      "ee": config.Component(
          sim_device.SiteDevice,
          name="ee",
          site_name=constant.EE_SITE,
      ),
      "peg_hole_geometry": config.Component(
          mdp.PegHoleGeometryDevice,
          peg_site_names=peg_site_names,
      ),
  }
  cfg.task_layer.command_processors = {
      "action_scale": config.Component(
          processors.TransformActions,
          output_key="robot/ctrl",
          scale=constant.ACTION_SCALE,
          offset=constant.ACTION_OFFSET,
      ),
  }
  cfg.task_layer.feature_producers = {
      "peg_hole_state": config.Component(
          mdp.PegHoleStateProducer,
          plate_edge_rings=plate_edge_rings,
          plate_top_z=plate_top_z,
          catch_site_count=16,
          insertion_depth=0.002,
          free_height=0.001,
      ),
  }
  cfg.task_layer.rewards = {
      "catch": config.Reward(
          mdp.reward_from_feature,
          feature_key="peg_hole_state/catch_event",
          weight=1.0,
      ),
      "insertion": config.Reward(
          mdp.reward_from_feature,
          feature_key="peg_hole_state/insertion_event",
          weight=1.0,
      ),
  }
  cfg.resetters["peg_in_hole"] = config.Component(
      reset.PegInHoleResetter,
  )
  cfg.observation_adapter = {
      "policy": config.ObsGroup(terms=observations),
      "critic": config.ObsGroup(terms=observations),
  }
  return cfg
