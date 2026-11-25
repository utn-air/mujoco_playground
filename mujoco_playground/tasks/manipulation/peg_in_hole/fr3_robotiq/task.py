"""FR3 + Robotiq peg-in-hole task configuration."""

from collections.abc import Mapping
from pathlib import Path

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import mujoco
from mujoco_playground.robots.fr3_robotiq.sim import constant
from mujoco_playground.robots.fr3_robotiq.sim import scene
from mujoco_playground.tasks.manipulation.peg_in_hole import mdp
from mujoco_playground.tasks.manipulation.peg_in_hole import process_mjcf
from mujoco_playground.tasks.manipulation.peg_in_hole import reset
from mujoco_playground.tasks.manipulation.peg_in_hole import task as peg_task
from mujoco_playground.tasks.manipulation.peg_in_hole import world_assets
import numpy as np
from reaf.core import commands_processor
from reaf.simulation import torch as sim_device
from reaf.simulation.torch import action_history
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch._src.core import device as device_module
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch

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
_ACTOR_HISTORY_FRAMES = 30


class _GripperKinematicsDevice(device_module.Device):
  """Measures gripper position, Euler orientation, and world-frame twist."""

  _SENSOR_SUFFIXES = ("pos", "quat", "linvel", "angvel")

  def __init__(self, *, site_name: str):
    self._site_name = site_name
    self._sensor_names = {
        suffix: f"peg_in_hole_ee_{suffix}" for suffix in self._SENSOR_SUFFIXES
    }
    self._sensor_slices: dict[str, slice] = {}

  @property
  def name(self) -> str:
    return "ee_kinematics"

  def configure_spec(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
    for suffix, sensor_type in zip(
        self._SENSOR_SUFFIXES,
        (
            mujoco.mjtSensor.mjSENS_FRAMEPOS,
            mujoco.mjtSensor.mjSENS_FRAMEQUAT,
            mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
        ),
        strict=True,
    ):
      spec.add_sensor(
          name=self._sensor_names[suffix],
          type=sensor_type,
          objtype=mujoco.mjtObj.mjOBJ_SITE,
          objname=self._site_name,
      )
    return spec

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    for suffix, name in self._sensor_names.items():
      sensor = simulator.mj_model.sensor(name)
      start = int(sensor.adr[0])
      self._sensor_slices[suffix] = slice(start, start + int(sensor.dim[0]))

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {"ee_kinematics/pose_twist": specs.Array((12,), np.float32)}

  def get_measurements(
      self, simulator: simulator_module.Simulator
  ) -> Mapping[str, torch.Tensor]:
    sensors = simulator.data.sensordata
    pos = sensors[:, self._sensor_slices["pos"]]
    quat = sensors[:, self._sensor_slices["quat"]]
    w, x, y, z = quat.unbind(dim=-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    euler = torch.stack((roll, pitch, yaw), dim=-1)
    twist = torch.cat(
        (
            sensors[:, self._sensor_slices["linvel"]],
            sensors[:, self._sensor_slices["angvel"]],
        ),
        dim=-1,
    )
    return {"ee_kinematics/pose_twist": torch.cat((pos, euler, twist), dim=-1)}


class _ArmTargetProcessor(commands_processor.CommandsProcessor):
  """Adds the fixed closed-gripper command to seven arm position targets."""

  @property
  def name(self) -> str:
    return "arm_target"

  def consumed_commands_spec(
      self,
  ) -> Mapping[str, gdmr_types.AnyArraySpec]:
    return {
        "actions": gdmr_types.UnboundedArraySpec(
            shape=(len(constant.ARM_ACTUATORS),), dtype=np.float32
        )
    }

  def produced_commands_keys(self) -> set[str]:
    return {"robot/ctrl"}

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    del env_mask

  def process_commands(
      self, consumed_commands: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    arm_targets = consumed_commands["actions"]
    closed_gripper = torch.full_like(arm_targets[:, :1], 255.0)
    return {"robot/ctrl": torch.cat((arm_targets, closed_gripper), dim=-1)}


def _actor_state(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """Returns the 33D actor state, excluding the gripper actuator."""
  return torch.cat(
      (
          features["robot_state/joint_pos"][:, :7],
          features["robot_state/joint_vel"][:, :7],
          features["ee_kinematics/pose_twist"],
          features["actions"][:, :7],
      ),
      dim=-1,
  )


def _peg_pose_relative_to_plate(
    features: Mapping[str, torch.Tensor],
) -> torch.Tensor:
  """Returns peg translation and roll-pitch-yaw in the plate frame."""
  plate_rot = features["peg_hole_geometry/plate_xmat"].reshape(-1, 3, 3)
  peg_rot = features["peg_hole_geometry/peg_xmat"].reshape(-1, 3, 3)
  plate_rot_t = plate_rot.transpose(-1, -2)
  relative_pos = torch.einsum(
      "bij,bj->bi",
      plate_rot_t,
      features["peg_hole_geometry/peg_pos"]
      - features["peg_hole_geometry/plate_pos"],
  )
  relative_rot = plate_rot_t @ peg_rot
  roll = torch.atan2(relative_rot[:, 2, 1], relative_rot[:, 2, 2])
  pitch = torch.asin(torch.clamp(-relative_rot[:, 2, 0], -1.0, 1.0))
  yaw = torch.atan2(relative_rot[:, 1, 0], relative_rot[:, 0, 0])
  return torch.cat(
      (relative_pos, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1
  )


def _log_uniform(bounds: tuple[float, float]) -> float:
  return float(np.exp(np.random.uniform(np.log(bounds[0]), np.log(bounds[1]))))


def _robot_spec_with_contact_scales(
    scales: dict[str, float],
    gravity_compensation: bool = True,
) -> mujoco.MjSpec:
  """Builds the FR3 spec with joint impedance and sampled contact parameters."""
  spec = scene.fr3_robotiq()
  if gravity_compensation:
    for body in spec.bodies:
      if body.name != "world":
        body.gravcomp = 1.0
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
    gravity_compensation: bool = True,
    nworld: int = 1,
) -> config.Environment:
  """Builds the peg-in-hole task around the FR3 + Robotiq model.

  When ``generate_mjfit_object`` is true, one randomized peg and fixture pair
  is generated per world and remains fixed across resets. Otherwise,
  ``xml_path`` must point to a preprocessed peg/plate MJCF; relative filenames
  also resolve under
  ``peg_xmls`` when they are not found from the current working directory.
  ``gravity_compensation`` defaults to true and compensates the robot and
  gripper bodies; the free peg and fixture retain their own gravity settings.
  """
  world_paths = None
  world_sites = None
  world_plate_sites = None
  peg_assets = plate_assets = None
  if nworld < 1:
    raise ValueError("nworld must be positive.")
  if generate_mjfit_object:
    output_dir = (
        Path(mjfit_output_dir).expanduser()
        if mjfit_output_dir is not None
        else Path.home() / ".cache" / "mujoco_playground" / "peg_in_hole"
    )
    rng = np.random.default_rng(np.random.randint(0, 2**32))
    world_paths = [
        str(
            process_mjcf.generate_randomized_mjfit_xml(
                output_dir, parameters=mjfit_parameters, rng=rng
            )
        )
        for _ in range(nworld)
    ]
    xml_path = world_paths[0]
    if nworld > 1:
      peg_assets, plate_assets, world_sites, world_rims, world_heights = (
          world_assets.world_geometry(world_paths)
      )
    else:
      world_paths = None
  else:
    peg_xml_dir = Path(__file__).resolve().parents[1] / "peg_xmls"
    if xml_path is None:
      xml_path = str(peg_xml_dir / "peg_with_sites.xml")
    else:
      requested_xml = Path(xml_path).expanduser()
      if not requested_xml.is_absolute() and not requested_xml.is_file():
        packaged_xml = peg_xml_dir / requested_xml
        if packaged_xml.is_file():
          xml_path = str(packaged_xml)

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
    return _robot_spec_with_contact_scales(
        robot_contact_scales, gravity_compensation=gravity_compensation
    )

  cfg = peg_task.peg_in_hole_env(
      xml_path=xml_path,
      plate_pos=plate_pos,
      peg_pos=peg_pos,
      peg_quat=peg_quat,
  )
  cfg.sim.nworld = nworld
  cfg.episode_length_secs = 5.0
  cfg.task_layer.terminations["peg_far_from_plate"] = config.Term(
      mdp.terminate_peg_far_from_plate,
      max_distance=0.10,
  )
  cfg.task_layer.terminations["lost_grasp"] = config.Term(
      mdp.terminate_lost_grasp
  )
  cfg.task_layer.terminations["insertion"] = config.Term(
      mdp.terminate_on_insertion
  )
  peg_site_names, plate_edge_rings, plate_top_z = mdp.site_names_from_spec(
      cfg.base_scene
  )
  if world_paths is not None:
    plate = cfg.base_scene.body("plate")
    for i in range(len(plate_edge_rings[0]), world_rims.shape[1]):
      plate.add_site(
          name=f"plate_edge_0_{i}",
          pos=(0.0, 0.0, 0.0),
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=cfg.base_scene.site("plate_edge_0_0").size,
          rgba=(0.0, 0.8, 1.0, 1.0),
      )
    world_plate_sites = np.concatenate(
        (
            world_rims,
            np.broadcast_to(
                world_heights[:, None, None], (*world_rims.shape[:2], 1)
            ),
        ),
        axis=-1,
    )
    plate_edge_rings = (world_rims,)
    plate_top_z = tuple(world_heights)
  critic_observations = {
      "contact_labels": (
          config.Obs(mdp.contact_loss_labels).configure(output_shape=(64, 3))
      ),
      "peg_pose_relative": (
          config.Obs(_peg_pose_relative_to_plate).configure(output_shape=(6,))
      ),
      "actor_state": config.Obs(_actor_state).configure(output_shape=(33,)),
  }
  actor_observations = {
      "current": config.Obs(_actor_state).configure(output_shape=(33,)),
      "temporal": (
          config.Obs(_actor_state).configure(
              output_shape=(33,),
              history_length=_ACTOR_HISTORY_FRAMES - 1,
          )
      ),
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
      "ee_kinematics": config.Component(
          _GripperKinematicsDevice,
          site_name=constant.EE_SITE,
      ),
      "peg_hole_geometry": config.Component(
          mdp.PegHoleGeometryDevice,
          peg_site_names=peg_site_names,
          world_peg_sites=world_sites,
          world_plate_sites=world_plate_sites,
      ),
      "left_peg_contact": (
          config.Component(
              sim_device.ContactSensorDevice,
              fields=("found",),
              history_length=100,
          ).configure_spec(
              objtype="geom",
              objname=("robot/robotiq/left_pad1", "robot/robotiq/left_pad2"),
              reftype="body",
              refname="peg",
          )
      ),
      "right_peg_contact": (
          config.Component(
              sim_device.ContactSensorDevice,
              fields=("found",),
              history_length=100,
          ).configure_spec(
              objtype="geom",
              objname=("robot/robotiq/right_pad1", "robot/robotiq/right_pad2"),
              reftype="body",
              refname="peg",
          )
      ),
  }
  if world_paths is not None:
    cfg.device_layer.devices["peg_assets"] = config.Component(
        sim_device.MultiAssetDevice,
        body_name="peg",
        assets=peg_assets,
        resample_on_reset=False,
    )
    cfg.device_layer.devices["plate_assets"] = config.Component(
        world_assets.MocapMultiAssetDevice,
        body_name="plate",
        assets=plate_assets,
        resample_on_reset=False,
    )
  cfg.task_layer.command_processors = {
      "arm_target": config.Component(_ArmTargetProcessor),
  }
  cfg.task_layer.command_feature_producers["action_history"] = config.Component(
      action_history.ActionHistory,
      key="actions",
      history_length=0,
      feature_shape=(len(constant.ARM_ACTUATORS),),
  )
  cfg.task_layer.feature_producers = {
      "peg_hole_state": config.Component(
          mdp.PegHoleStateProducer,
          plate_edge_rings=plate_edge_rings,
          plate_top_z=plate_top_z,
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
      world_xml_paths=world_paths,
      robot_spec_fn=robot_spec_fn if world_paths is not None else None,
  )
  cfg.observation_adapter = {
      "policy": config.ObsGroup(terms=actor_observations),
      "critic": config.ObsGroup(terms=critic_observations),
  }
  return cfg
