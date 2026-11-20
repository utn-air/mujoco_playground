"""FR3 + Robotiq configuration for the Pick Cube task."""

import functools

import mujoco
from mujoco_playground.robots.fr3_robotiq.sim import constant
from mujoco_playground.robots.fr3_robotiq.sim import scene
from mujoco_playground.tasks.manipulation.pick_cube import mdp
from mujoco_playground.tasks.manipulation.pick_cube import task as pick_task
from reaf.simulation import torch as sim_device
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import processors


@functools.cache
def _nominal_ee_xmat() -> tuple[float, ...]:
  """Returns the gripper pinch-frame orientation at the FR3 home pose."""
  spec = scene.fr3_robotiq()
  model = spec.compile()
  data = mujoco.MjData(model)
  local_joint_names = constant.ARM_JOINTS + [
      f"{constant.GRIPPER_PREFIX}{constant.GRIPPER_JOINT}"
  ]
  for joint_name, value in zip(local_joint_names, constant.HOME_QPOS):
    data.joint(joint_name).qpos = value
  mujoco.mj_forward(model, data)
  return tuple(float(x) for x in data.site("robotiq/pinch").xmat)


def fr3_robotiq_pick_cube_env() -> config.Environment:
  """Builds the shared Pick Cube task around the FR3 + Robotiq model."""
  cfg = pick_task.pick_cube_env()
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 512

  cfg.device_layer.devices["robot_state"] = config.Component(
      sim_device.ArticulatedDevice,
      joint_names=constant.JOINTS,
      home_qpos=constant.HOME_QPOS,
      measurements=[
          "home_qpos",
          "joint_pos",
          "joint_vel",
          "joint_range",
      ],
  ).attach(scene.fr3_robotiq, prefix=constant.ATTACHMENT_PREFIX)

  cfg.device_layer.devices["robot"] = config.Component(
      sim_device.ActuatorDevice,
      targets=constant.ACTUATORS,
      home_ctrl=constant.HOME_CTRL,
  )

  cfg.device_layer.devices["ee"] = config.Component(
      sim_device.SiteDevice,
      name="ee",
      site_name=constant.EE_SITE,
  )

  cfg.device_layer.devices["floor_contact"].configure_spec(
      objname=r"robot/(fr3_link[0-7]|robotiq/.*)"
  )

  policy_terms = cfg.observation_adapter["policy"].terms
  critic_terms = cfg.observation_adapter["critic"].terms
  for terms in (policy_terms, critic_terms):
    terms["joint_pos"].output_shape = (len(constant.JOINTS),)
    terms["joint_vel"].output_shape = (len(constant.JOINTS),)
    terms["last_action"].output_shape = (len(constant.ACTUATORS),)

  cfg.task_layer.command_feature_producers["action_history"].runtime_params[
      "feature_shape"
  ] = (len(constant.ACTUATORS),)

  cfg.task_layer.command_processors = {
      "action_scale": config.Component(
          processors.TransformActions,
          output_key="robot/ctrl",
          scale=constant.ACTION_SCALE,
          offset=constant.ACTION_OFFSET,
      ),
  }

  # FR3 can reach farther than the SO101; keep the cube and goal within a
  # conservative near-field region around the arm base.
  resetter_params = cfg.resetters["pick_cube"].runtime_params
  resetter_params["object_pos_range"] = (
      (0.35, 0.50),
      (-0.15, 0.15),
      (0.018, 0.018),
  )
  resetter_params["target_pos_range"] = (
      (0.30, 0.55),
      (-0.20, 0.20),
      (0.12, 0.25),
  )
  resetter_params["joint_perturbation"] = 0.025

  dr_terms = cfg.resetters["domain_randomization"].runtime_params["terms"]
  dr_terms["geom_friction"].params["pattern"] = r"robot/.*"
  dr_terms["actuator_gains"].params["pattern"] = r"robot/.*"
  dr_terms["dof_damping"].params["pattern"] = r"robot/.*"

  rewards = cfg.task_layer.rewards
  qpos_key = "robot_state/joint_pos"
  qvel_key = "robot_state/joint_vel"
  joint_range_key = "robot_state/joint_range"
  home_qpos_key = "robot_state/home_qpos"
  rewards["joint_vel_hinge"].params["qvel_key"] = qvel_key
  rewards["joint_pos_limits"].params["qpos_key"] = qpos_key
  rewards["joint_pos_limits"].params["joint_range_key"] = joint_range_key
  rewards["grasp"] = config.Reward(
      mdp.reward_gripper_close,
      ee_xpos_key="ee/xpos",
      cube_xpos_key="object/xpos",
      qpos_key=qpos_key,
      gripper_joint_idx=-1,
      open_pos=0.0,
      closed_pos=0.8,
      reaching_std=0.025,
      weight=1.5,
  )
  rewards["arm_nominal_pose"] = config.Reward(
      mdp.reward_arm_nominal_pose,
      qpos_key=qpos_key,
      home_qpos_key=home_qpos_key,
      num_arm_joints=len(constant.ARM_JOINTS),
      std=1.0,
      weight=0.05,
  )
  rewards["ee_nominal_orientation"] = config.Reward(
      mdp.reward_ee_nominal_orientation,
      xmat_key="ee/xmat",
      nominal_xmat=_nominal_ee_xmat(),
      std=0.5,
      weight=0.05,
  )

  # Cube height is centered 1.8 cm above the ground plane.
  rewards["lift"].params["table_z"] = 0.018
  rewards["bringing"].params["table_z"] = 0.018
  rewards["bringing"].params["std"] = 0.08
  rewards["success"].params["distance_threshold"] = 0.05
  rewards["success"].params["lift_height_threshold"] = 0.04
  rewards["floor_contact"].params["penalty_scale"] = -1.0
  rewards["action_rate"].params["penalty_scale"] = -0.02
  rewards["joint_vel_hinge"].params["max_vel"] = 0.4
  rewards["joint_vel_hinge"].params["penalty_scale"] = -0.015

  cfg.task_layer.metrics["success"].params["distance_threshold"] = 0.05
  cfg.task_layer.metrics["success"].params["lift_height_threshold"] = 0.04
  cfg.task_layer.metrics["cube_lifted"].params["lift_height_threshold"] = 0.04

  return cfg
