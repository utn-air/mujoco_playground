"""Composable MjSpec for a Franka FR3 carrying a Robotiq 2F-85."""

import mujoco
from mujoco_playground import asset


def fr3_robotiq() -> mujoco.MjSpec:
  """Loads the Menagerie FR3 and mounts the Robotiq gripper at its flange."""
  spec = asset.get(
      "github:google-deepmind/mujoco_menagerie/franka_fr3/fr3.xml"
  ).spec()
  gripper = asset.get(
      "github:google-deepmind/mujoco_menagerie/robotiq_2f85/2f85.xml"
  ).spec()
  spec.option.cone = gripper.option.cone
  spec.option.impratio = gripper.option.impratio
  spec.attach(gripper, site="attachment_site", prefix="robotiq/")

  # The FR3 keyframe predates the attached gripper and has no gripper control.
  # The environment resetter supplies a complete home state instead.
  for key in list(spec.keys):
    spec.delete(key)

  return spec
