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

"""Runs a deterministic scripted peg drop and reports its site labels."""

from collections.abc import Sequence
from contextlib import nullcontext
import math
from pathlib import Path
import time

from absl import app
from absl import flags
from mujoco_playground._src.rl import env_registry
import torch

_DEFAULT_XML = Path(__file__).resolve().parents[4] / "peg_with_sites.xml"
_XML_PATH = flags.DEFINE_string(
    "xml_path",
    str(_DEFAULT_XML),
    "Processed MJCF containing peg and plate sites.",
)
_VISUALIZE = flags.DEFINE_bool(
    "visualize", False, "Open the MuJoCo viewer while running the script."
)
_POSES = (
    ("free", 0.005, 2),
    ("contact", 0.0005, 2),
    ("catch", -0.0005, 3),
    ("inserted", -0.003, 3),
)


def _set_peg_clearance(
    model, data, clearance: float, tilt_degrees: float = 0.0
) -> None:
  """Teleports the peg so its bottom sample ring has the requested clearance."""
  plate_id = model.body("plate").id
  peg_joint_id = model.joint("peg_joint").id
  peg_qpos = model.jnt_qposadr[peg_joint_id]
  peg_dof = model.jnt_dofadr[peg_joint_id]
  peg_site_ids = [model.site(f"peg_{i}").id for i in range(64)]
  edge_site_id = model.site("plate_edge_0_0").id

  plate_pos = data.xpos[plate_id]
  plate_rot = data.xmat[plate_id].reshape(3, 3)
  plate_top_local_z = model.site_pos[edge_site_id, 2]
  tilt = math.radians(tilt_degrees)
  cos_tilt = math.cos(tilt)
  sin_tilt = math.sin(tilt)
  peg_sites = model.site_pos[peg_site_ids]
  rotated_site_z = sin_tilt * peg_sites[:, 1] + cos_tilt * peg_sites[:, 2]
  peg_bottom_local_z = rotated_site_z.min()
  peg_pos_plate = [
      0.0,
      0.0,
      plate_top_local_z + clearance - peg_bottom_local_z,
  ]
  data.qpos[peg_qpos : peg_qpos + 3] = plate_pos + plate_rot @ peg_pos_plate
  data.qpos[peg_qpos + 3 : peg_qpos + 7] = (
      math.cos(tilt / 2),
      math.sin(tilt / 2),
      0.0,
      0.0,
  )
  data.qvel[peg_dof : peg_dof + 6] = 0.0


def _report_step(env, action, step: int, phase: str, viewer=None) -> None:
  env.step(action)
  if viewer is not None:
    viewer.sync()
    time.sleep(0.2)
  features = env._latest_features
  labels = features["peg_hole_state/site_labels"]
  free_count = int(torch.sum(labels == -1).item())
  contact_count = int(torch.sum(labels == 0).item())
  inside_count = int(torch.sum(labels == 1).item())
  catch = float(features["peg_hole_state/catch_event"][0].item())
  insertion = float(features["peg_hole_state/insertion_event"][0].item())
  print(
      f"step={step:02d} phase={phase:8s} "
      f"free={free_count:2d} contact={contact_count:2d} "
      f"inside={inside_count:2d} "
      f"catch_reward={catch:.0f} insertion_reward={insertion:.0f}"
  )


def main(argv: Sequence[str]) -> None:
  del argv
  xml_path = Path(_XML_PATH.value).expanduser().resolve()
  if not xml_path.is_file():
    raise app.UsageError(f"MJCF file does not exist: {xml_path}")

  env = env_registry.get("FR3RobotiqPegInHole").env_builder(
      xml_path=str(xml_path), nworld=1, device="cpu", impl="cpp"
  )
  try:
    env.reset()
    model = env.device_layer.sim.mj_model
    data = env.device_layer.sim.mj_data
    # Keep the directly scripted peg poses fixed while the robot sim advances.
    model.opt.gravity[:] = 0.0
    hold_q = [
        float(data.joint(f"robot/fr3_joint{i}").qpos[0]) for i in range(1, 8)
    ]
    action = {"actions": torch.tensor([hold_q], dtype=torch.float32)}
    step = 0
    if _VISUALIZE.value:
      import mujoco.viewer

      viewer_context = mujoco.viewer.launch_passive(model, data)
    else:
      viewer_context = nullcontext()
    with viewer_context as viewer:
      for phase, clearance, repeats in _POSES:
        for _ in range(repeats):
          if viewer is not None and not viewer.is_running():
            return
          _set_peg_clearance(model, data, clearance)
          _report_step(env, action, step, phase, viewer)
          step += 1
        if phase == "contact" and (viewer is None or viewer.is_running()):
          _set_peg_clearance(model, data, -0.0005, tilt_degrees=3.0)
          _report_step(env, action, step, "tilted", viewer)
          step += 1
  finally:
    env.close()


if __name__ == "__main__":
  app.run(main)
