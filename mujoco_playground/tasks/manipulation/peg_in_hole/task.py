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

"""Shared peg-in-hole task configuration."""

import mujoco
from reaf.simulation.torch import flat_scene as config


def peg_in_hole_env(
    xml_path: str,
    plate_pos: tuple[float, float, float] | None = None,
    peg_pos: tuple[float, float, float] | None = None,
    peg_quat: tuple[float, float, float, float] | None = None,
) -> config.Environment:
  """Builds the shared peg-in-hole scene from an MJCF file.

  The XML must define top-level bodies named ``plate`` and ``peg``. The plate
  is placed 45 cm in front of the robot, retaining its XML height, unless
  ``plate_pos`` is provided. By default the peg retains its XML pose relative
  to the plate. ``peg_pos`` overrides its world position, and ``peg_quat``
  overrides its world orientation (MuJoCo wxyz).
  """
  base_scene = mujoco.MjSpec.from_file(xml_path)
  floor = base_scene.geom("floor")
  if floor is not None:
    floor.pos[2] = -0.22
  plate = base_scene.body("plate")
  peg = base_scene.body("peg")
  if plate is None or peg is None:
    missing = [
        name for name, body in (("plate", plate), ("peg", peg)) if body is None
    ]
    raise ValueError(
        "Peg-in-hole XML must define top-level body/bodies named: "
        f"{', '.join(missing)}"
    )

  if plate_pos is None:
    plate_pos = (0.45, 0.0, float(plate.pos[2]))
  plate_offset = tuple(
      target - original for target, original in zip(plate_pos, plate.pos)
  )
  peg_xml_pos = tuple(peg.pos)
  plate.pos = plate_pos
  peg.pos = (
      peg_pos
      if peg_pos is not None
      else tuple(
          position + offset
          for position, offset in zip(peg_xml_pos, plate_offset)
      )
  )
  if peg_quat is not None:
    peg.quat = peg_quat

  return config.Environment(
      base_scene=base_scene,
      sim=config.Sim(
          simulation_timestep=0.001,
          control_timestep=0.02,
          nconmax=128,
          njmax=512,
      ),
  )
