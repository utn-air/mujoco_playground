"""Paired peg and fixture geometry for individual simulation worlds."""

from collections.abc import Sequence

import mujoco
from mujoco_playground.tasks.manipulation.peg_in_hole import mdp
import numpy as np
from reaf.simulation import torch as sim_device


def _body_asset(spec: mujoco.MjSpec, name: str) -> sim_device.MultiAsset:
  spec = spec.copy()
  for body in list(spec.worldbody.bodies):
    if body.name != name:
      spec.delete(body)
  body = spec.body(name)
  body.mocap = False
  for item in (*list(body.joints), *list(body.sites)):
    spec.delete(item)
  return sim_device.MultiAsset(spec=spec)


def world_geometry(
    paths: Sequence[str],
) -> tuple[
    list[sim_device.MultiAsset],
    list[sim_device.MultiAsset],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
  """Load paired assets and contact geometry in world order."""
  peg_assets, plate_assets, sites, rings, heights = [], [], [], [], []
  for path in paths:
    spec = mujoco.MjSpec.from_file(path)
    names, edges, height = mdp.site_names_from_spec(spec)
    peg_assets.append(_body_asset(spec, "peg"))
    plate_assets.append(_body_asset(spec, "plate"))
    sites.append([spec.site(name).pos.copy() for name in names])
    if len(edges) != 1:
      raise ValueError("Randomized stadium fixtures must have one rim ring.")
    rings.append(np.asarray(edges[0], dtype=np.float32))
    heights.append(height)
  max_edges = max(len(ring) for ring in rings)
  padded = np.stack([
      np.pad(ring, ((0, max_edges - len(ring)), (0, 0)), mode="edge")
      for ring in rings
  ])
  return (
      peg_assets,
      plate_assets,
      np.asarray(sites),
      padded,
      np.asarray(heights),
  )


class MocapMultiAssetDevice(sim_device.MultiAssetDevice):
  """Use MultiAssetDevice with a mocap fixture body."""

  def configure_spec(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
    body = spec.body(self._body_name)
    if body is None or not body.mocap:
      raise ValueError("The fixture must be a mocap body.")
    body.mocap = False
    joint = body.add_freejoint(name=f"{self.name}/temporary_joint")
    try:
      spec = super().configure_spec(spec)
    finally:
      spec.delete(joint)
      body.mocap = True
    return spec
