"""Paired peg and fixture geometry for individual simulation worlds."""

from collections.abc import Sequence
import time

from absl import logging
import mujoco
from mujoco_playground.tasks.manipulation.peg_in_hole import mdp
import numpy as np
from reaf.simulation import torch as sim_device


def log_progress(
    stage: str, completed: int, total: int, started: float
) -> None:
  """Report the first item, every 100 items, and the final item."""
  if completed == 1 or completed % 100 == 0 or completed == total:
    logging.info(
        "Peg-in-hole %s: %d/%d (%.1f s)",
        stage,
        completed,
        total,
        time.monotonic() - started,
    )


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


def _copy_inline_meshes(
    spec: mujoco.MjSpec, prefix: str, asset_spec: mujoco.MjSpec
) -> bool:
  """Copy simple inline convex meshes; leave other assets to the generic path."""
  if asset_spec.assets or asset_spec.materials or asset_spec.textures:
    return False
  body = asset_spec.worldbody.first_body()
  if any(geom.material for geom in body.geoms):
    return False
  names = dict.fromkeys(geom.meshname for geom in body.geoms if geom.meshname)
  meshes = [asset_spec.mesh(name) for name in names]
  if any(
      mesh is None
      or mesh.file
      or mesh.material
      or mesh.needsdf
      or mesh.plugin.active
      or not len(mesh.uservert)
      or len(mesh.userface)
      or len(mesh.usernormal)
      or len(mesh.usertexcoord)
      or len(mesh.userfacenormal)
      or len(mesh.userfacetexcoord)
      or len(mesh.usernode)
      or len(mesh.usertet)
      for mesh in meshes
  ):
    return False
  for mesh in meshes:
    spec.add_mesh(
        name=prefix + mesh.name,
        uservert=list(mesh.uservert),
        scale=mesh.scale,
        refpos=mesh.refpos,
        refquat=mesh.refquat,
        inertia=mesh.inertia,
        smoothnormal=mesh.smoothnormal,
        maxhullvert=mesh.maxhullvert,
    )
  return True


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
  logging.info("Peg-in-hole: preparing geometry for %d worlds", len(paths))
  started = time.monotonic()
  for index, path in enumerate(paths, start=1):
    spec = mujoco.MjSpec.from_file(path)
    names, edges, height = mdp.site_names_from_spec(spec)
    peg_assets.append(_body_asset(spec, "peg"))
    plate_assets.append(_body_asset(spec, "plate"))
    sites.append([spec.site(name).pos.copy() for name in names])
    if len(edges) != 1:
      raise ValueError("Randomized stadium fixtures must have one rim ring.")
    rings.append(np.asarray(edges[0], dtype=np.float32))
    heights.append(height)
    log_progress("geometry preparation", index, len(paths), started)
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


class LoggedMultiAssetDevice(sim_device.MultiAssetDevice):
  """Reports the stages of compiling and installing multi-world geometry."""

  def __init__(self, *, name, body_name, assets, resample_on_reset=False):
    started = time.monotonic()
    logging.info("Peg-in-hole %s: compiling %d assets", name, len(assets))
    super().__init__(
        name=name,
        body_name=body_name,
        assets=assets,
        resample_on_reset=resample_on_reset,
        progress_callback=self._log_asset_progress,
        asset_copy_callback=_copy_inline_meshes,
    )
    logging.info(
        "Peg-in-hole %s: compiled %d assets (%.1f s)",
        name,
        len(assets),
        time.monotonic() - started,
    )

  def configure_spec(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
    started = time.monotonic()
    self._addition_started = started
    logging.info("Peg-in-hole %s: adding assets to the model", self.name)
    spec = super().configure_spec(spec)
    logging.info(
        "Peg-in-hole %s: assets added (%.1f s)",
        self.name,
        time.monotonic() - started,
    )
    return spec

  def _log_asset_progress(self, completed: int, total: int) -> None:
    log_progress(
        f"{self.name} asset addition", completed, total, self._addition_started
    )

  def initialize(self, simulator) -> None:
    started = time.monotonic()
    logging.info("Peg-in-hole %s: assigning world geometry", self.name)
    super().initialize(simulator)
    logging.info(
        "Peg-in-hole %s: world geometry assigned (%.1f s)",
        self.name,
        time.monotonic() - started,
    )


class MocapMultiAssetDevice(LoggedMultiAssetDevice):
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
    logging.info("Peg-in-hole: compiling the shared MuJoCo/Warp model")
    return spec
