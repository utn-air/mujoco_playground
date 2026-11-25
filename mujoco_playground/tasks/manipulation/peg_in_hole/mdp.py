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

"""Site labels and rewards for peg-in-hole insertion."""

from collections.abc import Mapping
import re

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation.torch import Device
from reaf.simulation.torch import termination_checker
from reaf.simulation.torch._src.core import feature_producer
from reaf.simulation.torch._src.core import simulator as simulator_module
import torch

_SITE_COUNT = 64
_CATCH_SITE_COUNT = 31
_CATCH_DEPTH = 0.001
_CATCH_HEIGHT_TOLERANCE = 0
_FREE_HEIGHT = 0.001
_INSERTION_DEPTH = 0.003
_TERMINATE = termination_checker.TerminationResult.TERMINATE
_DO_NOT_TERMINATE = termination_checker.TerminationResult.DO_NOT_TERMINATE


def terminate_peg_far_from_plate(
    features: Mapping[str, torch.Tensor], max_distance: float = 0.05
) -> torch.Tensor:
  """Terminates when the peg insertion end is too far from the fixture."""
  distance = torch.linalg.vector_norm(
      features["peg_hole_geometry/peg_sites"].mean(dim=-2)
      - features["peg_hole_geometry/plate_pos"],
      dim=-1,
  )
  return torch.where(distance > max_distance, _TERMINATE, _DO_NOT_TERMINATE)


def terminate_lost_grasp(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
  """Terminate if either jaw has no peg contact for the history window."""
  left = features["left_peg_contact/found_history"].gt(0).any(dim=(1, 2, 3))
  right = features["right_peg_contact/found_history"].gt(0).any(dim=(1, 2, 3))
  settled = features["peg_hole_geometry/episode_step"] >= 5
  return torch.where(settled & ~(left & right), _TERMINATE, _DO_NOT_TERMINATE)


def terminate_on_insertion(
    features: Mapping[str, torch.Tensor],
) -> torch.Tensor:
  """Terminate on the same step as the one-shot insertion reward."""
  inserted = features["peg_hole_state/insertion_event"] > 0
  return torch.where(inserted, _TERMINATE, _DO_NOT_TERMINATE)


def site_names_from_spec(
    spec: mujoco.MjSpec,
) -> tuple[tuple[str, ...], tuple[tuple[tuple[float, float], ...], ...], float]:
  """Reads the 64 peg sites and ordered plate edge rings from an MJCF spec."""
  peg = spec.body("peg")
  plate = spec.body("plate")
  if peg is None or plate is None:
    raise ValueError("The MJCF must define bodies named 'peg' and 'plate'.")

  peg_sites = {}
  for site in peg.sites:
    match = re.fullmatch(r"peg_(\d+)", site.name)
    if match:
      peg_sites[int(match.group(1))] = site.name
  if sorted(peg_sites) != list(range(_SITE_COUNT)):
    raise ValueError(
        f"Expected peg_0 through peg_{_SITE_COUNT - 1}; found "
        f"{len(peg_sites)} peg sites. Run process_mjcf.py first."
    )

  edge_sites: dict[int, dict[int, object]] = {}
  for site in plate.sites:
    match = re.fullmatch(r"plate_edge_(\d+)_(\d+)", site.name)
    if match:
      ring_id, point_id = (int(value) for value in match.groups())
      edge_sites.setdefault(ring_id, {})[point_id] = site
  if not edge_sites:
    raise ValueError(
        "The plate has no plate_edge_<ring>_<point> sites; "
        "run process_mjcf.py first."
    )

  rings = []
  for ring_id in range(max(edge_sites) + 1):
    ring = edge_sites.get(ring_id, {})
    if sorted(ring) != list(range(len(ring))) or len(ring) < 3:
      raise ValueError(f"Plate edge ring {ring_id} has invalid point indices.")
    rings.append(tuple(tuple(ring[i].pos[:2]) for i in range(len(ring))))
  plate_top_z = float(next(iter(edge_sites[0].values())).pos[2])
  return (
      tuple(peg_sites[i] for i in range(_SITE_COUNT)),
      tuple(rings),
      plate_top_z,
  )


def classify_sites(
    peg_sites_plate: torch.Tensor,
    plate_edge_rings: tuple[torch.Tensor, ...],
    plate_top_z: float | torch.Tensor,
    free_height: float = _FREE_HEIGHT,
) -> torch.Tensor:
  """Labels peg sites geometrically as free (-1), contact (0), or inside (1).

  Args:
    peg_sites_plate: Peg sample points in plate coordinates, shape (..., 64, 3).
    plate_edge_rings: Ordered XY vertices, shared or batched by world.
    plate_top_z: Plate top height, scalar or shape (nworld, 1).
    free_height: Distance threshold from the plate top and rim.

  The opening uses an even-odd fill rule across its rings. Thus the first ring
  is the opening boundary and additional rings mark islands or holes.
  Sites below the top plane and inside the opening are inside. Sites more than
  ``free_height`` above the top plane are free. Sites above the opening are
  also free when their distance to the rim exceeds ``free_height``. All other
  sites are contact. These labels do not depend on MuJoCo collision contacts.
  """
  labels, _ = _classify_sites_and_opening(
      peg_sites_plate, plate_edge_rings, plate_top_z, free_height
  )
  print(labels)
  return labels


def _classify_sites_and_opening(
    peg_sites_plate: torch.Tensor,
    plate_edge_rings: tuple[torch.Tensor, ...],
    plate_top_z: float | torch.Tensor,
    free_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns site labels and the shared XY opening mask."""
  xy = peg_sites_plate[..., :2]
  inside_opening = torch.zeros(
      xy.shape[:-1], dtype=torch.bool, device=xy.device
  )
  rim_distance_sq = torch.full_like(xy[..., 0], float("inf"))
  px = xy[..., 0]
  py = xy[..., 1]
  for ring in plate_edge_rings:
    start = ring if ring.ndim == 2 else ring[:, None, :, :]
    end = torch.roll(start, shifts=-1, dims=-2)
    xi, yi = start[..., 0], start[..., 1]
    xj, yj = end[..., 0], end[..., 1]
    crosses_y = (yi > py[..., None]) != (yj > py[..., None])
    x_intersection = xi + (py[..., None] - yi) * (xj - xi) / torch.where(
        yj != yi, yj - yi, torch.ones_like(yj)
    )
    crosses_x = px[..., None] < x_intersection
    ring_inside = torch.remainder(
        torch.sum(crosses_y & crosses_x, dim=-1), 2
    ).bool()
    inside_opening ^= ring_inside

    edge = end - start
    offset = xy[..., None, :] - start
    edge_length_sq = (edge * edge).sum(dim=-1)
    projection = (offset * edge).sum(dim=-1) / edge_length_sq.clamp_min(
        torch.finfo(xy.dtype).tiny
    )
    nearest_offset = offset - projection.clamp(0.0, 1.0)[..., None] * edge
    ring_distance_sq = (
        (nearest_offset * nearest_offset).sum(dim=-1).amin(dim=-1)
    )
    rim_distance_sq = torch.minimum(rim_distance_sq, ring_distance_sq)

  z = peg_sites_plate[..., 2]
  inside = inside_opening & (z < plate_top_z)
  free = z > plate_top_z + free_height
  free |= (
      inside_opening
      & (z >= plate_top_z)
      & (rim_distance_sq + (z - plate_top_z).square() > free_height**2)
  )
  return torch.where(inside, 1.0, torch.where(free, -1.0, 0.0)), inside_opening


class PegHoleGeometryDevice(Device):
  """Exposes peg-site positions and the plate frame from MuJoCo."""

  def __init__(
      self,
      *,
      name: str,
      peg_site_names: tuple[str, ...],
      world_peg_sites: np.ndarray | None = None,
      world_plate_sites: np.ndarray | None = None,
  ):
    self._name = name
    self._peg_site_names = peg_site_names
    self._world_peg_sites = world_peg_sites
    self._world_plate_sites = world_plate_sites
    self._peg_site_ids: list[int] = []
    self._plate_body_id: int | None = None
    self._peg_body_id: int | None = None

  @property
  def name(self) -> str:
    return self._name

  def initialize(self, simulator: simulator_module.Simulator) -> None:
    model = simulator.mj_model
    self._peg_site_ids = [model.site(name).id for name in self._peg_site_names]
    if self._world_peg_sites is not None:
      if len(self._world_peg_sites) != simulator.nworld:
        raise ValueError("Expected one set of peg sites per world.")
      target = simulator.model.site_pos
      target[:, self._peg_site_ids] = torch.as_tensor(
          self._world_peg_sites, dtype=target.dtype, device=target.device
      )
    if self._world_plate_sites is not None:
      ids = [
          model.site(f"plate_edge_0_{i}").id
          for i in range(self._world_plate_sites.shape[1])
      ]
      target = simulator.model.site_pos
      target[:, ids] = torch.as_tensor(
          self._world_plate_sites, dtype=target.dtype, device=target.device
      )
    self._plate_body_id = model.body("plate").id
    self._peg_body_id = model.body("peg").id

  def reset(
      self,
      simulator: simulator_module.Simulator,
      env_mask: torch.Tensor | None = None,
  ) -> None:
    del simulator, env_mask

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {
        f"{self._name}/peg_sites": specs.Array(
            shape=(_SITE_COUNT, 3), dtype=np.float32
        ),
        f"{self._name}/plate_pos": specs.Array(shape=(3,), dtype=np.float32),
        f"{self._name}/plate_xmat": specs.Array(shape=(9,), dtype=np.float32),
        f"{self._name}/peg_pos": specs.Array(shape=(3,), dtype=np.float32),
        f"{self._name}/peg_xmat": specs.Array(shape=(9,), dtype=np.float32),
        f"{self._name}/episode_step": specs.Array(shape=(), dtype=np.int64),
    }

  def get_measurements(
      self, simulator: simulator_module.Simulator
  ) -> Mapping[str, torch.Tensor]:
    data = simulator.data
    return {
        f"{self._name}/episode_step": simulator.clock.episode_step,
        f"{self._name}/peg_sites": data.site_xpos[:, self._peg_site_ids],
        f"{self._name}/plate_pos": data.xpos[:, self._plate_body_id],
        f"{self._name}/plate_xmat": (
            data.xmat[:, self._plate_body_id].reshape(-1, 9)
        ),
        f"{self._name}/peg_pos": data.xpos[:, self._peg_body_id],
        f"{self._name}/peg_xmat": (
            data.xmat[:, self._peg_body_id].reshape(-1, 9)
        ),
    }


class PegHoleStateProducer(feature_producer.FeatureProducer):
  """Computes site labels and one-shot catch/insertion rewards per policy step."""

  def __init__(
      self,
      *,
      name: str,
      plate_edge_rings: tuple[tuple | torch.Tensor, ...],
      plate_top_z: float | tuple[float, ...] | torch.Tensor,
      nworld: int = 1,
      xpu_device: str = "cpu",
      peg_sites_key: str = "peg_hole_geometry/peg_sites",
      plate_pos_key: str = "peg_hole_geometry/plate_pos",
      plate_xmat_key: str = "peg_hole_geometry/plate_xmat",
      catch_site_count: int = _CATCH_SITE_COUNT,
      insertion_depth: float = _INSERTION_DEPTH,
      free_height: float = _FREE_HEIGHT,
  ):
    super().__init__()
    if not 1 <= catch_site_count <= _SITE_COUNT:
      raise ValueError(f"catch_site_count must be between 1 and {_SITE_COUNT}.")
    self._name = name
    self._device = torch.device(xpu_device)
    self._plate_edge_rings = tuple(
        torch.as_tensor(ring, dtype=torch.float32, device=self._device)
        for ring in plate_edge_rings
    )
    self._plate_top_z = torch.as_tensor(
        plate_top_z, dtype=torch.float32, device=self._device
    )
    if self._plate_top_z.ndim:
      if self._plate_top_z.numel() != nworld:
        raise ValueError("Expected one plate top height per world.")
      self._plate_top_z = self._plate_top_z.reshape(nworld, 1)
    self._peg_sites_key = peg_sites_key
    self._plate_pos_key = plate_pos_key
    self._plate_xmat_key = plate_xmat_key
    self._catch_site_count = catch_site_count
    self._insertion_depth = insertion_depth
    self._free_height = free_height
    self._caught = torch.zeros(nworld, dtype=torch.bool, device=self._device)
    self._inserted = torch.zeros(nworld, dtype=torch.bool, device=self._device)
    self._site_labels = torch.zeros(
        (nworld, _SITE_COUNT), dtype=torch.float32, device=self._device
    )
    self._catch_event = torch.zeros(
        nworld, dtype=torch.float32, device=self._device
    )
    self._insertion_event = torch.zeros_like(self._catch_event)
    self._inside_count = torch.zeros_like(self._catch_event)
    self._mean_inside_depth = torch.zeros_like(self._catch_event)

  @property
  def name(self) -> str:
    return self._name

  def required_features_keys(self) -> set[str]:
    return {
        self._peg_sites_key,
        self._plate_pos_key,
        self._plate_xmat_key,
    }

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    return {
        f"{self._name}/site_labels": specs.Array(
            shape=(_SITE_COUNT,), dtype=np.float32
        ),
        f"{self._name}/catch_event": specs.Array(shape=(), dtype=np.float32),
        f"{self._name}/insertion_event": specs.Array(
            shape=(), dtype=np.float32
        ),
        f"{self._name}/inside_count": specs.Array(shape=(), dtype=np.float32),
        f"{self._name}/mean_inside_depth": specs.Array(
            shape=(), dtype=np.float32
        ),
    }

  def _evaluate(
      self,
      required_features: Mapping[str, torch.Tensor],
      env_mask: torch.Tensor | None,
      *,
      emit_events: bool,
  ) -> None:
    plate_pos = required_features[self._plate_pos_key]
    plate_rot = required_features[self._plate_xmat_key].reshape(-1, 3, 3)
    peg_sites_world = required_features[self._peg_sites_key]
    peg_sites_plate = torch.einsum(
        "bij,bpj->bpi",
        plate_rot.transpose(-1, -2),
        peg_sites_world - plate_pos[:, None, :],
    )
    labels, inside_opening = _classify_sites_and_opening(
        peg_sites_plate,
        self._plate_edge_rings,
        self._plate_top_z,
        free_height=self._free_height,
    )
    inside = labels == 1
    inside_count = inside.sum(dim=-1).to(torch.float32)
    site_depth = (self._plate_top_z - peg_sites_plate[..., 2]).clamp_min(0.0)
    mean_inside_depth = torch.where(inside, site_depth, 0.0).sum(dim=-1) / (
        inside_count.clamp_min(1.0)
    )
    catch_count = (
        inside_opening
        & (
            peg_sites_plate[..., 2]
            <= self._plate_top_z + _CATCH_HEIGHT_TOLERANCE
        )
    ).sum(dim=-1)
    catch = (catch_count >= self._catch_site_count) & (
        mean_inside_depth >= _CATCH_DEPTH
    )
    insertion = (inside_count == _SITE_COUNT) & (
        mean_inside_depth >= self._insertion_depth
    )

    if env_mask is None:
      env_mask = torch.ones_like(self._caught)
    new_catch = (
        catch & ~self._caught if emit_events else torch.zeros_like(catch)
    )
    new_insertion = (
        insertion & ~self._inserted
        if emit_events
        else torch.zeros_like(insertion)
    )
    self._site_labels[env_mask] = labels[env_mask]
    self._inside_count[env_mask] = inside_count[env_mask]
    self._mean_inside_depth[env_mask] = mean_inside_depth[env_mask]
    self._catch_event[env_mask] = new_catch[env_mask].to(torch.float32)
    self._insertion_event[env_mask] = new_insertion[env_mask].to(torch.float32)
    if emit_events:
      self._caught[env_mask] |= catch[env_mask]
      self._inserted[env_mask] |= insertion[env_mask]

  def reset(self, env_mask: torch.Tensor | None = None) -> None:
    if env_mask is None:
      env_mask = torch.ones_like(self._caught)
    self._caught[env_mask] = False
    self._inserted[env_mask] = False
    self._site_labels[env_mask] = 0.0
    self._catch_event[env_mask] = 0.0
    self._insertion_event[env_mask] = 0.0
    self._inside_count[env_mask] = 0.0
    self._mean_inside_depth[env_mask] = 0.0

  def on_step(self, required_features: Mapping[str, torch.Tensor]) -> None:
    self._evaluate(required_features, None, emit_events=True)

  def on_begin_stepping(
      self,
      required_features: Mapping[str, torch.Tensor],
      env_mask: torch.Tensor | None = None,
  ) -> None:
    self._evaluate(required_features, env_mask, emit_events=False)

  def produce_features(
      self, required_features: Mapping[str, torch.Tensor]
  ) -> Mapping[str, torch.Tensor]:
    del required_features
    return {
        f"{self._name}/site_labels": self._site_labels,
        f"{self._name}/catch_event": self._catch_event,
        f"{self._name}/insertion_event": self._insertion_event,
        f"{self._name}/inside_count": self._inside_count,
        f"{self._name}/mean_inside_depth": self._mean_inside_depth,
    }


def reward_from_feature(
    features: Mapping[str, torch.Tensor], feature_key: str
) -> torch.Tensor:
  """Returns a precomputed one-shot reward feature."""
  return features[feature_key]


def contact_loss_labels(
    features: Mapping[str, torch.Tensor],
    site_labels_key: str = "peg_hole_state/site_labels",
) -> torch.Tensor:
  """Returns one-hot [free, contact, inside] targets for each peg site."""
  labels = features[site_labels_key].to(torch.long) + 1
  return torch.nn.functional.one_hot(labels, num_classes=3).to(torch.float32)
