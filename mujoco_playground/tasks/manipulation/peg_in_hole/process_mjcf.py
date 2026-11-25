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

"""Adds peg sample sites and the plate-hole outline to an MJCF file."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

from absl import app
from absl import flags
from etils import epath
import mjfit
import mujoco
import numpy as np
from shapely import geometry
from shapely import union_all

_POINT_COUNT = 64
_SITE_RADIUS = 0.0015
_UNION_GRID_SIZE = 1e-8
_MESH_SEAM_TOLERANCE = 1e-7
_INPUT_XML = flags.DEFINE_string("input_xml", None, "Input MJCF file.")
_OUTPUT_XML = flags.DEFINE_string(
    "output_xml",
    None,
    "Output MJCF file (defaults to <input>_with_sites.xml).",
)
_HANDLE_INSIDE = flags.DEFINE_bool(
    "handle-inside",
    False,
    "Include inner peg and plate-opening boundaries.",
)
_PLATE_EDGE_TOLERANCE = flags.DEFINE_float(
    "plate-edge-tolerance",
    0.0001,
    "Maximum outline simplification error in meters (default: 0.1 mm).",
)

_RANDOMIZED_PART_RANGES = {
    "scale": (1.35, 2.45),
    "opening_width": (0.038, 0.091),
    "opening_height": (0.020, 0.054),
    "aspect_ratio": (1.25, 2.5),
    "peg_length": (0.050, 0.060),
    "clearance": (0.001, 0.0015),
    "mass": (0.35, 0.675),
    "fixture_friction": (0.5, 2.0),
    "fixture_contact_timeconst": (0.0035, 0.0071),
}


@dataclass(frozen=True)
class MjfitParameterResolution:
  """Grid spacing for sampled asset parameters (linear values in mm)."""

  scale: float = 0.01
  opening_mm: float = 0.5
  length_mm: float = 0.5
  aspect_ratio: float = 0.01
  clearance_mm: float = 0.05
  mass_kg: float = 0.005
  fixture_friction: float = 0.01
  fixture_contact_timeconst_ms: float = 0.1

  def __post_init__(self) -> None:
    values = (
        self.scale,
        self.opening_mm,
        self.length_mm,
        self.aspect_ratio,
        self.clearance_mm,
        self.mass_kg,
        self.fixture_friction,
        self.fixture_contact_timeconst_ms,
    )
    if not np.isfinite(values).all() or any(value <= 0 for value in values):
      raise ValueError(
          "Every parameter resolution must be finite and positive."
      )


_DEFAULT_PARAMETER_RESOLUTION = MjfitParameterResolution()


def _quantize(
    value: float,
    step: float,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
  """Rounds a value to a grid, optionally aligned to a lower bound."""
  origin = 0.0 if lower is None else lower
  result = origin + round((value - origin) / step) * step
  if lower is not None:
    result = max(result, lower)
  if upper is not None:
    result = min(result, upper)
  return float(result)


def _sample_log_uniform(
    rng: np.random.Generator, lower: float, upper: float
) -> float:
  """Samples x with log(x) uniform between log(lower) and log(upper)."""
  return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))


def sample_mjfit_parameters(
    rng: np.random.Generator | None = None,
    resolution: MjfitParameterResolution = _DEFAULT_PARAMETER_RESOLUTION,
) -> dict[str, float]:
  """Samples paper-specified dimensions for an area-preserving stadium peg.

  The opening ranges describe final dimensions, so scale applies to the plate
  and peg length only. Preserve the sampled stadium area while drawing a
  log-uniform aspect ratio, then reject dimensions outside the stated ranges.
  """
  rng = rng or np.random.default_rng()
  ranges = _RANDOMIZED_PART_RANGES
  opening_step = resolution.opening_mm / 1000
  length_step = resolution.length_mm / 1000
  clearance_step = resolution.clearance_mm / 1000
  timeconst_step = resolution.fixture_contact_timeconst_ms / 1000
  for _ in range(10_000):
    scale = _quantize(
        rng.uniform(*ranges["scale"]),
        resolution.scale,
        *ranges["scale"],
    )
    base_width = _quantize(
        rng.uniform(*ranges["opening_width"]),
        opening_step,
        *ranges["opening_width"],
    )
    base_height = _quantize(
        rng.uniform(*ranges["opening_height"]),
        opening_step,
        *ranges["opening_height"],
    )
    minor = min(base_width, base_height)
    major = max(base_width, base_height)
    area = minor * (major - minor) + np.pi * minor**2 / 4
    aspect_ratio = _quantize(
        _sample_log_uniform(rng, *ranges["aspect_ratio"]),
        resolution.aspect_ratio,
        *ranges["aspect_ratio"],
    )
    height = _quantize(
        np.sqrt(area / (aspect_ratio - 1 + np.pi / 4)), opening_step
    )
    width = _quantize(
        np.sqrt(area / (aspect_ratio - 1 + np.pi / 4)) * aspect_ratio,
        opening_step,
    )
    clearance = _quantize(
        rng.uniform(*ranges["clearance"]),
        clearance_step,
        *ranges["clearance"],
    )
    plate_size = 0.080 * scale
    # Validate the actual dimensions after rounding, including the ratio.
    if (
        ranges["opening_width"][0] <= width <= ranges["opening_width"][1]
        and ranges["opening_height"][0] <= height <= ranges["opening_height"][1]
        and ranges["aspect_ratio"][0]
        <= width / height
        <= ranges["aspect_ratio"][1]
        and height > 2 * clearance
        and width + 2 * clearance < plate_size
    ):
      break
  else:
    raise ValueError(
        "Could not sample a valid peg; check the parameter ranges and"
        " resolution."
    )
  return {
      "scale": float(scale),
      "base_width": float(base_width),
      "base_height": float(base_height),
      "aspect_ratio": float(width / height),
      "width": float(width),
      "height": float(height),
      "peg_length": _quantize(
          rng.uniform(*ranges["peg_length"]) * scale, length_step
      ),
      "clearance": float(clearance),
      "plate_thickness": float(0.010 * scale),
      "plate_size": float(plate_size),
      "mass": _quantize(
          rng.uniform(*ranges["mass"]),
          resolution.mass_kg,
          *ranges["mass"],
      ),
      "fixture_friction": _quantize(
          _sample_log_uniform(rng, *ranges["fixture_friction"]),
          resolution.fixture_friction,
          *ranges["fixture_friction"],
      ),
      "fixture_contact_timeconst": _quantize(
          _sample_log_uniform(rng, *ranges["fixture_contact_timeconst"]),
          timeconst_step,
          *ranges["fixture_contact_timeconst"],
      ),
  }


def _randomized_output_path(
    output_dir: Path,
    parameters: dict[str, float],
    handle_inside: bool,
    plate_edge_tolerance: float,
) -> Path:
  """Builds a readable, exact-parameter cache filename."""
  identity = {
      **parameters,
      "handle_inside": handle_inside,
      "plate_edge_tolerance": plate_edge_tolerance,
  }
  digest = hashlib.sha256(
      json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()[:12]
  readable = (
      f"stadium_w{parameters['width'] * 1000:.2f}mm"
      f"_h{parameters['height'] * 1000:.2f}mm"
      f"_s{parameters['scale']:.4f}"
      f"_ar{parameters['aspect_ratio']:.4f}"
      f"_L{parameters['peg_length'] * 1000:.2f}mm"
      f"_c{parameters['clearance'] * 1000:.3f}mm"
      f"_d{parameters['plate_thickness'] * 1000:.2f}mm"
      f"_p{parameters['plate_size'] * 1000:.2f}mm"
      f"_m{parameters['mass']:.4f}kg"
      f"_mu{parameters['fixture_friction']:.2f}"
      f"_tc{parameters['fixture_contact_timeconst'] * 1000:.2f}ms"
  )
  inside_tag = "_inside" if handle_inside else ""
  return output_dir / f"{readable}{inside_tag}_{digest}_with_sites.xml"


def generate_randomized_mjfit_xml(
    output_dir: str | Path,
    parameters: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
    resolution: MjfitParameterResolution = _DEFAULT_PARAMETER_RESOLUTION,
    handle_inside: bool = False,
    plate_edge_tolerance: float = 0.0001,
) -> Path:
  """Generates and site-processes a randomized MJFit peg/plate asset.

  The generated path is a deterministic cache key of all sampled MJFit and
  site-processing parameters. Passing the same ``parameters`` reuses the
  existing XML. The intermediate MJFit XML is removed after processing.
  """
  output_dir = Path(output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  if parameters is None:
    parameters = sample_mjfit_parameters(rng, resolution)
  output_path = _randomized_output_path(
      output_dir, parameters, handle_inside, plate_edge_tolerance
  )
  if output_path.exists():
    return output_path

  xml = mjfit.peg(
      hole=mjfit.shapes.stadium(parameters["width"], parameters["height"]),
      clearance=parameters["clearance"],
      peg_length=parameters["peg_length"],
      plate_thickness=parameters["plate_thickness"],
      plate_size=parameters["plate_size"],
      peg_mass=parameters["mass"],
      standalone=False,
      floor=False,
      center=True,
  )
  with tempfile.NamedTemporaryFile(
      mode="w",
      suffix=".xml",
      prefix=".mjfit_raw_",
      dir=output_dir,
      delete=False,
  ) as raw_file:
    raw_path = Path(raw_file.name)
    raw_file.write(xml)
  processed_path = raw_path.with_name(raw_path.stem + "_with_sites.xml")
  try:
    spec = mujoco.MjSpec.from_file(str(raw_path))
    plate = spec.body("plate")
    if plate is None:
      raise ValueError("MJFit output has no body named 'plate'.")
    for geom in plate.geoms:
      friction = np.asarray(geom.friction).copy()
      friction[0] = parameters["fixture_friction"]
      geom.friction = friction
      solref = np.asarray(geom.solref).copy()
      solref[0] = parameters["fixture_contact_timeconst"]
      geom.solref = solref
    spec.compile()
    spec.to_file(str(raw_path))
    add_peg_sites(
        str(raw_path),
        str(processed_path),
        handle_inside=handle_inside,
        plate_edge_tolerance=plate_edge_tolerance,
    )
    processed_path.replace(output_path)
  finally:
    raw_path.unlink(missing_ok=True)
    processed_path.unlink(missing_ok=True)
  return output_path


def _body_mesh_vertices(
    model: mujoco.MjModel, body_id: int
) -> list[np.ndarray]:
  """Returns mesh vertices transformed into the body's local frame."""
  mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
  geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
  if not len(geom_ids):
    raise ValueError("The peg body has no collision geoms.")

  meshes = []
  for geom_id in geom_ids:
    if model.geom_type[geom_id] != mesh_type:
      raise ValueError(
          "Every peg collision geom must be a mesh; found geom id "
          f"{geom_id} with type {model.geom_type[geom_id]}."
      )
    mesh_id = model.geom_dataid[geom_id]
    first = model.mesh_vertadr[mesh_id]
    last = first + model.mesh_vertnum[mesh_id]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    vertices = model.mesh_vert[first:last] @ rotation.reshape(3, 3).T
    meshes.append(vertices + model.geom_pos[geom_id])
  return meshes


def _peg_outline(model: mujoco.MjModel) -> tuple[geometry.Polygon, float]:
  """Unions the lowest convex prism profiles and returns their lower edge."""
  peg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
  if peg_id < 0:
    raise ValueError("The MJCF must define a body named 'peg'.")

  mesh_vertices = _body_mesh_vertices(model, peg_id)
  bottom_z = min(float(vertices[:, 2].min()) for vertices in mesh_vertices)
  insertion_meshes = [
      vertices
      for vertices in mesh_vertices
      if abs(float(vertices[:, 2].min()) - bottom_z) <= 1e-8
  ]
  profiles = [
      geometry.MultiPoint(vertices[:, :2]).convex_hull
      for vertices in insertion_meshes
  ]
  outline = union_all(profiles, grid_size=_UNION_GRID_SIZE)
  if not isinstance(outline, geometry.Polygon):
    raise ValueError(
        "The peg meshes must form one connected 2D outline; found "
        f"{outline.geom_type}."
    )
  if outline.is_empty or outline.length == 0:
    raise ValueError("The peg outline is empty or degenerate.")
  return outline, bottom_z


def _plate_hole_outline(
    model: mujoco.MjModel, handle_inside: bool = False
) -> tuple[geometry.Polygon, float]:
  """Extracts the plate's top surface and returns its hole boundary."""
  plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
  if plate_id < 0:
    raise ValueError("The MJCF must define a body named 'plate'.")

  mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
  geom_ids = np.flatnonzero(model.geom_bodyid == plate_id)
  if not len(geom_ids):
    raise ValueError("The plate body has no collision geoms.")

  top_triangles = []
  top_z = -np.inf
  for geom_id in geom_ids:
    if model.geom_type[geom_id] != mesh_type:
      raise ValueError(
          "Every plate collision geom must be a mesh; found geom id "
          f"{geom_id} with type {model.geom_type[geom_id]}."
      )
    mesh_id = model.geom_dataid[geom_id]
    first = model.mesh_vertadr[mesh_id]
    last = first + model.mesh_vertnum[mesh_id]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    vertices = (
        model.mesh_vert[first:last] @ rotation.reshape(3, 3).T
        + model.geom_pos[geom_id]
    )
    mesh_top_z = float(vertices[:, 2].max())
    top_z = max(top_z, mesh_top_z)
    face_first = model.mesh_faceadr[mesh_id]
    face_last = face_first + model.mesh_facenum[mesh_id]
    faces = model.mesh_face[face_first:face_last]
    for face in faces:
      triangle = vertices[face]
      if np.all(np.abs(triangle[:, 2] - mesh_top_z) <= 1e-8):
        top_triangles.append(geometry.Polygon(triangle[:, :2]))

  if not top_triangles:
    raise ValueError("No horizontal top faces were found on the plate.")
  surface = union_all(top_triangles, grid_size=_UNION_GRID_SIZE)
  # Close floating-point seams between adjacent compiled mesh pieces.
  surface = surface.buffer(_MESH_SEAM_TOLERANCE, join_style="mitre").buffer(
      -_MESH_SEAM_TOLERANCE, join_style="mitre"
  )
  solids = list(surface.geoms) if hasattr(surface, "geoms") else [surface]
  solids = [solid for solid in solids if isinstance(solid, geometry.Polygon)]
  openings = []
  for solid in solids:
    for ring in solid.interiors:
      opening = geometry.Polygon(ring)
      if handle_inside:
        islands = [
            other
            for other in solids
            if other is not solid
            and opening.contains(other.representative_point())
        ]
        if islands:
          opening = opening.difference(union_all(islands))
      if not opening.is_empty:
        openings.append(opening)
  if not openings:
    raise ValueError("No through-hole was found in the plate top surface.")
  hole = union_all(openings, grid_size=_UNION_GRID_SIZE)
  if hole.is_empty or hole.length == 0:
    raise ValueError("The plate opening is empty or degenerate.")
  return hole, top_z


def _sample_perimeter(
    outline: geometry.Polygon,
    count: int = _POINT_COUNT,
    handle_inside: bool = False,
) -> np.ndarray:
  """Samples the outline at uniform intervals, optionally excluding holes."""
  rings = [outline.exterior]
  if handle_inside:
    rings.extend(outline.interiors)
  perimeters = [geometry.LineString(ring.coords) for ring in rings]
  if count < len(perimeters):
    raise ValueError("Point count must cover every boundary loop.")
  lengths = np.asarray([perimeter.length for perimeter in perimeters])
  target_counts = lengths / lengths.sum() * count
  point_counts = np.maximum(1, np.floor(target_counts).astype(int))
  while point_counts.sum() > count:
    removable = np.flatnonzero(point_counts > 1)
    index = removable[
        np.argmax(point_counts[removable] - target_counts[removable])
    ]
    point_counts[index] -= 1
  while point_counts.sum() < count:
    index = int(np.argmax(target_counts - point_counts))
    point_counts[index] += 1

  points = []
  for perimeter, loop_count in zip(perimeters, point_counts):
    points.extend(
        perimeter.interpolate(perimeter.length * i / loop_count).coords[0]
        for i in range(loop_count)
    )
  return np.asarray(points, dtype=np.float64)


def add_peg_sites(
    input_xml: str,
    output_xml: str,
    handle_inside: bool = False,
    plate_edge_tolerance: float = 0.0001,
) -> geometry.Polygon:
  """Adds peg sample sites and plate opening edge sites to an MJCF copy."""
  if plate_edge_tolerance < 0:
    raise ValueError("Plate edge tolerance must be nonnegative.")
  spec = mujoco.MjSpec.from_file(input_xml)
  model = spec.compile()
  outline, bottom_z = _peg_outline(model)
  points = _sample_perimeter(outline, handle_inside=handle_inside)
  peg = spec.body("peg")
  if peg is None:
    raise ValueError("The MJCF must define a body named 'peg'.")

  existing_names = {site.name for site in peg.sites}
  collisions = [
      f"peg_{i}" for i in range(_POINT_COUNT) if f"peg_{i}" in existing_names
  ]
  if collisions:
    raise ValueError("The peg already has sites named " + ", ".join(collisions))

  for i, (x, y) in enumerate(points):
    peg.add_site(
        name=f"peg_{i}",
        pos=(float(x), float(y), bottom_z),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(_SITE_RADIUS,),
        rgba=(1.0, 0.8, 0.0, 1.0),
        group=0,
    )

  hole, plate_top_z = _plate_hole_outline(model, handle_inside=handle_inside)
  hole = hole.simplify(plate_edge_tolerance, preserve_topology=True)
  polygons = list(hole.geoms) if hasattr(hole, "geoms") else [hole]
  edge_rings = []
  for polygon in polygons:
    edge_rings.append(polygon.exterior)
    if handle_inside:
      edge_rings.extend(polygon.interiors)
  edge_points = [
      np.asarray(ring.coords[:-1], dtype=np.float64) for ring in edge_rings
  ]
  if any(len(points) < 3 for points in edge_points):
    raise ValueError(
        "Each simplified plate opening edge needs at least 3 points."
    )
  plate = spec.body("plate")
  existing_edge_names = {site.name for site in plate.sites}
  collisions = [
      f"plate_edge_{ring}_{point}"
      for ring, points in enumerate(edge_points)
      for point in range(len(points))
      if f"plate_edge_{ring}_{point}" in existing_edge_names
  ]
  if collisions:
    raise ValueError(
        "The plate already has sites named " + ", ".join(collisions)
    )
  for ring, points in enumerate(edge_points):
    for point, (x, y) in enumerate(points):
      plate.add_site(
          name=f"plate_edge_{ring}_{point}",
          pos=(float(x), float(y), plate_top_z),
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=(_SITE_RADIUS,),
          rgba=(0.0, 0.8, 1.0, 1.0),
          group=0,
      )

  spec.compile()
  spec.to_file(output_xml)
  return outline


def _default_output_path(input_xml: str) -> str:
  path = epath.Path(input_xml)
  return str(path.with_name(f"{path.stem}_with_sites{path.suffix}"))


def main(argv: Sequence[str]) -> None:
  del argv
  if not _INPUT_XML.value:
    raise app.UsageError("--input_xml is required.")
  output_xml = _OUTPUT_XML.value or _default_output_path(_INPUT_XML.value)
  input_path = epath.Path(_INPUT_XML.value).resolve()
  output_path = epath.Path(output_xml).resolve()
  if input_path == output_path:
    raise app.UsageError("--output_xml must differ from --input_xml.")

  outline = add_peg_sites(
      str(input_path),
      str(output_path),
      handle_inside=_HANDLE_INSIDE.value,
      plate_edge_tolerance=_PLATE_EDGE_TOLERANCE.value,
  )
  print(
      f"Wrote {output_path}: {_POINT_COUNT} peg sites and plate edge sites; "
      f"peg perimeter {outline.length:.6f} m."
  )


if __name__ == "__main__":
  app.run(main)
