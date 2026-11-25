"""Tests for processing peg and fixture geometry."""

import xml.etree.ElementTree as ET

import mjfit
import mujoco
from mujoco_playground.tasks.manipulation.peg_in_hole import process_mjcf
import numpy as np
import pytest
from shapely import geometry


def test_randomized_dimensions_respect_final_ranges():
  rng = np.random.default_rng(5)
  for _ in range(500):
    p = process_mjcf.sample_mjfit_parameters(rng)
    assert 0.038 <= p["width"] <= 0.091
    assert 0.020 <= p["height"] <= 0.054
    assert 1.25 <= p["width"] / p["height"] <= 2.5
    assert p["aspect_ratio"] == pytest.approx(p["width"] / p["height"])
    assert p["height"] > 2 * p["clearance"]
    assert p["width"] + 2 * p["clearance"] < p["plate_size"]
    minor = min(p["base_width"], p["base_height"])
    major = max(p["base_width"], p["base_height"])
    original_area = minor * (major - minor) + np.pi * minor**2 / 4
    final_area = (
        p["height"] * (p["width"] - p["height"]) + np.pi * p["height"] ** 2 / 4
    )
    assert final_area == pytest.approx(original_area, abs=3e-5)


def test_plate_outline_closes_small_mesh_seams(tmp_path, monkeypatch):
  opening = mjfit.shapes.stadium(0.028, 0.015, n=32)
  xml = mjfit.peg(
      hole=opening,
      clearance=0.0004,
      plate_size=0.1,
      plate_thickness=0.008,
      peg_length=0.03,
  )
  root = ET.fromstring(xml)
  meshes = {
      mesh.attrib["name"]: (
          np.fromstring(mesh.attrib["vertex"], sep=" ").reshape(-1, 3)
      )
      for mesh in root.findall("./asset/mesh")
      if "vertex" in mesh.attrib
  }
  for geom in root.findall(".//body[@name='plate']/geom"):
    center = meshes[geom.attrib["mesh"]][:, :2].mean(axis=0)
    offset = center / np.linalg.norm(center) * 5e-8
    geom.set("pos", f"{offset[0]} {offset[1]} 0")
  input_xml = tmp_path / "peg.xml"
  output_xml = tmp_path / "peg_with_sites.xml"
  input_xml.write_text(ET.tostring(root, encoding="unicode"))
  model = mujoco.MjModel.from_xml_path(str(input_xml))

  with monkeypatch.context() as patch:
    patch.setattr(process_mjcf, "_MESH_SEAM_TOLERANCE", 0.0)
    with pytest.raises(ValueError, match="No through-hole"):
      process_mjcf._plate_hole_outline(model)

  hole, top_z = process_mjcf._plate_hole_outline(model)
  expected = geometry.Polygon(opening)
  assert hole.boundary.hausdorff_distance(expected.boundary) < 1e-6
  assert top_z == pytest.approx(0.004, abs=1e-6)
  process_mjcf.add_peg_sites(str(input_xml), str(output_xml))
  processed = mujoco.MjModel.from_xml_path(str(output_xml))
  assert all(processed.site(f"peg_{i}").id >= 0 for i in range(64))
  edge_sites = [
      processed.site(i)
      for i in range(processed.nsite)
      if processed.site(i).name.startswith("plate_edge_0_")
  ]
  assert len(edge_sites) >= 3
  for site in edge_sites:
    assert hole.boundary.distance(geometry.Point(site.pos[:2])) < 1e-6
    assert site.pos[2] == pytest.approx(top_z, abs=1e-6)


def test_plate_outline_handles_islands():
  xml = mjfit.peg(
      hole=mjfit.shapes.rect(0.03, 0.02),
      islands=[mjfit.shapes.rect(0.006, 0.004)],
      plate_size=0.1,
      include_peg=False,
  )
  model = mujoco.MjModel.from_xml_string(xml)
  outside, _ = process_mjcf._plate_hole_outline(model)
  inside, _ = process_mjcf._plate_hole_outline(model, handle_inside=True)
  assert len(outside.interiors) == 0
  assert len(inside.interiors) == 1
  assert outside.contains(geometry.Point(0, 0))
  assert not inside.contains(geometry.Point(0, 0))
