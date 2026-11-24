"""Tests peg-in-hole reward conditions."""

from mujoco_playground.tasks.manipulation.peg_in_hole import mdp
from mujoco_playground.tasks.manipulation.peg_in_hole.fr3_robotiq import task
import torch


def test_site_classification_uses_plate_height_and_opening():
  ring = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
  sites = torch.tensor([[
      [2.0, 0.0, 0.0011],  # Above the top face.
      [2.0, 0.0, 0.0005],  # Near the top face, away from the rim.
      [2.0, 0.0, 0.0],  # On the top face.
      [0.0, 0.0, 0.0011],  # More than 1 mm above the opening.
      [0.0, 0.0, 0.001],  # Exactly 1 mm above the opening.
      [0.0, 0.0, 0.0005],  # Near the plane, above the empty opening.
      [0.0, 0.0, 0.0],  # On the plane, above the opening.
      [0.0, 0.0, -0.0001],  # Below the top plane inside the opening.
      [2.0, 0.0, -0.0001],  # Penetrating the plate, outside the opening.
  ]])
  assert mdp.classify_sites(sites, (ring,), 0.0).tolist() == [
      [-1, 0, 0, -1, 0, 0, 0, 1, 0]
  ]


def test_episode_timeout_is_five_seconds():
  config = task.fr3_robotiq_peg_in_hole_env()
  assert config.episode_length_secs == 5.0
  assert config.sim.control_timestep == 0.02


def test_catch_accepts_sites_slightly_above_opening():
  producer = mdp.PegHoleStateProducer(
      name="test",
      plate_edge_rings=(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),),
      plate_top_z=0.0,
      nworld=4,
  )
  sites = torch.zeros((4, 64, 3))
  sites[:, :, 0] = 2.0
  sites[:, :, 2] = 0.00019
  sites[0, :31, 0] = 0.0
  sites[1, :32, 0] = 0.0
  sites[2, :32, 0] = 0.0
  sites[2, :32, 2] = 0.00021
  sites[3, :32, 0] = 0.0
  sites[3, :32, 2] = 0.0002
  features = {
      "peg_hole_geometry/peg_sites": sites,
      "peg_hole_geometry/plate_pos": torch.zeros((4, 3)),
      "peg_hole_geometry/plate_xmat": torch.eye(3).reshape(1, 9).repeat(4, 1),
  }
  producer.on_step(features)
  result = producer.produce_features(features)
  assert result["test/catch_event"].tolist() == [0, 1, 0, 1]
  assert result["test/inside_count"].tolist() == [0, 0, 0, 0]
  assert result["test/insertion_event"].sum() == 0
  assert (result["test/site_labels"] == 0).all()


def test_catch_requires_site_count_and_average_inside_depth(monkeypatch):
  monkeypatch.setattr(mdp, "_CATCH_DEPTH", 0.0002)
  producer = mdp.PegHoleStateProducer(
      name="test",
      plate_edge_rings=(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),),
      plate_top_z=0.0,
      nworld=5,
  )
  sites = torch.zeros((5, 64, 3))
  sites[:, :, 0] = 2.0  # Outside the opening.
  sites[:, :, 2] = -0.001
  sites[0, :31, 0] = 0.0  # Deep enough, but too few inside sites.
  sites[1, :32, 0] = 0.0
  sites[1, :32, 2] = -0.00019  # Too shallow.
  sites[2, :32, 0] = 0.0
  sites[2, :32, 2] = -0.0002  # Exactly at the margin.
  sites[3, :32, 0] = 0.0
  sites[3, :16, 2] = -0.0001
  sites[3, 16:32, 2] = -0.0003  # Average depth is 0.2 mm.
  sites[4, :, 0] = 0.0
  sites[4, :, 2] = -0.002  # Full insertion still earns both rewards.
  features = {
      "peg_hole_geometry/peg_sites": sites,
      "peg_hole_geometry/plate_pos": torch.zeros((5, 3)),
      "peg_hole_geometry/plate_xmat": torch.eye(3).reshape(1, 9).repeat(5, 1),
  }
  producer.on_step(features)
  result = producer.produce_features(features)
  assert result["test/catch_event"].tolist() == [0, 0, 1, 1, 1]
  assert result["test/insertion_event"].tolist() == [0, 0, 0, 0, 1]
  producer.on_step(features)
  assert producer.produce_features(features)["test/catch_event"].sum() == 0
