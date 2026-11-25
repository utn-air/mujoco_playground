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
      [-1, 0, 0, -1, -1, -1, -1, 1, 0]
  ]


def test_site_classification_near_rim_and_island():
  outer = torch.tensor(
      [[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01]],
      dtype=torch.float64,
  )
  island = torch.tensor(
      [[-0.002, -0.002], [0.002, -0.002], [0.002, 0.002], [-0.002, 0.002]],
      dtype=torch.float64,
  )
  sites = torch.tensor(
      [[
          [0.005, 0.0, 0.0302],  # Above empty space, far from either rim.
          [0.0095, 0.0, 0.0302],  # Within 1 mm of the outer rim.
          [0.0089, 0.0, 0.0300],  # More than 1 mm from the outer rim.
          [0.0094, 0.0, 0.03079],  # Just under 1 mm in 3D from the outer rim.
          [0.0094, 0.0, 0.03081],  # Just over 1 mm in 3D from the outer rim.
          [0.0025, 0.0, 0.0302],  # Near the island rim, over empty space.
          [0.0027, 0.0027, 0.0300],  # Within 1 mm of an island corner.
          [0.0028, 0.0028, 0.0300],  # More than 1 mm from the corner.
          [0.0, 0.0, 0.0302],  # Above the island's top face.
          [0.005, 0.0, 0.0299],  # Below the plane inside the opening.
          [0.0, 0.0, 0.0299],  # Below the plane inside the island.
          [0.0095, 0.0, 0.0311],  # More than 1 mm above the plane.
      ]],
      dtype=torch.float64,
  )
  assert mdp.classify_sites(sites, (outer, island), 0.03).tolist() == [
      [-1, 0, -1, 0, -1, 0, 0, -1, 0, 1, 0, -1]
  ]


def test_episode_timeout_is_five_seconds():
  config = task.fr3_robotiq_peg_in_hole_env()
  assert config.episode_length_secs == 5.0
  assert config.sim.control_timestep == 0.02


def test_distance_termination_uses_insertion_end():
  sites = torch.zeros((2, 64, 3))
  sites[1, :, 0] = 0.11
  features = {
      "peg_hole_geometry/peg_sites": sites,
      "peg_hole_geometry/plate_pos": torch.zeros((2, 3)),
      "peg_hole_geometry/peg_pos": torch.tensor([[0.0, 0.0, 0.12]] * 2),
  }
  result = mdp.terminate_peg_far_from_plate(features, max_distance=0.10)
  assert result.tolist() == [mdp._DO_NOT_TERMINATE, mdp._TERMINATE]


def test_site_classification_uses_each_worlds_opening_and_height():
  square = torch.tensor(
      [[-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [-0.01, 0.01]]
  )
  rings = (torch.stack((square, square * 2)),)
  sites = torch.tensor([[[0.015, 0.0, 0.0]], [[0.015, 0.0, 0.0]]])
  heights = torch.tensor([[0.0], [0.001]])
  assert mdp.classify_sites(sites, rings, heights).tolist() == [[0], [1]]
  producer = mdp.PegHoleStateProducer(
      name="test", plate_edge_rings=rings, plate_top_z=(0.0, 0.001), nworld=2
  )
  features = {
      "peg_hole_geometry/peg_sites": sites.repeat(1, 64, 1),
      "peg_hole_geometry/plate_pos": torch.zeros((2, 3)),
      "peg_hole_geometry/plate_xmat": torch.eye(3).reshape(1, 9).repeat(2, 1),
  }
  producer.on_step(features)
  labels = producer.produce_features(features)["test/site_labels"]
  assert (labels[0] == 0).all()
  assert (labels[1] == 1).all()


def test_catch_accepts_sites_slightly_above_opening(monkeypatch):
  monkeypatch.setattr(mdp, "_CATCH_DEPTH", 0.0)
  monkeypatch.setattr(mdp, "_CATCH_HEIGHT_TOLERANCE", 0.0002)
  producer = mdp.PegHoleStateProducer(
      name="test",
      plate_edge_rings=(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),),
      plate_top_z=0.0,
      nworld=4,
      catch_site_count=32,
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
  assert (
      result["test/site_labels"] == torch.where(sites[:, :, 0] == 0, -1, 0)
  ).all()


def test_catch_requires_site_count_and_average_inside_depth(monkeypatch):
  monkeypatch.setattr(mdp, "_CATCH_DEPTH", 0.0002)
  producer = mdp.PegHoleStateProducer(
      name="test",
      plate_edge_rings=(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),),
      plate_top_z=0.0,
      nworld=5,
      catch_site_count=32,
      insertion_depth=0.002,
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
