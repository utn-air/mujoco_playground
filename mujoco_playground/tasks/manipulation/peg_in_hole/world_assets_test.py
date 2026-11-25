"""Tests world-specific peg geometry and grasp calibration."""

import mujoco
from mujoco_playground._src.rl import env_registry
from mujoco_playground.tasks.manipulation.peg_in_hole import process_mjcf
from mujoco_playground.tasks.manipulation.peg_in_hole import reset
import numpy as np
from reaf.simulation.torch.viewer._src import sim_backend
from reaf.simulation.torch.viewer._src import viewer
import torch
import warp as wp


def test_world_assets_and_grasps_persist_across_resets(tmp_path, monkeypatch):
  monkeypatch.setattr(wp.config, "kernel_cache_dir", "/tmp/peg_warp_cache")
  monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 4)
  calls = []
  measure = reset.PegInHoleResetter._measure_closed_gripper_pose

  def counted(self, offset, tilt):
    calls.append((offset, tilt))
    return measure(self, offset, tilt)

  monkeypatch.setattr(
      reset.PegInHoleResetter, "_measure_closed_gripper_pose", counted
  )
  env = env_registry.get("FR3RobotiqPegInHole").env_builder(
      nworld=2,
      device="cpu",
      impl="warp",
      generate_mjfit_object=True,
      mjfit_output_dir=str(tmp_path),
  )
  try:
    resetter = env._environment_reset._resetters[0]
    assert len(set(resetter._world_xml_paths)) == 2
    assert len(calls) == 18
    assert not np.allclose(
        resetter._world_grasps[0][4], resetter._world_grasps[1][4]
    )
    sim = env.device_layer.sim
    ids = [sim.mj_model.site(f"peg_{i}").id for i in range(64)]
    initial_sites = sim.model.site_pos[:, ids].clone()
    edge_ids = [
        sim.mj_model.site(i).id
        for i in range(sim.mj_model.nsite)
        if sim.mj_model.site(i).name.startswith("plate_edge_0_")
    ]
    initial_edges = sim.model.site_pos[:, edge_ids].clone()
    geom_ids = torch.tensor([
        sim.mj_model.geom(i).id
        for i in range(sim.mj_model.ngeom)
        if sim.mj_model.geom(i).name.startswith("peg_assets/")
    ])
    initial_meshes = sim.model.geom_dataid[:, geom_ids].clone()
    assert not torch.equal(initial_meshes[0], initial_meshes[1])
    plate_geom_ids = torch.tensor([
        sim.mj_model.geom(i).id
        for i in range(sim.mj_model.ngeom)
        if sim.mj_model.geom(i).name.startswith("plate_assets/")
    ])
    plate_meshes = sim.model.geom_dataid[:, plate_geom_ids].clone()
    for i, path in enumerate(resetter._world_xml_paths):
      spec = mujoco.MjSpec.from_file(path)
      expected = np.array([spec.site(f"peg_{j}").pos for j in range(64)])
      np.testing.assert_allclose(
          initial_sites[i].cpu().numpy(), expected, atol=1e-7
      )
      edge_count = sum(
          site.name.startswith("plate_edge_0_")
          for site in spec.body("plate").sites
      )
      expected_edges = np.array(
          [spec.site(f"plate_edge_0_{j}").pos for j in range(edge_count)]
      )
      np.testing.assert_allclose(
          initial_edges[i, :edge_count].cpu().numpy(), expected_edges, atol=1e-7
      )
      for mesh_ids, prefix in (
          (initial_meshes, "peg_assets"),
          (plate_meshes, "plate_assets"),
      ):
        names = [
            sim.mj_model.mesh(int(mesh_id)).name for mesh_id in mesh_ids[i]
        ]
        assert any(name.startswith(f"{prefix}/{i}/") for name in names)
    for _ in range(2):
      env.reset()
      assert torch.equal(sim.model.site_pos[:, ids], initial_sites)
      assert torch.equal(sim.model.site_pos[:, edge_ids], initial_edges)
      assert torch.equal(sim.model.geom_dataid[:, geom_ids], initial_meshes)
      assert torch.equal(sim.model.geom_dataid[:, plate_geom_ids], plate_meshes)
      assert len(calls) == 18
    resetter._grasp_offset_range = (0.0, 0.0)
    resetter._grasp_tilt_range = (0.0, 0.0)
    env.reset()
    warp_data = sim.warp_data
    count = int(warp_data.nacon.numpy()[0])
    pairs = warp_data.contact.geom.numpy()[:count]
    worlds = warp_data.contact.worldid.numpy()[:count]
    for world in range(2):
      sides = set()
      for pair in pairs[worlds == world]:
        for pad, other in (pair, pair[::-1]):
          if sim.mj_model.geom_bodyid[other] != sim.mj_model.body("peg").id:
            continue
          name = sim.mj_model.geom(int(pad)).name
          for side in ("left", "right"):
            if name in (
                f"robot/robotiq/{side}_pad1",
                f"robot/robotiq/{side}_pad2",
            ):
              sides.add(side)
      assert sides == {"left", "right"}
    for world in range(2):
      model = sim.mj_spec.compile()
      data = mujoco.MjData(model)
      backend = sim_backend.TorchSimBackend(
          env,
          viewer._ZeroActionPolicy(env.action_spec(), xpu_device="cpu"),
          sim,
          "cpu",
          world,
      )
      backend.sync_to_viewer(model, data)
      np.testing.assert_array_equal(
          model.geom_dataid[geom_ids], initial_meshes[world].numpy()
      )
      np.testing.assert_allclose(
          model.site_pos[ids], initial_sites[world].numpy(), atol=1e-7
      )
      np.testing.assert_allclose(
          model.site_pos[edge_ids], initial_edges[world].numpy(), atol=1e-7
      )
  finally:
    env.close()


def test_grasp_calibration_is_shared_for_reused_assets(tmp_path, monkeypatch):
  monkeypatch.setattr(wp.config, "kernel_cache_dir", "/tmp/peg_warp_cache")
  params = process_mjcf.sample_mjfit_parameters(np.random.default_rng(3))
  calls = []
  measure = reset.PegInHoleResetter._measure_closed_gripper_pose

  def counted(self, offset, tilt):
    calls.append((offset, tilt))
    return measure(self, offset, tilt)

  monkeypatch.setattr(
      reset.PegInHoleResetter, "_measure_closed_gripper_pose", counted
  )
  env = env_registry.get("FR3RobotiqPegInHole").env_builder(
      nworld=2,
      device="cpu",
      impl="warp",
      generate_mjfit_object=True,
      mjfit_output_dir=str(tmp_path),
      mjfit_parameters=params,
  )
  try:
    resetter = env._environment_reset._resetters[0]
    assert resetter._world_xml_paths[0] == resetter._world_xml_paths[1]
    assert resetter._world_grasps[0] is resetter._world_grasps[1]
    assert len(calls) == 9
    env.reset()
    env.reset()
    assert len(calls) == 9
  finally:
    env.close()


def test_single_randomized_world_supports_cpp(tmp_path):
  env = env_registry.get("FR3RobotiqPegInHole").env_builder(
      nworld=1,
      device="cpu",
      impl="cpp",
      generate_mjfit_object=True,
      mjfit_output_dir=str(tmp_path),
  )
  try:
    env.reset()
    assert env.device_layer.sim.mj_model.site("peg_63").id >= 0
  finally:
    env.close()
