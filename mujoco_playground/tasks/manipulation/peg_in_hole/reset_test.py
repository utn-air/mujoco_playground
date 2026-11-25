"""Checks fixture reset and playback synchronization."""

import mujoco
from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
from mujoco_playground.tasks.manipulation.peg_in_hole import reset
from mujoco_playground.tasks.manipulation.peg_in_hole.fr3_robotiq import task
import numpy as np
import pytest
from reaf.simulation.torch import flat_scene
from reaf.simulation.torch.viewer._src import sim_backend
from reaf.simulation.torch.viewer._src import viewer
import torch
import warp as wp


@pytest.mark.parametrize("fixture_yaw", [0.0, 0.4])
@pytest.mark.parametrize("grasp_tilt", [-8.0, 0.0, 8.0])
def test_nominal_grasp_aligns_jaws_and_preserves_peg_yaw(
    fixture_yaw, grasp_tilt, monkeypatch
):
  """The closed gripper must not twist an aligned peg toward its home yaw."""
  monkeypatch.setattr(reset, "RANDOMIZE_FIXTURE_POSITION", False)
  monkeypatch.setattr(reset, "RANDOMIZE_FIXTURE_TILT", False)
  cfg = task.fr3_robotiq_peg_in_hole_env(
      plate_pos=(0.58, 0.02, 0.03),
      robot_contact_scales={
          "finger_friction": 1.0,
          "finger_contact_timeconst": 1.0,
          "gripper_force_cap": 1.0,
      },
  )
  cfg.base_scene.body("plate").quat = reset._quat_from_rpy(
      np.array([0.0, 0.0, fixture_yaw])
  )
  cfg.sim.impl, cfg.sim.xpu_device, cfg.sim.nworld = "cpp", "cpu", 1
  cfg.resetters["peg_in_hole"] = flat_scene.Component(
      reset.PegInHoleResetter,
      peg_offset_range=(0.0, 0.0),
      grasp_offset_range=(0.0, 0.0),
      grasp_tilt_range=(grasp_tilt, grasp_tilt),
  )
  env = flat_scene.build_env(cfg)
  try:
    env.reset()
    data = env.device_layer.sim.mj_data
    plate_rot = data.body("plate").xmat.reshape(3, 3).copy()
    closing_axis = plate_rot.T @ (
        data.geom("robot/robotiq/right_pad1").xpos
        - data.geom("robot/robotiq/left_pad1").xpos
    )
    # Check the physical pads, not merely the requested IK orientation.
    assert abs(np.arctan2(closing_axis[0], closing_axis[1])) < 0.002
    # Grasp randomization rotates the object about the corrected jaw axis.
    half_tilt = np.deg2rad(grasp_tilt) / 2
    expected_quat = np.concatenate([
        [np.cos(half_tilt)],
        np.sin(half_tilt) * closing_axis / np.linalg.norm(closing_axis),
    ])
    expected_rot = np.empty(9)
    mujoco.mju_quat2Mat(expected_rot, expected_quat)
    np.testing.assert_allclose(
        plate_rot.T @ data.body("peg").xmat.reshape(3, 3),
        expected_rot.reshape(3, 3),
        atol=2e-4,
    )
    peg_sites = np.array([data.site(f"peg_{i}").xpos for i in range(64)])
    plate_pos = data.body("plate").xpos
    lowest_site = ((peg_sites - plate_pos) @ plate_rot)[:, 2].min()
    plate_top = data.model.site("plate_edge_0_0").pos[2]
    plate_bottom, plate_top_mesh = reset._body_mesh_z_bounds(
        data.model, data, data.model.body("plate").id
    )
    np.testing.assert_allclose(
        lowest_site, plate_top + (plate_top_mesh - plate_bottom) / 2, atol=1e-5
    )
    if grasp_tilt != 0.0:
      return
    hold_q = torch.tensor(
        [[data.joint(f"robot/fr3_joint{i}").qpos[0] for i in range(1, 8)]],
        dtype=torch.float32,
    )
    for _ in range(25):
      env.step({"actions": hold_q})
    peg_rot = plate_rot.T @ data.body("peg").xmat.reshape(3, 3)
    assert abs(np.arctan2(peg_rot[1, 0], peg_rot[0, 0])) < 0.05
  finally:
    env.close()


@pytest.mark.parametrize(("impl", "nworld"), [("cpp", 1), ("warp", 2)])
def test_fixture_reset_playback_and_termination(
    impl, nworld, tmp_path, monkeypatch
):
  monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp"))
  monkeypatch.setattr(reset, "RANDOMIZE_FIXTURE_POSITION", True)
  monkeypatch.setattr(reset, "RANDOMIZE_FIXTURE_TILT", True)
  entry = env_registry.get("FR3RobotiqPegInHole")
  env = entry.env_builder(nworld=nworld, device="cpu", impl=impl)
  try:
    vec = vecenv_wrapper.RslRlVecEnvWrapper(env)
    sim = env.device_layer.sim
    model = sim.mj_model
    plate_id = model.body("plate").id
    peg_id = model.body("peg").id
    render_model = sim.mj_spec.compile()
    render_data = mujoco.MjData(render_model)
    backend = sim_backend.TorchSimBackend(
        env,
        viewer._ZeroActionPolicy(env.action_spec(), xpu_device="cpu"),
        sim,
        "cpu",
        0,
    )

    positions = []
    for _ in range(2):
      env.reset()
      backend.sync_to_viewer(render_model, render_data)
      plate = sim.data.xpos[0, plate_id].cpu().numpy().copy()
      peg = sim.data.xpos[0, peg_id].cpu().numpy().copy()
      np.testing.assert_allclose(
          render_data.body("plate").xpos, plate, atol=1e-5
      )
      assert np.linalg.norm(peg - plate) < 0.10
      positions.append(plate)
    assert not np.allclose(*positions)

    peg_qpos = int(model.jnt_qposadr[model.joint("peg_joint").id])
    sim.data.qpos[0, peg_qpos] += 0.25
    sim.forward()
    actions = vec.get_observations()["policy"][:, :7].clone()
    _, _, done, _ = vec.step(actions)
    assert done[0].item() == 1
    if nworld > 1:
      assert done[1].item() == 0
    assert (
        torch.linalg.vector_norm(
            sim.data.xpos[0, peg_id] - sim.data.xpos[0, plate_id]
        )
        < 0.10
    )
  finally:
    env.close()


@pytest.mark.parametrize("randomize_position", [False, True])
@pytest.mark.parametrize("randomize_tilt", [False, True])
def test_fixture_randomization_switches(
    randomize_position, randomize_tilt, monkeypatch
):
  monkeypatch.setattr(reset, "RANDOMIZE_FIXTURE_POSITION", randomize_position)
  monkeypatch.setattr(reset, "RANDOMIZE_FIXTURE_TILT", randomize_tilt)
  entry = env_registry.get("FR3RobotiqPegInHole")
  env = entry.env_builder(
      nworld=1, device="cpu", impl="cpp", plate_pos=(0.58, 0.02, 0.03)
  )
  try:
    sim = env.device_layer.sim
    plate_id = sim.mj_model.body("plate").id
    peg_id = sim.mj_model.body("peg").id
    plate_poses, peg_positions = [], []
    for _ in range(3):
      env.reset()
      plate_poses.append((
          sim.data.xpos[0, plate_id].clone().numpy(),
          sim.data.xquat[0, plate_id].clone().numpy(),
      ))
      peg_positions.append(sim.data.xpos[0, peg_id].clone().numpy())
    if randomize_position:
      assert not np.allclose(plate_poses[0][0], plate_poses[1][0])
    else:
      for position, _ in plate_poses:
        np.testing.assert_allclose(position, (0.58, 0.02, 0.03), atol=1e-6)
    if randomize_tilt:
      assert not np.allclose(plate_poses[0][1], plate_poses[1][1])
    else:
      for _, quat in plate_poses:
        np.testing.assert_allclose(quat, (1, 0, 0, 0), atol=1e-6)
    assert not np.allclose(peg_positions[0], peg_positions[1])
  finally:
    env.close()
