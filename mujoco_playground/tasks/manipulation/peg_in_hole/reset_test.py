"""Checks fixture reset and playback synchronization."""

import mujoco
import numpy as np
import pytest
from mujoco_playground._src.rl import env_registry
from mujoco_playground._src.rl.rsl_rl import vecenv_wrapper
from reaf.simulation.torch.viewer._src import sim_backend
from reaf.simulation.torch.viewer._src import viewer
import torch
import warp as wp


@pytest.mark.parametrize(("impl", "nworld"), [("cpp", 1), ("warp", 2)])
def test_fixture_reset_playback_and_termination(impl, nworld, tmp_path, monkeypatch):
  monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp"))
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
      np.testing.assert_allclose(render_data.body("plate").xpos, plate, atol=1e-5)
      assert np.linalg.norm(peg - plate) < 0.05
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
    assert torch.linalg.vector_norm(
        sim.data.xpos[0, peg_id] - sim.data.xpos[0, plate_id]
    ) < 0.05
  finally:
    env.close()
