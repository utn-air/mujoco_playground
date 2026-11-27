"""Tactile observations must use the integrated simulation state."""

import mujoco
from mujoco_playground.tasks.manipulation.peg_in_hole.fr3_robotiq import tactile_task
import mujoco_warp as mjw
import pytest
import torch
import warp as wp


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)
def test_tactile_step_refreshes_poses_and_contacts():
  model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.002" gravity="0 0 0"/><worldbody>
      <body name="pad"><geom name="pad_geom" type="box" size=".02 .02 .005"/>
        <site name="pad_site" pos="0 0 .005" quat="0 1 0 0"/>
      </body>
      <body name="peg" pos="0 0 .0155"><freejoint/>
        <geom name="peg_geom" type="box" size=".008 .007 .01" mass=".1"/>
      </body>
    </worldbody></mujoco>
  """)
  data = mujoco.MjData(model)
  data.qvel[2] = -1
  mujoco.mj_forward(model, data)
  wp.init()
  with wp.ScopedDevice("cuda:0"):
    wm = mjw.put_model(model)
    wd = mjw.put_data(model, data, nworld=1)
    mjw.step(wm, wd)
    assert int(wp.to_torch(wd.nacon)[0]) == 0

    class Simulator:

      def forward(self):
        mjw.forward(wm, wd)

    class SnapshotSensor:

      def render(self, world_mask=None):
        assert world_mask is None
        self.z = float(
            wp.to_torch(wd.geom_xpos)[0, model.geom("peg_geom").id, 2]
        )
        self.contacts = int(wp.to_torch(wd.nacon)[0])
        return torch.zeros((1, 2, 3, 64, 96), device="cuda")

    device = tactile_task.TaximDevice(raster_backend="warp")
    device._warp_sensor = SnapshotSensor()
    device._images = {
        side: torch.zeros((1, 3, 64, 96), device="cuda")
        for side in ("left", "right")
    }
    device.on_step(Simulator())
    assert device._warp_sensor.z == pytest.approx(
        float(wp.to_torch(wd.qpos)[0, 2])
    )
    assert device._warp_sensor.contacts > 0
