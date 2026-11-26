"""FR3 peg insertion with Taxim DIGIT sensors on the Robotiq fingers."""

from pathlib import Path

from absl import logging
from dm_env import specs
import mujoco
from mujoco_playground import asset
from mujoco_playground.robots.fr3_robotiq.sim import constant
from mujoco_playground.tasks.manipulation.peg_in_hole.fr3_robotiq import task
import mujoco_warp as mjw
import numpy as np
from reaf.simulation.torch import flat_scene as config
from reaf.simulation.torch import types
from reaf.simulation.torch._src.core import device as device_module
import torch

_TAXIM_GRIPPER = (
    Path(__file__).resolve().parents[6]
    / "mujoco-taxim/assets/robotiq_2f85/robotiq_2f85.xml"
)


def _digit_robot_spec(gripper_xml: str) -> mujoco.MjSpec:
  """Attach the DIGIT-equipped Robotiq to the existing FR3."""
  spec = asset.get(
      "github:google-deepmind/mujoco_menagerie/franka_fr3/fr3.xml"
  ).spec()
  gripper = mujoco.MjSpec.from_file(gripper_xml)
  if gripper.site("pinch") is None:
    gripper.body("base").add_site(name="pinch", pos=(0.0, 0.0, 0.145))
  spec.option.cone = gripper.option.cone
  spec.option.impratio = gripper.option.impratio
  spec.attach(gripper, site="attachment_site", prefix="robotiq/")
  for key in list(spec.keys):
    spec.delete(key)
  return spec


class TaximDevice(device_module.Device):
  """Owns one Taxim sensor per fingertip and world."""

  def __init__(self, raster_backend: str = "cuda", batch_size: int = 8):
    self._raster_backend = raster_backend
    self._batch_size = batch_size
    self.sensors = []
    self.data = []
    self._images = {}
    self._warp_sensor = None

  @property
  def name(self) -> str:
    return "taxim"

  def initialize(self, simulator) -> None:
    try:
      from TaximSensor import TaximSensor  # pylint: disable=g-import-not-at-top
    except ImportError as exc:
      raise ImportError(
          "The tactile task requires mujoco-taxim and its norm2tex dependency. "
          "Install the CUDA branch and norm2tex in the Playground environment."
      ) from exc

    model = simulator.mj_model
    peg_id = model.body("peg").id
    peg_geoms = [
        i for i in range(model.ngeom) if model.geom_bodyid[i] == peg_id
    ]
    if len(peg_geoms) != 1:
      raise ValueError(
          "Taxim currently requires a peg with exactly one geom; "
          f"found {len(peg_geoms)}."
      )
    geom = model.geom(peg_geoms[0])
    geom_name = geom.name
    if not geom_name:
      raise ValueError("Taxim requires a named peg geom.")
    if self._raster_backend == "warp":
      if simulator.impl != types.Impl.WARP:
        raise ValueError("Taxim warp mode requires the MuJoCo Warp simulator")
      from TaximSensor.warp import WarpTaximSensor  # pylint: disable=g-import-not-at-top

      self._warp_sensor = WarpTaximSensor(
          model,
          simulator.data.obj,
          site_names=[
              f"robot/robotiq/{side}_digit_pad" for side in ("left", "right")
          ],
          pad_geom_names=[
              f"robot/robotiq/{side}_digit_pad" for side in ("left", "right")
          ],
          object_geom_name=geom_name,
          device=simulator.xpu_device,
          image_size=(64, 96),
          batch_size=self._batch_size,
      )
      self._images = {
          side: (
              self._warp_sensor.blank.expand(
                  simulator.nworld, -1, -1, -1
              ).clone()
          )
          for side in ("left", "right")
      }
      return
    mesh_name = (
        model.mesh(int(model.geom_dataid[geom.id])).name
        if model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_MESH
        else ""
    )
    xpu = torch.device(simulator.xpu_device)
    cuda_device = xpu.index or 0
    blank_images = {"left": [], "right": []}
    for _ in range(simulator.nworld):
      data = mujoco.MjData(model)
      mujoco.mj_forward(model, data)
      self.data.append(data)
      fingers = {}
      for side in ("left", "right"):
        site = f"robot/robotiq/{side}_digit_pad"
        pad = f"robot/robotiq/{side}_digit_pad"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
          raise ValueError(f"DIGIT site {site!r} is missing from the gripper.")
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pad) < 0:
          raise ValueError(f"DIGIT pad {pad!r} is missing from the gripper.")
        sensor = TaximSensor(
            sensor_type="digit",
            raster_backend=self._raster_backend,
            cuda_device=cuda_device,
            resize=(96, 64),
        )
        sensor.add_camera_mujoco(site, model, data)
        sensor.set_sensor_pad_geom(pad)
        sensor.add_geom_mujoco(geom_name, model, data, mesh_name=mesh_name)
        fingers[side] = sensor
        image = np.ascontiguousarray(sensor.render_blank_taxim())
        blank_images[side].append(torch.from_numpy(image).permute(2, 0, 1))
      self.sensors.append(fingers)
    self._images = {
        side: torch.stack(images).to(device=xpu, dtype=torch.float32) / 255.0
        for side, images in blank_images.items()
    }
    logging.info(
        "Initialized Taxim for %d worlds with two DIGIT sensors per world",
        simulator.nworld,
    )

  def reset(self, simulator, env_mask=None) -> None:
    del simulator, env_mask

  def substep_measurements_spec(self):
    return {
        f"taxim/{side}": specs.Array((3, 64, 96), np.float32)
        for side in ("left", "right")
    }

  def get_measurements(self, simulator):
    del simulator
    return {f"taxim/{side}": image for side, image in self._images.items()}

  def on_step(self, simulator) -> None:
    self._render(simulator, None)

  def on_begin_stepping(self, simulator, env_mask=None) -> None:
    self._render(simulator, env_mask)

  def _render(self, simulator, env_mask) -> None:
    if self._warp_sensor is not None:
      images = self._warp_sensor.render()
      for side_index, side in enumerate(("left", "right")):
        if env_mask is None:
          self._images[side].copy_(images[:, side_index])
        else:
          self._images[side].copy_(
              torch.where(
                  env_mask[:, None, None, None],
                  images[:, side_index],
                  self._images[side],
              )
          )
      return
    model = simulator.mj_model
    worlds = (
        range(simulator.nworld)
        if env_mask is None
        else torch.nonzero(env_mask).flatten().cpu().tolist()
    )
    for world in worlds:
      data = self.data[world]
      if simulator.impl == types.Impl.CPP:
        state_spec = (
            int(mujoco.mjtState.mjSTATE_PHYSICS)
            | int(mujoco.mjtState.mjSTATE_MOCAP_POS)
            | int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
        )
        state = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, simulator.mj_data, state, state_spec)
        mujoco.mj_setState(model, data, state, state_spec)
      else:
        mjw.get_data_into(data, model, simulator.data.obj, world_id=world)
      mujoco.mj_forward(model, data)
      for side, sensor in self.sensors[world].items():
        image, _, _ = sensor.render_taxim(
            model,
            data,
            get_depth=False,
            img_noise_sigma=0,
            visualize=False,
            cycle_bg=False,
        )
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        self._images[side][world].copy_(image.to(self._images[side]) / 255.0)


def _tactile_observation(features, side: str):
  return torch.nn.functional.interpolate(
      features[f"taxim/{side}"],
      size=(64, 64),
      mode="bilinear",
      align_corners=False,
  )


def fr3_robotiq_tactile_peg_in_hole_env(
    nworld: int = 2,
    gripper_xml: str | None = None,
    raster_backend: str = "warp",
    taxim_batch_size: int = 8,
    **kwargs,
) -> config.Environment:
  """Builds the peg task with initialized DIGIT sensors for each world."""
  if kwargs.get("generate_mjfit_object", False):
    raise ValueError(
        "Taxim per-world randomized geometry is not supported yet."
    )
  path = Path(gripper_xml).expanduser() if gripper_xml else _TAXIM_GRIPPER
  if not path.is_file():
    raise FileNotFoundError(f"DIGIT Robotiq XML not found: {path}")
  cfg = task.fr3_robotiq_peg_in_hole_env(
      nworld=nworld,
      robot_spec_factory=lambda: _digit_robot_spec(str(path.resolve())),
      **kwargs,
  )
  cfg.device_layer.devices["taxim"] = config.Component(
      TaximDevice, raster_backend=raster_backend, batch_size=taxim_batch_size
  )
  for side in ("left", "right"):
    cfg.observation_adapter[f"tactile_{side}"] = config.ObsGroup(
        terms={
            "image": (
                config.Obs(_tactile_observation, side=side).configure(
                    output_shape=(3, 64, 64)
                )
            )
        }
    )
    cfg.device_layer.devices[f"{side}_peg_contact"] = config.Component(
        task.sim_device.ContactSensorDevice,
        fields=("found",),
        history_length=100,
    ).configure_spec(
        objtype="geom",
        objname=(
            f"robot/robotiq/{side}_pad1",
            f"robot/robotiq/{side}_pad2",
            f"robot/robotiq/{side}_digit_pad",
        ),
        reftype="body",
        refname="peg",
    )
  return cfg
