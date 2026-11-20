import time

from absl import app
from absl import flags
import mujoco.viewer
from mujoco_playground._src.rl import env_registry

_ENV = flags.DEFINE_string(
    "env", "FR3RobotiqPickCube", "Environment to preview."
)
_CHECK_ONLY = flags.DEFINE_bool(
    "check_only",
    False,
    "Build and reset the environment without opening the GUI.",
)


def main(_):
  env = env_registry.get(_ENV.value).env_builder(
      nworld=1, device="cpu", impl="cpp"
  )
  try:
    env.reset()  # Shows the task's reset pose and placements.
    sim = env.device_layer.sim
    print(
        f"Loaded {_ENV.value}: nq={sim.mj_model.nq}, nu={sim.mj_model.nu}, "
        f"action_shape={tuple(env.action_spec()['actions'].shape)}"
    )
    if _CHECK_ONLY.value:
      return
    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
      try:
        while viewer.is_running():
          time.sleep(1 / 60)
      except KeyboardInterrupt:
        pass
  finally:
    env.close()


if __name__ == "__main__":
  app.run(main)
