import json
import time

from absl import app
from absl import flags
import mujoco.viewer
from mujoco_playground._src.rl import env_registry

_ENV = flags.DEFINE_string(
    "env", "FR3RobotiqPickCube", "Environment to preview."
)
_ENV_OVERRIDES = flags.DEFINE_string(
    "env_overrides",
    None,
    "JSON dict string of environment config overrides.",
)
_CHECK_ONLY = flags.DEFINE_bool(
    "check_only",
    False,
    "Build and reset the environment without opening the GUI.",
)
_RESET_INTERVAL = flags.DEFINE_float(
    "reset_interval",
    0.0,
    "Periodically reset the environment after this many seconds; 0"
    " disables it.",
)


def main(_):
  try:
    env_overrides = (
        json.loads(_ENV_OVERRIDES.value) if _ENV_OVERRIDES.value else {}
    )
  except json.JSONDecodeError as exc:
    raise app.UsageError("--env_overrides must be a JSON object.") from exc
  if not isinstance(env_overrides, dict):
    raise app.UsageError("--env_overrides must be a JSON object.")

  env = env_registry.get(_ENV.value).env_builder(
      nworld=1, device="cpu", impl="cpp", **env_overrides
  )
  try:
    env.reset()  # Shows the task's reset pose and placements.
    sim = env.device_layer.sim
    control_timestep = env.device_layer.control_timestep
    n_substeps = round(control_timestep / sim.mj_model.opt.timestep)
    print(
        f"Loaded {_ENV.value}: nq={sim.mj_model.nq}, nu={sim.mj_model.nu}, "
        f"action_shape={tuple(env.action_spec()['actions'].shape)}"
    )
    if _CHECK_ONLY.value:
      return
    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
      next_step = time.perf_counter()
      next_reset = (
          next_step + _RESET_INTERVAL.value
          if _RESET_INTERVAL.value > 0
          else float("inf")
      )
      try:
        while viewer.is_running():
          if time.perf_counter() >= next_reset:
            env.reset()
            print(f"Reset {_ENV.value}")
            next_reset = time.perf_counter() + _RESET_INTERVAL.value
          sim.step_n(n_substeps)
          viewer.sync()
          next_step += control_timestep
          time.sleep(max(0.0, next_step - time.perf_counter()))
      except KeyboardInterrupt:
        pass
  finally:
    env.close()


if __name__ == "__main__":
  app.run(main)
