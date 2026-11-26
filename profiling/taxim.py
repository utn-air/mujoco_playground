"""Profile Taxim during a short RSL-RL run; accepts the training flags."""

import atexit
from collections import defaultdict
import functools
import time

from absl import app
from mujoco_playground.rl.rsl_rl import train
from reaf.simulation.torch._src.core import environment
from TaximSensor.cuda.backend import CudaRasterBackend
from TaximSensor.warp import WarpTaximSensor
import torch

_records = defaultdict(list)


def _instrument(cls, method: str, gpu: bool = False) -> None:
  original = getattr(cls, method)

  @functools.wraps(original)
  def timed(*args, **kwargs):
    start = time.perf_counter()
    if gpu:
      first = torch.cuda.Event(enable_timing=True)
      last = torch.cuda.Event(enable_timing=True)
      first.record()
    result = original(*args, **kwargs)
    if gpu:
      last.record()
      _records[f'{method}_gpu'].append((first, last))
    _records[f'{method}_host'].append(time.perf_counter() - start)
    return result

  setattr(cls, method, timed)


@atexit.register
def _report() -> None:
  if not _records:
    return
  torch.cuda.synchronize()
  print('\nTaxim timings (nested; do not add these totals):', flush=True)
  for name, samples in _records.items():
    if name.endswith('_gpu'):
      samples = [start.elapsed_time(end) / 1000 for start, end in samples]
    total = sum(samples)
    print(
        f'  {name}: {len(samples)} calls, {total:.3f} s total, '
        f'{1000 * total / len(samples):.2f} ms/call',
        flush=True,
    )


_instrument(CudaRasterBackend, 'render_frames', gpu=True)
_instrument(WarpTaximSensor, 'render', gpu=True)
_instrument(environment.Environment, 'step')
_instrument(environment.Environment, '_reset')

if __name__ == '__main__':
  app.run(train.main)
