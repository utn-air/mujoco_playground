"""Peg-in-hole actor architecture and PPO contact auxiliary loss."""

from __future__ import annotations

from itertools import chain

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTOR_STATE_DIM = 33
_HISTORY_FRAMES = 30
_TEMPORAL_LATENT_DIM = 32
_CONTACT_SITES = 64
_CONTACT_CLASSES = 3
_DELTA_SCALE = 0.02


def _mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
) -> nn.Sequential:
  activation_cls = getattr(nn, activation.capitalize(), None)
  if activation_cls is None:
    activation_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}.get(
        activation.lower()
    )
  if activation_cls is None:
    raise ValueError(f"Unsupported activation {activation!r}.")
  layers: list[nn.Module] = []
  width = input_dim
  for hidden_dim in hidden_dims:
    layers.extend((nn.Linear(width, hidden_dim), activation_cls()))
    width = hidden_dim
  layers.append(nn.Linear(width, output_dim))
  return nn.Sequential(*layers)


class PegInHoleActor(MLPModel):
  """TCN actor with a weighted contact-state auxiliary decoder."""

  def __init__(
      self,
      obs: TensorDict,
      obs_groups: dict[str, list[str]],
      obs_set: str,
      output_dim: int,
      hidden_dims: tuple[int, ...] | list[int] = (256, 256, 128),
      activation: str = "elu",
      obs_normalization: bool = False,
      distribution_cfg: dict | None = None,
  ):
    if output_dim != 7:
      raise ValueError(f"PegInHoleActor needs 7 actions, got {output_dim}.")
    super().__init__(
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims,
        activation,
        obs_normalization,
        distribution_cfg,
    )
    if self.distribution is None:
      raise ValueError("PegInHoleActor requires a stochastic distribution.")

    self.temporal_encoder = nn.Sequential(
        nn.Conv1d(_ACTOR_STATE_DIM, 64, kernel_size=3, padding=1),
        nn.ELU(),
        nn.Conv1d(64, 64, kernel_size=3, padding=1),
        nn.ELU(),
        nn.Flatten(start_dim=1),
        nn.Linear(64 * _HISTORY_FRAMES, _TEMPORAL_LATENT_DIM),
        nn.ELU(),
    )
    self.mlp = _mlp(
        _ACTOR_STATE_DIM + _TEMPORAL_LATENT_DIM,
        self.distribution.input_dim,
        hidden_dims,
        activation,
    )
    self.distribution.init_mlp_weights(self.mlp)
    self.contact_decoder = nn.Sequential(
        nn.Linear(_TEMPORAL_LATENT_DIM, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, _CONTACT_SITES * _CONTACT_CLASSES),
    )
    self.register_buffer(
        "_delta_scale", torch.tensor(_DELTA_SCALE, dtype=torch.float32)
    )
    self._current_q: torch.Tensor | None = None

  def _split_observation(
      self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor]:
    policy_obs = obs["policy"]
    expected = _ACTOR_STATE_DIM * (1 + _HISTORY_FRAMES)
    if policy_obs.shape[-1] != expected:
      raise ValueError(
          f"Expected policy observation width {expected}, got"
          f" {policy_obs.shape[-1]}."
      )
    current = policy_obs[..., :_ACTOR_STATE_DIM]
    history = policy_obs[..., _ACTOR_STATE_DIM:].reshape(
        -1, _HISTORY_FRAMES, _ACTOR_STATE_DIM
    )
    return current, history

  def encode_history(self, obs: TensorDict) -> torch.Tensor:
    _, history = self._split_observation(obs)
    return self.temporal_encoder(history.transpose(1, 2))

  def contact_logits(self, obs: TensorDict) -> torch.Tensor:
    temporal_latent = self.encode_history(obs)
    return self.contact_decoder(temporal_latent).reshape(
        -1, _CONTACT_SITES, _CONTACT_CLASSES
    )

  def forward(
      self,
      obs: TensorDict,
      masks: torch.Tensor | None = None,
      hidden_state=None,
      stochastic_output: bool = False,
  ) -> torch.Tensor:
    del hidden_state
    if masks is not None and not self.is_recurrent:
      obs = unpad_trajectories(obs, masks)
    current, history = self._split_observation(obs)
    temporal_latent = self.temporal_encoder(history.transpose(1, 2))
    self._current_q = current[..., :7]
    mlp_output = self.mlp(torch.cat((current, temporal_latent), -1))
    self.distribution.update(mlp_output)
    delta = (
        self.distribution.sample()
        if stochastic_output
        else self.distribution.deterministic_output(mlp_output)
    )
    return self._current_q + self._delta_scale * delta

  def get_output_log_prob(self, target_q: torch.Tensor) -> torch.Tensor:
    if self._current_q is None:
      raise RuntimeError("Actor must run before requesting action log-prob.")
    delta = (target_q - self._current_q) / self._delta_scale
    return self.distribution.log_prob(delta)


class PegInHolePPO(PPO):
  """PPO with weighted cross-entropy for the actor contact decoder."""

  def __init__(self, *args, contact_loss_coef: float = 1.0, **kwargs):
    super().__init__(*args, **kwargs)
    if not isinstance(self._raw_actor, PegInHoleActor):
      raise TypeError("PegInHolePPO requires a PegInHoleActor.")
    self.contact_loss_coef = float(contact_loss_coef)
    self.contact_class_weights = torch.tensor(
        (0.25, 3.0, 2.0), dtype=torch.float32, device=self.device
    )

  def update(self) -> dict[str, float]:
    if self.rnd is not None or self.symmetry is not None:
      raise NotImplementedError(
          "PegInHolePPO currently supports PPO without RND or symmetry."
      )
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_contact_loss = 0.0
    generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
    )

    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
              batch.advantages.std() + 1e-8
          )

      self.actor(batch.observations, stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
              batch.old_distribution_params,
              self.actor.output_distribution_params,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(
                kl_mean, op=torch.distributed.ReduceOp.SUM
            )
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

      ratio = torch.exp(
          actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
      )
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss = torch.max(
            (values - batch.returns).pow(2),
            (value_clipped - batch.returns).pow(2),
        ).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      critic_obs = batch.observations["critic"]
      labels = (
          critic_obs[:, : _CONTACT_SITES * _CONTACT_CLASSES]
          .reshape(-1, _CONTACT_SITES, _CONTACT_CLASSES)
          .argmax(dim=-1)
      )
      contact_logits = self._raw_actor.contact_logits(batch.observations)
      contact_loss = F.cross_entropy(
          contact_logits.reshape(-1, _CONTACT_CLASSES),
          labels.reshape(-1),
          weight=self.contact_class_weights,
      )
      loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy.mean()
          + self.contact_loss_coef * contact_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_contact_loss += contact_loss.item()

    updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    return {
        "value": mean_value_loss / updates,
        "surrogate": mean_surrogate_loss / updates,
        "entropy": mean_entropy / updates,
        "contact": mean_contact_loss / updates,
    }
