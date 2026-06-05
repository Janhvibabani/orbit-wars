# classes
from __future__ import annotations

import torch
from torch.distributions import Categorical
from dataclasses import dataclass
from .policy import PolicyOutput


@dataclass(slots=True)
class SampledAction:
  target_index: torch.Tensor
  log_prob: torch.Tensor
  entropy: torch.Tensor


@dataclass(slots=True)
class TransitionBatch:
  self_features: torch.Tensor
  candidate_features: torch.Tensor
  global_features: torch.Tensor
  candidate_mask: torch.Tensor
  target_index: torch.Tensor
  log_prob: torch.Tensor
  returns: torch.Tensor
  advantages: torch.Tensor


# Calc. Entropy and log probability
def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

  tgt_logits = safe_tgt_logits(outputs.target_logits)
  tgt_dist = Categorical(logits=tgt_logits)

  tgt_log_prob = tgt_dist.log_prob(target_index)
  tgt_entropy = tgt_dist.entropy()

  return tgt_log_prob, tgt_entropy


# Fix invalid logits
def safe_tgt_logits(target_logits: torch.Tensor) -> torch.Tensor:
  invalid_rows = ~torch.isfinite(target_logits).any(dim=-1)

  if not invalid_rows.any():
    return target_logits

  safe_logits = target_logits.clone()
  safe_logits[invalid_rows, 0] = 0.0
  return safe_logits


# choose an action from policy o/p and return SampledAction obj
def sample_actions(outputs: PolicyOutput, deterministic: bool) -> SampledAction:

  # getting target_idx
  tgt_logits = safe_tgt_logits(outputs.target_logits)
  tgt_dist = Categorical(logits=tgt_logits)

  tgt_idx = (
      target_logits.argmax(dim=-1)
      if deterministic
      else tgt_dist.sample()
  )

  # cal. log prob and entropy
  log_prob, entropy = action_log_prob_and_entropy(outputs, tgt_idx)

  return SampledAction(
      target_index=tgt_idx,
      log_prob=log_prob,
      entropy=entropy
  )


# Actual ppo training step
def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:

  # Empty batch check
  if batch.self_features.shape[0] == 0:
    return {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0
    }

  # Move tensors to device
  self_feat = batch.self_features.to(device)
  candidate_feat = batch.candidate_features.to(device)
  global_feat = batch.global_features.to(device)
  candidate_mask = batch.candidate_mask.to(device).bool()
  old_log_prob = batch.log_prob.to(device)
  tgt_idx = batch.target_index.to(device)
  returns = batch.returns.to(device)
  adv = batch.advantages.to(device)

  # Adv normalization
  adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

  # MiniBatchs
  size = self_feat.shape[0]
  minibatch_size = min(size, max(1, minibatch_size))

  metrics = {
      "loss": 0.0,
      "policy_loss": 0.0,
      "value_loss": 0.0,
      "entropy": 0.0
  }

  updates = 0

  for _ in range(epochs):

    # shuffle
    order = torch.randperm(size, device=device)

    for start in range(0, size, minibatch_size):
      idx = order[start:start + minibatch_size]

      # forward pass - gives logits, values
      outputs = policy(
          self_feat[idx],
          candidate_feat[idx],
          global_feat[idx],
          candidate_mask[idx]
      )

      # log prob. of chosen action
      new_log_prob, entropy = action_log_prob_and_entropy(outputs, tgt_idx[idx])

      # PPO ratio
      ratio = (new_log_prob - old_log_prob[idx]).exp()

      # Policy loss
      policy_loss = torch.maximum(
          -adv[idx] * ratio,
          -adv[idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
      ).mean()

      # Value loss
      value_loss = 0.5 * (returns[idx] - outputs.value).pow(2).mean()

      entropy = entropy.mean()

      loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

      # Backprop
      optimizer.zero_grad(set_to_none=True)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
      optimizer.step()

      # Metrics accumulation
      metrics["loss"] += float(loss.detach().cpu())
      metrics["policy_loss"] += float(policy_loss.detach().cpu())
      metrics["value_loss"] += float(value_loss.detach().cpu())
      metrics["entropy"] += float(entropy.detach().cpu())

      updates += 1

  return {key: value / max(updates, 1) for key, value in metrics.items()}
