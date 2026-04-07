from __future__ import annotations

import copy
import math
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from rocket_lander.config import AppConfig
from rocket_lander.environment import OBSERVATION_NAMES, RocketLanderEnv
from rocket_lander.ppo import ACTION_NAMES, ActorCritic


@dataclass
class GenerationReport:
    generation_index: int
    episodes_run: int
    landing_rate: float
    best_score: float
    mean_score: float
    scores: list[float] = field(default_factory=list)
    landed_episodes: int = 0
    total_steps: int = 0
    current_state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    best_state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    best_updated: bool = False
    partial_generation: bool = False
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    current_normalizer_state: dict[str, Any] = field(default_factory=dict)
    best_normalizer_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_index": self.generation_index,
            "episodes_run": self.episodes_run,
            "landing_rate": self.landing_rate,
            "best_score": self.best_score,
            "mean_score": self.mean_score,
            "scores": list(self.scores),
            "landed_episodes": self.landed_episodes,
            "total_steps": self.total_steps,
            "best_updated": self.best_updated,
            "partial_generation": self.partial_generation,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "current_normalizer_state": clone_normalizer_state(
                self.current_normalizer_state
            ),
            "best_normalizer_state": clone_normalizer_state(
                self.best_normalizer_state
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationReport":
        return cls(
            generation_index=int(data.get("generation_index", 0)),
            episodes_run=int(data.get("episodes_run", 0)),
            landing_rate=float(data.get("landing_rate", 0.0)),
            best_score=float(data.get("best_score", 0.0)),
            mean_score=float(data.get("mean_score", 0.0)),
            scores=list(data.get("scores", [])),
            landed_episodes=int(data.get("landed_episodes", 0)),
            total_steps=int(data.get("total_steps", 0)),
            best_updated=bool(data.get("best_updated", False)),
            partial_generation=bool(data.get("partial_generation", False)),
            policy_loss=float(data.get("policy_loss", 0.0)),
            value_loss=float(data.get("value_loss", 0.0)),
            entropy=float(data.get("entropy", 0.0)),
            current_normalizer_state=clone_normalizer_state(
                data.get("current_normalizer_state", {})
            ),
            best_normalizer_state=clone_normalizer_state(
                data.get("best_normalizer_state", {})
            ),
        )


@dataclass
class TrainingSummary:
    status: str
    history: list[GenerationReport]
    current_state_dict: dict[str, torch.Tensor]
    best_state_dict: dict[str, torch.Tensor]
    best_metrics: dict[str, float]
    current_optimizer_state_dict: dict[str, Any]
    current_normalizer_state: dict[str, Any]
    best_normalizer_state: dict[str, Any]
    checkpoint_metadata: dict[str, Any]


@dataclass
class EpisodeTrajectory:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    score: float = 0.0
    final_info: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        value: float,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.score += float(reward)


@dataclass
class ActiveEpisode:
    env: RocketLanderEnv
    raw_observation: np.ndarray
    trajectory: EpisodeTrajectory = field(default_factory=EpisodeTrajectory)


class ObservationNormalizer:
    def __init__(
        self,
        dimension: int,
        clip_value: float = 5.0,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.dimension = dimension
        self.clip_value = float(clip_value)
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.var = np.ones(dimension, dtype=np.float64)
        self.count = 1e-4

        if state:
            self.mean = np.asarray(
                state.get("mean", self.mean),
                dtype=np.float64,
            )
            self.var = np.maximum(
                np.asarray(state.get("var", self.var), dtype=np.float64),
                1e-6,
            )
            self.count = float(state.get("count", self.count))
            self.clip_value = float(state.get("clip_value", self.clip_value))

    def update(self, batch: np.ndarray) -> None:
        array = np.asarray(batch, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[0] == 0:
            return

        batch_mean = array.mean(axis=0)
        batch_var = array.var(axis=0)
        batch_count = array.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        m2 = mean_a + mean_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = np.maximum(m2 / total_count, 1e-6)
        self.count = total_count

    def normalize(self, batch: np.ndarray) -> np.ndarray:
        array = np.asarray(batch, dtype=np.float32)
        normalized = (array - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8
        )
        return np.clip(normalized, -self.clip_value, self.clip_value).astype(
            np.float32
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.astype(np.float32).copy(),
            "var": self.var.astype(np.float32).copy(),
            "count": float(self.count),
            "clip_value": float(self.clip_value),
        }


class RolloutBuffer:
    def __init__(self) -> None:
        self.episodes: list[EpisodeTrajectory] = []

    def add_episode(self, trajectory: EpisodeTrajectory) -> None:
        if trajectory.rewards:
            self.episodes.append(trajectory)

    def __len__(self) -> int:
        return sum(len(episode.rewards) for episode in self.episodes)

    def as_tensors(
        self,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        log_probs: list[np.ndarray] = []
        advantages: list[np.ndarray] = []
        returns: list[np.ndarray] = []

        for episode in self.episodes:
            rewards = np.asarray(episode.rewards, dtype=np.float32)
            values = np.asarray(episode.values, dtype=np.float32)
            episode_advantages = np.zeros_like(rewards, dtype=np.float32)

            last_gae = 0.0
            next_value = 0.0
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * next_value - values[step]
                last_gae = delta + gamma * gae_lambda * last_gae
                episode_advantages[step] = last_gae
                next_value = values[step]

            observations.append(
                np.asarray(episode.observations, dtype=np.float32)
            )
            actions.append(np.asarray(episode.actions, dtype=np.float32))
            log_probs.append(np.asarray(episode.log_probs, dtype=np.float32))
            advantages.append(episode_advantages)
            returns.append(episode_advantages + values)

        return {
            "observations": torch.as_tensor(
                np.concatenate(observations, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.as_tensor(
                np.concatenate(actions, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "log_probs": torch.as_tensor(
                np.concatenate(log_probs, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "advantages": torch.as_tensor(
                np.concatenate(advantages, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            "returns": torch.as_tensor(
                np.concatenate(returns, axis=0),
                dtype=torch.float32,
                device=device,
            ),
        }


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def clone_optimizer_state_dict(state_dict: dict[str, Any] | None) -> dict[str, Any]:
    if not state_dict:
        return {}
    return clone_to_cpu(state_dict)


def clone_normalizer_state(state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    mean = state.get("mean")
    var = state.get("var")
    if mean is None or var is None:
        return {}
    return {
        "mean": np.asarray(mean, dtype=np.float32).copy(),
        "var": np.asarray(var, dtype=np.float32).copy(),
        "count": float(state.get("count", 1e-4)),
        "clip_value": float(state.get("clip_value", 5.0)),
    }


def normalize_observation_array(
    observation: np.ndarray,
    normalizer_state: dict[str, Any] | None,
) -> np.ndarray:
    if not normalizer_state:
        return np.asarray(observation, dtype=np.float32)
    observation_array = np.asarray(observation, dtype=np.float32)
    try:
        normalizer = ObservationNormalizer(
            dimension=observation_array.shape[-1],
            state=normalizer_state,
        )
        normalized = normalizer.normalize(observation_array)
        return np.asarray(normalized, dtype=np.float32)
    except Exception:
        return observation_array


def _safe_bool_probe(check: Callable[[], Any]) -> bool:
    try:
        return bool(check())
    except Exception:
        return False


def _backend_label(device: torch.device) -> str:
    if device.type == "cuda" and getattr(torch.version, "hip", None):
        return "rocm"
    return device.type


def _backend_available(device: torch.device) -> bool:
    if device.type == "cpu":
        return True
    if device.type == "cuda":
        return _safe_bool_probe(torch.cuda.is_available)
    if device.type == "xpu":
        xpu_backend = getattr(torch, "xpu", None)
        is_available = getattr(xpu_backend, "is_available", None)
        if is_available is None:
            return False
        return _safe_bool_probe(is_available)
    return False


def _synchronize_backend(device: torch.device) -> None:
    try:
        if device.type == "cuda" and hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
        elif device.type == "xpu":
            xpu_backend = getattr(torch, "xpu", None)
            synchronize = getattr(xpu_backend, "synchronize", None)
            if synchronize is not None:
                synchronize()
    except Exception:
        pass


def _probe_training_backend(device: torch.device) -> None:
    sample = torch.randn((8, 4), dtype=torch.float32, device=device)
    layer = nn.Linear(4, 4).to(device)
    output = layer(sample)
    distribution = torch.distributions.Normal(output, torch.ones_like(output))
    loss = output.square().mean() - 0.01 * distribution.log_prob(output).mean()
    loss.backward()
    _ = torch.randperm(8, device=device)
    _ = next(layer.parameters()).grad.detach().cpu().sum().item()
    _ = output.detach().cpu().mean().item()
    _synchronize_backend(device)


def detect_training_device(
    preferred: str | torch.device | None = None,
) -> tuple[torch.device, str, list[str]]:
    warnings: list[str] = []
    candidates: list[torch.device] = []

    if preferred is not None:
        try:
            candidates.append(torch.device(preferred))
        except Exception as exc:
            warnings.append(
                f"Requested training device {preferred!r} is invalid; "
                f"falling back to auto-detection. ({exc})"
            )
    else:
        candidates.extend([torch.device("cuda"), torch.device("xpu")])

    for candidate in candidates:
        label = _backend_label(candidate)
        if not _backend_available(candidate):
            warnings.append(
                f"{label.upper()} is not available; falling back to another backend."
            )
            continue
        try:
            _probe_training_backend(candidate)
            return candidate, label, warnings
        except Exception as exc:
            warnings.append(
                f"{label.upper()} probe failed; falling back to CPU. ({exc})"
            )
            break

    return torch.device("cpu"), "cpu", warnings


def seed_torch_backends(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    elif device.type == "xpu":
        try:
            xpu_backend = getattr(torch, "xpu", None)
            manual_seed_all = getattr(xpu_backend, "manual_seed_all", None)
            if manual_seed_all is not None:
                manual_seed_all(seed)
        except Exception:
            pass


def is_better_candidate(candidate: dict[str, float], best: dict[str, float]) -> bool:
    if candidate["landing_rate"] > best["landing_rate"] + 1e-9:
        return True
    if math.isclose(candidate["landing_rate"], best["landing_rate"], rel_tol=1e-9, abs_tol=1e-9):
        return candidate["mean_score"] > best["mean_score"]
    return False


class TrainerSession:
    def __init__(
        self,
        config: AppConfig,
        device: str | torch.device | None = None,
        initial_history: list[GenerationReport] | None = None,
        initial_state_dict: dict[str, torch.Tensor] | None = None,
        initial_best_state_dict: dict[str, torch.Tensor] | None = None,
        initial_best_metrics: dict[str, float] | None = None,
        initial_optimizer_state_dict: dict[str, Any] | None = None,
        initial_observation_normalizer_state: dict[str, Any] | None = None,
        initial_best_observation_normalizer_state: dict[str, Any] | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.device, self.device_label, self.device_warnings = detect_training_device(
            device
        )
        seed_torch_backends(self.config.ppo.seed, self.device)

        try:
            self.model = ActorCritic(
                observation_dim=len(OBSERVATION_NAMES),
                action_dim=len(ACTION_NAMES),
                network_config=self.config.network,
                init_std=self.config.ppo.init_std,
            ).to(self.device)
        except Exception as exc:
            if self.device.type != "cpu":
                self.device_warnings.append(
                    f"{self.device_label.upper()} initialization failed; using CPU instead. ({exc})"
                )
                self.device = torch.device("cpu")
                self.device_label = "cpu"
                seed_torch_backends(self.config.ppo.seed, self.device)
                self.model = ActorCritic(
                    observation_dim=len(OBSERVATION_NAMES),
                    action_dim=len(ACTION_NAMES),
                    network_config=self.config.network,
                    init_std=self.config.ppo.init_std,
                ).to(self.device)
            else:
                raise
        if initial_state_dict is not None:
            self.model.load_state_dict(initial_state_dict)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.ppo.learning_rate,
        )
        if initial_optimizer_state_dict:
            try:
                self.optimizer.load_state_dict(initial_optimizer_state_dict)
            except Exception:
                pass
        self.history = [
            GenerationReport.from_dict(report.to_dict())
            for report in (initial_history or [])
        ]
        self.generation_offset = len(self.history)
        self.stop_requested = False
        self.pause_requested = False
        self.parallel_env_count = max(
            1,
            min(self.config.ppo.games_per_generation, 8),
        )
        self.observation_normalizer = ObservationNormalizer(
            dimension=len(OBSERVATION_NAMES),
            state=initial_observation_normalizer_state,
        )
        self.best_state_dict = clone_state_dict(
            initial_best_state_dict or self.model.state_dict()
        )
        self.best_observation_normalizer_state = clone_normalizer_state(
            initial_best_observation_normalizer_state
            or self.observation_normalizer.state_dict()
        )
        self.best_metrics = dict(
            initial_best_metrics
            or {
                "landing_rate": -1.0,
                "mean_score": float("-inf"),
            }
        )

    def request_stop(self) -> None:
        self.stop_requested = True

    def request_pause(self) -> None:
        self.pause_requested = True

    def train(
        self,
        generations: int | None = None,
        on_generation: Callable[[GenerationReport], None] | None = None,
    ) -> TrainingSummary:
        target_generations = generations or self.config.ppo.target_generations
        status = "completed"

        for generation_index in range(
            self.generation_offset,
            self.generation_offset + target_generations,
        ):
            if self.stop_requested:
                status = "stopped"
                break

            report = self._run_generation(generation_index)
            metrics = {
                "landing_rate": report.landing_rate,
                "mean_score": report.mean_score,
            }
            if (not report.partial_generation) and is_better_candidate(
                metrics,
                self.best_metrics,
            ):
                self.best_metrics = metrics
                self.best_state_dict = clone_state_dict(self.model.state_dict())
                self.best_observation_normalizer_state = clone_normalizer_state(
                    self.observation_normalizer.state_dict()
                )
                report.best_updated = True

            report.current_state_dict = clone_state_dict(self.model.state_dict())
            report.best_state_dict = clone_state_dict(self.best_state_dict)
            report.current_normalizer_state = clone_normalizer_state(
                self.observation_normalizer.state_dict()
            )
            report.best_normalizer_state = clone_normalizer_state(
                self.best_observation_normalizer_state
            )
            self.history.append(report)
            if on_generation is not None:
                on_generation(report)

            if self.stop_requested:
                status = "stopped"
                break
            if self.pause_requested:
                status = "paused"
                break

        return TrainingSummary(
            status=status,
            history=list(self.history),
            current_state_dict=clone_state_dict(self.model.state_dict()),
            best_state_dict=clone_state_dict(self.best_state_dict),
            best_metrics=dict(self.best_metrics),
            current_optimizer_state_dict=clone_optimizer_state_dict(
                self.optimizer.state_dict()
            ),
            current_normalizer_state=clone_normalizer_state(
                self.observation_normalizer.state_dict()
            ),
            best_normalizer_state=clone_normalizer_state(
                self.best_observation_normalizer_state
            ),
            checkpoint_metadata=build_checkpoint_metadata(
                history=self.history,
                best_metrics=self.best_metrics,
                source_label="best",
                observation_normalizer_state=self.best_observation_normalizer_state,
                has_resume_state=True,
            ),
        )

    def _run_generation(self, generation_index: int) -> GenerationReport:
        buffer = RolloutBuffer()
        scores: list[float] = []
        landed_episodes = 0
        total_steps = 0
        generation_seed_offset = self.config.ppo.seed + generation_index * 10000
        target_episodes = self.config.ppo.games_per_generation
        environments = [
            RocketLanderEnv(
                physics=self.config.physics,
                rewards=self.config.rewards,
                seed=generation_seed_offset + slot_index,
            )
            for slot_index in range(self.parallel_env_count)
        ]
        active_episodes: list[ActiveEpisode | None] = [None] * len(environments)
        episodes_started = 0

        def launch_episode(slot_index: int) -> None:
            nonlocal episodes_started
            if episodes_started >= target_episodes:
                return
            episode_seed = generation_seed_offset + episodes_started
            raw_observation = environments[slot_index].reset(
                seed=episode_seed,
                dramatic=True,
            )
            active_episodes[slot_index] = ActiveEpisode(
                env=environments[slot_index],
                raw_observation=np.asarray(raw_observation, dtype=np.float32),
            )
            episodes_started += 1

        for slot_index in range(len(active_episodes)):
            launch_episode(slot_index)

        while any(episode is not None for episode in active_episodes):
            active_indices = [
                index
                for index, episode in enumerate(active_episodes)
                if episode is not None
            ]
            raw_batch = np.stack(
                [
                    np.asarray(active_episodes[index].raw_observation, dtype=np.float32)
                    for index in active_indices
                ],
                axis=0,
            )
            self.observation_normalizer.update(raw_batch)
            normalized_batch = self.observation_normalizer.normalize(raw_batch)

            observation_tensor = torch.as_tensor(
                normalized_batch,
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor, _ = self.model.act(
                    observation_tensor,
                    deterministic=False,
                )

            actions = action_tensor.cpu().numpy()
            log_probs = log_prob_tensor.cpu().numpy()
            values = value_tensor.cpu().numpy()

            for batch_index, slot_index in enumerate(active_indices):
                active_episode = active_episodes[slot_index]
                assert active_episode is not None

                next_observation, reward, done, info = active_episode.env.step(
                    actions[batch_index]
                )
                active_episode.trajectory.add(
                    observation=normalized_batch[batch_index],
                    action=actions[batch_index],
                    reward=reward,
                    log_prob=float(log_probs[batch_index]),
                    value=float(values[batch_index]),
                )
                total_steps += 1
                active_episode.raw_observation = np.asarray(
                    next_observation,
                    dtype=np.float32,
                )

                if not done:
                    continue

                active_episode.trajectory.final_info = dict(info)
                buffer.add_episode(active_episode.trajectory)
                scores.append(active_episode.trajectory.score)
                landed_episodes += int(bool(info.get("landed", False)))
                active_episodes[slot_index] = None

                if (
                    episodes_started < target_episodes
                    and not self.stop_requested
                ):
                    launch_episode(slot_index)

        if len(buffer) == 0:
            return GenerationReport(
                generation_index=generation_index,
                episodes_run=0,
                landing_rate=0.0,
                best_score=0.0,
                mean_score=0.0,
                total_steps=0,
            )

        optimization_stats = self._update_policy(buffer)
        episodes_run = max(1, len(scores))
        return GenerationReport(
            generation_index=generation_index,
            episodes_run=episodes_run,
            landing_rate=landed_episodes / episodes_run,
            best_score=max(scores),
            mean_score=float(np.mean(scores)),
            scores=scores,
            landed_episodes=landed_episodes,
            total_steps=total_steps,
            partial_generation=episodes_run < target_episodes,
            policy_loss=optimization_stats["policy_loss"],
            value_loss=optimization_stats["value_loss"],
            entropy=optimization_stats["entropy"],
        )

    def _update_policy(self, buffer: RolloutBuffer) -> dict[str, float]:
        batch = buffer.as_tensors(
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda,
            device=self.device,
        )
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = batch["observations"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        batch_size = observations.shape[0]
        minibatch_size = min(self.config.ppo.minibatch_size, batch_size)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        updates = 0

        for _ in range(self.config.ppo.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    observations[indices],
                    actions[indices],
                )
                ratio = (new_log_probs - old_log_probs[indices]).exp()
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo.clip_range,
                    1.0 + self.config.ppo.clip_range,
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns[indices])
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.ppo.value_coef * value_loss
                    - self.config.ppo.entropy_coef * entropy_bonus
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.ppo.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_bonus.item())
                updates += 1

        if updates:
            for key in metrics:
                metrics[key] /= updates
        return metrics


def build_policy_from_state(
    config: AppConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
) -> ActorCritic:
    policy = ActorCritic(
        observation_dim=len(OBSERVATION_NAMES),
        action_dim=len(ACTION_NAMES),
        network_config=config.network,
        init_std=config.ppo.init_std,
    ).to(device)
    if state_dict is not None:
        policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def build_checkpoint_metadata(
    history: list[GenerationReport],
    best_metrics: dict[str, float],
    source_label: str,
    observation_normalizer_state: dict[str, Any] | None,
    has_resume_state: bool = False,
) -> dict[str, Any]:
    best_report: GenerationReport | None = None
    best_candidate = {
        "landing_rate": -1.0,
        "mean_score": float("-inf"),
    }
    for report in history:
        candidate = {
            "landing_rate": report.landing_rate,
            "mean_score": report.mean_score,
        }
        if (not report.partial_generation) and is_better_candidate(
            candidate,
            best_candidate,
        ):
            best_report = report
            best_candidate = candidate

    last_report = history[-1] if history else None
    return {
        "schema_version": 2,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_label": source_label,
        "generation_count": len(history),
        "total_episodes": int(sum(report.episodes_run for report in history)),
        "total_steps": int(sum(report.total_steps for report in history)),
        "best_metrics": dict(best_metrics),
        "last_report": last_report.to_dict() if last_report is not None else None,
        "best_report": best_report.to_dict() if best_report is not None else None,
        "resume_supported": bool(has_resume_state),
        "observation_normalizer_state": clone_normalizer_state(
            observation_normalizer_state
        ),
    }


def save_brain_checkpoint(
    path: Path | str,
    config: AppConfig,
    state_dict: dict[str, torch.Tensor],
    best_metrics: dict[str, float],
    history: list[GenerationReport],
    source_label: str,
    current_state_dict: dict[str, torch.Tensor] | None = None,
    best_state_dict: dict[str, torch.Tensor] | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    observation_normalizer_state: dict[str, Any] | None = None,
    best_observation_normalizer_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_metadata = metadata or build_checkpoint_metadata(
        history=history,
        best_metrics=best_metrics,
        source_label=source_label,
        observation_normalizer_state=observation_normalizer_state,
        has_resume_state=bool(current_state_dict or optimizer_state_dict),
    )
    payload = {
        "config": config.to_dict(),
        "state_dict": clone_state_dict(state_dict),
        "current_state_dict": clone_state_dict(current_state_dict or state_dict),
        "best_state_dict": clone_state_dict(best_state_dict or state_dict),
        "optimizer_state_dict": clone_optimizer_state_dict(optimizer_state_dict),
        "best_metrics": dict(best_metrics),
        "history": [report.to_dict() for report in history],
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "source_label": source_label,
        "observation_normalizer_state": clone_normalizer_state(
            observation_normalizer_state
        ),
        "best_observation_normalizer_state": clone_normalizer_state(
            best_observation_normalizer_state
            or observation_normalizer_state
        ),
        "metadata": checkpoint_metadata,
    }
    torch.save(payload, Path(path))


def load_brain_checkpoint(path: Path | str) -> dict[str, Any]:
    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = AppConfig.from_dict(raw["config"])
    history = [GenerationReport.from_dict(item) for item in raw.get("history", [])]
    return {
        "config": config,
        "state_dict": raw["state_dict"],
        "current_state_dict": raw.get("current_state_dict", raw["state_dict"]),
        "best_state_dict": raw.get("best_state_dict", raw["state_dict"]),
        "optimizer_state_dict": clone_optimizer_state_dict(
            raw.get("optimizer_state_dict", {})
        ),
        "best_metrics": dict(raw.get("best_metrics", {})),
        "history": history,
        "observation_names": raw.get("observation_names", list(OBSERVATION_NAMES)),
        "action_names": raw.get("action_names", list(ACTION_NAMES)),
        "source_label": raw.get("source_label", "best"),
        "observation_normalizer_state": clone_normalizer_state(
            raw.get("observation_normalizer_state", {})
        ),
        "best_observation_normalizer_state": clone_normalizer_state(
            raw.get(
                "best_observation_normalizer_state",
                raw.get("observation_normalizer_state", {}),
            )
        ),
        "metadata": dict(
            raw.get(
                "metadata",
                build_checkpoint_metadata(
                    history=history,
                    best_metrics=dict(raw.get("best_metrics", {})),
                    source_label=raw.get("source_label", "best"),
                    observation_normalizer_state=raw.get(
                        "observation_normalizer_state",
                        {},
                    ),
                    has_resume_state=bool(raw.get("current_state_dict")),
                ),
            )
        ),
        "path": str(path),
    }


def smoke_test(seed: int = 11) -> tuple[bool, str]:
    try:
        config = AppConfig()
        config.ppo.target_generations = 2
        config.ppo.games_per_generation = 4
        config.ppo.ppo_epochs = 2
        config.ppo.minibatch_size = 64
        config.ppo.seed = seed

        env = RocketLanderEnv(config.physics, config.rewards, seed=seed)
        observation = env.reset()
        if observation.shape[0] != len(OBSERVATION_NAMES):
            return False, "Smoke test failed: unexpected observation size."

        for _ in range(10):
            observation, _, done, _ = env.step(np.array([0.1, -0.2], dtype=np.float32))
            if done:
                observation = env.reset()

        session = TrainerSession(config=config, device="cpu")
        summary = session.train()
        if not summary.history:
            return False, "Smoke test failed: no training history was produced."

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "rocket_best.pt"
            save_brain_checkpoint(
                path=checkpoint_path,
                config=config,
                state_dict=summary.best_state_dict,
                best_metrics=summary.best_metrics,
                history=summary.history,
                source_label="best",
                current_state_dict=summary.current_state_dict,
                best_state_dict=summary.best_state_dict,
                optimizer_state_dict=summary.current_optimizer_state_dict,
                observation_normalizer_state=summary.best_normalizer_state,
                best_observation_normalizer_state=summary.best_normalizer_state,
            )
            payload = load_brain_checkpoint(checkpoint_path)
            if len(payload["history"]) != len(summary.history):
                return False, "Smoke test failed: checkpoint history mismatch."

        last_report = summary.history[-1]
        return (
            True,
            "Smoke test passed: "
            f"{len(summary.history)} generations, "
            f"landing_rate={last_report.landing_rate:.3f}, "
            f"mean_score={last_report.mean_score:.3f}",
        )
    except Exception as exc:  # pragma: no cover - smoke output is user facing
        return False, f"Smoke test failed with exception: {exc}"
