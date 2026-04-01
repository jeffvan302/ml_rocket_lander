from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ACTIVATION_OPTIONS = [
    "relu",
    "tanh",
    "elu",
    "leaky_relu",
    "selu",
    "identity",
]


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError:
            continue
    return values


def find_invalid_float_tokens(text: str) -> list[str]:
    invalid: list[str] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            float(stripped)
        except ValueError:
            invalid.append(stripped)
    return invalid


@dataclass
class LayerConfig:
    units: int
    activation: str = "relu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerConfig":
        return cls(
            units=int(data.get("units", 8)),
            activation=str(data.get("activation", "relu")),
        )


@dataclass
class PhysicsConfig:
    world_width: float = 100.0
    world_height: float = 140.0
    dt: float = 0.1
    gravity: float = 9.5
    gravity_multi_mode: bool = False
    gravity_values_text: str = ""
    main_thrust: float = 19.0
    drag_coefficient: float = 0.03
    wind_strength: float = 0.0
    fuel_capacity: float = 100.0
    fuel_burn_rate: float = 1.15
    angular_acceleration: float = 4.0
    angular_damping: float = 0.92
    rocket_height: float = 8.0
    rocket_width: float = 2.8
    pad_width: float = 16.0
    landing_angle_deg: float = 12.0
    landing_max_vx: float = 3.0
    landing_max_vy: float = 4.0
    max_steps: int = 500
    spawn_x_extent: float = 44.0
    spawn_y_min: float = 78.0
    spawn_y_max: float = 130.0
    spawn_angle_deg: float = 38.0
    spawn_velocity_x: float = 3.0
    spawn_velocity_y: float = 2.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parsed_gravity_values(self) -> list[float]:
        return parse_float_list(self.gravity_values_text)

    def gravity_candidates(self) -> list[float]:
        values = self.parsed_gravity_values()
        if self.gravity_multi_mode and values:
            return values
        return [float(self.gravity)]


@dataclass
class RewardConfig:
    landing_bonus: float = 220.0
    close_to_pad_bonus: float = 4.0
    progress_scale: float = 80.0
    alive_bonus: float = 0.15
    center_bonus: float = 4.0
    fuel_bonus: float = 2.0
    upright_bonus: float = 3.0
    delta_x_penalty: float = 1.5
    delta_y_penalty: float = 0.8
    step_delta_x_penalty: float = 0.6
    step_delta_y_penalty: float = 0.4
    crash_penalty: float = 90.0
    offscreen_penalty: float = 120.0
    timeout_penalty: float = 60.0
    turn_penalty: float = 0.0
    throttle_penalty: float = 0.0
    step_penalty: float = 0.08
    velocity_penalty: float = 1.2
    spin_penalty: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PPOConfig:
    target_generations: int = 150
    games_per_generation: int = 24
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.7
    ppo_epochs: int = 8
    minibatch_size: int = 256
    max_grad_norm: float = 0.7
    init_std: float = 0.45
    seed: int = 7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkConfig:
    hidden_layers: list[LayerConfig] = field(
        default_factory=lambda: [
            LayerConfig(8, "relu"),
            LayerConfig(40, "relu"),
            LayerConfig(8, "relu"),
        ]
    )
    output_activation: str = "tanh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": [layer.to_dict() for layer in self.hidden_layers],
            "output_activation": self.output_activation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkConfig":
        layers = [
            LayerConfig.from_dict(item)
            for item in data.get("hidden_layers", [])
        ]
        if not layers:
            layers = [
                LayerConfig(8, "relu"),
                LayerConfig(40, "relu"),
                LayerConfig(8, "relu"),
            ]
        return cls(
            hidden_layers=layers,
            output_activation=str(data.get("output_activation", "tanh")),
        )


@dataclass
class AppConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "physics": self.physics.to_dict(),
            "rewards": self.rewards.to_dict(),
            "ppo": self.ppo.to_dict(),
            "network": self.network.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return cls(
            physics=PhysicsConfig(**data.get("physics", {})),
            rewards=RewardConfig(**data.get("rewards", {})),
            ppo=PPOConfig(**data.get("ppo", {})),
            network=NetworkConfig.from_dict(data.get("network", {})),
        )
