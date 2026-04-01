from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rocket_lander.config import PhysicsConfig, RewardConfig


OBSERVATION_NAMES = [
    "delta_y_from_pad",
    "delta_x_from_pad",
    "angle_to_pad_center",
    "sin_angle_to_pad",
    "cos_angle_to_pad",
    "fuel_left",
    "steps",
    "velocity_y",
    "velocity_x",
    "facing_angle",
    "velocity",
    "angular_velocity",
    "distance_to_pad",
]


@dataclass
class RocketState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    fuel: float
    steps: int = 0
    score: float = 0.0
    last_distance: float = 0.0
    throttle: float = 0.0
    turn: float = 0.0
    trail: list[tuple[float, float]] = field(default_factory=list)


class RocketLanderEnv:
    def __init__(
        self,
        physics: PhysicsConfig | None = None,
        rewards: RewardConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.physics = physics or PhysicsConfig()
        self.rewards = rewards or RewardConfig()
        self.rng = random.Random(seed)
        self.state: RocketState | None = None
        self.pad_x = 0.0
        self.pad_y = 0.0
        self.last_info: dict[str, Any] = {}

    @property
    def observation_size(self) -> int:
        return len(OBSERVATION_NAMES)

    @property
    def action_size(self) -> int:
        return 2

    def reset(self, seed: int | None = None, dramatic: bool = True) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        side = -1.0 if self.rng.random() < 0.5 else 1.0
        edge_bias = 0.60 + 0.40 * (self.rng.random() ** 0.35)
        x = side * edge_bias * self.physics.spawn_x_extent
        if not dramatic:
            x *= 0.5
        if self.rng.random() < 0.20:
            x *= self.rng.uniform(0.15, 0.55)

        y = self.rng.uniform(
            self.physics.spawn_y_min,
            self.physics.spawn_y_max,
        )

        inbound_velocity = -math.copysign(
            self.rng.uniform(0.5, self.physics.spawn_velocity_x),
            x if abs(x) > 1e-6 else side,
        )
        vx = 0.7 * inbound_velocity + 0.3 * self.rng.uniform(
            -self.physics.spawn_velocity_x,
            self.physics.spawn_velocity_x,
        )
        vy = -self.rng.uniform(0.4, self.physics.spawn_velocity_y)

        angle_deg = self.rng.uniform(
            -self.physics.spawn_angle_deg,
            self.physics.spawn_angle_deg,
        )
        if abs(x) > 20.0:
            angle_deg += -math.copysign(self.rng.uniform(0.0, 12.0), x)
        angle = math.radians(angle_deg)

        state = RocketState(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            angle=angle,
            angular_velocity=0.0,
            fuel=self.physics.fuel_capacity,
        )
        state.last_distance = self._distance_to_pad(state.x, state.y)
        state.trail.append((state.x, state.y))
        self.state = state
        self.last_info = {
            "landed": False,
            "crashed": False,
            "timeout": False,
            "offscreen": False,
        }
        return self._get_observation()

    def set_physics(self, physics: PhysicsConfig) -> None:
        self.physics = physics

    def set_rewards(self, rewards: RewardConfig) -> None:
        self.rewards = rewards

    def _distance_to_pad(self, x: float, y: float) -> float:
        return math.hypot(x - self.pad_x, y - self.pad_y)

    def _angle_to_pad_center(self) -> float:
        assert self.state is not None
        altitude = max(self.state.y - self.pad_y, 1e-6)
        return math.atan2(self.pad_x - self.state.x, altitude)

    def _max_distance(self) -> float:
        return math.hypot(self.physics.world_width, self.physics.world_height)

    def _max_speed(self) -> float:
        return max(
            10.0,
            self.physics.main_thrust * 0.9,
            self.physics.gravity * 2.5,
        )

    def _max_angular_speed(self) -> float:
        return max(1.5, self.physics.angular_acceleration * self.physics.dt * 7.0)

    def _get_observation(self) -> np.ndarray:
        assert self.state is not None
        dx = self.state.x - self.pad_x
        dy = self.state.y - self.pad_y
        angle_to_pad = self._angle_to_pad_center()
        speed = math.hypot(self.state.vx, self.state.vy)
        obs = np.array(
            [
                dy / max(self.physics.world_height, 1.0),
                dx / max(self.physics.world_width * 0.5, 1.0),
                angle_to_pad / math.pi,
                math.sin(angle_to_pad),
                math.cos(angle_to_pad),
                self.state.fuel / max(self.physics.fuel_capacity, 1.0),
                self.state.steps / max(self.physics.max_steps, 1),
                self.state.vy / self._max_speed(),
                self.state.vx / self._max_speed(),
                self.state.angle / math.pi,
                speed / self._max_speed(),
                self.state.angular_velocity / self._max_angular_speed(),
                self._distance_to_pad(self.state.x, self.state.y) / self._max_distance(),
            ],
            dtype=np.float32,
        )
        return obs

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, float],
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        assert self.state is not None
        throttle_signal = float(np.clip(action[0], -1.0, 1.0))
        turn_signal = float(np.clip(action[1], -1.0, 1.0))
        throttle = np.clip((throttle_signal + 1.0) * 0.5, 0.0, 1.0)

        dt = self.physics.dt
        fuel_cost = throttle * self.physics.fuel_burn_rate * dt
        if fuel_cost > self.state.fuel:
            throttle = self.state.fuel / max(self.physics.fuel_burn_rate * dt, 1e-6)
            fuel_cost = self.state.fuel
        self.state.fuel = max(0.0, self.state.fuel - fuel_cost)

        self.state.angular_velocity += (
            turn_signal * self.physics.angular_acceleration * dt
        )
        self.state.angular_velocity *= self.physics.angular_damping
        self.state.angle += self.state.angular_velocity * dt
        self.state.angle = ((self.state.angle + math.pi) % (2.0 * math.pi)) - math.pi

        thrust_acc = throttle * self.physics.main_thrust
        wind = self.physics.wind_strength * math.sin(
            0.035 * self.state.steps + 0.07 * self.state.x
        )
        drag_x = self.physics.drag_coefficient * self.state.vx
        drag_y = self.physics.drag_coefficient * self.state.vy
        ax = math.sin(self.state.angle) * thrust_acc + wind - drag_x
        ay = math.cos(self.state.angle) * thrust_acc - self.physics.gravity - drag_y

        self.state.vx += ax * dt
        self.state.vy += ay * dt
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt
        self.state.steps += 1
        self.state.throttle = throttle
        self.state.turn = turn_signal
        self.state.trail.append((self.state.x, self.state.y))
        if len(self.state.trail) > 80:
            self.state.trail = self.state.trail[-80:]

        reward, done, info = self._calculate_reward_and_termination(
            throttle=throttle,
            turn_signal=turn_signal,
        )
        self.state.score += reward
        self.last_info = info
        return self._get_observation(), reward, done, info

    def _calculate_reward_and_termination(
        self,
        throttle: float,
        turn_signal: float,
    ) -> tuple[float, bool, dict[str, Any]]:
        assert self.state is not None
        dx = self.state.x - self.pad_x
        dy = self.state.y - self.pad_y
        distance = self._distance_to_pad(self.state.x, self.state.y)
        progress = self.state.last_distance - distance
        self.state.last_distance = distance

        speed = math.hypot(self.state.vx, self.state.vy)
        dx_norm = min(abs(dx) / max(self.physics.world_width * 0.5, 1.0), 2.0)
        dy_norm = min(abs(dy) / max(self.physics.world_height, 1.0), 2.0)
        distance_norm = min(distance / self._max_distance(), 2.0)
        step_ratio = self.state.steps / max(self.physics.max_steps, 1)
        speed_norm = min(speed / self._max_speed(), 2.0)
        spin_norm = min(
            abs(self.state.angular_velocity) / self._max_angular_speed(),
            2.0,
        )
        upright_score = 1.0 - min(abs(self.state.angle) / math.pi, 1.0)
        fuel_ratio = self.state.fuel / max(self.physics.fuel_capacity, 1.0)

        reward = 0.0
        reward += self.rewards.alive_bonus
        reward += self.rewards.progress_scale * (progress / self._max_distance())
        reward += self.rewards.close_to_pad_bonus * (1.0 - distance_norm) * 0.08
        reward += self.rewards.center_bonus * (1.0 - dx_norm) * 0.07
        reward += self.rewards.fuel_bonus * fuel_ratio * 0.02
        reward += self.rewards.upright_bonus * upright_score * 0.05
        reward -= self.rewards.delta_x_penalty * dx_norm * 0.12
        reward -= self.rewards.delta_y_penalty * dy_norm * 0.10
        reward -= self.rewards.step_delta_x_penalty * step_ratio * dx_norm * 0.08
        reward -= self.rewards.step_delta_y_penalty * step_ratio * dy_norm * 0.08
        reward -= self.rewards.turn_penalty * abs(turn_signal) * 0.05
        reward -= self.rewards.throttle_penalty * throttle * 0.05
        reward -= self.rewards.velocity_penalty * speed_norm * 0.05
        reward -= self.rewards.spin_penalty * spin_norm * 0.05
        reward -= self.rewards.step_penalty

        done = False
        landed = False
        crashed = False
        offscreen = False
        timeout = False
        event = "flying"

        touchdown_height = self.physics.rocket_height * 0.5
        if self.state.y <= touchdown_height:
            self.state.y = touchdown_height
            landed = (
                abs(dx) <= self.physics.pad_width * 0.5
                and abs(math.degrees(self.state.angle)) <= self.physics.landing_angle_deg
                and abs(self.state.vx) <= self.physics.landing_max_vx
                and abs(self.state.vy) <= self.physics.landing_max_vy
                and spin_norm <= 1.0
            )
            done = True
            if landed:
                event = "landed"
                reward += self.rewards.landing_bonus
                reward += self.rewards.fuel_bonus * fuel_ratio * 10.0
                reward += self.rewards.upright_bonus * upright_score
                reward += 15.0 * (1.0 - min(speed_norm, 1.0))
                self.state.vx = 0.0
                self.state.vy = 0.0
                self.state.angular_velocity = 0.0
            else:
                event = "crashed"
                crashed = True
                reward -= self.rewards.crash_penalty

        x_limit = self.physics.world_width * 0.65
        if not done and (
            abs(self.state.x) > x_limit
            or self.state.y > self.physics.world_height * 1.08
            or self.state.y < -4.0
        ):
            done = True
            offscreen = True
            event = "offscreen"
            reward -= self.rewards.offscreen_penalty

        if not done and self.state.steps >= self.physics.max_steps:
            done = True
            timeout = True
            event = "timeout"
            reward -= self.rewards.timeout_penalty

        info = {
            "event": event,
            "landed": landed,
            "crashed": crashed,
            "offscreen": offscreen,
            "timeout": timeout,
            "distance_to_pad": distance,
            "fuel_ratio": fuel_ratio,
            "speed": speed,
            "score": self.state.score + reward,
        }
        return reward, done, info

    def snapshot(self) -> dict[str, Any]:
        if self.state is None:
            return {
                "physics": self.physics.to_dict(),
                "pad_x": self.pad_x,
                "pad_y": self.pad_y,
                "state": None,
                "info": self.last_info,
            }

        return {
            "physics": self.physics.to_dict(),
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "state": {
                "x": self.state.x,
                "y": self.state.y,
                "vx": self.state.vx,
                "vy": self.state.vy,
                "angle": self.state.angle,
                "angular_velocity": self.state.angular_velocity,
                "fuel": self.state.fuel,
                "steps": self.state.steps,
                "score": self.state.score,
                "throttle": self.state.throttle,
                "turn": self.state.turn,
                "trail": list(self.state.trail),
            },
            "info": dict(self.last_info),
        }
