from __future__ import annotations

import math
from dataclasses import dataclass, field

from rocket_lander.config import (
    ACTIVATION_OPTIONS,
    AppConfig,
    find_invalid_float_tokens,
)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_app_config(config: AppConfig) -> ValidationResult:
    result = ValidationResult()
    physics = config.physics
    rewards = config.rewards
    ppo = config.ppo
    network = config.network

    if physics.world_width <= 10.0:
        result.errors.append("World width must be greater than 10.")
    if physics.world_height <= 10.0:
        result.errors.append("World height must be greater than 10.")
    if physics.dt <= 0.0:
        result.errors.append("Physics dt must be positive.")
    if physics.gravity <= 0.0:
        result.errors.append("Gravity must be positive.")
    gravity_values = physics.parsed_gravity_values()
    invalid_gravity_tokens = find_invalid_float_tokens(physics.gravity_values_text)
    if physics.gravity_multi_mode:
        if not physics.gravity_values_text.strip():
            result.errors.append(
                "Gravity list mode is enabled, but the gravity list is empty."
            )
        elif not gravity_values:
            result.errors.append(
                "Gravity list mode is enabled, but no valid gravity values were found."
            )
        if invalid_gravity_tokens:
            result.errors.append(
                "Gravity list contains invalid values: "
                + ", ".join(invalid_gravity_tokens[:4])
            )
        for gravity_value in gravity_values:
            if gravity_value <= 0.0:
                result.errors.append("Gravity list values must be positive.")
            if gravity_value < 0.1 or gravity_value > 50.0:
                result.errors.append(
                    "Gravity list values must stay within the supported range of 0.1 to 50.0."
                )
    if physics.main_thrust <= 0.0:
        result.errors.append("Main thrust must be positive.")
    if physics.drag_coefficient < 0.0:
        result.errors.append("Drag cannot be negative.")
    if physics.fuel_capacity <= 0.0:
        result.errors.append("Fuel capacity must be positive.")
    if physics.fuel_burn_rate <= 0.0:
        result.errors.append("Fuel burn rate must be positive.")
    if physics.angular_acceleration <= 0.0:
        result.errors.append("Angular acceleration must be positive.")
    if not 0.0 < physics.angular_damping <= 1.0:
        result.errors.append("Angular damping must be between 0 and 1.")
    if physics.rocket_height <= 0.0 or physics.rocket_width <= 0.0:
        result.errors.append("Rocket dimensions must be positive.")
    if physics.pad_width <= 0.0:
        result.errors.append("Pad width must be positive.")
    if physics.landing_angle_deg <= 0.0 or physics.landing_angle_deg > 90.0:
        result.errors.append("Landing angle must be between 0 and 90 degrees.")
    if physics.landing_max_vx <= 0.0 or physics.landing_max_vy <= 0.0:
        result.errors.append("Landing max velocities must be positive.")
    if physics.max_steps < 25:
        result.errors.append("Max steps must be at least 25.")
    if physics.spawn_x_extent < 0.0:
        result.errors.append("Spawn x extent cannot be negative.")
    if physics.spawn_y_min <= 0.0 or physics.spawn_y_max <= 0.0:
        result.errors.append("Spawn y values must be positive.")
    if physics.spawn_y_min > physics.spawn_y_max:
        result.errors.append("Spawn y min must be less than or equal to spawn y max.")
    if physics.spawn_angle_deg < 0.0 or physics.spawn_angle_deg > 180.0:
        result.errors.append("Spawn angle must be between 0 and 180 degrees.")
    if physics.spawn_velocity_x < 0.0 or physics.spawn_velocity_y < 0.0:
        result.errors.append("Spawn velocities cannot be negative.")
    if physics.pad_width >= physics.world_width:
        result.errors.append("Pad width must be smaller than the world width.")
    if physics.spawn_y_max >= physics.world_height * 1.5:
        result.warnings.append("Spawn y max is very high relative to the world height.")
    if physics.spawn_x_extent >= physics.world_width * 0.75:
        result.warnings.append("Spawn x extent is close to the world edge, which makes landing harder.")
    gravity_reference = max(gravity_values) if (physics.gravity_multi_mode and gravity_values) else physics.gravity
    if physics.main_thrust <= gravity_reference * 1.02:
        result.warnings.append("Main thrust is only slightly stronger than gravity, so control may be weak.")
    if physics.gravity_multi_mode and len(gravity_values) >= 2:
        gravity_span = max(gravity_values) - min(gravity_values)
        if gravity_span >= 6.0:
            result.warnings.append(
                "Gravity list spans a wide range, so the agent may need longer training to generalize."
            )

    if ppo.target_generations < 1:
        result.errors.append("Target generations must be at least 1.")
    if ppo.games_per_generation < 1:
        result.errors.append("Games per generation must be at least 1.")
    if ppo.learning_rate <= 0.0:
        result.errors.append("Learning rate must be positive.")
    if not 0.0 < ppo.gamma <= 0.999999:
        result.errors.append("Gamma must be in the range (0, 0.999999].")
    if not 0.0 < ppo.gae_lambda <= 0.999999:
        result.errors.append("GAE lambda must be in the range (0, 0.999999].")
    if not 0.0 < ppo.clip_range <= 1.0:
        result.errors.append("Clip range must be in the range (0, 1].")
    if ppo.entropy_coef < 0.0:
        result.errors.append("Entropy coef cannot be negative.")
    if ppo.value_coef < 0.0:
        result.errors.append("Value coef cannot be negative.")
    if ppo.ppo_epochs < 1:
        result.errors.append("PPO epochs must be at least 1.")
    if ppo.minibatch_size < 8:
        result.errors.append("Minibatch size must be at least 8.")
    if ppo.max_grad_norm <= 0.0:
        result.errors.append("Max grad norm must be positive.")
    if ppo.init_std <= 0.0:
        result.errors.append("Action std must be positive.")

    if not network.hidden_layers:
        result.errors.append("At least one hidden layer is required.")
    for index, layer in enumerate(network.hidden_layers, start=1):
        if layer.units < 1:
            result.errors.append(f"Hidden layer {index} must have at least one unit.")
        if layer.activation not in ACTIVATION_OPTIONS:
            result.errors.append(
                f"Hidden layer {index} uses unsupported activation '{layer.activation}'."
            )
    if network.output_activation not in ACTIVATION_OPTIONS:
        result.errors.append(
            f"Output activation '{network.output_activation}' is not supported."
        )

    reward_values = rewards.to_dict()
    for name, value in reward_values.items():
        if not math.isfinite(float(value)):
            result.errors.append(f"Reward value '{name}' must be finite.")

    if rewards.crash_penalty <= 0.0:
        result.warnings.append("Crash penalty is not positive, so the agent may not avoid bad touchdowns.")
    if rewards.offscreen_penalty <= 0.0:
        result.warnings.append("Offscreen penalty is not positive, so leaving the world may not be discouraged.")
    if rewards.timeout_penalty <= 0.0:
        result.warnings.append("Timeout penalty is not positive, so long indecisive flights may persist.")

    return result
