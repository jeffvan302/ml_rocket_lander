from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import queue

import numpy as np

from rocket_lander.config import AppConfig
from rocket_lander.environment import OBSERVATION_NAMES, RocketLanderEnv
from rocket_lander.training import (
    TrainerSession,
    load_brain_checkpoint,
    save_brain_checkpoint,
)
from rocket_lander.ui_app import MainApplication
from rocket_lander.validation import validate_app_config


class RocketLanderSmokeTests(unittest.TestCase):
    def test_environment_observation_shape(self) -> None:
        env = RocketLanderEnv(seed=4)
        obs = env.reset()
        self.assertEqual(obs.shape[0], len(OBSERVATION_NAMES))
        next_obs, reward, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertEqual(next_obs.shape[0], len(OBSERVATION_NAMES))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("event", info)

    def test_trainer_runs_and_checkpoint_roundtrip(self) -> None:
        config = AppConfig()
        config.ppo.target_generations = 1
        config.ppo.games_per_generation = 3
        config.ppo.ppo_epochs = 2
        config.ppo.minibatch_size = 32

        summary = TrainerSession(config=config, device="cpu").train()
        self.assertEqual(summary.status, "completed")
        self.assertEqual(len(summary.history), 1)
        self.assertFalse(summary.history[0].partial_generation)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "brain.pt"
            save_brain_checkpoint(
                path=checkpoint_path,
                config=config,
                state_dict=summary.best_state_dict,
                best_metrics=summary.best_metrics,
                history=summary.history,
                source_label="best",
                observation_normalizer_state=summary.best_normalizer_state,
                metadata=summary.checkpoint_metadata,
            )
            payload = load_brain_checkpoint(checkpoint_path)
            self.assertEqual(len(payload["history"]), 1)
            self.assertIn("state_dict", payload)
            self.assertIn("current_state_dict", payload)
            self.assertIn("best_state_dict", payload)
            self.assertIn("optimizer_state_dict", payload)
            self.assertIn("metadata", payload)
            self.assertEqual(payload["metadata"]["schema_version"], 2)
            self.assertIn("observation_normalizer_state", payload)

            resumed = TrainerSession(
                config=payload["config"],
                device="cpu",
                initial_history=payload["history"],
                initial_state_dict=payload["current_state_dict"],
                initial_best_state_dict=payload["best_state_dict"],
                initial_best_metrics=payload["best_metrics"],
                initial_optimizer_state_dict=payload["optimizer_state_dict"],
                initial_observation_normalizer_state=payload[
                    "observation_normalizer_state"
                ],
                initial_best_observation_normalizer_state=payload[
                    "best_observation_normalizer_state"
                ],
            ).train(generations=1)
            self.assertEqual(resumed.status, "completed")
            self.assertEqual(len(resumed.history), 2)
            self.assertEqual(resumed.history[-1].generation_index, 1)

    def test_validation_catches_cross_field_errors(self) -> None:
        config = AppConfig()
        config.physics.spawn_y_min = 120.0
        config.physics.spawn_y_max = 80.0
        config.physics.pad_width = config.physics.world_width + 1.0
        validation = validate_app_config(config)
        self.assertFalse(validation.is_valid)
        self.assertTrue(
            any("Spawn y min" in error for error in validation.errors)
        )
        self.assertTrue(
            any("Pad width" in error for error in validation.errors)
        )

    def test_poll_training_queue_survives_bridge_reset(self) -> None:
        app = object.__new__(MainApplication)
        app.closed = False
        scheduled: list[tuple[int, object]] = []
        app.root = SimpleNamespace(
            after=lambda delay, callback: scheduled.append((delay, callback))
        )
        app.training_bridge = SimpleNamespace(queue=queue.Queue())
        app.training_bridge.queue.put(("finished", "paused"))

        handled: list[str] = []
        app._handle_generation_report = lambda payload: handled.append("generation")

        def mark_finished(payload: str) -> None:
            handled.append(payload)
            app.training_bridge = None

        app._training_finished = mark_finished
        app._training_failed = lambda payload: handled.append(f"failed:{payload}")

        MainApplication._poll_training_queue(app)

        self.assertEqual(handled, ["paused"])
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][0], 100)


if __name__ == "__main__":
    unittest.main()
