# Rocket Landing Lab

Rocket Landing Lab is a Python 3.12 desktop application and training tool for a self-implemented rocket landing game powered by Deep Reinforcement Learning with PPO in PyTorch.

The project combines:

- a custom 2D rocket landing environment
- a PyTorch actor-critic PPO trainer
- a desktop GUI for tuning physics, rewards, and network architecture
- live evaluation playback of the active policy
- save/load support for training sessions and resumable checkpoints

For the detailed tracked scope of the project, see [project_requirements.md](./project_requirements.md).

**You can create your own launcher with a ml trainer by giving a Coding LLM the [project_requirements.md](./project_requirements.md) and asking it to implement it.**

## Highlights

- Launch the full GUI from `run.py`
- Train from the GUI or from the command line
- Resume training from a saved session checkpoint
- Change physics and reward settings, then continue training with the loaded brain
- View the rocket simulation, training graph, and neural network side by side
- Track both the current brain and the best brain so far
- Save the session state, including model weights, settings, history, and normalizer state

## Requirements

- Python 3.12
- `pip`
- Packages from `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running The App

Launch the GUI:

```bash
python run.py
```

Explicit GUI mode:

```bash
python run.py gui
```

Load a saved session directly into the GUI:

```bash
python run.py gui --load checkpoint.pt
```

## Command Line Training

Run a headless training session:

```bash
python run.py headless-train --generations 5 --games 8
```

Resume from a saved checkpoint:

```bash
python run.py headless-train --load checkpoint.pt --generations 3 --games 8
```

Run the built-in smoke test:

```bash
python run.py smoke-test
```

## GUI Overview

The application is organized into three main areas:

- Left panel: scrollable controls for session actions, PPO settings, network layers, physics, landing criteria, and rewards
- Center panel: the rocket landing view on top and the training graph on the bottom
- Right panel: a resizable neural-network visualization showing nodes and weighted connections

### Left Panel Controls

The left panel includes:

- `Start / Resume Training`
- `Pause After Generation`
- `Stop`
- `Apply Physics`
- brain source toggle for current vs best brain
- session save/load controls
- editable hidden layers and activation functions
- PPO hyperparameters
- physics configuration
- reward and penalty configuration

### Training Behavior

- While training is running, the live game rendering pauses to keep training faster.
- The graph and network view update after each completed generation.
- `Pause` waits for the current generation to finish before stopping.
- `Stop` finishes active episodes and then stops without starting new ones.
- After training is paused or completed, the GUI continuously evaluates the selected brain in the game view.

### Current Brain vs Best Brain

The application tracks:

- the current brain being trained
- the best brain so far, chosen by landing rate first and mean score second

You can switch the evaluation and network view between those two policies from the left panel.

## Save, Load, And Resume

Session checkpoints are designed to support real resume workflows. A saved session includes:

- current brain weights
- best brain weights
- optimizer state when available
- configuration from the left panel
- training history
- best metrics
- observation normalization state
- checkpoint metadata

When a checkpoint is loaded, training can continue from the saved current brain. If you adjust physics or reward values after loading, resumed training continues with the loaded brain under the updated environment settings, as long as the network architecture still matches.

## Project Structure

- `run.py`: main entry point and CLI dispatcher
- `requirements.txt`: Python dependencies
- `project_requirements.md`: detailed project scope and validation requirements
- `rocket_lander/environment.py`: custom rocket landing physics and observations
- `rocket_lander/ppo.py`: actor-critic neural network and PPO-related model code
- `rocket_lander/training.py`: PPO training loop, checkpointing, resume support, and smoke test helpers
- `rocket_lander/validation.py`: config validation rules and cross-field checks
- `rocket_lander/ui_app.py`: main GUI application and training coordination
- `rocket_lander/ui_controls.py`: scrollable configuration controls
- `rocket_lander/ui_views.py`: game, graph, and neural-network visualization canvases
- `rocket_lander/ui_common.py`: shared UI helpers
- `rocket_lander/tk_gui.py`: GUI launcher wrapper
- `tests/test_smoke.py`: smoke and regression tests

## Testing

Recommended validation commands:

```bash
python -m unittest discover -s tests
python run.py smoke-test
python run.py headless-train --generations 1 --games 4
python run.py headless-train --load checkpoint.pt --generations 1 --games 2
```

## Notes

- The environment is self-implemented and does not rely on Gym or a third-party landing simulator.
- PPO uses batched multi-environment rollout collection for faster data gathering.
- Observation normalization is shared between training and evaluation so resumed policies behave consistently.

For exact tracked requirements, acceptance details, and validation expectations, use [project_requirements.md](./project_requirements.md) as the source of truth.
