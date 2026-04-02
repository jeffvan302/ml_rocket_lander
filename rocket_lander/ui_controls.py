from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from typing import Any

from rocket_lander.config import ACTIVATION_OPTIONS, AppConfig, LayerConfig
from rocket_lander.validation import ValidationResult, validate_app_config


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    minimum: float
    maximum: float
    increment: float
    numeric_type: type
    width: int = 12

    @property
    def is_integer(self) -> bool:
        return self.numeric_type is int

    def make_var(self) -> tk.Variable:
        return tk.IntVar() if self.is_integer else tk.DoubleVar()


PPO_FIELD_SPECS = [
    FieldSpec("target_generations", "Target generations", 1, 50000, 1, int),
    FieldSpec("games_per_generation", "Games per generation", 1, 5000, 1, int),
    FieldSpec("learning_rate", "Learning rate", 1e-6, 1.0, 0.00005, float),
    FieldSpec("gamma", "Gamma", 0.5, 0.999999, 0.001, float),
    FieldSpec("gae_lambda", "GAE lambda", 0.5, 0.999999, 0.001, float),
    FieldSpec("clip_range", "Clip range", 0.01, 1.0, 0.01, float),
    FieldSpec("entropy_coef", "Entropy coef", 0.0, 1.0, 0.001, float),
    FieldSpec("value_coef", "Value coef", 0.0, 10.0, 0.05, float),
    FieldSpec("ppo_epochs", "PPO epochs", 1, 128, 1, int),
    FieldSpec("minibatch_size", "Minibatch size", 8, 100000, 8, int),
    FieldSpec("max_grad_norm", "Max grad norm", 0.01, 10.0, 0.05, float),
    FieldSpec("init_std", "Action std", 0.01, 2.0, 0.01, float),
    FieldSpec("seed", "Seed", 0, 999999, 1, int),
]

PHYSICS_FIELD_SPECS = [
    FieldSpec("world_width", "World width", 20.0, 500.0, 1.0, float, 10),
    FieldSpec("world_height", "World height", 20.0, 500.0, 1.0, float, 10),
    FieldSpec("dt", "Physics dt", 0.01, 1.0, 0.01, float, 10),
    FieldSpec("gravity", "Gravity", 0.1, 50.0, 0.1, float, 10),
    FieldSpec("main_thrust", "Main thrust", 0.1, 50.0, 0.1, float, 10),
    FieldSpec("drag_coefficient", "Drag", 0.0, 2.0, 0.01, float, 10),
    FieldSpec("wind_strength", "Wind strength", 0.0, 10.0, 0.1, float, 10),
    FieldSpec("fuel_capacity", "Fuel capacity", 1.0, 500.0, 1.0, float, 10),
    FieldSpec("fuel_burn_rate", "Fuel burn rate", 0.01, 10.0, 0.01, float, 10),
    FieldSpec("angular_acceleration", "Angular accel", 0.1, 20.0, 0.1, float, 10),
    FieldSpec("angular_damping", "Angular damping", 0.1, 1.0, 0.01, float, 10),
    FieldSpec("rocket_height", "Rocket height", 1.0, 20.0, 0.5, float, 10),
    FieldSpec("rocket_width", "Rocket width", 0.5, 20.0, 0.5, float, 10),
    FieldSpec("pad_width", "Pad width", 1.0, 40.0, 0.5, float, 10),
    FieldSpec("landing_angle_deg", "Landing angle", 1.0, 90.0, 1.0, float, 10),
    FieldSpec("landing_max_vx", "Landing max vx", 0.1, 20.0, 0.1, float, 10),
    FieldSpec("landing_max_vy", "Landing max vy", 0.1, 20.0, 0.1, float, 10),
    FieldSpec("max_steps", "Max steps", 25, 50000, 1, int, 10),
    FieldSpec("spawn_x_extent", "Spawn x extent", 0.0, 80.0, 0.5, float, 10),
    FieldSpec("spawn_y_min", "Spawn y min", 1.0, 300.0, 1.0, float, 10),
    FieldSpec("spawn_y_max", "Spawn y max", 1.0, 300.0, 1.0, float, 10),
    FieldSpec("spawn_angle_deg", "Spawn angle", 0.0, 180.0, 1.0, float, 10),
    FieldSpec("spawn_velocity_x", "Spawn vx", 0.0, 20.0, 0.1, float, 10),
    FieldSpec("spawn_velocity_y", "Spawn vy", 0.0, 20.0, 0.1, float, 10),
]

REWARD_FIELD_SPECS = [
    FieldSpec("landing_bonus", "Landing bonus", -5000.0, 5000.0, 1.0, float, 10),
    FieldSpec("close_to_pad_bonus", "Close to pad bonus", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("progress_scale", "Progress scale", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("alive_bonus", "Alive bonus", -5000.0, 5000.0, 0.01, float, 10),
    FieldSpec("center_bonus", "Center bonus", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("fuel_bonus", "Fuel bonus", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("upright_bonus", "Facing-up bonus", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("delta_x_penalty", "Delta x penalty", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("delta_y_penalty", "Delta y penalty", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("step_delta_x_penalty", "Step * delta x penalty", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("step_delta_y_penalty", "Step * delta y penalty", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("crash_penalty", "Crash penalty", -5000.0, 5000.0, 1.0, float, 10),
    FieldSpec("offscreen_penalty", "Offscreen penalty", -5000.0, 5000.0, 1.0, float, 10),
    FieldSpec("timeout_penalty", "Timeout penalty", -5000.0, 5000.0, 1.0, float, 10),
    FieldSpec("turn_penalty", "Turn penalty", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("throttle_penalty", "Throttle penalty", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("step_penalty", "Step penalty", -5000.0, 5000.0, 0.01, float, 10),
    FieldSpec("velocity_penalty", "Velocity penalty", -5000.0, 5000.0, 0.1, float, 10),
    FieldSpec("spin_penalty", "Spin penalty", -5000.0, 5000.0, 0.1, float, 10),
]

ADAPTIVE_TWO_COLUMN_MIN_WIDTH = 320
COMPACT_PANEL_WIDTH = 320


def adaptive_grid_columns_for_width(width: int) -> int:
    return 2 if width >= ADAPTIVE_TWO_COLUMN_MIN_WIDTH else 1


def compact_panel_mode_for_width(width: int) -> bool:
    return width < COMPACT_PANEL_WIDTH


class LayerEditor(ttk.Frame):
    def __init__(self, master, layers: list[LayerConfig]) -> None:
        super().__init__(master, style="PanelInner.TFrame")
        self.rows: list[dict[str, Any]] = []
        self.compact_layout = False
        self.rows_frame = ttk.Frame(self, style="PanelInner.TFrame")
        self.rows_frame.pack(fill="x", expand=True)
        ttk.Button(
            self,
            text="Add Hidden Layer",
            style="AccentGlow.TButton",
            command=self.add_row,
        ).pack(
            anchor="w",
            pady=(8, 0),
        )
        self.set_layers(layers)

    def add_row(self, units: int = 8, activation: str = "relu") -> None:
        row_frame = ttk.Frame(self.rows_frame, style="PanelInner.TFrame")
        row_frame.pack(fill="x", pady=3)
        label = ttk.Label(
            row_frame,
            text=f"Layer {len(self.rows) + 1}",
            width=9,
            style="PanelLabel.TLabel",
        )
        label.pack(side="left")
        units_var = tk.IntVar(value=units)
        activation_var = tk.StringVar(value=activation)
        units_widget = ttk.Spinbox(
            row_frame,
            from_=1,
            to=256,
            textvariable=units_var,
            width=8,
            style="Neon.TSpinbox",
        )
        activation_widget = ttk.Combobox(
            row_frame,
            values=ACTIVATION_OPTIONS,
            textvariable=activation_var,
            state="readonly",
            width=12,
            style="Neon.TCombobox",
        )
        remove_button = ttk.Button(
            row_frame,
            text="Remove",
            style="DangerGlow.TButton",
            command=lambda: self.remove_row(row_frame),
        )
        self.rows.append(
            {
                "frame": row_frame,
                "label": label,
                "units": units_var,
                "activation": activation_var,
                "units_widget": units_widget,
                "activation_widget": activation_widget,
                "remove_button": remove_button,
            }
        )
        self._layout_row(self.rows[-1])
        self._sync_labels()

    def remove_row(self, row_frame) -> None:
        if len(self.rows) <= 1:
            return
        for row in list(self.rows):
            if row["frame"] is row_frame:
                self.rows.remove(row)
                row_frame.destroy()
                break
        self._sync_labels()

    def _sync_labels(self) -> None:
        for index, row in enumerate(self.rows):
            row["label"].configure(text=f"Layer {index + 1}")

    def _layout_row(self, row: dict[str, Any]) -> None:
        row["label"].pack_forget()
        row["units_widget"].pack_forget()
        row["activation_widget"].pack_forget()
        row["remove_button"].pack_forget()
        if self.compact_layout:
            row["label"].pack(anchor="w")
            row["units_widget"].pack(fill="x", pady=(4, 0))
            row["activation_widget"].pack(fill="x", pady=(4, 0))
            row["remove_button"].pack(anchor="e", pady=(6, 0))
        else:
            row["label"].pack(side="left")
            row["units_widget"].pack(side="left", padx=4)
            row["activation_widget"].pack(side="left", padx=4)
            row["remove_button"].pack(side="left", padx=4)

    def set_compact(self, compact: bool) -> None:
        if self.compact_layout == compact:
            return
        self.compact_layout = compact
        for row in self.rows:
            self._layout_row(row)

    def get_layers(self) -> list[LayerConfig]:
        return [
            LayerConfig(
                units=max(1, int(row["units"].get())),
                activation=row["activation"].get(),
            )
            for row in self.rows
        ]

    def set_layers(self, layers: list[LayerConfig]) -> None:
        for row in self.rows:
            row["frame"].destroy()
        self.rows.clear()
        for layer in layers or [LayerConfig(8, "relu")]:
            self.add_row(layer.units, layer.activation)


class ControlPanel(ttk.Frame):
    def __init__(self, master, config: AppConfig) -> None:
        super().__init__(master, padding=(12, 12), style="Panel.TFrame")
        self.ppo_vars: dict[str, tk.Variable] = {}
        self.physics_vars: dict[str, tk.Variable] = {}
        self.reward_vars: dict[str, tk.Variable] = {}
        self.field_widgets: dict[str, ttk.Spinbox] = {}
        self.physics_item_frames: list[ttk.Frame] = []
        self.reward_item_frames: list[ttk.Frame] = []
        self.physics_grid: ttk.Frame | None = None
        self.reward_grid: ttk.Frame | None = None
        self._layout_after_id: str | None = None
        self._compact_mode = False
        self.brain_source_var = tk.StringVar(value="best")
        self.gravity_multi_var = tk.BooleanVar(value=False)
        self.gravity_values_var = tk.StringVar(value="")
        self.gravity_mode_hint_var = tk.StringVar(value="Single gravity active")
        self.telemetry_var = tk.StringVar(
            value=(
                "Telemetry\n"
                "Fuel n/a | Speed n/a\n"
                "Angle n/a | Gravity n/a\n"
                "Steps n/a | Event n/a"
            )
        )
        self.session_status_var = tk.StringVar(value="Ready to evaluate and train.")
        self.evaluation_status_var = tk.StringVar(
            value=(
                "Last eval: awaiting result\n"
                "The active brain will keep playing while training is paused."
            )
        )
        self.evaluation_totals_var = tk.StringVar(
            value="Landings 0 | Crashes 0 | Offscreen 0 | Timeouts 0"
        )
        self.best_metrics_var = tk.StringVar(
            value="Best landing rate: n/a\nBest mean score: n/a"
        )
        self.current_metrics_var = tk.StringVar(
            value="Latest generation: n/a\nPolicy loss / value loss / entropy: n/a"
        )

        self.title_label = ttk.Label(self, text="Rocket Landing Lab", style="Title.TLabel")
        self.title_label.pack(
            anchor="w"
        )
        self.subtitle_label = ttk.Label(
            self,
            text="Tune the lander, shape the rewards, and train a PPO brain.",
            style="HeroMuted.TLabel",
            wraplength=300,
            justify="left",
        )
        self.subtitle_label.pack(anchor="w", pady=(2, 12))

        self._build_monitor_section().pack(fill="x", pady=(0, 6))
        self._build_session_section().pack(fill="x", pady=6)
        self._build_brain_section().pack(fill="x", pady=6)
        self._build_training_section().pack(fill="x", pady=6)
        self._build_network_section(config).pack(fill="x", pady=6)
        self._build_physics_section().pack(fill="x", pady=6)
        self._build_rewards_section().pack(fill="x", pady=6)
        self.set_config(config)
        self.bind("<Configure>", lambda _event: self._schedule_responsive_layout())
        self._schedule_responsive_layout()

    def _section(self, title: str) -> ttk.LabelFrame:
        return ttk.LabelFrame(
            self,
            text=title,
            padding=(10, 10),
            style="Panel.TLabelframe",
        )

    def _build_session_section(self) -> ttk.LabelFrame:
        box = self._section("Session")
        self.start_button = ttk.Button(
            box,
            text="Start / Resume Training",
            style="PrimaryGlow.TButton",
        )
        self.pause_button = ttk.Button(
            box,
            text="Pause After Generation",
            style="SecondaryGlow.TButton",
        )
        self.stop_button = ttk.Button(
            box,
            text="Stop",
            style="DangerGlow.TButton",
        )
        self.start_button.pack(fill="x", pady=3)
        self.pause_button.pack(fill="x", pady=3)
        self.stop_button.pack(fill="x", pady=3)
        self.session_status_label = ttk.Label(
            box,
            textvariable=self.session_status_var,
            style="PanelMuted.TLabel",
            wraplength=290,
            justify="left",
        )
        self.session_status_label.pack(fill="x", pady=(8, 0))
        return box

    def _build_monitor_section(self) -> ttk.LabelFrame:
        box = self._section("Live Monitor")
        self.telemetry_label = ttk.Label(
            box,
            textvariable=self.telemetry_var,
            style="MetricCard.TLabel",
            justify="left",
            anchor="w",
        )
        self.telemetry_label.pack(fill="x", pady=(0, 8))
        self.evaluation_label = ttk.Label(
            box,
            textvariable=self.evaluation_status_var,
            style="MetricCard.TLabel",
            wraplength=290,
            justify="left",
            anchor="w",
        )
        self.evaluation_label.pack(fill="x", pady=(0, 6))
        self.evaluation_totals_label = ttk.Label(
            box,
            textvariable=self.evaluation_totals_var,
            style="PanelMuted.TLabel",
            wraplength=290,
            justify="left",
        )
        self.evaluation_totals_label.pack(fill="x")
        return box

    def _build_brain_section(self) -> ttk.LabelFrame:
        box = self._section("Brains")
        self.radio_row = ttk.Frame(box, style="PanelInner.TFrame")
        self.radio_row.pack(fill="x", pady=(0, 8))
        self.current_brain_radio = ttk.Radiobutton(
            self.radio_row,
            text="Current",
            variable=self.brain_source_var,
            value="current",
            style="Panel.TRadiobutton",
        )
        self.best_brain_radio = ttk.Radiobutton(
            self.radio_row,
            text="Best so far",
            variable=self.brain_source_var,
            value="best",
            style="Panel.TRadiobutton",
        )
        self._layout_brain_source_controls(compact=False)

        self.button_row = ttk.Frame(box, style="PanelInner.TFrame")
        self.button_row.pack(fill="x")
        self.save_button = ttk.Button(
            self.button_row,
            text="Save Session",
            style="AccentGlow.TButton",
        )
        self.load_button = ttk.Button(
            self.button_row,
            text="Load Session",
            style="PrimaryGlow.TButton",
        )
        self._layout_brain_action_buttons(compact=False)

        self.best_metrics_label = ttk.Label(
            box,
            textvariable=self.best_metrics_var,
            style="MetricCard.TLabel",
            justify="left",
            anchor="w",
        )
        self.best_metrics_label.pack(fill="x", pady=(10, 6))
        self.current_metrics_label = ttk.Label(
            box,
            textvariable=self.current_metrics_var,
            style="MetricCard.TLabel",
            justify="left",
            anchor="w",
        )
        self.current_metrics_label.pack(fill="x")
        return box

    def _build_training_section(self) -> ttk.LabelFrame:
        box = self._section("PPO Trainer")
        grid = ttk.Frame(box, style="PanelInner.TFrame")
        grid.pack(fill="x")
        for row, spec in enumerate(PPO_FIELD_SPECS):
            var = spec.make_var()
            self.ppo_vars[spec.key] = var
            ttk.Label(grid, text=spec.label, style="PanelLabel.TLabel").grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=3,
            )
            widget = self._create_spinbox(grid, spec, var)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            self.field_widgets[spec.key] = widget
        grid.columnconfigure(1, weight=1)
        return box

    def _build_network_section(self, config: AppConfig) -> ttk.LabelFrame:
        box = self._section("Network")
        ttk.Label(box, text="Hidden layers", style="PanelLabel.TLabel").pack(anchor="w")
        self.layer_editor = LayerEditor(box, config.network.hidden_layers)
        self.layer_editor.pack(fill="x", pady=(4, 8))
        self.output_row = ttk.Frame(box, style="PanelInner.TFrame")
        self.output_row.pack(fill="x")
        self.output_activation_label = ttk.Label(
            self.output_row,
            text="Output activation",
            style="PanelLabel.TLabel",
        )
        self.output_activation_var = tk.StringVar(
            value=config.network.output_activation
        )
        self.output_activation_combo = ttk.Combobox(
            self.output_row,
            values=ACTIVATION_OPTIONS,
            textvariable=self.output_activation_var,
            state="readonly",
            width=12,
            style="Neon.TCombobox",
        )
        self._layout_output_row(compact=False)
        self.network_help_label = ttk.Label(
            box,
            text=(
                "The actor outputs throttle and gimbal actions. "
                "The critic value stream trains alongside it."
            ),
            style="PanelMuted.TLabel",
            wraplength=290,
            justify="left",
        )
        self.network_help_label.pack(fill="x", pady=(8, 0))
        return box

    def _build_physics_section(self) -> ttk.LabelFrame:
        box = self._section("Physics, Spawn, and Landing")
        self.physics_grid = ttk.Frame(box, style="PanelInner.TFrame")
        self.physics_grid.pack(fill="x")
        self.physics_item_frames.clear()
        for spec in PHYSICS_FIELD_SPECS:
            var = spec.make_var()
            self.physics_vars[spec.key] = var
            item, widget = self._create_field_item(
                self.physics_grid,
                spec,
                var,
                with_gravity_toggle=(spec.key == "gravity"),
            )
            self.physics_item_frames.append(item)
            self.field_widgets[spec.key] = widget
        self.gravity_multi_frame = ttk.Frame(box, style="PanelInner.TFrame")
        ttk.Label(
            self.gravity_multi_frame,
            text="Gravity list",
            style="PanelLabel.TLabel",
        ).pack(anchor="w")
        ttk.Entry(
            self.gravity_multi_frame,
            textvariable=self.gravity_values_var,
            style="Neon.TEntry",
        ).pack(fill="x", pady=(4, 2))
        self.gravity_list_help_label = ttk.Label(
            self.gravity_multi_frame,
            text="Comma-separated values used per episode, for example: 6.8, 8, 9.5",
            style="PanelMuted.TLabel",
            wraplength=290,
            justify="left",
        )
        self.gravity_list_help_label.pack(fill="x")
        self.gravity_mode_hint_label = ttk.Label(
            box,
            textvariable=self.gravity_mode_hint_var,
            style="PanelMuted.TLabel",
            wraplength=290,
            justify="left",
        )
        self.gravity_mode_hint_label.pack(fill="x", pady=(6, 2))
        self.apply_physics_button = ttk.Button(
            box,
            text="Apply Physics",
            style="AccentGlow.TButton",
        )
        self.apply_physics_button.pack(fill="x", pady=(10, 4))
        self.physics_help_label = ttk.Label(
            box,
            text=(
                "Applying physics updates the live evaluation world immediately. "
                "A running generation keeps its current settings for consistency."
            ),
            style="PanelMuted.TLabel",
            wraplength=290,
            justify="left",
        )
        self.physics_help_label.pack(fill="x")
        self._update_gravity_mode_ui()
        return box

    def _toggle_gravity_mode(self) -> None:
        self.gravity_multi_var.set(not self.gravity_multi_var.get())
        self._update_gravity_mode_ui()

    def _update_gravity_mode_ui(self) -> None:
        gravity_widget = self.field_widgets.get("gravity")
        if self.gravity_multi_var.get():
            self.gravity_mode_hint_var.set(
                "Gravity pool active. Each episode samples one value from the list."
            )
            if gravity_widget is not None:
                gravity_widget.state(["disabled"])
            self.gravity_multi_frame.pack(fill="x", pady=(2, 8), before=self.apply_physics_button)
        else:
            self.gravity_mode_hint_var.set(
                "Single gravity active. Toggle the gear to use a gravity pool."
            )
            if gravity_widget is not None:
                gravity_widget.state(["!disabled"])
            self.gravity_multi_frame.pack_forget()

    def _build_rewards_section(self) -> ttk.LabelFrame:
        box = self._section("Rewards and Penalties")
        self.reward_grid = ttk.Frame(box, style="PanelInner.TFrame")
        self.reward_grid.pack(fill="x")
        self.reward_item_frames.clear()
        for spec in REWARD_FIELD_SPECS:
            var = spec.make_var()
            self.reward_vars[spec.key] = var
            item, widget = self._create_field_item(
                self.reward_grid,
                spec,
                var,
            )
            self.reward_item_frames.append(item)
            self.field_widgets[spec.key] = widget
        return box

    def _create_field_item(
        self,
        master,
        spec: FieldSpec,
        variable: tk.Variable,
        with_gravity_toggle: bool = False,
    ) -> tuple[ttk.Frame, ttk.Spinbox]:
        item = ttk.Frame(master, style="PanelInner.TFrame")
        header = ttk.Frame(item, style="PanelInner.TFrame")
        header.pack(fill="x")
        label = ttk.Label(
            header,
            text=spec.label,
            style="PanelLabel.TLabel",
            justify="left",
        )
        label.pack(side="left", fill="x", expand=True)
        if with_gravity_toggle:
            ttk.Button(
                header,
                text="⚙",
                width=3,
                style="SecondaryGlow.TButton",
                command=self._toggle_gravity_mode,
            ).pack(side="right", padx=(6, 0))

        widget = self._create_spinbox(item, spec, variable)
        widget.pack(fill="x", pady=(4, 0))
        item.bind(
            "<Configure>",
            lambda event, target=label, reserve=(44 if with_gravity_toggle else 12): target.configure(
                wraplength=max(event.width - reserve, 110)
            ),
        )
        return item, widget

    def _schedule_responsive_layout(self) -> None:
        if self._layout_after_id is not None:
            self.after_cancel(self._layout_after_id)
        self._layout_after_id = self.after(1, self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        self._layout_after_id = None
        panel_width = max(self.winfo_width(), 1)
        self._apply_compact_mode(compact_panel_mode_for_width(panel_width), panel_width)
        self._layout_adaptive_grid(self.physics_grid, self.physics_item_frames)
        self._layout_adaptive_grid(self.reward_grid, self.reward_item_frames)

    def _apply_compact_mode(self, compact: bool, panel_width: int) -> None:
        self._compact_mode = compact
        wraplength = max(panel_width - 64, 150)
        metric_wraplength = max(panel_width - 72, 150)

        self.subtitle_label.configure(wraplength=wraplength)
        self.session_status_label.configure(wraplength=wraplength)
        self.telemetry_label.configure(wraplength=metric_wraplength)
        self.evaluation_label.configure(wraplength=metric_wraplength)
        self.evaluation_totals_label.configure(wraplength=wraplength)
        self.best_metrics_label.configure(wraplength=metric_wraplength)
        self.current_metrics_label.configure(wraplength=metric_wraplength)
        self.network_help_label.configure(wraplength=wraplength)
        self.gravity_list_help_label.configure(wraplength=wraplength)
        self.gravity_mode_hint_label.configure(wraplength=wraplength)
        self.physics_help_label.configure(wraplength=wraplength)

        self._layout_brain_source_controls(compact)
        self._layout_brain_action_buttons(compact)
        self._layout_output_row(compact)
        self.layer_editor.set_compact(compact)

    def _layout_brain_source_controls(self, compact: bool) -> None:
        self.current_brain_radio.pack_forget()
        self.best_brain_radio.pack_forget()
        if compact:
            self.current_brain_radio.pack(anchor="w")
            self.best_brain_radio.pack(anchor="w", pady=(6, 0))
        else:
            self.current_brain_radio.pack(side="left")
            self.best_brain_radio.pack(side="left", padx=(12, 0))

    def _layout_brain_action_buttons(self, compact: bool) -> None:
        self.save_button.pack_forget()
        self.load_button.pack_forget()
        if compact:
            self.save_button.pack(fill="x", pady=(0, 6))
            self.load_button.pack(fill="x")
        else:
            self.save_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
            self.load_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _layout_output_row(self, compact: bool) -> None:
        self.output_activation_label.pack_forget()
        self.output_activation_combo.pack_forget()
        if compact:
            self.output_activation_label.pack(anchor="w")
            self.output_activation_combo.pack(fill="x", pady=(6, 0))
        else:
            self.output_activation_label.pack(side="left")
            self.output_activation_combo.pack(side="right")

    def _layout_adaptive_grid(
        self,
        grid: ttk.Frame | None,
        items: list[ttk.Frame],
    ) -> None:
        if grid is None or not items:
            return
        available_width = max(grid.winfo_width(), 1)
        columns = adaptive_grid_columns_for_width(available_width)
        current_columns = getattr(grid, "_current_columns", None)
        if current_columns == columns:
            return

        for item in items:
            item.grid_forget()
        for column_index in range(2):
            grid.columnconfigure(column_index, weight=0, uniform="")

        for index, item in enumerate(items):
            row = index // columns
            column = index % columns
            padx = (0, 8) if columns == 2 and column == 0 else (0, 0)
            item.grid(
                row=row,
                column=column,
                sticky="ew",
                padx=padx,
                pady=4,
            )

        for column_index in range(columns):
            grid.columnconfigure(
                column_index,
                weight=1,
                uniform=f"adaptive_{id(grid)}",
            )
        setattr(grid, "_current_columns", columns)

    def _create_spinbox(
        self,
        master,
        spec: FieldSpec,
        variable: tk.Variable,
    ) -> ttk.Spinbox:
        kwargs: dict[str, Any] = {
            "from_": spec.minimum,
            "to": spec.maximum,
            "textvariable": variable,
            "width": spec.width,
            "increment": spec.increment,
            "style": "Neon.TSpinbox",
        }
        if not spec.is_integer:
            kwargs["format"] = "%.6f"
        return ttk.Spinbox(master, **kwargs)

    def get_config(self) -> AppConfig:
        config = AppConfig()
        for spec in PPO_FIELD_SPECS:
            setattr(config.ppo, spec.key, self._read_numeric_value(spec, self.ppo_vars[spec.key]))
        for spec in PHYSICS_FIELD_SPECS:
            setattr(
                config.physics,
                spec.key,
                self._read_numeric_value(spec, self.physics_vars[spec.key]),
            )
        config.physics.gravity_multi_mode = bool(self.gravity_multi_var.get())
        config.physics.gravity_values_text = self.gravity_values_var.get().strip()
        for spec in REWARD_FIELD_SPECS:
            setattr(
                config.rewards,
                spec.key,
                self._read_numeric_value(spec, self.reward_vars[spec.key]),
            )
        config.network.hidden_layers = self.layer_editor.get_layers()
        config.network.output_activation = self.output_activation_var.get()
        return config

    def get_config_and_validation(self) -> tuple[AppConfig, ValidationResult]:
        config = self.get_config()
        return config, validate_app_config(config)

    def _read_numeric_value(self, spec: FieldSpec, variable: tk.Variable) -> int | float:
        try:
            raw_value = variable.get()
            if spec.is_integer:
                value = int(raw_value)
            else:
                value = float(raw_value)
        except (tk.TclError, ValueError):
            value = spec.minimum
            variable.set(value)
        if value < spec.minimum:
            value = spec.minimum
            variable.set(value)
        if value > spec.maximum:
            value = spec.maximum
            variable.set(value)
        return int(value) if spec.is_integer else float(value)

    def set_config(self, config: AppConfig) -> None:
        for spec in PPO_FIELD_SPECS:
            self.ppo_vars[spec.key].set(getattr(config.ppo, spec.key))
        for spec in PHYSICS_FIELD_SPECS:
            self.physics_vars[spec.key].set(getattr(config.physics, spec.key))
        self.gravity_multi_var.set(bool(config.physics.gravity_multi_mode))
        self.gravity_values_var.set(config.physics.gravity_values_text)
        for spec in REWARD_FIELD_SPECS:
            self.reward_vars[spec.key].set(getattr(config.rewards, spec.key))
        self.layer_editor.set_layers(config.network.hidden_layers)
        self.output_activation_var.set(config.network.output_activation)
        self._update_gravity_mode_ui()

    def selected_brain_source(self) -> str:
        return self.brain_source_var.get()

    def set_session_status(self, text: str) -> None:
        self.session_status_var.set(text)

    def set_live_telemetry(self, text: str) -> None:
        self.telemetry_var.set(text)

    def set_evaluation_status(
        self,
        text: str,
        totals_text: str,
        detail_text: str = "",
    ) -> None:
        combined = text if not detail_text else f"{text}\n{detail_text}"
        self.evaluation_status_var.set(combined)
        self.evaluation_totals_var.set(totals_text)

    def set_best_metrics(
        self,
        landing_rate: float | None,
        mean_score: float | None,
    ) -> None:
        if landing_rate is None or mean_score is None:
            self.best_metrics_var.set("Best landing rate: n/a\nBest mean score: n/a")
            return
        self.best_metrics_var.set(
            f"Best landing rate: {landing_rate * 100.0:.1f}%\n"
            f"Best mean score: {mean_score:.2f}"
        )

    def set_current_metrics(
        self,
        generation_text: str,
        policy_loss: float | None,
        value_loss: float | None,
        entropy: float | None,
    ) -> None:
        if policy_loss is None or value_loss is None or entropy is None:
            self.current_metrics_var.set(
                f"{generation_text}\nPolicy loss / value loss / entropy: n/a"
            )
            return
        self.current_metrics_var.set(
            f"{generation_text}\n"
            f"Policy loss / value loss / entropy: "
            f"{policy_loss:.4f} / {value_loss:.4f} / {entropy:.4f}"
        )

    def set_training_running(self, running: bool, pause_pending: bool = False) -> None:
        self.start_button.state(["disabled"] if running else ["!disabled"])
        self.save_button.state(["disabled"] if running else ["!disabled"])
        self.load_button.state(["disabled"] if running else ["!disabled"])
        self.pause_button.state(
            ["!disabled"] if running and not pause_pending else ["disabled"]
        )
        self.stop_button.state(["!disabled"] if running else ["disabled"])
        self.pause_button.configure(
            text="Pause Requested" if pause_pending else "Pause After Generation"
        )

    def validation_message(self, validation: ValidationResult) -> str:
        parts: list[str] = []
        if validation.errors:
            parts.append("Validation errors: " + "; ".join(validation.errors[:4]))
        if validation.warnings:
            parts.append("Warnings: " + "; ".join(validation.warnings[:3]))
        return " ".join(parts) if parts else "Configuration looks valid."
