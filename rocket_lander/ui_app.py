from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from rocket_lander.config import AppConfig
from rocket_lander.environment import OBSERVATION_NAMES, RocketLanderEnv
from rocket_lander.ppo import ACTION_NAMES
from rocket_lander.training import (
    GenerationReport,
    ObservationNormalizer,
    TrainerSession,
    TrainingSummary,
    build_policy_from_state,
    clone_normalizer_state,
    clone_state_dict,
    is_better_candidate,
    load_brain_checkpoint,
    normalize_observation_array,
    save_brain_checkpoint,
)
from rocket_lander.ui_common import ScrollableFrame
from rocket_lander.ui_controls import ControlPanel
from rocket_lander.ui_views import GameCanvas, GraphCanvas, NetworkCanvas


class TrainingBridge:
    def __init__(
        self,
        config: AppConfig,
        history: list[GenerationReport],
        current_state_dict: dict | None,
        best_state_dict: dict | None,
        best_metrics: dict[str, float] | None,
        current_optimizer_state_dict: dict[str, Any] | None,
        current_normalizer_state: dict[str, Any] | None,
        best_normalizer_state: dict[str, Any] | None,
    ) -> None:
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._pause_requested = False
        self._stop_requested = False
        self.session: TrainerSession | None = None
        self.thread = threading.Thread(
            target=self._run,
            args=(
                config,
                history,
                current_state_dict,
                best_state_dict,
                best_metrics,
                current_optimizer_state_dict,
                current_normalizer_state,
                best_normalizer_state,
            ),
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _run(
        self,
        config: AppConfig,
        history: list[GenerationReport],
        current_state_dict: dict | None,
        best_state_dict: dict | None,
        best_metrics: dict[str, float] | None,
        current_optimizer_state_dict: dict[str, Any] | None,
        current_normalizer_state: dict[str, Any] | None,
        best_normalizer_state: dict[str, Any] | None,
    ) -> None:
        try:
            self.session = TrainerSession(
                config=config,
                initial_history=history,
                initial_state_dict=current_state_dict,
                initial_best_state_dict=best_state_dict,
                initial_best_metrics=best_metrics,
                initial_optimizer_state_dict=current_optimizer_state_dict,
                initial_observation_normalizer_state=current_normalizer_state,
                initial_best_observation_normalizer_state=best_normalizer_state,
            )
            if self._pause_requested:
                self.session.request_pause()
            if self._stop_requested:
                self.session.request_stop()
            summary = self.session.train(
                on_generation=lambda report: self.queue.put(("generation", report))
            )
            self.queue.put(("finished", summary))
        except Exception as exc:  # pragma: no cover - GUI failure path
            self.queue.put(("failed", str(exc)))

    def request_pause(self) -> None:
        self._pause_requested = True
        if self.session is not None:
            self.session.request_pause()

    def request_stop(self) -> None:
        self._stop_requested = True
        if self.session is not None:
            self.session.request_stop()


def describe_evaluation_outcome(info: dict[str, Any]) -> dict[str, str]:
    event = str(info.get("event", "unknown"))
    speed = float(info.get("speed", 0.0))
    score = float(info.get("score", 0.0))
    gravity = float(info.get("gravity", 0.0))
    detail_suffix = f" | g {gravity:.2f}" if gravity > 0.0 else ""

    if info.get("landed"):
        return {
            "kind": "success",
            "headline": "Last eval: SUCCESS LANDING",
            "detail": (
                f"Event landed | score {score:.1f} | speed {speed:.2f}{detail_suffix}"
            ),
            "counter_key": "landed",
        }
    if info.get("crashed") or event == "crashed":
        return {
            "kind": "failure",
            "headline": "Last eval: CRASH",
            "detail": (
                f"Event crashed | score {score:.1f} | speed {speed:.2f}{detail_suffix}"
            ),
            "counter_key": "crashed",
        }
    if info.get("offscreen") or event == "offscreen":
        return {
            "kind": "failure",
            "headline": "Last eval: OFFSCREEN",
            "detail": (
                f"Event offscreen | score {score:.1f} | speed {speed:.2f}{detail_suffix}"
            ),
            "counter_key": "offscreen",
        }
    if info.get("timeout") or event == "timeout":
        return {
            "kind": "failure",
            "headline": "Last eval: TIMEOUT",
            "detail": (
                f"Event timeout | score {score:.1f} | speed {speed:.2f}{detail_suffix}"
            ),
            "counter_key": "timeout",
        }
    return {
        "kind": "neutral",
        "headline": f"Last eval: {event.upper()}",
        "detail": f"Score {score:.1f} | speed {speed:.2f}{detail_suffix}",
        "counter_key": "other",
    }


class MainApplication:
    def __init__(self, startup_payload: dict[str, Any] | None = None) -> None:
        self.root = tk.Tk()
        self.root.title("Rocket Landing Lab")
        self.root.geometry("1680x980")
        self.root.configure(bg="#050816")
        self._configure_style()

        self.config = AppConfig()
        self.history: list[GenerationReport] = []
        self.best_metrics = {"landing_rate": -1.0, "mean_score": float("-inf")}
        self.current_state_dict: dict | None = None
        self.best_state_dict: dict | None = None
        self.current_optimizer_state_dict: dict[str, Any] = {}
        self.current_normalizer_state: dict[str, Any] = {}
        self.best_normalizer_state: dict[str, Any] = {}
        self.current_policy = None
        self.best_policy = None
        self.eval_env = RocketLanderEnv(
            self.config.physics,
            self.config.rewards,
            seed=self.config.ppo.seed,
        )
        self.eval_observation = None
        self.eval_outcome_counts = {
            "landed": 0,
            "crashed": 0,
            "offscreen": 0,
            "timeout": 0,
        }
        self.training_bridge: TrainingBridge | None = None
        self.pause_pending = False
        self.status_var = tk.StringVar(value="Ready.")
        self.closed = False

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.control_panel.brain_source_var.trace_add(
            "write",
            lambda *_: self._brain_source_changed(),
        )

        if startup_payload:
            self._apply_loaded_payload(startup_payload)
        else:
            self._reset_brains(self.config)
            self._update_metric_labels()

        self._start_evaluation()
        self._poll_training_queue()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        bg_app = "#050816"
        bg_shell = "#070d1b"
        bg_panel = "#0b1430"
        bg_panel_alt = "#101c3d"
        bg_field = "#081121"
        border = "#244c88"
        border_soft = "#162f56"
        text_main = "#dceaff"
        text_muted = "#87a2cf"
        neon_cyan = "#6ef7ff"
        neon_pink = "#ff74f7"
        neon_lime = "#a8ff6b"
        neon_amber = "#ffb85e"
        neon_red = "#ff7c91"

        def configure_glow_button(
            style_name: str,
            foreground: str,
            background: str,
            bordercolor: str,
            active_background: str,
        ) -> None:
            style.configure(
                style_name,
                padding=(12, 10),
                font=("Consolas", 10, "bold"),
                foreground=foreground,
                background=background,
                bordercolor=bordercolor,
                lightcolor=bordercolor,
                darkcolor=background,
                focuscolor=bordercolor,
                relief="solid",
                borderwidth=2,
            )
            style.map(
                style_name,
                foreground=[
                    ("disabled", "#566680"),
                    ("pressed", foreground),
                    ("active", foreground),
                ],
                background=[
                    ("disabled", "#0a1220"),
                    ("pressed", active_background),
                    ("active", active_background),
                ],
                bordercolor=[
                    ("disabled", border_soft),
                    ("pressed", bordercolor),
                    ("active", bordercolor),
                ],
                lightcolor=[
                    ("disabled", border_soft),
                    ("pressed", bordercolor),
                    ("active", bordercolor),
                ],
            )

        style.configure("TFrame", background=bg_shell)
        style.configure("Panel.TFrame", background=bg_app)
        style.configure("PanelInner.TFrame", background=bg_panel)
        style.configure("StatusBar.TFrame", background=bg_app)
        style.configure("TPanedwindow", background=bg_app)
        style.configure("TLabel", background=bg_shell, foreground=text_main)
        style.configure(
            "TLabelframe",
            background=bg_shell,
            bordercolor=border,
            lightcolor=border,
            darkcolor=bg_shell,
        )
        style.configure(
            "TLabelframe.Label",
            background=bg_shell,
            foreground=text_main,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Title.TLabel",
            background=bg_app,
            foreground=neon_cyan,
            font=("Consolas", 18, "bold"),
        )
        style.configure(
            "HeroMuted.TLabel",
            background=bg_app,
            foreground=text_muted,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Panel.TLabelframe",
            background=bg_panel,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border_soft,
            relief="solid",
            borderwidth=2,
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background=bg_panel,
            foreground=neon_pink,
            font=("Consolas", 10, "bold"),
        )
        style.configure(
            "PanelLabel.TLabel",
            background=bg_panel,
            foreground=neon_cyan,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "PanelMuted.TLabel",
            background=bg_panel,
            foreground=text_muted,
            font=("Segoe UI", 9),
        )
        style.configure(
            "MetricCard.TLabel",
            background=bg_panel_alt,
            foreground=text_main,
            padding=10,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Status.TLabel",
            background=bg_app,
            foreground="#7fcfff",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Panel.TRadiobutton",
            background=bg_panel,
            foreground=neon_lime,
            indicatorcolor=neon_lime,
            font=("Consolas", 10, "bold"),
        )
        style.map(
            "Panel.TRadiobutton",
            foreground=[
                ("selected", neon_lime),
                ("active", "#d8ffb3"),
                ("!selected", "#93b9b9"),
            ],
            background=[("active", bg_panel)],
            indicatorcolor=[
                ("selected", neon_lime),
                ("!selected", "#324669"),
            ],
        )
        style.configure(
            "Neon.TSpinbox",
            foreground=text_main,
            fieldbackground=bg_field,
            background=bg_panel_alt,
            arrowcolor=neon_cyan,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border_soft,
            insertcolor=neon_cyan,
            padding=4,
        )
        style.map(
            "Neon.TSpinbox",
            fieldbackground=[("readonly", bg_field), ("disabled", "#09101b")],
            foreground=[("disabled", "#53657c")],
            bordercolor=[("focus", neon_cyan), ("active", border)],
        )
        style.configure(
            "Neon.TCombobox",
            foreground=text_main,
            fieldbackground=bg_field,
            background=bg_panel_alt,
            arrowcolor=neon_pink,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border_soft,
            padding=4,
        )
        style.map(
            "Neon.TCombobox",
            fieldbackground=[("readonly", bg_field), ("disabled", "#09101b")],
            foreground=[("disabled", "#53657c")],
            bordercolor=[("focus", neon_pink), ("active", border)],
            arrowcolor=[("active", "#ffacf7")],
        )
        style.configure(
            "Neon.TEntry",
            foreground=text_main,
            fieldbackground=bg_field,
            background=bg_panel_alt,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border_soft,
            insertcolor=neon_cyan,
            padding=5,
        )
        style.map(
            "Neon.TEntry",
            foreground=[("disabled", "#53657c")],
            fieldbackground=[("disabled", "#09101b")],
            bordercolor=[("focus", neon_cyan), ("active", border)],
        )
        style.configure(
            "Panel.Vertical.TScrollbar",
            troughcolor="#060d1e",
            background=bg_panel_alt,
            bordercolor=border_soft,
            arrowcolor=neon_cyan,
            darkcolor=bg_panel_alt,
            lightcolor=border,
        )
        style.map(
            "Panel.Vertical.TScrollbar",
            background=[("active", "#173460")],
            arrowcolor=[("active", "#b5ffff")],
        )

        configure_glow_button(
            "PrimaryGlow.TButton",
            neon_cyan,
            "#122240",
            "#3cf3ff",
            "#18355f",
        )
        configure_glow_button(
            "SecondaryGlow.TButton",
            neon_amber,
            "#2f1d07",
            "#ffb85e",
            "#4a2b09",
        )
        configure_glow_button(
            "AccentGlow.TButton",
            neon_lime,
            "#112710",
            "#7dff85",
            "#193817",
        )
        configure_glow_button(
            "DangerGlow.TButton",
            neon_red,
            "#32111f",
            "#ff5f8f",
            "#4a1830",
        )

    def _build_layout(self) -> None:
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left_wrapper = ttk.Frame(main, width=360, style="Panel.TFrame")
        left_wrapper.pack_propagate(False)
        self.scrollable = ScrollableFrame(left_wrapper)
        self.scrollable.pack(fill="both", expand=True)
        self.control_panel = ControlPanel(self.scrollable.inner, self.config)
        self.control_panel.pack(fill="x", expand=True)
        self.control_panel.start_button.configure(command=self.start_training)
        self.control_panel.pause_button.configure(command=self.request_pause)
        self.control_panel.stop_button.configure(command=self.request_stop)
        self.control_panel.apply_physics_button.configure(command=self.apply_physics)
        self.control_panel.save_button.configure(command=self.save_best_brain)
        self.control_panel.load_button.configure(command=self.load_brain)
        main.add(left_wrapper, weight=1)

        center_right = ttk.Panedwindow(main, orient=tk.HORIZONTAL)
        main.add(center_right, weight=5)

        center = ttk.Panedwindow(center_right, orient=tk.VERTICAL)
        center_right.add(center, weight=3)
        self.game_canvas = GameCanvas(center)
        self.graph_canvas = GraphCanvas(center)
        center.add(self.game_canvas, weight=3)
        center.add(self.graph_canvas, weight=2)

        self.network_canvas = NetworkCanvas(center_right)
        center_right.add(self.network_canvas, weight=2)

        status_bar = ttk.Frame(self.root, padding=(10, 4), style="StatusBar.TFrame")
        status_bar.pack(fill="x")
        ttk.Label(
            status_bar,
            textvariable=self.status_var,
            style="Status.TLabel",
        ).pack(anchor="w")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _evaluation_totals_text(self) -> str:
        counts = self.eval_outcome_counts
        return (
            f"Landings {counts['landed']} | "
            f"Crashes {counts['crashed']} | "
            f"Offscreen {counts['offscreen']} | "
            f"Timeouts {counts['timeout']}"
        )

    def _reset_evaluation_monitor(self) -> None:
        self.eval_outcome_counts = {
            "landed": 0,
            "crashed": 0,
            "offscreen": 0,
            "timeout": 0,
        }
        headline = "Last eval: awaiting result"
        detail = "The active brain will keep playing while training is paused."
        totals = self._evaluation_totals_text()
        self.control_panel.set_evaluation_status(headline, totals)
        self.game_canvas.set_evaluation_outcome(headline, detail, kind="neutral")

    def _record_evaluation_outcome(self, info: dict[str, Any]) -> None:
        outcome = describe_evaluation_outcome(info)
        counter_key = outcome["counter_key"]
        if counter_key in self.eval_outcome_counts:
            self.eval_outcome_counts[counter_key] += 1
        totals = self._evaluation_totals_text()
        self.control_panel.set_evaluation_status(outcome["headline"], totals)
        self.game_canvas.set_evaluation_outcome(
            outcome["headline"],
            outcome["detail"],
            kind=outcome["kind"],
        )

    def _default_normalizer_state(self) -> dict[str, Any]:
        return clone_normalizer_state(
            ObservationNormalizer(len(OBSERVATION_NAMES)).state_dict()
        )

    def _validation_ok(self, config: AppConfig, action_label: str) -> bool:
        validation = self.control_panel.get_config_and_validation()[1]
        message = self.control_panel.validation_message(validation)
        if validation.errors:
            self.control_panel.set_session_status(message)
            self._set_status(message)
            messagebox.showerror(
                title=f"{action_label} blocked",
                message="\n".join(validation.errors),
                parent=self.root,
            )
            return False
        if validation.warnings:
            self.control_panel.set_session_status(message)
            self._set_status(message)
        return True

    def _apply_loaded_payload(self, payload: dict[str, Any]) -> None:
        self.config = payload["config"]
        self.history = list(payload.get("history", []))
        loaded_metrics = dict(payload.get("best_metrics", self.best_metrics))
        self.best_metrics = {
            "landing_rate": loaded_metrics.get("landing_rate", -1.0),
            "mean_score": loaded_metrics.get("mean_score", float("-inf")),
        }
        loaded_norm = clone_normalizer_state(
            payload.get("observation_normalizer_state", {})
        )
        loaded_best_norm = clone_normalizer_state(
            payload.get("best_observation_normalizer_state", {})
        )
        self.current_state_dict = clone_state_dict(
            payload.get("current_state_dict", payload["state_dict"])
        )
        self.best_state_dict = clone_state_dict(
            payload.get("best_state_dict", payload["state_dict"])
        )
        self.current_optimizer_state_dict = payload.get("optimizer_state_dict", {}) or {}
        self.current_normalizer_state = loaded_norm or self._default_normalizer_state()
        self.best_normalizer_state = (
            loaded_best_norm
            or loaded_norm
            or self._default_normalizer_state()
        )
        self.control_panel.set_config(self.config)
        self.graph_canvas.set_history(self.history)
        self._rebuild_policies()
        self._update_metric_labels()
        self._reset_evaluation_episode()

        metadata = payload.get("metadata", {})
        saved_at = metadata.get("saved_at_utc", "unknown time")
        generation_count = metadata.get("generation_count", len(self.history))
        self.control_panel.set_session_status(
            f"Loaded checkpoint from {payload.get('path', 'disk')} "
            f"saved at {saved_at} with {generation_count} generations."
        )
        self._set_status("Training session loaded.")

    def _reset_brains(self, config: AppConfig) -> None:
        self.config = config
        fresh = build_policy_from_state(self.config, device="cpu")
        self.current_state_dict = clone_state_dict(fresh.state_dict())
        self.best_state_dict = clone_state_dict(fresh.state_dict())
        self.current_optimizer_state_dict = {}
        self.current_normalizer_state = self._default_normalizer_state()
        self.best_normalizer_state = self._default_normalizer_state()
        self.best_metrics = {"landing_rate": -1.0, "mean_score": float("-inf")}
        self._rebuild_policies()
        self._reset_evaluation_episode()

    def _rebuild_policies(self) -> None:
        self.current_policy = build_policy_from_state(
            self.config,
            state_dict=self.current_state_dict,
            device="cpu",
        )
        best_state = self.best_state_dict or self.current_state_dict
        self.best_policy = build_policy_from_state(
            self.config,
            state_dict=best_state,
            device="cpu",
        )
        self._refresh_network_visualization()

    def _selected_policy(self):
        source = self.control_panel.selected_brain_source()
        self.game_canvas.set_brain_source(source)
        if source == "current" or self.best_policy is None:
            return self.current_policy
        return self.best_policy

    def _selected_normalizer_state(self) -> dict[str, Any]:
        source = self.control_panel.selected_brain_source()
        if source == "current":
            return self.current_normalizer_state
        return self.best_normalizer_state or self.current_normalizer_state

    def _refresh_network_visualization(self) -> None:
        policy = self._selected_policy()
        if policy is None:
            self.network_canvas.set_visualization(None, "Active Brain")
            return
        title = (
            "Current Brain"
            if self.control_panel.selected_brain_source() == "current"
            else "Best Brain So Far"
        )
        self.network_canvas.set_visualization(
            policy.export_visualization_data(
                list(OBSERVATION_NAMES),
                list(ACTION_NAMES),
            ),
            title,
        )

    def _derive_best_metrics(self) -> dict[str, float]:
        best = {"landing_rate": -1.0, "mean_score": float("-inf")}
        for report in self.history:
            candidate = {
                "landing_rate": report.landing_rate,
                "mean_score": report.mean_score,
            }
            if (not report.partial_generation) and is_better_candidate(
                candidate,
                best,
            ):
                best = candidate
        return best

    def _update_metric_labels(self) -> None:
        if self.best_metrics["landing_rate"] < 0:
            self.control_panel.set_best_metrics(None, None)
        else:
            self.control_panel.set_best_metrics(
                self.best_metrics["landing_rate"],
                self.best_metrics["mean_score"],
            )
        if not self.history:
            self.control_panel.set_current_metrics(
                "Latest generation: n/a",
                None,
                None,
                None,
            )
            return
        latest = self.history[-1]
        suffix = " (partial)" if latest.partial_generation else ""
        generation_text = (
            f"Latest generation {latest.generation_index + 1}{suffix}: "
            f"landings {latest.landing_rate * 100.0:.1f}% | "
            f"mean {latest.mean_score:.2f}"
        )
        self.control_panel.set_current_metrics(
            generation_text,
            latest.policy_loss,
            latest.value_loss,
            latest.entropy,
        )

    def _reset_evaluation_episode(self) -> None:
        self.eval_env = RocketLanderEnv(
            self.config.physics,
            self.config.rewards,
            seed=self.config.ppo.seed,
        )
        self.eval_observation = self.eval_env.reset(dramatic=True)
        self._reset_evaluation_monitor()
        self.game_canvas.set_snapshot(self.eval_env.snapshot())

    def _evaluation_tick(self) -> None:
        if self.closed:
            return
        if self.training_bridge is not None:
            self.root.after(33, self._evaluation_tick)
            return

        policy = self._selected_policy()
        if policy is not None:
            if self.eval_observation is None:
                self.eval_observation = self.eval_env.reset(dramatic=True)
            normalized_observation = normalize_observation_array(
                self.eval_observation,
                self._selected_normalizer_state(),
            )
            action = policy.act_numpy(
                normalized_observation,
                device="cpu",
                deterministic=True,
            )
            self.eval_observation, _, done, info = self.eval_env.step(action)
            if done:
                self._record_evaluation_outcome(info)
                self.eval_observation = self.eval_env.reset(dramatic=True)
            self.game_canvas.set_snapshot(self.eval_env.snapshot())

        self.root.after(33, self._evaluation_tick)

    def _start_evaluation(self) -> None:
        self.game_canvas.set_training_mode(False)
        self._reset_evaluation_episode()
        self.root.after(33, self._evaluation_tick)

    def _brain_source_changed(self) -> None:
        self._refresh_network_visualization()
        if self.training_bridge is None:
            self._reset_evaluation_episode()

    def apply_physics(self) -> None:
        config, validation = self.control_panel.get_config_and_validation()
        if validation.errors:
            self.control_panel.set_session_status(
                self.control_panel.validation_message(validation)
            )
            self._set_status(self.control_panel.validation_message(validation))
            messagebox.showerror(
                title="Apply Physics blocked",
                message="\n".join(validation.errors),
                parent=self.root,
            )
            return

        self.config.physics = config.physics
        self.config.rewards = config.rewards
        self._reset_evaluation_episode()
        message = (
            "Updated evaluation physics and rewards. "
            "The running generation keeps its current settings."
            if self.training_bridge
            else "Updated evaluation physics and rewards."
        )
        if validation.warnings:
            message = (
                f"{message} Warnings: "
                + "; ".join(validation.warnings[:3])
            )
        self.control_panel.set_session_status(message)
        self._set_status(message)

    def start_training(self) -> None:
        if self.training_bridge is not None:
            return

        new_config, validation = self.control_panel.get_config_and_validation()
        if validation.errors:
            message = self.control_panel.validation_message(validation)
            self.control_panel.set_session_status(message)
            self._set_status(message)
            messagebox.showerror(
                title="Training blocked",
                message="\n".join(validation.errors),
                parent=self.root,
            )
            return

        architecture_changed = self.config.network.to_dict() != new_config.network.to_dict()
        self.config = new_config

        if architecture_changed:
            self.history = []
            self.graph_canvas.set_history(self.history)
            self._reset_brains(self.config)
            self._set_status(
                "Network architecture changed, so current and best brains were reset."
            )
        else:
            self._rebuild_policies()

        self.pause_pending = False
        self.control_panel.set_training_running(True, pause_pending=False)
        status_message = (
            "Training is running. Pause waits for the generation boundary; "
            "Stop finishes active episodes and then stops."
        )
        if validation.warnings:
            status_message += " Warnings: " + "; ".join(validation.warnings[:3])
        self.control_panel.set_session_status(status_message)
        self._set_status(status_message)
        self.game_canvas.set_training_mode(
            True,
            "Training in progress.\nGame rendering is paused for faster learning.",
        )
        self.training_bridge = TrainingBridge(
            config=self.config,
            history=list(self.history),
            current_state_dict=(
                clone_state_dict(self.current_state_dict)
                if self.current_state_dict
                else None
            ),
            best_state_dict=(
                clone_state_dict(self.best_state_dict)
                if self.best_state_dict
                else None
            ),
            best_metrics=self.best_metrics,
            current_optimizer_state_dict=self.current_optimizer_state_dict,
            current_normalizer_state=self.current_normalizer_state,
            best_normalizer_state=self.best_normalizer_state,
        )
        self.training_bridge.start()

    def request_pause(self) -> None:
        if self.training_bridge is None:
            return
        self.pause_pending = True
        self.training_bridge.request_pause()
        self.control_panel.set_training_running(True, pause_pending=True)
        self.control_panel.set_session_status(
            "Pause requested. The trainer will finish the current generation first."
        )
        self._set_status("Pause requested.")

    def request_stop(self) -> None:
        if self.training_bridge is None:
            return
        self.training_bridge.request_stop()
        self.control_panel.set_session_status(
            "Stop requested. The trainer will finish active episodes and stop."
        )
        self._set_status("Stop requested.")

    def _poll_training_queue(self) -> None:
        if self.closed:
            return
        bridge = self.training_bridge
        if bridge is not None:
            while True:
                try:
                    event_name, payload = bridge.queue.get_nowait()
                except queue.Empty:
                    break
                if event_name == "generation":
                    self._handle_generation_report(payload)
                elif event_name == "finished":
                    self._training_finished(payload)
                elif event_name == "failed":
                    self._training_failed(payload)
                if bridge is not self.training_bridge:
                    break
        self.root.after(100, self._poll_training_queue)

    def _handle_generation_report(self, report: GenerationReport) -> None:
        self.history.append(report)
        self.graph_canvas.set_history(self.history)
        self.current_state_dict = clone_state_dict(report.current_state_dict)
        self.best_state_dict = clone_state_dict(report.best_state_dict)
        self.current_normalizer_state = clone_normalizer_state(
            report.current_normalizer_state
        ) or self.current_normalizer_state
        self.best_normalizer_state = clone_normalizer_state(
            report.best_normalizer_state
        ) or self.best_normalizer_state
        self.best_metrics = self._derive_best_metrics()
        self._rebuild_policies()
        self._update_metric_labels()
        partial_note = " (partial generation)" if report.partial_generation else ""
        message = (
            f"Generation {report.generation_index + 1} finished{partial_note}. "
            f"Landing rate {report.landing_rate * 100.0:.1f}% | "
            f"mean score {report.mean_score:.2f}"
        )
        self.control_panel.set_session_status(message)
        self._set_status(message)

    def _training_finished(self, summary: TrainingSummary) -> None:
        self.current_state_dict = clone_state_dict(summary.current_state_dict)
        self.best_state_dict = clone_state_dict(summary.best_state_dict)
        self.current_optimizer_state_dict = summary.current_optimizer_state_dict
        self.current_normalizer_state = clone_normalizer_state(
            summary.current_normalizer_state
        ) or self.current_normalizer_state
        self.best_normalizer_state = clone_normalizer_state(
            summary.best_normalizer_state
        ) or self.best_normalizer_state
        self.best_metrics = dict(summary.best_metrics)
        self.history = list(summary.history)
        self.training_bridge = None
        self.graph_canvas.set_history(self.history)
        self._rebuild_policies()
        self._update_metric_labels()
        self.control_panel.set_training_running(False)
        self.pause_pending = False
        message = {
            "paused": "Training paused after the generation boundary.",
            "stopped": "Training stopped after active episodes completed.",
            "completed": "Training completed.",
        }.get(summary.status, f"Training ended with status: {summary.status}")
        self.control_panel.set_session_status(message)
        self._set_status(message)
        self.game_canvas.set_training_mode(False, message)
        self._reset_evaluation_episode()

    def _training_failed(self, message: str) -> None:
        self.training_bridge = None
        self.pause_pending = False
        self.control_panel.set_training_running(False)
        self.control_panel.set_session_status(f"Training failed: {message}")
        self._set_status(f"Training failed: {message}")
        self.game_canvas.set_training_mode(
            False,
            "Training failed. Evaluation loop restored.",
        )
        self._reset_evaluation_episode()

    def save_best_brain(self) -> None:
        state = self.best_state_dict or self.current_state_dict
        if state is None:
            self._set_status("No brain is available to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Best Brain",
            initialfile="rocket_best_brain.pt",
            defaultextension=".pt",
            filetypes=[("PyTorch Checkpoints", "*.pt *.pth")],
        )
        if not path:
            return
        save_brain_checkpoint(
            path=Path(path),
            config=self.control_panel.get_config(),
            state_dict=state,
            best_metrics=self.best_metrics,
            history=self.history,
            source_label="best",
            current_state_dict=self.current_state_dict,
            best_state_dict=self.best_state_dict,
            optimizer_state_dict=self.current_optimizer_state_dict,
            observation_normalizer_state=self.current_normalizer_state,
            best_observation_normalizer_state=self.best_normalizer_state,
        )
        self.control_panel.set_session_status(
            f"Saved training session to {path}."
        )
        self._set_status("Training session saved.")

    def load_brain(self) -> None:
        if self.training_bridge is not None:
            return
        path = filedialog.askopenfilename(
            title="Load Brain",
            filetypes=[("PyTorch Checkpoints", "*.pt *.pth")],
        )
        if not path:
            return
        payload = load_brain_checkpoint(path)
        self._apply_loaded_payload(payload)

    def _on_close(self) -> None:
        self.closed = True
        if self.training_bridge is not None:
            self.training_bridge.request_stop()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def launch_gui(startup_payload: dict[str, Any] | None = None) -> int:
    return MainApplication(startup_payload=startup_payload).run()
