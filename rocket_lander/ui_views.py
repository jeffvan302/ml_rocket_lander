from __future__ import annotations

import math
import tkinter as tk
from typing import Any

from rocket_lander.ui_common import lerp_color


class GameCanvas(tk.Canvas):
    def __init__(self, master) -> None:
        super().__init__(master, highlightthickness=0, bg="#071c2c")
        self.snapshot: dict[str, Any] | None = None
        self.training_active = False
        self.overlay_text = "Evaluating the active brain."
        self.brain_source = "best"
        self.last_outcome_text = "Last eval: awaiting result"
        self.last_outcome_detail = (
            "The active brain will keep playing while training is paused."
        )
        self.last_outcome_kind = "neutral"
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.redraw()

    def set_training_mode(self, active: bool, overlay_text: str | None = None) -> None:
        self.training_active = active
        if overlay_text is not None:
            self.overlay_text = overlay_text
        self.redraw()

    def set_brain_source(self, source: str) -> None:
        self.brain_source = source
        self.redraw()

    def set_evaluation_outcome(
        self,
        text: str,
        detail: str,
        kind: str = "neutral",
    ) -> None:
        self.last_outcome_text = text
        self.last_outcome_detail = detail
        self.last_outcome_kind = kind
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 10)
        height = max(self.winfo_height(), 10)
        self._gradient_background(width, height)
        self._draw_stars(width, height)
        self._draw_ground(width, height)

        if not self.snapshot or not self.snapshot.get("state"):
            self.create_text(
                width / 2,
                height / 2,
                text="No active episode yet.\nStart training or let evaluation play.",
                fill="#f8fcff",
                font=("Segoe UI", 16, "bold"),
                justify="center",
            )
            return

        physics = self.snapshot["physics"]
        state = self.snapshot["state"]
        info = self.snapshot.get("info", {})
        x_span = max(float(physics["world_width"]) * 0.7, 1.0)
        y_span = max(float(physics["world_height"]) * 1.05, 1.0)
        ground_y = height - 38
        top_margin = 26

        def world_to_screen(x: float, y: float) -> tuple[float, float]:
            sx = width / 2 + (x / x_span) * width * 0.42
            sy = ground_y - (y / y_span) * (ground_y - top_margin)
            return sx, sy

        pad_half = float(physics["pad_width"]) * 0.5
        pad_left, _ = world_to_screen(-pad_half, 0.0)
        pad_right, _ = world_to_screen(pad_half, 0.0)
        self.create_rectangle(
            pad_left,
            ground_y - 12,
            pad_right,
            ground_y - 2,
            fill="#1bdfb8",
            outline="#a8fff3",
            width=2,
        )
        for lamp in range(5):
            lamp_x = pad_left + (lamp + 0.5) * (pad_right - pad_left) / 5
            self.create_oval(
                lamp_x - 2,
                ground_y - 16,
                lamp_x + 2,
                ground_y - 12,
                fill="#fff2a1",
                outline="",
            )

        trail = state.get("trail", [])
        for index in range(len(trail) - 1):
            x1, y1 = world_to_screen(*trail[index])
            x2, y2 = world_to_screen(*trail[index + 1])
            color = lerp_color("#ffe19d", "#ff9042", index / max(len(trail) - 1, 1))
            self.create_line(x1, y1, x2, y2, fill=color, width=2)

        rocket_x, rocket_y = world_to_screen(state["x"], state["y"])
        self._draw_rocket(
            rocket_x,
            rocket_y,
            angle=float(state["angle"]),
            throttle=float(state["throttle"]),
            step_count=int(state["steps"]),
        )
        self._draw_hud(width, state, info)
        self.create_rectangle(
            width - 240,
            12,
            width - 12,
            42,
            fill="#faf6ea",
            outline="",
        )
        self.create_text(
            width - 126,
            27,
            text=(
                f"{'TRAINING' if self.training_active else 'EVALUATION'}"
                f" • {self.brain_source.upper()}"
            ),
            fill="#17303e",
            font=("Segoe UI", 10, "bold"),
        )

        self._draw_outcome_panel(width, height)

        if self.training_active:
            self.create_rectangle(
                0,
                0,
                width,
                height,
                fill="#071c2c",
                stipple="gray50",
                outline="",
            )
            self.create_text(
                width / 2,
                height / 2,
                text=self.overlay_text,
                fill="#f8fcff",
                font=("Segoe UI", 18, "bold"),
                justify="center",
            )

    def _draw_outcome_panel(self, width: int, height: int) -> None:
        palette = {
            "success": ("#0e332e", "#74ffd5", "#ecfffa"),
            "failure": ("#3c1424", "#ff7ba0", "#fff0f5"),
            "neutral": ("#10223f", "#79d4ff", "#e8f7ff"),
        }
        fill, border, text_fill = palette.get(
            self.last_outcome_kind,
            palette["neutral"],
        )
        x0 = 18
        y0 = height - 88
        x1 = max(x0 + 220, min(width * 0.62, 408))
        y1 = height - 18
        self.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=fill,
            outline=border,
            width=2,
        )
        self.create_text(
            x0 + 14,
            y0 + 16,
            text=self.last_outcome_text,
            fill=text_fill,
            font=("Consolas", 10, "bold"),
            anchor="w",
        )
        self.create_text(
            x0 + 14,
            y0 + 40,
            text=self.last_outcome_detail,
            fill="#d8ebff",
            font=("Segoe UI", 9),
            anchor="w",
        )

    def _gradient_background(self, width: int, height: int) -> None:
        steps = 30
        for index in range(steps):
            t = index / max(steps - 1, 1)
            if t < 0.33:
                color = lerp_color("#071c2c", "#0c2f47", t / 0.33)
            elif t < 0.66:
                color = lerp_color("#0c2f47", "#16526d", (t - 0.33) / 0.33)
            else:
                color = lerp_color("#16526d", "#e79a4d", (t - 0.66) / 0.34)
            y0 = height * index / steps
            y1 = height * (index + 1) / steps
            self.create_rectangle(0, y0, width, y1, fill=color, outline=color)

    def _draw_stars(self, width: int, height: int) -> None:
        for index in range(28):
            x = 18 + (index * 53) % max(width - 36, 1)
            y = 14 + (index * 37) % max(int(height * 0.55), 1)
            radius = 1 + index % 3
            self.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#fff4d6",
                outline="",
            )

    def _draw_ground(self, width: int, height: int) -> None:
        self.create_polygon(
            0,
            height - 40,
            width * 0.12,
            height * 0.72,
            width * 0.24,
            height * 0.58,
            width * 0.35,
            height * 0.76,
            width * 0.52,
            height * 0.50,
            width * 0.68,
            height * 0.74,
            width * 0.82,
            height * 0.54,
            width,
            height * 0.76,
            width,
            height,
            0,
            height,
            fill="#091a22",
            outline="",
        )
        self.create_rectangle(0, height - 38, width, height, fill="#254d31", outline="")

    def _draw_rocket(
        self,
        cx: float,
        cy: float,
        angle: float,
        throttle: float,
        step_count: int,
    ) -> None:
        def rotate(points: list[tuple[float, float]]) -> list[float]:
            rotated: list[float] = []
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            for x, y in points:
                rx = x * cos_a - y * sin_a
                ry = x * sin_a + y * cos_a
                rotated.extend([cx + rx, cy + ry])
            return rotated

        body = [(-14, 22), (-14, -44), (0, -74), (14, -44), (14, 22), (0, 38)]
        fin_left = [(-14, 2), (-26, 26), (-14, 24)]
        fin_right = [(14, 2), (26, 26), (14, 24)]
        nose = [(-10, -48), (0, -74), (10, -48)]
        window = rotate([(-7, -10), (7, -10), (7, 4), (-7, 4)])
        nozzle = rotate([(-7, 30), (7, 30), (7, 40), (-7, 40)])
        self.create_polygon(rotate(body), fill="#c3d0db", outline="#eef8ff", width=2)
        self.create_polygon(rotate(fin_left), fill="#d15546", outline="")
        self.create_polygon(rotate(fin_right), fill="#d15546", outline="")
        self.create_polygon(rotate(nose), fill="#e86c52", outline="")
        self.create_oval(window[0], window[1], window[4], window[5], fill="#73d8ff", outline="")
        self.create_polygon(nozzle, fill="#27343c", outline="")

        if throttle <= 0.02:
            return

        flicker = 0.78 + 0.22 * math.sin(step_count * 0.45)
        flame_length = 18 + 48 * throttle * flicker
        flame = [(-7.5, 39), (0, 39 + flame_length), (7.5, 39)]
        core = [(-4, 39), (0, 28 + flame_length), (4, 39)]
        self.create_polygon(rotate(flame), fill="#ffa839", outline="")
        self.create_polygon(rotate(core), fill="#fff5ba", outline="")

    def _draw_hud(
        self,
        width: int,
        state: dict[str, Any],
        info: dict[str, Any],
    ) -> None:
        self.create_rectangle(18, 18, 232, 136, fill="#091a26", outline="#b7eee5", width=1)
        self.create_text(
            32,
            32,
            text="Telemetry",
            fill="#f8fcff",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        lines = [
            f"Fuel: {state['fuel']:.1f}",
            f"Velocity: {math.hypot(state['vx'], state['vy']):.2f}",
            f"Angle: {math.degrees(state['angle']):.1f} deg",
            f"Steps: {state['steps']}",
            f"Event: {info.get('event', 'flying')}",
        ]
        for index, line in enumerate(lines):
            self.create_text(
                32,
                56 + index * 16,
                text=line,
                fill="#d6eef7",
                font=("Segoe UI", 9),
                anchor="w",
            )


class GraphCanvas(tk.Canvas):
    def __init__(self, master) -> None:
        super().__init__(master, highlightthickness=0, bg="#08101f")
        self.history: list[Any] = []
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_history(self, history: list[Any]) -> None:
        self.history = list(history)
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 10)
        height = max(self.winfo_height(), 10)
        for index in range(24):
            color = lerp_color("#08101f", "#101c34", index / 23)
            self.create_rectangle(
                0,
                height * index / 24,
                width,
                height * (index + 1) / 24,
                fill=color,
                outline=color,
            )
        self.create_text(
            20,
            20,
            text="Training Progress",
            fill="#6ff7ff",
            font=("Consolas", 12, "bold"),
            anchor="w",
        )

        if not self.history:
            self.create_text(
                width / 2,
                height / 2,
                text=(
                    "Landing rate, generation best score, and mean score "
                    "appear here after training starts."
                ),
                fill="#87a2cf",
                font=("Segoe UI", 11),
                width=width * 0.8,
                justify="center",
            )
            return

        upper = (54, 42, width - 32, height * 0.40)
        lower = (54, height * 0.50, width - 32, height - 36)
        self._draw_plot_box(*upper)
        self._draw_plot_box(*lower)
        self._draw_grid(*upper)
        self._draw_grid(*lower)

        landing_rates = [float(item.landing_rate) for item in self.history]
        best_scores = [float(item.best_score) for item in self.history]
        mean_scores = [float(item.mean_score) for item in self.history]
        landing_rate_color = "#50f6ff"
        best_score_color = "#ffab4f"
        mean_score_color = "#ff74f7"
        self._draw_series(upper, landing_rates, 0.0, 1.0, landing_rate_color)

        score_min = min(best_scores + mean_scores + [0.0])
        score_max = max(best_scores + mean_scores + [1.0])
        if math.isclose(score_min, score_max):
            score_max = score_min + 1.0
        pad = (score_max - score_min) * 0.08
        self._draw_series(
            lower,
            best_scores,
            score_min - pad,
            score_max + pad,
            best_score_color,
        )
        self._draw_series(
            lower,
            mean_scores,
            score_min - pad,
            score_max + pad,
            mean_score_color,
        )

        self.create_text(18, upper[1], text="100%", fill="#8db3d1", font=("Segoe UI", 9), anchor="w")
        self.create_text(18, upper[3] - 2, text="0%", fill="#8db3d1", font=("Segoe UI", 9), anchor="sw")
        self.create_text(18, lower[1], text=f"{score_max + pad:.0f}", fill="#8db3d1", font=("Segoe UI", 9), anchor="w")
        self.create_text(18, lower[3] - 2, text=f"{score_min - pad:.0f}", fill="#8db3d1", font=("Segoe UI", 9), anchor="sw")
        self.create_text(20, upper[1] - 14, text="Landing rate", fill="#72f7ff", font=("Consolas", 10, "bold"), anchor="w")
        self.create_text(20, lower[1] - 14, text="Generation best and mean score", fill="#72f7ff", font=("Consolas", 10, "bold"), anchor="w")

        legend = [
            ("Landing rate", landing_rate_color, f"{landing_rates[-1] * 100:.1f}%"),
            ("Best score", best_score_color, f"{best_scores[-1]:.1f}"),
            ("Mean score", mean_score_color, f"{mean_scores[-1]:.1f}"),
        ]
        self.create_rectangle(width - 220, 12, width - 16, 80, fill="#0b1630", outline="#244c88")
        for index, (label, color, value) in enumerate(legend):
            y = 28 + index * 18
            self.create_oval(width - 208, y - 4, width - 200, y + 4, fill=color, outline="")
            self.create_text(width - 194, y, text=label, fill="#cfe7ff", font=("Segoe UI", 9), anchor="w")
            self.create_text(width - 26, y, text=value, fill="#cfe7ff", font=("Segoe UI", 9), anchor="e")

    def _draw_plot_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.create_rectangle(x0, y0, x1, y1, fill="#081121", outline="#244c88")

    def _draw_grid(self, x0: float, y0: float, x1: float, y1: float) -> None:
        for index in range(1, 4):
            y = y0 + (y1 - y0) * index / 4
            self.create_line(x0, y, x1, y, fill="#16365e", dash=(4, 3))

    def _draw_series(
        self,
        rect,
        values: list[float],
        minimum: float,
        maximum: float,
        color: str,
    ) -> None:
        x0, y0, x1, y1 = rect
        if len(values) == 1:
            values = [values[0], values[0]]
        points = []
        for index, value in enumerate(values):
            x = x0 + (x1 - x0) * index / max(len(values) - 1, 1)
            norm = (value - minimum) / max(maximum - minimum, 1e-6)
            y = y1 - norm * (y1 - y0)
            points.extend([x, y])
        self.create_line(*points, fill=color, width=3, smooth=True)
        for index in range(0, len(points), 2):
            self.create_oval(
                points[index] - 3,
                points[index + 1] - 3,
                points[index] + 3,
                points[index + 1] + 3,
                fill=color,
                outline="",
            )


class NetworkCanvas(tk.Canvas):
    def __init__(self, master) -> None:
        super().__init__(master, highlightthickness=0, bg="#0b1723")
        self.visual_data: dict[str, Any] | None = None
        self.title = "Active Brain"
        self._visual_signature: Any = None
        self._layout_cache_key: Any = None
        self._layout_cache: dict[str, Any] = {}
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_visualization(self, visual_data: dict[str, Any] | None, title: str) -> None:
        signature = self._build_visual_signature(visual_data)
        if signature == self._visual_signature and title == self.title:
            return
        self.visual_data = visual_data
        self.title = title
        self._visual_signature = signature
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 10)
        height = max(self.winfo_height(), 10)
        for index in range(26):
            color = lerp_color("#060d18", "#101d35", index / 25)
            self.create_rectangle(
                0,
                height * index / 26,
                width,
                height * (index + 1) / 26,
                fill=color,
                outline=color,
            )
        self.create_text(
            18,
            22,
            text=self.title,
            fill="#72f7ff",
            font=("Consolas", 12, "bold"),
            anchor="w",
        )

        if not self.visual_data:
            self.create_text(
                width / 2,
                height / 2,
                text="The actor network will appear here once a brain is available.",
                fill="#8aa5c8",
                font=("Segoe UI", 11),
                width=width * 0.8,
                justify="center",
            )
            return

        layer_sizes = list(self.visual_data["layer_sizes"])
        weight_matrices = list(self.visual_data["weights"])
        activations = list(self.visual_data["activations"])
        input_names = list(self.visual_data.get("input_names", []))
        output_names = list(self.visual_data.get("output_names", []))
        log_std = list(self.visual_data.get("log_std", []))

        layout = self._get_layout(width, height, layer_sizes)
        layers = layout["layers"]
        x_positions = layout["x_positions"]
        left = layout["left"]
        right = layout["right"]
        max_abs = 1e-6
        for matrix in weight_matrices:
            max_abs = max(max_abs, abs(matrix).max())

        for layer_index, matrix in enumerate(weight_matrices):
            for out_idx, right_point in enumerate(layers[layer_index + 1]):
                for in_idx, left_point in enumerate(layers[layer_index]):
                    weight = float(matrix[out_idx][in_idx])
                    norm = abs(weight) / max_abs
                    color = "#4af7ff" if weight >= 0 else "#ff74f7"
                    width_px = 0.6 + 2.8 * (norm ** 1.3)
                    kwargs = {
                        "fill": color,
                        "width": width_px,
                        "smooth": True,
                        "splinesteps": 12,
                    }
                    if norm <= 0.35:
                        kwargs["stipple"] = "gray50"
                    self.create_line(
                        left_point[0],
                        left_point[1],
                        right_point[0],
                        right_point[1],
                        **kwargs,
                    )

        radius = layout["radius"]
        for layer_index, layer_points in enumerate(layers):
            for node_index, (x, y) in enumerate(layer_points):
                glow = radius + 3
                glow_fill = (
                    "#123d4d"
                    if layer_index == 0
                    else "#4b214d"
                    if layer_index == len(layers) - 1
                    else "#18355d"
                )
                self.create_oval(
                    x - glow,
                    y - glow,
                    x + glow,
                    y + glow,
                    fill=glow_fill,
                    outline="",
                )
                fill = (
                    "#2ae8ff"
                    if layer_index == 0
                    else "#ff74f7"
                    if layer_index == len(layers) - 1
                    else "#d7edff"
                )
                self.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline="#f3fbff",
                )
                if layer_index == 0 and node_index < len(input_names):
                    self.create_text(
                        left - 12,
                        y,
                        text=input_names[node_index],
                        fill="#cce4ff",
                        font=("Segoe UI", 8),
                        anchor="e",
                    )
                if layer_index == len(layers) - 1 and node_index < len(output_names):
                    self.create_text(
                        right + 12,
                        y,
                        text=output_names[node_index],
                        fill="#ffd3fb",
                        font=("Segoe UI", 8),
                        anchor="w",
                    )

        for index, activation in enumerate(activations):
            if index + 1 >= len(x_positions):
                continue
            x = (x_positions[index] + x_positions[index + 1]) / 2
            self.create_rectangle(
                x - 32,
                34,
                x + 32,
                56,
                fill="#0f1c34",
                outline="#2eeeff",
            )
            self.create_text(
                x,
                45,
                text=activation,
                fill="#e8fbff",
                font=("Consolas", 9, "bold"),
            )

        topology = " -> ".join(str(size) for size in layer_sizes)
        self.create_text(
            18,
            height - 38,
            text=f"Topology: {topology}",
            fill="#b9d7ff",
            font=("Consolas", 9),
            anchor="w",
        )
        if log_std:
            log_std_text = ", ".join(f"{value:.2f}" for value in log_std[:2])
            self.create_text(
                18,
                height - 18,
                text=f"Policy log std: {log_std_text}",
                fill="#b9d7ff",
                font=("Consolas", 9),
                anchor="w",
            )

    def _build_visual_signature(
        self,
        visual_data: dict[str, Any] | None,
    ) -> Any:
        if visual_data is None:
            return None
        weights = visual_data.get("weights", [])
        return (
            tuple(visual_data.get("layer_sizes", [])),
            tuple(matrix.shape for matrix in weights),
            tuple(hash(matrix.tobytes()) for matrix in weights),
            tuple(visual_data.get("activations", [])),
            tuple(visual_data.get("log_std", [])),
        )

    def _get_layout(
        self,
        width: int,
        height: int,
        layer_sizes: list[int],
    ) -> dict[str, Any]:
        key = (width, height, tuple(layer_sizes))
        if key == self._layout_cache_key:
            return self._layout_cache

        left = 92
        top = 84
        right = width - 92
        bottom = height - 72
        x_positions = [
            left + (right - left) * index / max(len(layer_sizes) - 1, 1)
            for index in range(len(layer_sizes))
        ]
        max_nodes = max(layer_sizes)

        def positions(count: int, x: float) -> list[tuple[float, float]]:
            if count == 1:
                return [(x, (top + bottom) / 2)]
            spacing = (bottom - top) / max(count - 1, 1)
            return [(x, top + idx * spacing) for idx in range(count)]

        layers = [
            positions(size, x_positions[index])
            for index, size in enumerate(layer_sizes)
        ]
        radius = max(4, min(10, int((bottom - top) / max_nodes * 0.24)))
        self._layout_cache_key = key
        self._layout_cache = {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "x_positions": x_positions,
            "layers": layers,
            "radius": radius,
        }
        return self._layout_cache
