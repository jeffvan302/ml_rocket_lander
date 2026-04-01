from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def lerp_color(a: str, b: str, t: float) -> str:
    a = a.lstrip("#")
    b = b.lstrip("#")
    rgb = []
    for index in range(3):
        av = int(a[index * 2 : index * 2 + 2], 16)
        bv = int(b[index * 2 : index * 2 + 2], 16)
        value = int(av + (bv - av) * t)
        rgb.append(f"{value:02x}")
    return f"#{''.join(rgb)}"


class ScrollableFrame(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#efe6da")
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.inner,
            anchor="nw",
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind("<Configure>", self._resize_inner)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _resize_inner(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if not self.winfo_ismapped():
            return
        x = self.winfo_pointerx() - self.winfo_rootx()
        y = self.winfo_pointery() - self.winfo_rooty()
        if 0 <= x <= self.winfo_width() and 0 <= y <= self.winfo_height():
            self.canvas.yview_scroll(int(-event.delta / 120), "units")
