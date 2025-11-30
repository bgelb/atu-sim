from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .tuning_algos.types import Topology

# Color scale shared by VSWR maps
_BOUNDS = [
    1.0,
    1.2,
    1.3,
    1.4,
    1.5,
    1.7,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.5,
    8.0,
    9.5,
    11.0,
]
_COLORS = [
    "#006d2c",  # <1.2
    "#238b45",  # <1.3
    "#41ae76",  # <1.4
    "#66c2a4",  # <1.5
    "#8dd3c7",  # <1.7
    "#fee391",  # <2.0
    "#fec44f",  # <2.5
    "#fe9929",  # <3.0
    "#fdae61",  # <4.0
    "#f46d43",  # <5.0
    "#d53e4f",  # <6.5
    "#9e0142",  # <8.0
    "#542788",  # <9.5
    "#d9d9d9",  # >=9.5 up to 11 (light gray)
]


def fmt_swr(swr: float) -> str:
    if np.isinf(swr):
        return ">= 9.99"
    return f"{swr:.2f}"


def new_plot(title: str, sw: int):
    fig, ax = plt.subplots(figsize=(16, 14), dpi=150)
    ax.set_title(f"{title} (SW={sw})")
    ax.set_xlabel("Capacitor bitmask (0-127)")
    ax.set_ylabel("Inductor bitmask (0-127)")
    return fig, ax


def plot_vswr_map(ax, swr_map: Sequence[Sequence[float]]):
    data = np.array(swr_map, dtype=float)
    cmap = plt.matplotlib.colors.ListedColormap(_COLORS)
    norm = plt.matplotlib.colors.BoundaryNorm(_BOUNDS, cmap.N)
    im = ax.imshow(data, origin="upper", cmap=cmap, norm=norm, aspect="equal")
    ticks = [1.1, 1.25, 1.35, 1.45, 1.55, 1.8, 2.25, 2.75, 3.5, 4.5, 5.75, 7.25, 9.0, 10.0]
    tick_labels = [
        "<1.2",
        "<1.3",
        "<1.4",
        "<1.5",
        "<1.7",
        "<2.0",
        "<2.5",
        "<3.0",
        "<4.0",
        "<5.0",
        "<6.5",
        "<8.0",
        "<9.5",
        ">=9.5",
    ]
    cbar = ax.figure.colorbar(im, ax=ax, boundaries=_BOUNDS, ticks=ticks, label="SWR")
    cbar.ax.set_yticklabels(tick_labels)
    return im


def overlay_trace(ax, trace: Iterable, sw: int, final_state: tuple[int, int] | None = None):
    trace_sw = [t for t in trace if getattr(t, "sw", None) == sw]
    if not trace_sw and final_state is None:
        return

    # Basic path overlay
    if trace_sw:
        xs = [t.c_bits for t in trace_sw]
        ys = [t.l_bits for t in trace_sw]
        ax.plot(xs, ys, color="white", linewidth=1.0, alpha=0.8, marker="o", markersize=3)
        ax.scatter(xs[:1], ys[:1], marker="s", color="cyan", edgecolors="black", s=30, label="start", alpha=0.9)

    if final_state:
        ax.scatter(
            final_state[0],
            final_state[1],
            marker="*",
            color="red",
            edgecolors="black",
            s=80,
            label="final",
            zorder=6,
        )
    if trace_sw or final_state:
        ax.legend(loc="upper right")


def save_plot(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
