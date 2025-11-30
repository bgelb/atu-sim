from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .tuning_algos.types import Topology, TuningPhase

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

    start_pts = []
    coarse_pts = []
    fine_pts = []
    coarse_best = None
    per_secondary: list[tuple[int, int]] = []

    for t in trace_sw:
        phase = getattr(t, "tuning_phase", None)
        if phase is None:
            continue
        point = (t.c_bits, t.l_bits)
        if phase in {TuningPhase.RESET, TuningPhase.START, TuningPhase.COARSE_START, TuningPhase.COARSE_RESET}:
            start_pts.append(point)
        if phase in {TuningPhase.COARSE_START, TuningPhase.COARSE_STEP, TuningPhase.COARSE_RESET, TuningPhase.COARSE_BEST}:
            coarse_pts.append(point)
            if phase == TuningPhase.COARSE_BEST:
                coarse_best = point
        elif phase == TuningPhase.SECONDARY_BEST:
            per_secondary.append(point)
        elif phase in {TuningPhase.FINE_START, TuningPhase.FINE_STEP, TuningPhase.FINE_BEST}:
            fine_pts.append(point)

    if coarse_pts:
        ax.scatter(
            [p[0] for p in coarse_pts],
            [p[1] for p in coarse_pts],
            marker="^",
            color="black",
            s=14,
            label="coarse steps",
            alpha=0.8,
        )
    if coarse_best:
        ax.scatter(
            coarse_best[0],
            coarse_best[1],
            marker="^",
            color="gold",
            edgecolors="black",
            s=40,
            label="best coarse",
            zorder=5,
        )
    if per_secondary:
        ax.scatter(
            [p[0] for p in per_secondary],
            [p[1] for p in per_secondary],
            marker="D",
            color="#ff69b4",
            edgecolors="black",
            s=26,
            label="best per secondary",
            alpha=0.9,
            zorder=5,
        )
    if start_pts:
        ax.scatter(
            [p[0] for p in start_pts],
            [p[1] for p in start_pts],
            marker="s",
            color="cyan",
            edgecolors="black",
            s=20,
            label="start/reset",
            alpha=0.8,
        )

    if fine_pts:
        ax.scatter(
            [p[0] for p in fine_pts],
            [p[1] for p in fine_pts],
            marker="o",
            color="white",
            edgecolors="black",
            s=18,
            label="fine steps",
            alpha=0.9,
        )
        for (x0, y0), (x1, y1) in zip(fine_pts[:-1], fine_pts[1:]):
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            )

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
    if coarse_pts or fine_pts or start_pts or per_secondary or final_state or coarse_best:
        ax.legend(loc="upper right")


def save_plot(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
