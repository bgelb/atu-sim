from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .core import (
    LCBank,
    SimFlags,
    TunerSim,
    brute_force_best,
    l_network_input_impedance,
    swr_from_z,
    swr_grid,
)


TABLE1 = [
    ("3.6 MHz", 3.6e6, complex(25, -615)),
    ("3.9 MHz", 3.9e6, complex(30, -500)),
    ("7.0 MHz", 7.0e6, complex(185, 510)),
    ("10.1 MHz", 10.1e6, complex(3360, 2245)),
    ("14.0 MHz", 14.0e6, complex(155, -805)),
]

TABLE3 = [
    ("3.6 MHz (70 ft)", 3.6e6, complex(30, -610)),
    ("3.9 MHz (70 ft)", 3.9e6, complex(35, -495)),
    ("7.0 MHz (70 ft)", 7.0e6, complex(165, 485)),
    ("10.1 MHz (70 ft)", 10.1e6, complex(3810, 2160)),
    ("14.0 MHz (70 ft)", 14.0e6, complex(155, -820)),
]


def fmt_swr(swr_int: int) -> str:
    if swr_int >= 999:
        return ">= 9.99"
    return f"{swr_int / 100.0:.2f}"


def _safe_label(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label)


def plot_swr_grid(
    grid: list[list[int]],
    label: str,
    sw: int,
    out_dir: Path,
    trace: list[dict] | None = None,
    final_state: tuple[int, int] | None = None,
    coarse_best: tuple[int, int] | None = None,
) -> Path:
    """
    Render a 128x128 SWR grid to a PNG file using matplotlib.
    Rows = inductor bitmask (0..127), cols = capacitor bitmask (0..127).
    """
    data = np.array(grid, dtype=float) / 100.0  # convert to SWR ratio
    bounds = [
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
    colors = [
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
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(16, 14), dpi=150)  # larger canvas; cells appear larger
    im = ax.imshow(data, origin="upper", cmap=cmap, norm=norm, aspect="equal")
    ax.set_title(f"{label} (SW={sw})")
    ax.set_xlabel("Capacitor bitmask (0-127)")
    ax.set_ylabel("Inductor bitmask (0-127)")
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
    cbar = fig.colorbar(im, ax=ax, boundaries=bounds, ticks=ticks, label="SWR")
    cbar.ax.set_yticklabels(tick_labels)

    if trace:
        trace_sw = [t for t in trace if t["SW"] == sw]
        coarse_pts = [
            (t["cap"], t["ind"])
            for t in trace_sw
            if "coarse" in t["phase"]
        ]
        start_pts = [
            (t["cap"], t["ind"])
            for t in trace_sw
            if t["phase"] in ("reset", "tune_start", "bg_start")
        ]
        sharp_pts = [
            (t["cap"], t["ind"])
            for t in trace_sw
            if "coarse" not in t["phase"] and t["phase"] not in ("reset", "tune_start", "bg_start")
        ]

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
        if sharp_pts:
            ax.scatter(
                [p[0] for p in sharp_pts],
                [p[1] for p in sharp_pts],
                marker="o",
                color="white",
                edgecolors="black",
                s=18,
                label="sharp steps",
                alpha=0.9,
            )
            for (x0, y0), (x1, y1) in zip(sharp_pts[:-1], sharp_pts[1:]):
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
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

        if coarse_pts or sharp_pts or start_pts:
            ax.legend(loc="upper right")

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"swr_grid_{_safe_label(label)}_sw{sw}.png"
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def continuous_input_impedance(
    freq_hz: float, z_load: complex, L: float, C: float, sw: int
) -> complex:
    w = 2 * math.pi * freq_hz
    j = 1j

    z_L_series = j * w * L if L > 0 else 0j
    y_C = j * w * C if C > 0 else 0j

    if sw == 0:
        y_total = (1.0 / z_load) + y_C
        if abs(y_total) < 1e-18:
            z_node = complex(1e12, 0)
        else:
            z_node = 1.0 / y_total
        return z_L_series + z_node

    z_series = z_L_series + z_load
    y_series = 1.0 / z_series if abs(z_series) > 1e-18 else 0j
    y_in = y_series + y_C
    if abs(y_in) < 1e-18:
        return complex(1e12, 0)
    return 1.0 / y_in


def solve_series_then_shunt(
    freq_hz: float, z_load: complex, z0: float
) -> dict | None:
    R = z_load.real
    X = z_load.imag
    if R <= 0 or R > z0:
        return None
    w = 2 * math.pi * freq_hz
    term = (R * (z0 - R))
    if term < 0:
        return None
    Xs = math.sqrt(term) - X  # choose branch that yields Bc > 0
    X_tot = X + Xs
    denom = (R * R) + (X_tot * X_tot)
    G = R / denom
    Bc = (X + Xs) / denom
    if Bc <= 0 or abs(G - (1.0 / z0)) > 1e-9:
        return None
    L = Xs / w
    C = Bc / w
    return {"sw": 0, "L": L, "C": C}


def solve_shunt_then_series(
    freq_hz: float, z_load: complex, z0: float
) -> dict | None:
    R = z_load.real
    X = z_load.imag
    if R <= 0 or R < z0:
        return None
    w = 2 * math.pi * freq_hz
    denom_base = (R * R) + (X * X)
    G = R / denom_base
    B_load = -X / denom_base
    delta = (G / z0) - (G * G)
    if delta < 0:
        return None
    B_target = math.sqrt(delta)
    Bc = B_target - B_load
    if Bc < 0:
        return None
    Xs = B_target * z0 / G
    if Xs <= 0:
        return None
    L = Xs / w
    C = Bc / w
    return {"sw": 1, "L": L, "C": C}


def find_ideal_match(freq_hz: float, z_load: complex, z0: float) -> dict:
    """
    Analytic L-network match (both topologies), unconstrained by relay steps.
    """
    R = z_load.real
    X = z_load.imag
    w = 2 * math.pi * freq_hz
    candidates: list[dict] = []

    # sw=0 in simulator: shunt first (at load), then series
    denom_base = (R * R) + (X * X)
    if denom_base > 0:
        G = R / denom_base
        B_load = -X / denom_base
        if 0 < G < (1.0 / z0):
            target = (G - z0 * G * G) / z0
            if target >= 0:
                root = math.sqrt(target)
                for sign in (1.0, -1.0):
                    B_total = sign * root
                    Bc = B_total - B_load
                    if Bc == 0:
                        continue
                    Xs = B_total / (G * G + B_total * B_total)
                    L_series = Xs / w
                    C_shunt = Bc / w  # negative => shunt L
                    z_in = continuous_input_impedance(freq_hz, z_load, L_series, C_shunt, 0)
                    swr = swr_from_z(z_in, z0)
                    candidates.append(
                        {"sw": 0, "L": L_series, "C": C_shunt, "z_in": z_in, "swr": swr}
                    )

    # sw=1 in simulator: series first, then shunt at input
    if R > 0 and R < z0:
        term = (R * z0) - (R * R)
        if term >= 0:
            root = math.sqrt(term)
            for sign in (1.0, -1.0):
                Xs = -X + sign * root
                X_tot = X + Xs
                denom = (R * R) + (X_tot * X_tot)
                if denom == 0:
                    continue
                Bc = X_tot / denom  # shunt susceptance
                if Bc == 0:
                    continue
                L_series = Xs / w
                C_shunt = Bc / w
                z_in = continuous_input_impedance(freq_hz, z_load, L_series, C_shunt, 1)
                swr = swr_from_z(z_in, z0)
                candidates.append(
                    {"sw": 1, "L": L_series, "C": C_shunt, "z_in": z_in, "swr": swr}
                )

    if not candidates:
        z_in = z_load
        return {"sw": 0, "L": 0.0, "C": 0.0, "z_in": z_in, "swr": swr_from_z(z_in, z0)}

    return min(candidates, key=lambda c: c["swr"])


def run_table(
    table, title: str, flags: SimFlags | None = None, return_last_sim: bool = False
) -> TunerSim | None:
    if flags is None:
        flags = SimFlags()

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"SimFlags: {flags}")
    freq_width = max(len("Freq"), max(len(label) for label, _, _ in table))
    load_width = 18
    ideal_width = 88
    best_width = 88
    algo_width = 88
    print(
        f"{'Freq':>{freq_width}} | {'Z_load (R+jX)':>{load_width}} | "
        f"{'Ideal (L/C, topology, Zin, SWR)':<{ideal_width}}| "
        f"{'Best discrete (L/C, topology, Zin, SWR)':<{best_width}}| "
        f"{'Tune algo (L/C, topology, Zin, SWR)':<{algo_width}}"
    )
    print("-" * (freq_width + 3 + load_width + 3 + ideal_width + 2 + best_width + 2 + algo_width))

    last_sim: TunerSim | None = None

    for label, freq, zL in table:
        sim = TunerSim(freq_hz=freq, z_load=zL, flags=flags)
        sim.atu_reset()
        last_sim = sim

        best_swr, best_state = brute_force_best(freq, zL)
        sim.tune()
        bank: LCBank = sim.bank

        def topo_label(sw: int, C_val: float) -> str:
            shunt = "shunt-C" if C_val > 0 else ("open" if C_val == 0 else "shunt-L")
            return f"{shunt}@{'load' if sw == 0 else 'input'}"

        ideal = find_ideal_match(freq, zL, sim.z0)
        z_ideal = ideal["z_in"]
        ideal_str = (
            f"L={ideal['L']*1e6:8.3f}u "
            f"C={ideal['C']*1e12:8.1f}p "
            f"{topo_label(ideal['sw'], ideal['C']):>12} "
            f"Zin={z_ideal.real:9.2f}+j{z_ideal.imag:9.2f} "
            f"SWR={fmt_swr(ideal['swr']):>6}"
        )

        z_best = l_network_input_impedance(
            freq, zL, best_state[0], best_state[1], best_state[2], bank
        )
        best_str = (
            f"L={bank.l_from_bits(best_state[0])*1e6:8.3f}u "
            f"C={bank.c_from_bits(best_state[1])*1e12:8.1f}p "
            f"{topo_label(best_state[2], bank.c_from_bits(best_state[1])):>12} "
            f"Zin={z_best.real:9.2f}+j{z_best.imag:9.2f} "
            f"SWR={fmt_swr(best_swr):>6}"
        )

        z_alg = l_network_input_impedance(freq, zL, sim.ind, sim.cap, sim.SW)
        algo_str = (
            f"L={bank.l_from_bits(sim.ind)*1e6:8.3f}u "
            f"C={bank.c_from_bits(sim.cap)*1e12:8.1f}p "
            f"{topo_label(sim.SW, bank.c_from_bits(sim.cap)):>12} "
            f"Zin={z_alg.real:9.2f}+j{z_alg.imag:9.2f} "
            f"SWR={fmt_swr(sim.SWR):>6}"
        )

        sign = "+" if zL.imag >= 0 else "-"
        load_str = f"{zL.real:7.0f} {sign} j{abs(zL.imag):5.0f}"
        print(
            f"{label:>{freq_width}} | "
            f"{load_str:>{load_width}} | "
            f"{ideal_str:<{ideal_width}}| "
            f"{best_str:<{best_width}}| "
            f"{algo_str:<{algo_width}}"
        )

    if return_last_sim:
        return last_sim
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ATU-10 firmware tuning on the Cebik 88' doublet tables."
    )
    parser.add_argument(
        "--list-flags",
        action="store_true",
        help="List available SimFlags, show defaults, and exit.",
    )
    parser.add_argument(
        "--force-all-coarse-strategies",
        action="store_true",
        help="Always run coarse strategies 2/3 (default matches firmware gating).",
    )
    parser.add_argument(
        "--debug-coarse",
        action="store_true",
        help="Print coarse tuning step-by-step details from the simulator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write SWR grid PNGs (default: ./plots).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating SWR grid PNGs.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["bg", "atu10"],
        default="bg",
        help="Tuning algorithm to use (default: bg).",
    )
    args = parser.parse_args()

    if args.list_flags:
        defaults = SimFlags()
        print("Available SimFlags (defaults in brackets):")
        print(
            f"  force_all_coarse_strategies [{defaults.force_all_coarse_strategies}] - "
            "always run coarse strategies 2/3 even if gated."
        )
        print(
            f"  debug_coarse [{defaults.debug_coarse}] - "
            "emit coarse tuning step-by-step debug logs."
        )
        print(
            f"  trace_steps [{defaults.trace_steps}] - "
            "record every relay-set step (for plotting the tuning path)."
        )
        return

    flags = SimFlags(
        force_all_coarse_strategies=args.force_all_coarse_strategies,
        debug_coarse=args.debug_coarse,
        algorithm=args.algorithm,
    )
    default_flags = SimFlags(algorithm=args.algorithm)
    run_table(TABLE1, "Cebik Table 1 - Free-space 88' doublet", default_flags)
    run_table(TABLE3, "Cebik Table 3 - 70 ft high 88' doublet", default_flags)

    detailed_label = "7.0 MHz (70 ft)"
    detailed_entry = next(item for item in TABLE3 if item[0] == detailed_label)
    analysis_flags = SimFlags(
        force_all_coarse_strategies=args.force_all_coarse_strategies,
        debug_coarse=args.debug_coarse,
        trace_steps=True,
        algorithm=args.algorithm,
    )
    sim_detail = run_table(
        [detailed_entry],
        "Detailed analysis - 7.0 MHz (70 ft)",
        analysis_flags,
        return_last_sim=True,
    )

    if args.skip_plots:
        return

    out_dir = args.output_dir
    plotting_flags = SimFlags(
        force_all_coarse_strategies=args.force_all_coarse_strategies,
        debug_coarse=args.debug_coarse,
        trace_steps=True,
        algorithm=args.algorithm,
    )
    print("Generating SWR grid PNGs for all table entries...")
    for label, freq, zL in TABLE1 + TABLE3:
        sim_plot = TunerSim(freq_hz=freq, z_load=zL, flags=plotting_flags)
        sim_plot.atu_reset()
        sim_plot.tune()
        grids = swr_grid(freq, zL)
        trace = sim_plot.trace
        final_state_sw0 = (
            (sim_plot.cap, sim_plot.ind) if sim_plot.SW == 0 else None
        )
        final_state_sw1 = (
            (sim_plot.cap, sim_plot.ind) if sim_plot.SW == 1 else None
        )
        coarse_best = None
        # best coarse was logged as bg_coarse_best; find it
        for t in trace:
            if t["phase"] == "bg_coarse_best":
                cb = (t["cap"], t["ind"])
                if t["SW"] == 0:
                    coarse_best = ("sw0", cb)
                else:
                    coarse_best = ("sw1", cb)
                break
        coarse_best_sw0 = coarse_best[1] if coarse_best and coarse_best[0] == "sw0" else None
        coarse_best_sw1 = coarse_best[1] if coarse_best and coarse_best[0] == "sw1" else None

        path0 = plot_swr_grid(
            grids[0],
            label,
            0,
            out_dir,
            trace=trace,
            final_state=final_state_sw0,
            coarse_best=coarse_best_sw0,
        )
        path1 = plot_swr_grid(
            grids[1],
            label,
            1,
            out_dir,
            trace=trace,
            final_state=final_state_sw1,
            coarse_best=coarse_best_sw1,
        )
        print(f"  {label}: {path0}, {path1}")


if __name__ == "__main__":
    main()
