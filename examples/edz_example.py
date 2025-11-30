from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from atu10_firmware_sim.cebik_tables import TABLE1, TABLE3
from atu10_firmware_sim.detectors import ATU10IntegerVSWRDetector
from atu10_firmware_sim.hardware import atu10_bank
from atu10_firmware_sim.lc_bank import LCBank, ShuntPosition
from atu10_firmware_sim.simulator import ATUSimulator
from atu10_firmware_sim.tuning_algos.atu10_reference import ATU10ReferenceAlgo
from atu10_firmware_sim.tuning_algos.bg_algo import BGAlgo
from atu10_firmware_sim.tuning_algos.types import Topology


def fmt_swr(swr_int: int) -> str:
    if swr_int >= 999:
        return ">= 9.99"
    return f"{swr_int / 100.0:.2f}"


def _safe_label(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label)


def swr_grid(bank: LCBank, detector: ATU10IntegerVSWRDetector, freq_hz: float, z_load: complex) -> dict[int, list[list[int]]]:
    grids = {0: [], 1: []}
    for sw in (0, 1):
        pos = ShuntPosition.LOAD if sw == 0 else ShuntPosition.SOURCE
        grid: list[list[int]] = []
        for ind in range(128):
            row: list[int] = []
            for cap in range(128):
                z_in = bank.input_impedance(freq_hz, z_load, ind, cap, pos)
                row.append(detector.measure(z_in))
            grid.append(row)
        grids[sw] = grid
    return grids


def brute_force_best(
    bank: LCBank, detector: ATU10IntegerVSWRDetector, freq_hz: float, z_load: complex
) -> tuple[int, tuple[int, int, int]]:
    best_swr = 999
    best_state = (0, 0, 0)
    for sw in (0, 1):
        pos = ShuntPosition.LOAD if sw == 0 else ShuntPosition.SOURCE
        for ind in range(128):
            for cap in range(128):
                z_in = bank.input_impedance(freq_hz, z_load, ind, cap, pos)
                swr = detector.measure(z_in)
                if swr < best_swr:
                    best_swr = swr
                    best_state = (ind, cap, sw)
    return best_swr, best_state


def plot_swr_grid(
    grid: list[list[int]],
    label: str,
    sw: int,
    out_dir: Path,
    trace: list[dict] | None = None,
    final_state: tuple[int, int] | None = None,
    coarse_best: tuple[int, int] | None = None,
    per_secondary: list[tuple[int, int]] | None = None,
) -> Path:
    data = np.array(grid, dtype=float) / 100.0
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

    fig, ax = plt.subplots(figsize=(16, 14), dpi=150)
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
        trace_sw = [t for t in trace if t["sw"] == sw]
        coarse_pts = [(t["cap"], t["ind"]) for t in trace_sw if "coarse" in t["phase"]]
        start_pts = [
            (t["cap"], t["ind"])
            for t in trace_sw
            if t["phase"] in ("bg_start",)
        ]
        sharp_pts = [
            (t["cap"], t["ind"])
            for t in trace_sw
            if "coarse" not in t["phase"] and t["phase"] not in ("bg_start",)
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

        if coarse_pts or sharp_pts or start_pts or per_secondary or final_state:
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


def find_ideal_match(
    freq_hz: float, z_load: complex, z0: float, detector: ATU10IntegerVSWRDetector
) -> dict:
    R = z_load.real
    X = z_load.imag
    w = 2 * math.pi * freq_hz
    candidates: list[dict] = []

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
                    swr = detector.swr_from_z(z_in, z0)
                    candidates.append(
                        {"sw": 0, "L": L_series, "C": C_shunt, "z_in": z_in, "swr": swr}
                    )

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
                Bc = X_tot / denom
                if Bc == 0:
                    continue
                L_series = Xs / w
                C_shunt = Bc / w
                z_in = continuous_input_impedance(freq_hz, z_load, L_series, C_shunt, 1)
                swr = detector.swr_from_z(z_in, z0)
                candidates.append(
                    {"sw": 1, "L": L_series, "C": C_shunt, "z_in": z_in, "swr": swr}
                )

    if not candidates:
        z_in = z_load
        return {"sw": 0, "L": 0.0, "C": 0.0, "z_in": z_in, "swr": detector.swr_from_z(z_in, z0)}

    return min(candidates, key=lambda c: c["swr"])


def topo_label(topo: Topology, c_val: float) -> str:
    shunt = "shunt-C" if c_val > 0 else ("open" if c_val == 0 else "shunt-L")
    return f"{shunt}@{'load' if topo == Topology.SHUNT_AT_LOAD else 'input'}"


def _build_algo(name: str, bank: LCBank, detector: ATU10IntegerVSWRDetector):
    name_l = name.lower()
    if name_l == "bg":
        return BGAlgo(bank, detector)
    if name_l == "atu10":
        return ATU10ReferenceAlgo(bank, detector)
    raise ValueError(f"Unknown algorithm {name}")


def run_table(table, title: str, algo: str, bank: LCBank, detector: ATU10IntegerVSWRDetector) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Algorithm: {algo}")
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

    for label, freq, zL in table:
        sim = ATUSimulator(algorithm=_build_algo(algo, bank, detector))
        result = sim.tune(freq, zL)

        ideal = find_ideal_match(freq, zL, 50.0, detector)
        z_ideal = ideal["z_in"]
        ideal_str = (
            f"L={ideal['L']*1e6:8.3f}u "
            f"C={ideal['C']*1e12:8.1f}p "
            f"{topo_label(Topology.SHUNT_AT_LOAD if ideal['sw']==0 else Topology.SHUNT_AT_SOURCE, ideal['C']):>12} "
            f"Zin={z_ideal.real:9.2f}+j{z_ideal.imag:9.2f} "
            f"SWR={fmt_swr(ideal['swr']):>6}"
        )

        best_swr, best_state = brute_force_best(bank, detector, freq, zL)
        z_best = bank.input_impedance(
            freq, zL, best_state[0], best_state[1], ShuntPosition.LOAD if best_state[2] == 0 else ShuntPosition.SOURCE
        )
        best_str = (
            f"L={bank.l_from_bits(best_state[0])*1e6:8.3f}u "
            f"C={bank.c_from_bits(best_state[1])*1e12:8.1f}p "
            f"{topo_label(Topology.SHUNT_AT_LOAD if best_state[2]==0 else Topology.SHUNT_AT_SOURCE, bank.c_from_bits(best_state[1])):>12} "
            f"Zin={z_best.real:9.2f}+j{z_best.imag:9.2f} "
            f"SWR={fmt_swr(best_swr):>6}"
        )

        z_alg = result.final_z_in
        algo_str = (
            f"L={bank.l_from_bits(result.final_config.l_bits)*1e6:8.3f}u "
            f"C={bank.c_from_bits(result.final_config.c_bits)*1e12:8.1f}p "
            f"{topo_label(result.final_config.topology, bank.c_from_bits(result.final_config.c_bits)):>12} "
            f"Zin={z_alg.real:9.2f}+j{z_alg.imag:9.2f} "
            f"SWR={fmt_swr(result.final_swr):>6}"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate tuner algorithms on the Cebik 88' doublet tables."
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

    bank = atu10_bank()
    detector = ATU10IntegerVSWRDetector()

    run_table(TABLE1, "Cebik Table 1 - Free-space 88' doublet", args.algorithm, bank, detector)
    run_table(TABLE3, "Cebik Table 3 - 70 ft high 88' doublet", args.algorithm, bank, detector)

    if args.skip_plots:
        return

    out_dir = args.output_dir
    print("Generating SWR grid PNGs for all table entries...")
    for label, freq, zL in TABLE1 + TABLE3:
        sim = ATUSimulator(algorithm=_build_algo(args.algorithm, bank, detector))
        result = sim.tune(freq, zL)
        grids = swr_grid(bank, detector, freq, zL)
        trace_dicts = [
            {
                "cap": t.c_bits,
                "ind": t.l_bits,
                "sw": t.sw,
                "phase": t.phase,
            }
            for t in result.trace
        ]
        coarse_best_by_sw: dict[int, tuple[int, int] | None] = {0: None, 1: None}
        per_secondary_by_sw: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
        for t in trace_dicts:
            if t["phase"] == "bg_coarse_best":
                coarse_best_by_sw[t["sw"]] = (t["cap"], t["ind"])
            if t["phase"] == "bg_sec_best":
                per_secondary_by_sw[t["sw"]].append((t["cap"], t["ind"]))
        final_state = (
            (result.final_config.c_bits, result.final_config.l_bits)
            if result.final_config.topology == Topology.SHUNT_AT_LOAD
            else (result.final_config.c_bits, result.final_config.l_bits)
        )
        path0 = plot_swr_grid(
            grids[0],
            label,
            0,
            out_dir,
            trace=trace_dicts,
            final_state=final_state if result.final_config.topology == Topology.SHUNT_AT_LOAD else None,
            coarse_best=coarse_best_by_sw[0],
            per_secondary=per_secondary_by_sw[0] if per_secondary_by_sw[0] else None,
        )
        path1 = plot_swr_grid(
            grids[1],
            label,
            1,
            out_dir,
            trace=trace_dicts,
            final_state=final_state if result.final_config.topology == Topology.SHUNT_AT_SOURCE else None,
            coarse_best=coarse_best_by_sw[1],
            per_secondary=per_secondary_by_sw[1] if per_secondary_by_sw[1] else None,
        )
        print(f"  {label}: {path0}, {path1}")


if __name__ == "__main__":
    main()
