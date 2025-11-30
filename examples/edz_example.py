from __future__ import annotations

import argparse
from pathlib import Path

from atu_sim.cebik_tables import TABLE1, TABLE3
from atu_sim.detectors import ATU10IntegerVSWRDetector
from atu_sim.hardware import atu10_bank
from atu_sim.lc_bank import LCBank, ShuntPosition
from atu_sim.plotting import fmt_swr, new_plot, overlay_trace, plot_vswr_map, save_plot
from atu_sim.simulator import ATUSimulator
from atu_sim.tuning_algos.atu10_reference import ATU10ReferenceAlgo
from atu_sim.tuning_algos.bg_algo import BGAlgo
from atu_sim.tuning_algos.types import Topology


def _build_algo(name: str, bank, detector):
    name_l = name.lower()
    if name_l == "bg":
        return BGAlgo(bank, detector)
    if name_l == "atu10":
        return ATU10ReferenceAlgo(bank, detector)
    raise ValueError(f"Unknown algorithm {name}")


def topo_label(topo: Topology, c_val: float) -> str:
    shunt = "shunt-C" if c_val > 0 else ("open" if c_val == 0 else "shunt-L")
    return f"{shunt}@{'load' if topo == Topology.SHUNT_AT_LOAD else 'input'}"


def run_table(table, title: str, algo_name: str, bank: LCBank, detector: ATU10IntegerVSWRDetector) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Algorithm: {algo_name}")
    freq_width = max(len("Freq"), max(len(label) for label, _, _ in table))
    load_width = 18
    best_width = 88
    algo_width = 88
    print(
        f"{'Freq':>{freq_width}} | {'Z_load (R+jX)':>{load_width}} | "
        f"{'Best discrete (L/C, topology, Zin, SWR)':<{best_width}}| "
        f"{'Tune algo (L/C, topology, Zin, SWR)':<{algo_width}}"
    )
    print("-" * (freq_width + 3 + load_width + 3 + best_width + 2 + algo_width))

    for label, freq, zL in table:
        algo = _build_algo(algo_name, bank, detector)
        sim = ATUSimulator(algorithm=algo)
        result = sim.tune(freq, zL)

        best_swr, best_l, best_c, best_pos = bank.get_best_swr(freq, zL, z0=50.0)
        z_best = bank.input_impedance(freq, zL, best_l, best_c, best_pos)
        best_str = (
            f"L={bank.l_from_bits(best_l)*1e6:8.3f}u "
            f"C={bank.c_from_bits(best_c)*1e12:8.1f}p "
            f"{topo_label(Topology.SHUNT_AT_LOAD if best_pos == ShuntPosition.LOAD else Topology.SHUNT_AT_SOURCE, bank.c_from_bits(best_c)):>12} "
            f"Zin={z_best.real:9.2f}+j{z_best.imag:9.2f} "
            f"SWR={fmt_swr(best_swr):>6}"
        )

        z_alg = result.final_z_in
        algo_str = (
            f"L={bank.l_from_bits(result.final_config.l_bits)*1e6:8.3f}u "
            f"C={bank.c_from_bits(result.final_config.c_bits)*1e12:8.1f}p "
            f"{topo_label(result.final_config.topology, bank.c_from_bits(result.final_config.c_bits)):>12} "
            f"Zin={z_alg.real:9.2f}+j{z_alg.imag:9.2f} "
            f"SWR={fmt_swr(result.final_swr / 100.0 if isinstance(result.final_swr, int) else result.final_swr):>6}"
        )

        sign = "+" if zL.imag >= 0 else "-"
        load_str = f"{zL.real:7.0f} {sign} j{abs(zL.imag):5.0f}"
        print(
            f"{label:>{freq_width}} | "
            f"{load_str:>{load_width}} | "
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
        algo = _build_algo(args.algorithm, bank, detector)
        sim = ATUSimulator(algorithm=algo)
        result = sim.tune(freq, zL)
        grids = {
            0: bank.get_swr_map(freq, zL, ShuntPosition.LOAD),
            1: bank.get_swr_map(freq, zL, ShuntPosition.SOURCE),
        }
        trace = result.trace
        final_state = (result.final_config.c_bits, result.final_config.l_bits)
        for sw in (0, 1):
            fig, ax = new_plot(label, sw)
            plot_vswr_map(ax, grids[sw])
            final_for_sw = (
                final_state
                if (sw == 0 and result.final_config.topology == Topology.SHUNT_AT_LOAD)
                or (sw == 1 and result.final_config.topology == Topology.SHUNT_AT_SOURCE)
                else None
            )
            overlay_trace(ax, trace, sw=sw, final_state=final_for_sw)
            out_path = out_dir / f"swr_map_{label.replace(' ', '_')}_sw{sw}.png"
            save_plot(fig, out_path)


if __name__ == "__main__":
    main()
