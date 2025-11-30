from __future__ import annotations

import argparse
from pathlib import Path

from atu10_firmware_sim.cebik_tables import TABLE1, TABLE3
from atu10_firmware_sim.detectors import ATU10IntegerVSWRDetector
from atu10_firmware_sim.hardware import atu10_bank
from atu10_firmware_sim.lc_bank import ShuntPosition
from atu10_firmware_sim.plotting import fmt_swr, new_plot, overlay_trace, plot_vswr_map, save_plot
from atu10_firmware_sim.simulator import ATUSimulator
from atu10_firmware_sim.tuning_algos.atu10_reference import ATU10ReferenceAlgo
from atu10_firmware_sim.tuning_algos.bg_algo import BGAlgo
from atu10_firmware_sim.tuning_algos.types import Topology


def _build_algo(name: str, bank, detector):
    name_l = name.lower()
    if name_l == "bg":
        return BGAlgo(bank, detector)
    if name_l == "atu10":
        return ATU10ReferenceAlgo(bank, detector)
    raise ValueError(f"Unknown algorithm {name}")


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

    print(f"Algorithm: {args.algorithm}")
    for table_name, table in (("Table 1", TABLE1), ("Table 3", TABLE3)):
        print()
        print(f"=== {table_name} ===")
        for label, freq, z_load in table:
            algo = _build_algo(args.algorithm, bank, detector)
            sim = ATUSimulator(algorithm=algo)
            result = sim.tune(freq, z_load)
            swr_val = fmt_swr(result.final_swr / 100.0 if isinstance(result.final_swr, int) else result.final_swr)
            print(f"{label:20s} load={z_load.real:+7.0f}+j{z_load.imag:+7.0f}  final SWR={swr_val}")

            if args.skip_plots:
                continue

            trace = result.trace
            final_state = (result.final_config.c_bits, result.final_config.l_bits)
            for sw, shunt_pos in ((0, ShuntPosition.LOAD), (1, ShuntPosition.SOURCE)):
                swr_map = bank.get_swr_map(freq, z_load, shunt_pos)
                fig, ax = new_plot(label, sw)
                plot_vswr_map(ax, swr_map)
                overlay_trace(ax, trace, sw=sw, final_state=final_state if (sw == 0 and result.final_config.topology == Topology.SHUNT_AT_LOAD) or (sw == 1 and result.final_config.topology == Topology.SHUNT_AT_SOURCE) else None)
                out_path = args.output_dir / f"swr_map_{label.replace(' ', '_')}_sw{sw}.png"
                save_plot(fig, out_path)


if __name__ == "__main__":
    main()
